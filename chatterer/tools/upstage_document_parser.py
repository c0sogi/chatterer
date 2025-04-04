"""Adopted from`langchain_upstage.document_parse"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import TYPE_CHECKING, Iterator, Literal, Optional, cast

import requests
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from ..common_types.io import BytesReadable
from ..language_model import DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION, Chatterer
from ..utils.base64_image import Base64Image

if TYPE_CHECKING:
    from pypdf import PdfReader

logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)

DOCUMENT_PARSE_BASE_URL = "https://api.upstage.ai/v1/document-ai/document-parse"
DEFAULT_NUM_PAGES = 10
DOCUMENT_PARSE_DEFAULT_MODEL = "document-parse"

OutputFormat = Literal["text", "html", "markdown"]
OCR = Literal["auto", "force"]
SplitType = Literal["none", "page", "element"]
Category = Literal[
    "paragraph",
    "table",
    "figure",
    "header",
    "footer",
    "caption",
    "equation",
    "heading1",
    "list",
    "index",
    "footnote",
    "chart",
]


class Content(BaseModel):
    text: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None


class Coordinate(BaseModel):
    x: float
    y: float


class Element(BaseModel):
    category: Category
    content: Content
    coordinates: list[Coordinate] = Field(default_factory=list)
    base64_encoding: str = ""
    id: int
    page: int

    def parse_text(self, parser: "UpstageDocumentParseParser") -> str:
        output_format: OutputFormat = parser.output_format
        chatterer: Optional[Chatterer] = parser.chatterer
        image_description_instruction: str = parser.image_description_instruction
        output: Optional[str] = None
        if output_format == "text":
            output = self.content.text
        elif output_format == "html":
            output = self.content.html
        elif output_format == "markdown":
            output = self.content.markdown
        if output is None:
            raise ValueError(f"Invalid output format: {output_format}")

        if chatterer is not None and self.category == "figure" and self.base64_encoding:
            image = Base64Image.from_string(f"data:image/jpeg;base64,{self.base64_encoding}")
            if image is None:
                raise ValueError(f"Invalid base64 encoding for image: {self.base64_encoding}")
            ocr_content = output.removeprefix("![image](/image/placeholder)\n")
            image_description = chatterer.describe_image(
                image.data_uri,
                image_description_instruction
                + f"\nHint: The OCR detected the following text:\n```\n{ocr_content}\n```",
            )
            output = f"\n\n<details>\n{image_description}\n</details>\n\n"

        return output


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


class UpstageDocumentParseParser(BaseBlobParser):
    """Upstage Document Parse Parser.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageDocumentParseParser

            loader = UpstageDocumentParseParser(split="page", output_format="text")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DOCUMENT_PARSE_BASE_URL,
        model: str = DOCUMENT_PARSE_DEFAULT_MODEL,
        split: SplitType = "none",
        ocr: OCR = "auto",
        output_format: OutputFormat = "markdown",
        coordinates: bool = True,
        base64_encoding: list[Category] = [],
        chatterer: Optional[Chatterer] = None,
        image_description_instruction: str = DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION,
    ) -> None:
        """
        Initializes an instance of the Upstage class.

        Args:
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None, in which case it will be
                                     fetched from the environment variable
                                     `UPSTAGE_API_KEY`.
            base_url (str, optional): The base URL for accessing the Upstage API.
            model (str): The model to be used for the document parse.
                         Defaults to "document-parse".
            split (SplitType, optional): The type of splitting to be applied.
                                         Defaults to "none" (no splitting).
            ocr (OCRMode, optional): Extract text from images in the document using OCR.
                                     If the value is "force", OCR is used to extract
                                     text from an image. If the value is "auto", text is
                                     extracted from a PDF. (An error will occur if the
                                     value is "auto" and the input is NOT in PDF format)
            output_format (OutputFormat, optional): Format of the inference results.
            coordinates (bool, optional): Whether to include the coordinates of the
                                          OCR in the output.
            base64_encoding (List[Category], optional): The category of the elements to
                                                        be encoded in base64.
            chatterer (Chatterer, optional): The Chatterer instance to use for image
                                             description.
            image_description_instruction (str, optional): The instruction to use for
                                                           image description.


        """
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY",
            api_key,
            "UPSTAGE_API_KEY",
            os.environ.get("UPSTAGE_API_KEY"),
        )
        self.base_url = base_url
        self.model = model
        self.split: SplitType = split
        self.ocr: OCR = ocr
        self.output_format: OutputFormat = output_format
        self.coordinates = coordinates
        self.base64_encoding: list[Category] = base64_encoding
        self.chatterer = chatterer
        self.image_description_instruction = image_description_instruction

    def _get_response(self, files: dict[str, BytesReadable]) -> list[Element]:
        """
        Sends a POST request to the API endpoint with the provided files and
        returns the response.

        Args:
            files (dict): A dictionary containing the files to be sent in the request.

        Returns:
            dict: The JSON response from the API.

        Raises:
            ValueError: If there is an error in the API call.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data={
                    "ocr": self.ocr,
                    "model": self.model,
                    "output_formats": f"['{self.output_format}']",
                    "coordinates": self.coordinates,
                    "base64_encoding": f"{self.base64_encoding}",
                },
            )
            response.raise_for_status()
            result: object = response.json().get("elements", [])
            if not isinstance(result, list):
                raise ValueError(f"Failed to parse JSON data: {result}")
            result = cast(list[object], result)
            return [Element.model_validate(element) for element in result]
        except requests.HTTPError as e:
            raise ValueError(f"HTTP error: {e.response.text}")
        except requests.RequestException as e:
            # Handle any request-related exceptions
            raise ValueError(f"Failed to send request: {e}")
        except json.JSONDecodeError as e:
            # Handle JSON decode errors
            raise ValueError(f"Failed to decode JSON response: {e}")
        except Exception as e:
            # Handle any other exceptions
            raise ValueError(f"An error occurred: {e}")

    def _split_and_request(
        self, full_docs: PdfReader, start_page: int, num_pages: int = DEFAULT_NUM_PAGES
    ) -> list[Element]:
        """
        Splits the full pdf document into partial pages and sends a request to the
        server.

        Args:
            full_docs (PdfReader): The full document to be split and requested.
            start_page (int): The starting page number for splitting the document.
            num_pages (int, optional): The number of pages to split the document
                                       into.
                                       Defaults to DEFAULT_NUMBER_OF_PAGE.

        Returns:
            response: The response from the server.
        """
        from pypdf import PdfWriter

        merger = PdfWriter()
        merger.append(
            full_docs,
            pages=(start_page, min(start_page + num_pages, full_docs.get_num_pages())),
        )

        with io.BytesIO() as buffer:
            merger.write(buffer)
            buffer.seek(0)
            return self._get_response({"document": buffer})

    def _element_document(self, element: Element, start_page: int = 0) -> Document:
        """
        Converts an elements into a Document object.

        Args:
            elements (Dict) : The elements to convert.
            start_page (int): The starting page number for splitting the document.
                              This number starts from zero.

        Returns:
            A list containing a single Document object.

        """
        metadata: dict[str, object] = element.model_dump(exclude_none=True)
        metadata["page"] = element.page + start_page
        return Document(
            page_content=element.parse_text(self),
            metadata=metadata,
        )

    def _page_document(self, elements: list[Element], start_page: int = 0) -> list[Document]:
        """
        Combines elements with the same page number into a single Document object.

        Args:
            elements (List): A list of elements containing page numbers.
            start_page (int): The starting page number for splitting the document.
                              This number starts from zero.

        Returns:
            List[Document]: A list of Document objects, each representing a page
                            with its content and metadata.
        """
        documents: list[Document] = []
        pages: list[int] = sorted(set(map(lambda x: x.page, elements)))
        page_group: list[list[Element]] = [[element for element in elements if element.page == x] for x in pages]
        for group in page_group:
            metadata: dict[str, object] = {
                "page": group[0].page + start_page,
            }
            if self.base64_encoding:
                metadata["base64_encodings"] = [element.base64_encoding for element in group if element.base64_encoding]
            if self.coordinates:
                metadata["coordinates"] = [element.coordinates for element in group if element.coordinates]
            documents.append(
                Document(
                    page_content=" ".join(element.parse_text(self) for element in group),
                    metadata=metadata,
                )
            )

        return documents

    def lazy_parse(self, blob: Blob, is_batch: bool = False) -> Iterator[Document]:
        """
        Lazily parses a document and yields Document objects based on the specified
        split type.

        Args:
            blob (Blob): The input document blob to parse.
            is_batch (bool, optional): Whether to parse the document in batches.
                                       Defaults to False (single page parsing)

        Yields:
            Document: The parsed document object.

        Raises:
            ValueError: If an invalid split type is provided.

        """
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        if is_batch:
            num_pages = DEFAULT_NUM_PAGES
        else:
            num_pages = 1

        full_docs: Optional[PdfReader] = None
        try:
            full_docs = PdfReader(str(blob.path))
            number_of_pages = full_docs.get_num_pages()
        except PdfReadError:
            number_of_pages = 1
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {e}")

        if self.split == "none":
            result = ""
            base64_encodings: list[str] = []
            coordinates: list[list[Coordinate]] = []

            if full_docs is not None:
                start_page = 0
                num_pages = DEFAULT_NUM_PAGES
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    elements = self._split_and_request(full_docs, start_page, num_pages)
                    for element in elements:
                        result += element.parse_text(self)
                        if self.base64_encoding and (base64_encoding := element.base64_encoding):
                            base64_encodings.append(base64_encoding)
                        if self.coordinates and (coords := element.coordinates):
                            coordinates.append(coords)

                    start_page += num_pages

            else:
                if not blob.path:
                    raise ValueError("Blob path is required for non-PDF files.")

                with open(blob.path, "rb") as f:
                    elements = self._get_response({"document": f})

                for element in elements:
                    result += element.parse_text(self)

                    if self.base64_encoding and (base64_encoding := element.base64_encoding):
                        base64_encodings.append(base64_encoding)
                    if self.coordinates and (coords := element.coordinates):
                        coordinates.append(coords)
            metadata: dict[str, object] = {"total_pages": number_of_pages}
            if self.coordinates:
                metadata["coordinates"] = coordinates
            if self.base64_encoding:
                metadata["base64_encodings"] = base64_encodings

            yield Document(
                page_content=result,
                metadata=metadata,
            )

        elif self.split == "element":
            if full_docs is not None:
                start_page = 0
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    elements = self._split_and_request(full_docs, start_page, num_pages)
                    for element in elements:
                        yield self._element_document(element, start_page)

                    start_page += num_pages

            else:
                if not blob.path:
                    raise ValueError("Blob path is required for non-PDF files.")
                with open(blob.path, "rb") as f:
                    elements = self._get_response({"document": f})

                for element in elements:
                    yield self._element_document(element)

        elif self.split == "page":
            if full_docs is not None:
                start_page = 0
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    elements = self._split_and_request(full_docs, start_page, num_pages)
                    yield from self._page_document(elements, start_page)

                    start_page += num_pages
            else:
                if not blob.path:
                    raise ValueError("Blob path is required for non-PDF files.")
                with open(blob.path, "rb") as f:
                    elements = self._get_response({"document": f})

                yield from self._page_document(elements)

        else:
            raise ValueError(f"Invalid split type: {self.split}")
