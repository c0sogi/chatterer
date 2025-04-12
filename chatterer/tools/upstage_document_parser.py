# -*- coding: utf-8 -*-
"""Adopted from `langchain_upstage.document_parse`"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Dict, Iterator, Literal, Optional, TypedDict, cast

import requests
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from ..common_types.io import BytesReadable
from ..language_model import DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION, Chatterer
from ..utils.base64_image import Base64Image
from ..utils.imghdr import what

if TYPE_CHECKING:
    from pypdf import PdfReader

logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)
parser_logger = logging.getLogger(__name__)  # Added logger for this module

DOCUMENT_PARSE_BASE_URL = "https://api.upstage.ai/v1/document-ai/document-parse"
DEFAULT_NUM_PAGES = 10
DOCUMENT_PARSE_DEFAULT_MODEL = "document-parse"
DEFAULT_IMAGE_DIR = "images"  # Added default image directory

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
        """
        Generates the text representation of the element.

        If the element is a figure with base64 encoding and no chatterer is provided,
        it generates a markdown link to a uniquely named image file and stores the
        image data in the parser's image_data dictionary. Otherwise, it uses the
        chatterer for description or returns the standard text/html/markdown.
        """
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
            # Fallback or raise error if needed, here using text as fallback
            output = self.content.text or ""
            # Or raise ValueError(f"Invalid output format or missing content: {output_format}")

        # --- Logic modification starts here ---
        if self.category == "figure" and self.base64_encoding:
            # Case 1: Chatterer is available - Generate description
            if chatterer is not None:
                # Check if base64 encoding is valid
                try:
                    # Decode base64 to check if valid
                    img_type = what(self.base64_encoding)
                    if not img_type:
                        parser_logger.warning(
                            f"Could not determine image type for figure element {self.id} (page {self.page})."
                        )
                        return output
                    image = Base64Image.from_string(f"data:image/{img_type};base64,{self.base64_encoding}")

                except (binascii.Error, ValueError) as e:
                    parser_logger.warning(
                        f"Could not decode base64 for figure element {self.id} (page {self.page}): {e}. Falling back to original output."
                    )
                    return output

                if image is None:
                    parser_logger.warning(
                        f"Invalid base64 encoding format for image element {self.id}, cannot create Base64Image object."
                    )
                    # Fallback to original output (placeholder/OCR)
                    return output

                ocr_content = ""
                if output_format == "markdown":
                    ocr_content = output.removeprefix("![image](/image/placeholder)\n")
                elif output_format == "text":
                    ocr_content = output

                image_description = chatterer.describe_image(
                    image.data_uri,
                    image_description_instruction
                    + f"\nHint: The OCR detected the following text:\n```\n{ocr_content}\n```",
                )
                # Return description within details tag (as original)
                output = f"\n\n<details>\n<summary>Image Description</summary>\n{image_description}\n</details>\n\n"

            # Case 2: Chatterer is NOT available - Generate file path and store data
            elif parser.image_dir is not None:
                try:
                    img_type = what(self.base64_encoding)
                    if not img_type:
                        parser_logger.warning(
                            f"Could not determine image type for figure element {self.id} (page {self.page})."
                        )
                        return output

                    image_bytes = base64.b64decode(self.base64_encoding)

                    # Generate unique filename and path
                    filename = f"{uuid.uuid4().hex}.{img_type}"  # Use default format
                    # Create relative path for markdown link, ensuring forward slashes
                    relative_path = os.path.join(parser.image_dir, filename).replace("\\", "/")

                    # Store the image data for the user to save later
                    parser.image_data[relative_path] = image_bytes

                    # Extract OCR content if present
                    ocr_content = ""
                    if output_format == "markdown" and output.startswith("![image]"):
                        ocr_content = output.split("\n", 1)[1] if "\n" in output else ""
                    elif output_format == "text":
                        ocr_content = output  # Assume text output is OCR for images

                    # Update output to be the markdown link + OCR
                    output = f"![image]({relative_path})\n{ocr_content}".strip()

                except (binascii.Error, ValueError) as e:
                    # Handle potential base64 decoding errors gracefully
                    parser_logger.warning(
                        f"Could not decode base64 for figure element {self.id} (page {self.page}): {e}. Falling back to original output."
                    )
                    # Keep the original 'output' value (placeholder or OCR)
                    pass

        return output


class Coordinates(TypedDict):
    id: int
    category: Category
    coordinates: list[Coordinate]


class PageCoordinates(Coordinates):
    page: int


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
            f" `{key}` as a named parameter."
        )


class UpstageDocumentParseParser(BaseBlobParser):
    """Upstage Document Parse Parser.

    Parses documents using the Upstage Document AI API. Can optionally extract
    images and return their data alongside the parsed documents.

    If a `chatterer` is provided, it will be used to generate descriptions for
    images (figures with base64 encoding).

    If `chatterer` is NOT provided, for figure elements with `base64_encoding`,
    this parser will:
    1. Generate a unique relative file path (e.g., "images/uuid.jpeg").
       The base directory can be configured with `image_dir`.
    2. Replace the element's content with a markdown image link pointing to this path.
    3. Store the actual image bytes in the `image_data` attribute dictionary,
       mapping the generated relative path to the bytes.

    The user is responsible for saving the files from the `image_data` dictionary
    after processing the documents yielded by `lazy_parse`.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageDocumentParseParser
            from langchain_core.documents import Blob
            import os

            # --- Setup ---
            # Ensure UPSTAGE_API_KEY is set in environment or passed as api_key
            # Create a dummy PDF or image file 'my_document.pdf' / 'my_image.png'

            # --- Parsing without chatterer (extracts images) ---
            parser = UpstageDocumentParseParser(
                split="page",
                output_format="markdown",
                base64_encoding=["figure"], # Important: Request base64 for figures
                image_dir="extracted_images" # Optional: specify image dir
            )
            blob = Blob.from_path("my_document.pdf") # Or your image file path
            documents = []
            for doc in parser.lazy_parse(blob):
                print("--- Document ---")
                print(f"Page: {get_metadata_from_document(doc).get('page')}")
                print(doc.page_content)
                documents.append(doc)

            print("\\n--- Extracted Image Data ---")
            if parser.image_data:
                # User saves the images
                for img_path, img_bytes in parser.image_data.items():
                    # Create directories if they don't exist
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    try:
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"Saved image: {img_path}")
                    except IOError as e:
                        print(f"Error saving image {img_path}: {e}")
            else:
                print("No images extracted.")

            # --- Parsing with chatterer (generates descriptions) ---
            # from langchain_upstage import UpstageChatter # Assuming this exists
            # chatterer = UpstageChatter() # Initialize your chatterer
            # parser_with_desc = UpstageDocumentParseParser(
            #     split="page",
            #     output_format="markdown",
            #     base64_encoding=["figure"], # Still need base64 for description
            #     chatterer=chatterer
            # )
            # documents_with_desc = list(parser_with_desc.lazy_parse(blob))
            # print("\\n--- Documents with Descriptions ---")
            # for doc in documents_with_desc:
            #     print(f"Page: {get_metadata_from_document(doc).get('page')}")
            #     print(doc.page_content)

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
        image_dir: Optional[str] = None,  # Added: Directory for image paths
    ) -> None:
        """
        Initializes an instance of the UpstageDocumentParseParser.

        Args:
            api_key (str, optional): Upstage API key. Defaults to env `UPSTAGE_API_KEY`.
            base_url (str, optional): Base URL for the Upstage API.
            model (str): Model for document parse. Defaults to "document-parse".
            split (SplitType, optional): Splitting type ("none", "page", "element").
                                          Defaults to "none".
            ocr (OCR, optional): OCR mode ("auto", "force"). Defaults to "auto".
            output_format (OutputFormat, optional): Output format ("text", "html", "markdown").
                                                     Defaults to "markdown".
            coordinates (bool, optional): Include coordinates in metadata. Defaults to True.
            base64_encoding (List[Category], optional): Categories to return as base64.
                                                       Crucial for image extraction/description.
                                                       Set to `["figure"]` to process images.
                                                       Defaults to [].
            chatterer (Chatterer, optional): Chatterer instance for image description.
                                             If None, images will be extracted to files.
                                             Defaults to None.
            image_description_instruction (str, optional): Instruction for image description.
                                                            Defaults to a standard instruction.
            image_dir (str, optional): The directory name to use when constructing
                                        relative paths for extracted images.
                                        Defaults to "images". This directory
                                        is NOT created by the parser.
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
        # Ensure 'figure' is requested if chatterer is None and user wants extraction implicitly
        # However, it's better to require the user to explicitly set base64_encoding=["figure"]
        self.base64_encoding: list[Category] = base64_encoding
        self.chatterer = chatterer
        self.image_description_instruction = image_description_instruction
        self.image_dir = image_dir  # Store output directory name

        # Initialize dictionary to store image data (path -> bytes)
        self.image_data: Dict[str, bytes] = {}

    def _get_response(self, files: dict[str, tuple[str, BytesReadable]]) -> list[Element]:
        """
        Sends a POST request to the API endpoint with the provided files and
        returns the parsed elements.
        """
        response: Optional[requests.Response] = None
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            # Convert list to string representation required by the API
            base64_encoding_str = str(self.base64_encoding) if self.base64_encoding else "[]"
            output_formats_str = f"['{self.output_format}']"

            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data={
                    "ocr": self.ocr,
                    "model": self.model,
                    "output_formats": output_formats_str,
                    "coordinates": str(self.coordinates).lower(),  # API might expect 'true'/'false'
                    "base64_encoding": base64_encoding_str,
                },
            )
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            # Check content type before parsing JSON
            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise ValueError(f"Unexpected content type: {content_type}. Response body: {response.text}")

            response_data = response.json()
            result: object = response_data.get("elements", [])

            if not isinstance(result, list):
                raise ValueError(f"API response 'elements' is not a list: {result}")
            result = cast(list[object], result)  # Cast to list of objects

            # Validate each element using Pydantic
            validated_elements: list[Element] = []
            for i, element_data in enumerate(result):
                try:
                    validated_elements.append(Element.model_validate(element_data))
                except Exception as e:  # Catch Pydantic validation errors etc.
                    parser_logger.error(f"Failed to validate element {i}: {element_data}. Error: {e}")
                    # Decide whether to skip the element or raise the error
                    # continue # Option: skip problematic element
                    raise ValueError(f"Failed to validate element {i}: {e}") from e  # Option: fail fast

            return validated_elements

        except requests.HTTPError as e:
            # Log more details from the response if available
            error_message = f"HTTP error: {e.response.status_code} {e.response.reason}"
            try:
                error_details = e.response.json()  # Try to get JSON error details
                error_message += f" - {error_details}"
            except json.JSONDecodeError:
                error_message += f" - Response body: {e.response.text}"
            raise ValueError(error_message) from e
        except requests.RequestException as e:
            raise ValueError(f"Failed to send request: {e}") from e
        except json.JSONDecodeError as e:
            # Include part of the response text that failed to parse
            raise ValueError(
                f"Failed to decode JSON response: {e}. Response text starts with: {response.text[:200] if response else 'No response'}"
            ) from e
        except Exception as e:  # Catch-all for other unexpected errors
            raise ValueError(f"An unexpected error occurred during API call: {e}") from e

    def _split_and_request(
        self, full_docs: PdfReader, start_page: int, num_pages: int = DEFAULT_NUM_PAGES
    ) -> list[Element]:
        """
        Splits the full pdf document into partial pages and sends a request.
        """
        # Need to import here if not globally available
        try:
            from pypdf import PdfWriter
        except ImportError:
            raise ImportError("pypdf is required for PDF splitting. Please install it with `pip install pypdf`.")

        merger = PdfWriter()
        total_pages = len(full_docs.pages)  # Use len(reader.pages) instead of get_num_pages()
        end_page = min(start_page + num_pages, total_pages)

        # Check if start_page is valid
        if start_page >= total_pages:
            parser_logger.warning(f"Start page {start_page} is out of bounds for document with {total_pages} pages.")
            return []

        # pypdf page indices are 0-based, slicing is exclusive of the end index
        # PdfWriter.append() expects pages=(start, stop) where stop is exclusive.
        # However, the example used pages=(start, end) which might behave differently depending on version?
        # Let's stick to add_page for clarity if possible, or ensure append range is correct.
        # merger.append(full_docs, pages=(start_page, end_page)) # This selects pages start_page..end_page-1

        # Alternative using add_page loop (more explicit)
        for i in range(start_page, end_page):
            merger.add_page(full_docs.pages[i])

        with io.BytesIO() as buffer:
            merger.write(buffer)
            buffer.seek(0)
            # Need to provide a filename for the 'files' dict
            return self._get_response({"document": ("partial_doc.pdf", buffer)})  # Provide a dummy filename

    def _element_document(self, element: Element, start_page: int = 0) -> Document:
        """Converts an element into a Document object."""
        # parse_text now handles image path generation and data storage if needed
        page_content = element.parse_text(self)
        metadata: dict[str, object] = element.model_dump(
            exclude={"content", "base64_encoding"}, exclude_none=True
        )  # Exclude raw content/base64
        metadata["page"] = element.page + start_page  # Adjust page number
        # Base64 encoding is not added to metadata if it was processed into image_data
        # Coordinates are kept if requested
        if not self.coordinates:
            metadata.pop("coordinates", None)

        return Document(
            page_content=page_content,
            metadata=metadata,
        )

    def _page_document(self, elements: list[Element], start_page: int = 0) -> list[Document]:
        """Combines elements with the same page number into a single Document object."""
        documents: list[Document] = []
        if not elements:
            return documents

        # Group elements by page (relative to the current batch)
        pages: list[int] = sorted(list(set(map(lambda x: x.page, elements))))
        page_groups: Dict[int, list[Element]] = {page: [] for page in pages}
        for element in elements:
            page_groups[element.page].append(element)

        for page_num, group in page_groups.items():
            actual_page_num = page_num + start_page
            page_content_parts: list[str] = []
            page_coordinates: list[Coordinates] = []
            # Base64 encodings are handled within parse_text now, not collected here

            for element in sorted(group, key=lambda x: x.id):  # Process elements in order
                page_content_parts.append(element.parse_text(self))
                if self.coordinates and element.coordinates:
                    page_coordinates.append({  # Store coordinates with element id/category for context
                        "id": element.id,
                        "category": element.category,
                        "coordinates": element.coordinates,
                    })

            metadata: dict[str, object] = {
                "page": actual_page_num,
            }
            if self.coordinates and page_coordinates:
                metadata["element_coordinates"] = page_coordinates  # Changed key for clarity

            # Combine content, typically with spaces or newlines
            # Using newline might be better for readability if elements are paragraphs etc.
            combined_page_content = "\n\n".join(part for part in page_content_parts if part)  # Join non-empty parts

            documents.append(
                Document(
                    page_content=combined_page_content,
                    metadata=metadata,
                )
            )

        return documents

    def lazy_parse(self, blob: Blob, is_batch: bool = False) -> Iterator[Document]:
        """
        Lazily parses a document blob.

        Yields Document objects based on the specified split type.
        If images are extracted (chatterer=None, base64_encoding=["figure"]),
        the image data will be available in `self.image_data` after iteration.

        Args:
            blob (Blob): The input document blob to parse. Requires `blob.path`.
            is_batch (bool, optional): Currently affects PDF page batch size.
                                       Defaults to False (process 1 page batch for PDF).
                                       *Note: API might have limits regardless.*

        Yields:
            Document: The parsed document object(s).

        Raises:
            ValueError: If blob.path is not set, API error occurs, or invalid config.
            ImportError: If pypdf is needed but not installed.
        """
        # Clear image data at the start of parsing for this specific call
        self.image_data = {}

        if not blob.path:
            # Non-PDF files and direct API calls require reading the file,
            # PDF splitting also requires the path.
            raise ValueError("Blob path is required for UpstageDocumentParseParser.")

        # Try importing pypdf here, only if needed
        PdfReader = None
        PdfReadError = None
        try:
            from pypdf import PdfReader as PyPdfReader
            from pypdf.errors import PdfReadError as PyPdfReadError

            PdfReader = PyPdfReader
            PdfReadError = PyPdfReadError
        except ImportError:
            # We only absolutely need pypdf if the file is a PDF and split is not 'none' maybe?
            # Let's attempt to read anyway, API might support non-PDFs directly.
            # We'll check for PdfReader later if we determine it's a PDF.
            pass

        full_docs: Optional[PdfReader] = None
        is_pdf = False
        number_of_pages = 1  # Default for non-PDF or single-page docs

        try:
            # Check if it's a PDF by trying to open it
            if PdfReader and PdfReadError:
                try:
                    # Use strict=False to be more lenient with potentially corrupted PDFs
                    full_docs = PdfReader(str(blob.path), strict=False)
                    number_of_pages = len(full_docs.pages)
                    is_pdf = True
                except (PdfReadError, FileNotFoundError, IsADirectoryError) as e:
                    parser_logger.warning(f"Could not read '{blob.path}' as PDF: {e}. Assuming non-PDF format.")
                except Exception as e:  # Catch other potential pypdf errors
                    parser_logger.error(f"Unexpected error reading PDF '{blob.path}': {e}")
                    raise ValueError(f"Failed to process PDF file: {e}") from e
            else:
                parser_logger.info("pypdf not installed. Treating input as a single non-PDF document for the API.")

        except Exception as e:
            raise ValueError(f"Failed to access or identify file type for: {blob.path}. Error: {e}") from e

        # --- Parsing Logic based on Split Type ---

        # Case 1: No Splitting (Combine all content)
        if self.split == "none":
            combined_result = ""
            all_coordinates: list[PageCoordinates] = []
            # Base64 handled by parse_text, data stored in self.image_data

            if is_pdf and full_docs and PdfReader:  # Process PDF page by page or in batches
                start_page = 0
                # Use a reasonable batch size for 'none' split to avoid huge requests
                batch_num_pages = DEFAULT_NUM_PAGES
                while start_page < number_of_pages:
                    elements = self._split_and_request(full_docs, start_page, batch_num_pages)
                    for element in sorted(elements, key=lambda x: (x.page, x.id)):
                        combined_result += element.parse_text(self) + "\n\n"  # Add separator
                        if self.coordinates and element.coordinates:
                            # Adjust page number for coordinates metadata
                            coords_with_page: PageCoordinates = {
                                "id": element.id,
                                "category": element.category,
                                "page": element.page + start_page,  # Actual page
                                "coordinates": element.coordinates,
                            }
                            all_coordinates.append(coords_with_page)
                    start_page += batch_num_pages
            else:  # Process non-PDF file as a single unit
                with open(blob.path, "rb") as f:
                    # Provide a filename for the 'files' dict
                    filename = os.path.basename(blob.path)
                    elements = self._get_response({"document": (filename, f)})

                for element in sorted(elements, key=lambda x: x.id):
                    combined_result += element.parse_text(self) + "\n\n"
                    if self.coordinates and element.coordinates:
                        all_coordinates.append({
                            "id": element.id,
                            "category": element.category,
                            "page": element.page,  # Page is relative to the single doc (usually 0 or 1)
                            "coordinates": element.coordinates,
                        })

            metadata: dict[str, object] = {"source": blob.path, "total_pages": number_of_pages}
            if self.coordinates and all_coordinates:
                metadata["element_coordinates"] = all_coordinates
            # self.image_data is populated, no need to add base64 to metadata

            yield Document(
                page_content=combined_result.strip(),
                metadata=metadata,
            )

        # Case 2: Split by Element
        elif self.split == "element":
            if is_pdf and full_docs and PdfReader:
                start_page = 0
                batch_num_pages = DEFAULT_NUM_PAGES if is_batch else 1  # Use smaller batches for element split?
                while start_page < number_of_pages:
                    elements = self._split_and_request(full_docs, start_page, batch_num_pages)
                    for element in sorted(elements, key=lambda x: (x.page, x.id)):
                        # _element_document handles metadata and adjusts page number
                        doc = self._element_document(element, start_page)
                        _get_metadata_from_document(doc)["source"] = blob.path  # Add source
                        yield doc
                    start_page += batch_num_pages
            else:  # Non-PDF
                with open(blob.path, "rb") as f:
                    filename = os.path.basename(blob.path)
                    elements = self._get_response({"document": (filename, f)})
                for element in sorted(elements, key=lambda x: x.id):
                    doc = self._element_document(element, 0)  # Start page is 0 for single doc
                    _get_metadata_from_document(doc)["source"] = blob.path  # Add source
                    yield doc

        # Case 3: Split by Page
        elif self.split == "page":
            if is_pdf and full_docs and PdfReader:
                start_page = 0
                batch_num_pages = DEFAULT_NUM_PAGES if is_batch else 1  # Process page-by-page if not is_batch
                while start_page < number_of_pages:
                    elements = self._split_and_request(full_docs, start_page, batch_num_pages)
                    # _page_document groups elements by page and creates Documents
                    page_docs = self._page_document(elements, start_page)
                    for doc in page_docs:
                        _get_metadata_from_document(doc)["source"] = blob.path  # Add source
                        yield doc
                    start_page += batch_num_pages
            else:  # Non-PDF (treat as single page)
                with open(blob.path, "rb") as f:
                    filename = os.path.basename(blob.path)
                    elements = self._get_response({"document": (filename, f)})
                page_docs = self._page_document(elements, 0)  # Process elements as page 0
                for doc in page_docs:
                    _get_metadata_from_document(doc)["source"] = blob.path  # Add source
                    yield doc

        else:
            raise ValueError(f"Invalid split type: {self.split}")


def _get_metadata_from_document(doc: Document) -> dict[object, object]:
    """
    Helper function to extract metadata from a Document object.
    This is a placeholder and should be adjusted based on actual metadata structure.
    """
    metadata: dict[object, object] = doc.metadata  # pyright: ignore[reportUnknownMemberType]
    return metadata
