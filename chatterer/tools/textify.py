from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
)

from ..common_types.io import PathOrReadable
from ..utils.bytesio import read_bytes_stream
from .pdf2md import PageIndexType, extract_text_from_pdf

if TYPE_CHECKING:
    from bs4 import Tag
    from openai import OpenAI
    from requests import Response, Session

try:
    from tiktoken import get_encoding, list_encoding_names

    enc = get_encoding(list_encoding_names()[-1])
except ImportError:
    enc = None


# Type definition for representing a file tree structure
type FileTree = dict[str, Optional[FileTree]]

# Type aliases for callback functions and file descriptors
CodeLanguageCallback: TypeAlias = Callable[["Tag"], Optional[str]]


class HtmlToMarkdownOptions(TypedDict):
    """
    TypedDict for options used in HTML to Markdown conversion.

    Contains various configuration options for controlling how HTML is converted to Markdown,
    including formatting preferences, escape behaviors, and styling options.
    """

    autolinks: NotRequired[bool]
    bullets: NotRequired[str]
    code_language: NotRequired[str]
    code_language_callback: NotRequired[CodeLanguageCallback]
    convert: NotRequired[list[str]]
    default_title: NotRequired[bool]
    escape_asterisks: NotRequired[bool]
    escape_underscores: NotRequired[bool]
    escape_misc: NotRequired[bool]
    heading_style: NotRequired[str]
    keep_inline_images_in: NotRequired[list[str]]
    newline_style: NotRequired[str]
    strip: NotRequired[list[str]]
    strip_document: NotRequired[str]
    strong_em_symbol: NotRequired[str]
    sub_symbol: NotRequired[str]
    sup_symbol: NotRequired[str]
    table_infer_header: NotRequired[bool]
    wrap: NotRequired[bool]
    wrap_width: NotRequired[int]


def get_default_html_to_markdown_options() -> HtmlToMarkdownOptions:
    """
    Returns the default options for HTML to Markdown conversion.

    This function provides a set of sensible defaults for the markdownify library,
    including settings for bullets, escaping, heading styles, and other formatting options.

    Returns:
        HtmlToMarkdownOptions: A dictionary of default conversion options.
    """
    from markdownify import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
        ASTERISK,
        SPACES,
        STRIP,
        UNDERLINED,
    )

    return {
        "autolinks": True,
        "bullets": "*+-",  # An iterable of bullet types.
        "code_language": "",
        "default_title": False,
        "escape_asterisks": True,
        "escape_underscores": True,
        "escape_misc": False,
        "heading_style": UNDERLINED,
        "keep_inline_images_in": [],
        "newline_style": SPACES,
        "strip_document": STRIP,
        "strong_em_symbol": ASTERISK,
        "sub_symbol": "",
        "sup_symbol": "",
        "table_infer_header": False,
        "wrap": False,
        "wrap_width": 80,
    }


def html_to_markdown(html: str, options: Optional[HtmlToMarkdownOptions]) -> str:
    """
    Convert HTML content to Markdown using the provided options.

    Args:
        html (str): HTML content to convert.
        options (HtmlToMarkdownOptions): Options for the conversion.

    Returns:
        str: The Markdown content.
    """
    from markdownify import markdownify  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

    return str(markdownify(html, **(options or {})))  # pyright: ignore[reportUnknownArgumentType]


def pdf_to_text(path_or_file: PathOrReadable, page_indices: Optional[PageIndexType] = None) -> str:
    """
    Convert a PDF file to plain text.

    Extracts text from each page of a PDF file and formats it with page markers.

    Args:
        path_or_file: Path to a PDF file or a readable object containing PDF data.
        page_indices: Optional list of page indices to extract. If None, all pages are extracted.
            If an integer is provided, it extracts that specific page.
            If a list is provided, it extracts the specified pages.

    Returns:
        str: Extracted text with page markers.

    Raises:
        FileNotFoundError: If the file cannot be found or opened.
    """
    from pymupdf import Document  # pyright: ignore[reportMissingTypeStubs]

    with read_bytes_stream(path_or_file) as stream:
        if stream is None:
            raise FileNotFoundError(path_or_file)
        with Document(stream=stream.read()) as doc:
            return "\n".join(
                f"<!-- Page {page_no} -->\n{text}\n"
                for page_no, text in extract_text_from_pdf(doc=doc, page_indices=page_indices).items()
            )


def anything_to_markdown(
    source: "str | Response | Path",
    requests_session: Optional["Session"] = None,
    llm_client: Optional["OpenAI"] = None,
    llm_model: Optional[str] = None,
    style_map: Optional[str] = None,
    exiftool_path: Optional[str] = None,
    docintel_endpoint: Optional[str] = None,
) -> str:
    """
    Convert various types of content to Markdown format.

    Uses the MarkItDown library to convert different types of content (URLs, files, API responses)
    to Markdown format.

    Args:
        source: The source content to convert (URL string, Response object, or Path).
        requests_session: Optional requests Session for HTTP requests.
        llm_client: Optional OpenAI client for LLM-based conversions.
        llm_model: Optional model name for the LLM.
        style_map: Optional style mapping configuration.
        exiftool_path: Optional path to exiftool for metadata extraction.
        docintel_endpoint: Optional Document Intelligence API endpoint.

    Returns:
        str: The converted Markdown content.
    """
    from markitdown import MarkItDown

    result = MarkItDown(
        requests_session=requests_session,
        llm_client=llm_client,
        llm_model=llm_model,
        style_map=style_map,
        exiftool_path=exiftool_path,
        docintel_endpoint=docintel_endpoint,
    ).convert(source)
    return result.text_content
