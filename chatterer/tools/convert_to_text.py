import os
from contextlib import contextmanager, suppress
from io import BufferedReader, BufferedWriter, BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, NotRequired, Optional, Sequence, TypeAlias, TypedDict

if TYPE_CHECKING:
    from bs4 import Tag
    from openai import OpenAI
    from requests import Response, Session

CodeLanguageCallback: TypeAlias = Callable[["Tag"], Optional[str]]
FileDescriptorOrPath: TypeAlias = int | str | bytes | os.PathLike[str] | os.PathLike[bytes]

BytesReadable: TypeAlias = BytesIO | BufferedReader
BytesWritable: TypeAlias = BytesIO | BufferedWriter
StringReadable: TypeAlias = StringIO | TextIOWrapper
StringWritable: TypeAlias = StringIO | TextIOWrapper

Readable: TypeAlias = BytesReadable | StringReadable
Writable: TypeAlias = BytesWritable | StringWritable

PathOrReadable: TypeAlias = FileDescriptorOrPath | Readable


class HtmlToMarkdownOptions(TypedDict):
    autolinks: NotRequired[bool]
    bullets: NotRequired[str]
    code_language: NotRequired[str]
    code_language_callback: NotRequired[CodeLanguageCallback]
    convert: NotRequired[Sequence[str]]
    default_title: NotRequired[bool]
    escape_asterisks: NotRequired[bool]
    escape_underscores: NotRequired[bool]
    escape_misc: NotRequired[bool]
    heading_style: NotRequired[str]
    keep_inline_images_in: NotRequired[Sequence[str]]
    newline_style: NotRequired[str]
    strip: NotRequired[Sequence[str]]
    strip_document: NotRequired[str]
    strong_em_symbol: NotRequired[str]
    sub_symbol: NotRequired[str]
    sup_symbol: NotRequired[str]
    table_infer_header: NotRequired[bool]
    wrap: NotRequired[bool]
    wrap_width: NotRequired[int]


def get_default_html_to_markdown_options() -> HtmlToMarkdownOptions:
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


def pdf_to_text(path_or_file: PathOrReadable) -> str:
    from pymupdf import Document  # pyright: ignore[reportMissingTypeStubs]

    with _open_stream(path_or_file) as stream:
        if stream is None:
            raise FileNotFoundError(path_or_file)
        return "\n".join(
            f"<!-- Page {page_no} -->\n{text.strip()}\n"
            for page_no, text in enumerate(
                (
                    page.get_textpage().extractText()  # pyright: ignore[reportUnknownMemberType]
                    for page in Document(stream=stream.read())
                ),
                1,
            )
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


@contextmanager
def _open_stream(
    path_or_file: PathOrReadable,
) -> Iterator[Optional[BytesReadable]]:
    stream: Optional[BytesReadable] = None
    try:
        with suppress(BaseException):
            if isinstance(path_or_file, BytesReadable):
                stream = path_or_file
            elif isinstance(path_or_file, StringReadable):
                stream = BytesIO(path_or_file.read().encode("utf-8"))
            else:
                stream = open(path_or_file, "rb")
        yield stream
    finally:
        if stream is not None:
            stream.close()
