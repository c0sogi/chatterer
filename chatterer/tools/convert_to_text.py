import ast
import importlib
import os
import re
import site
from contextlib import contextmanager, suppress
from fnmatch import fnmatch
from io import BufferedReader, BufferedWriter, BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    NamedTuple,
    NotRequired,
    Optional,
    Self,
    Sequence,
    TypeAlias,
    TypedDict,
)

if TYPE_CHECKING:
    from bs4 import Tag
    from openai import OpenAI
    from requests import Response, Session

try:
    from tiktoken import get_encoding, list_encoding_names

    enc = get_encoding(list_encoding_names()[-1])
except ImportError:
    enc = None


type FileTree = dict[str, Optional[FileTree]]

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


class CodeSnippets(NamedTuple):
    paths: list[Path]
    snippets_text: str
    base_dir: Path

    @classmethod
    def from_path_or_pkgname(cls, path_or_pkgname: str, ban_file_patterns: Optional[list[str]] = None) -> Self:
        paths: list[Path] = _get_pyscript_paths(path_or_pkgname=path_or_pkgname, ban_fn_patterns=ban_file_patterns)
        snippets_text: str = "".join(_get_a_snippet(p) for p in paths)
        return cls(
            paths=paths,
            snippets_text=snippets_text,
            base_dir=_get_base_dir(paths),
        )

    @property
    def metadata(self) -> str:
        file_paths: list[Path] = self.paths
        text: str = self.snippets_text

        base_dir: Path = _get_base_dir(file_paths)
        results: list[str] = [base_dir.as_posix()]

        file_tree: FileTree = {}
        for file_path in sorted(file_paths):
            rel_path = file_path.relative_to(base_dir)
            subtree: Optional[FileTree] = file_tree
            for part in rel_path.parts[:-1]:
                if subtree is not None:
                    subtree = subtree.setdefault(part, {})
            if subtree is not None:
                subtree[rel_path.parts[-1]] = None

        def _display_tree(tree: FileTree, prefix: str = "") -> None:
            items: list[tuple[str, Optional[FileTree]]] = sorted(tree.items())
            count: int = len(items)
            for idx, (name, subtree) in enumerate(items):
                branch: str = "└── " if idx == count - 1 else "├── "
                results.append(f"{prefix}{branch}{name}")
                if subtree is not None:
                    extension: str = "    " if idx == count - 1 else "│   "
                    _display_tree(tree=subtree, prefix=prefix + extension)

        _display_tree(file_tree)
        results.append(f"- Total files: {len(file_paths)}")
        if enc is not None:
            num_tokens: int = len(enc.encode(text, disallowed_special=()))
            results.append(f"- Total tokens: {num_tokens}")
        results.append(f"- Total lines: {text.count('\n') + 1}")
        return "\n".join(results)


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


pyscripts_to_snippets = CodeSnippets.from_path_or_pkgname


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """
    fnmatch 패턴을 정규표현식으로 변환합니다.
    여기서는 '**'는 모든 문자를(디렉토리 구분자 포함) 의미하도록 변환합니다.
    나머지 '*'는 디렉토리 구분자를 제외한 모든 문자, '?'는 단일 문자를 의미합니다.
    """
    # 먼저 패턴을 이스케이프
    pattern = re.escape(pattern)
    # '**'를 디렉토리 구분자 포함 모든 문자에 대응하는 '.*'로 변환
    pattern = pattern.replace(r"\*\*", ".*")
    # 그 후 단일 '*'는 디렉토리 구분자를 제외한 모든 문자에 대응하도록 변환
    pattern = pattern.replace(r"\*", "[^/]*")
    # '?'를 단일 문자 대응으로 변환
    pattern = pattern.replace(r"\?", ".")
    # 시작과 끝을 고정
    pattern = "^" + pattern + "$"
    return re.compile(pattern)


def _is_banned(p: Path, ban_patterns: list[str]) -> bool:
    """
    주어진 경로 p가 ban_patterns 중 하나와 fnmatch 기반 혹은 재귀적 패턴(즉, '**' 포함)으로
    매칭되는지 확인합니다.

    주의: 패턴은 POSIX 스타일의 경로(즉, '/' 구분자)를 사용해야 합니다.
    """
    p_str = p.as_posix()
    for pattern in ban_patterns:
        if "**" in pattern:
            regex = _pattern_to_regex(pattern)
            if regex.match(p_str):
                return True
        else:
            # 단순 fnmatch: '*'는 기본적으로 '/'와 매칭되지 않음
            if fnmatch(p_str, pattern):
                return True
    return False


def _get_a_snippet(fpath: Path) -> str:
    if not fpath.is_file():
        return ""

    cleaned_code: str = "\n".join(
        line for line in ast.unparse(ast.parse(fpath.read_text(encoding="utf-8"))).splitlines()
    )
    if site_dir := next(
        (d for d in reversed(site.getsitepackages()) if fpath.is_relative_to(d)),
        None,
    ):
        display_path = fpath.relative_to(site_dir)
    elif fpath.is_relative_to(cwd := Path.cwd()):
        display_path = fpath.relative_to(cwd)
    else:
        display_path = fpath.absolute()
    return f"```{display_path}\n{cleaned_code}\n```\n\n"


def _get_base_dir(target_files: Sequence[Path]) -> Path:
    return sorted(
        {file_path.parent for file_path in target_files},
        key=lambda p: len(p.parts),
    )[0]


def _get_pyscript_paths(path_or_pkgname: str, ban_fn_patterns: Optional[list[str]] = None) -> list[Path]:
    path = Path(path_or_pkgname)
    pypaths: list[Path]
    if path.is_dir():
        pypaths = list(path.rglob("*.py", case_sensitive=False))
    elif path.is_file():
        pypaths = [path]
    else:
        pypaths = [
            p
            for p in Path(next(iter(importlib.import_module(path_or_pkgname).__path__))).rglob(
                "*.py", case_sensitive=False
            )
            if p.is_file()
        ]
    return [p for p in pypaths if ban_fn_patterns and not _is_banned(p, ban_fn_patterns)]


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
