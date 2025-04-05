import ast
import importlib
import os
import re
import site
from fnmatch import fnmatch
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    NamedTuple,
    NotRequired,
    Optional,
    Self,
    Sequence,
    TypeAlias,
    TypedDict,
)

from ..common_types.io import PathOrReadable
from ..utils.bytesio import read_bytes_stream
from .convert_pdf_to_markdown import extract_text_from_pdf

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


class CodeSnippets(NamedTuple):
    """
    A named tuple that represents code snippets extracted from Python files.

    Contains the paths to the files, the concatenated text of all snippets,
    and the base directory of the files.
    """

    paths: list[Path]
    snippets_text: str
    base_dir: Path

    @classmethod
    def from_path_or_pkgname(
        cls,
        path_or_pkgname: str,
        glob_patterns: str | list[str] = "*.py",
        case_sensitive: bool = False,
        ban_file_patterns: Optional[list[str]] = None,
    ) -> Self:
        """
        Creates a CodeSnippets instance from a file path or package name.

        Args:
            path_or_pkgname: Path to a file/directory or a Python package name.
            ban_file_patterns: Optional list of patterns to exclude files.

        Returns:
            A new CodeSnippets instance with extracted code snippets.
        """
        paths: list[Path] = _get_filepaths(
            path_or_pkgname=path_or_pkgname,
            glob_patterns=glob_patterns,
            case_sensitive=case_sensitive,
            ban_fn_patterns=ban_file_patterns,
        )
        snippets_text: str = "".join(_get_a_snippet(p) for p in paths)
        return cls(
            paths=paths,
            snippets_text=snippets_text,
            base_dir=_get_base_dir(paths),
        )

    @property
    def metadata(self) -> str:
        """
        Generates metadata about the code snippets.

        Returns a string containing information about the file tree structure,
        total number of files, tokens (if tiktoken is available), and lines.

        Returns:
            str: Formatted metadata string.
        """
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
            """
            Helper function to recursively display a file tree structure.

            Args:
                tree: The file tree dictionary to display.
                prefix: Current line prefix for proper indentation.
            """
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


def pdf_to_text(path_or_file: PathOrReadable, page_indices: Iterable[int] | int | None = None) -> str:
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
                for page_no, text in extract_text_from_pdf(doc, page_indices).items()
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


# Alias for CodeSnippets.from_path_or_pkgname for backward compatibility
pyscripts_to_snippets = CodeSnippets.from_path_or_pkgname


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """
    Converts an fnmatch pattern to a regular expression.

    In this function, '**' is converted to match any character including directory separators.
    The remaining '*' matches any character except directory separators, and '?' matches a single character.

    Args:
        pattern: The fnmatch pattern to convert.

    Returns:
        A compiled regular expression pattern.
    """
    # First escape the pattern
    pattern = re.escape(pattern)
    # Convert '**' to match any character including directory separators ('.*')
    pattern = pattern.replace(r"\*\*", ".*")
    # Then convert single '*' to match any character except directory separators
    pattern = pattern.replace(r"\*", "[^/]*")
    # Convert '?' to match a single character
    pattern = pattern.replace(r"\?", ".")
    # Anchor the pattern to start and end
    pattern = "^" + pattern + "$"
    return re.compile(pattern)


def _is_banned(p: Path, ban_patterns: list[str]) -> bool:
    """
    Checks if a given path matches any of the ban patterns.

    Determines if the path p matches any pattern in ban_patterns using either
    fnmatch-based or recursive patterns (i.e., containing '**').

    Note: Patterns should use POSIX-style paths (i.e., '/' separators).

    Args:
        p: The path to check.
        ban_patterns: List of patterns to match against.

    Returns:
        bool: True if the path matches any ban pattern, False otherwise.
    """
    p_str = p.as_posix()
    for pattern in ban_patterns:
        if "**" in pattern:
            regex = _pattern_to_regex(pattern)
            if regex.match(p_str):
                return True
        else:
            # Simple fnmatch: '*' by default doesn't match '/'
            if fnmatch(p_str, pattern):
                return True
    return False


def _get_a_snippet(fpath: Path) -> str:
    """
    Extracts a code snippet from a Python file.

    Reads the file, parses it as Python code, and returns a formatted code snippet
    with the relative path as a header in markdown code block format.

    Args:
        fpath: Path to the Python file.

    Returns:
        str: Formatted code snippet or empty string if the file doesn't exist.
    """
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
    """
    Determines the common base directory for a sequence of file paths.

    Finds the directory with the shortest path that is a parent to at least one file.

    Args:
        target_files: Sequence of file paths.

    Returns:
        Path: The common base directory.
    """
    return Path(os.path.commonpath(target_files))


def _get_filepaths(
    path_or_pkgname: str,
    glob_patterns: str | list[str] = "*.py",
    case_sensitive: bool = False,
    ban_fn_patterns: Optional[list[str]] = None,
) -> list[Path]:
    """
    Gets paths to files from a directory, file, or Python package name.

    If path_or_pkgname is a directory, finds all `glob_pattern` matching files recursively.
    If it's a file, returns just that file.
    If it's a package name, imports the package and finds all .py files in its directory.

    Args:
        path_or_pkgname: Path to directory/file or package name.
        glob_pattern: Pattern to match files.
        case_sensitive: Whether to match files case-sensitively.
        ban_fn_patterns: Optional list of patterns to exclude files.

    Returns:
        list[Path]: List of paths to Python files.
    """
    path = Path(path_or_pkgname)
    pypaths: list[Path]
    if path.is_dir():
        glob_patterns = glob_patterns if isinstance(glob_patterns, (tuple, list)) else [glob_patterns]
        pypaths = []
        for pattern in glob_patterns:
            if "**" in pattern:
                regex = _pattern_to_regex(pattern)
                pypaths.extend(
                    p for p in path.rglob("**/*", case_sensitive=case_sensitive) if regex.match(p.as_posix())
                )
            else:
                pypaths += list(path.rglob(pattern, case_sensitive=case_sensitive))

        # pypaths = list(path.rglob(glob_pattern, case_sensitive=case_sensitive))
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
    return [p for p in pypaths if not ban_fn_patterns or not _is_banned(p, ban_fn_patterns)]
