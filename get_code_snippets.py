import ast
import importlib
import re
import site
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple, Optional, Self, Sequence

from tiktoken import get_encoding, list_encoding_names

enc = get_encoding(list_encoding_names()[-1])
ban_file_patterns: list[str] = [".venv/*", Path(__file__).relative_to(Path.cwd()).as_posix()]

type FileTree = dict[str, Optional[FileTree]]


def pattern_to_regex(pattern: str) -> re.Pattern[str]:
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


def is_banned(p: Path, ban_patterns: list[str]) -> bool:
    """
    주어진 경로 p가 ban_patterns 중 하나와 fnmatch 기반 혹은 재귀적 패턴(즉, '**' 포함)으로
    매칭되는지 확인합니다.

    주의: 패턴은 POSIX 스타일의 경로(즉, '/' 구분자)를 사용해야 합니다.
    """
    p_str = p.as_posix()
    for pattern in ban_patterns:
        if "**" in pattern:
            regex = pattern_to_regex(pattern)
            if regex.match(p_str):
                return True
        else:
            # 단순 fnmatch: '*'는 기본적으로 '/'와 매칭되지 않음
            if fnmatch(p_str, pattern):
                return True
    return False


class CodeSnippets(NamedTuple):
    paths: list[Path]
    snippets_text: str
    metadata: str
    base_dir: Path

    @classmethod
    def from_path_or_pkgname(cls, path_or_pkgname: str, ban_file_patterns: Optional[list[str]] = None) -> Self:
        paths: list[Path] = get_pyscript_paths(path_or_pkgname=path_or_pkgname, ban_fn_patterns=ban_file_patterns)
        snippets_text: str = "".join(get_a_snippet(p) for p in paths)
        metadata = get_metadata(file_paths=paths, text=snippets_text)
        return cls(
            paths=paths,
            snippets_text=snippets_text,
            metadata=metadata,
            base_dir=get_base_dir(paths),
        )


def get_a_snippet(fpath: Path) -> str:
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


def get_base_dir(target_files: Sequence[Path]) -> Path:
    return sorted(
        {file_path.parent for file_path in target_files},
        key=lambda p: len(p.parts),
    )[0]


def get_metadata(file_paths: list[Path], text: str) -> str:
    base_dir: Path = get_base_dir(file_paths)
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
    num_tokens: int = len(enc.encode(text, disallowed_special=()))
    results.append(f"- Total tokens: {num_tokens}")
    results.append(f"- Total lines: {text.count('\n') + 1}")
    return "\n".join(results)


def get_pyscript_paths(path_or_pkgname: str, ban_fn_patterns: Optional[list[str]] = None) -> list[Path]:
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
    return [p for p in pypaths if ban_fn_patterns and not is_banned(p, ban_fn_patterns)]


if __name__ == "__main__":
    print("Enter a package name or path: ", end="")
    path_or_pkgname: str = input()
    code_snippets = CodeSnippets.from_path_or_pkgname(
        path_or_pkgname=path_or_pkgname, ban_file_patterns=ban_file_patterns
    )
    print(
        f"{code_snippets.metadata}\nWould you like to save the result? [y/n]: ",
        end="",
    )
    if input().lower() == "y":
        save_path: Path = Path(__file__).with_suffix(".txt")
        save_path.write_text(code_snippets.snippets_text, encoding="utf-8")
        print(f"Saved to {save_path}")
    else:
        print(code_snippets.snippets_text)
