import sys
from pathlib import Path

sys.path.append(".")
from chatterer import ArgumentSpec, BaseArguments, CodeSnippets


class GetCodeSnippetsArgs(BaseArguments):
    path_or_pkgname: ArgumentSpec[str] = ArgumentSpec(
        ["path_or_pkgname"], help="Path to the package or file from which to extract code snippets."
    )
    ban_file_patterns: ArgumentSpec[list[str]] = ArgumentSpec(
        ["--ban-file-patterns"],
        nargs="*",
        default=[".venv/*", Path(__file__).relative_to(Path.cwd()).as_posix()],
        help="List of file patterns to ignore.",
    )
    glob_patterns: ArgumentSpec[list[str]] = ArgumentSpec(
        ["--glob-patterns"], nargs="*", default=["*.py"], help="List of glob patterns to include."
    )
    case_sensitive: ArgumentSpec[bool] = ArgumentSpec(
        ["--case-sensitive"],
        action="store_true",
        default=False,
        help="Enable case-sensitive matching for glob patterns.",
    )
    out: ArgumentSpec[Path] = ArgumentSpec(["--out"], default=None, help="Path to save the code snippets text file.")


def main() -> None:
    GetCodeSnippetsArgs.load()
    path_or_pkgname = GetCodeSnippetsArgs.path_or_pkgname.value_not_none
    cs = CodeSnippets.from_path_or_pkgname(
        path_or_pkgname=path_or_pkgname,
        ban_file_patterns=GetCodeSnippetsArgs.ban_file_patterns.value,
        glob_patterns=GetCodeSnippetsArgs.glob_patterns.value_not_none,
        case_sensitive=GetCodeSnippetsArgs.case_sensitive.value_not_none,
    )
    (out := GetCodeSnippetsArgs.out.value or Path(__file__).with_suffix(".txt")).write_text(
        cs.snippets_text, encoding="utf-8"
    )
    print(f"[*] Extracted code snippets from `{path_or_pkgname}` and saved to `{out}`.")


if __name__ == "__main__":
    main()
