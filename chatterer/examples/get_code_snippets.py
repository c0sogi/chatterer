def resolve_import_path_and_get_logger():
    # ruff: noqa: E402
    import logging
    import sys

    if __name__ == "__main__" and "." not in sys.path:
        sys.path.append(".")

    logger = logging.getLogger(__name__)
    return logger


logger = resolve_import_path_and_get_logger()
from pathlib import Path
from typing import Optional

from spargear import ArgumentSpec, BaseArguments

from chatterer import CodeSnippets


class GetCodeSnippetsArgs(BaseArguments):
    path_or_pkgname: ArgumentSpec[str] = ArgumentSpec(
        ["path_or_pkgname"], help="Path to the package or file from which to extract code snippets."
    )
    out_path: Optional[str] = None
    ban_file_patterns: list[str] = [".venv/*", Path(__file__).relative_to(Path.cwd()).as_posix()]
    """List of file patterns to ignore."""
    glob_patterns: list[str] = ["*.py"]
    """List of glob patterns to include."""
    case_sensitive: bool = False
    """Enable case-sensitive matching for glob patterns."""
    prevent_save_file: bool = False
    """Prevent saving the extracted code snippets to a file."""

    def run(self) -> CodeSnippets:
        path_or_pkgname = self.path_or_pkgname.unwrap()
        if not self.prevent_save_file:
            if not self.out_path:
                out_path = Path(__file__).with_suffix(".txt")
            else:
                out_path = Path(self.out_path)
        else:
            out_path = None

        cs = CodeSnippets.from_path_or_pkgname(
            path_or_pkgname=path_or_pkgname,
            ban_file_patterns=self.ban_file_patterns,
            glob_patterns=self.glob_patterns,
            case_sensitive=self.case_sensitive,
        )
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cs.snippets_text, encoding="utf-8")
            logger.info(f"Extracted code snippets from `{path_or_pkgname}` and saved to `{out_path}`.")
        else:
            logger.info(f"Extracted code snippets from `{path_or_pkgname}`.")
        return cs


if __name__ == "__main__":
    GetCodeSnippetsArgs().run()
