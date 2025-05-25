import logging
from pathlib import Path
from typing import Optional

from spargear import BaseArguments

from chatterer import CodeSnippets

logger = logging.getLogger(__name__)


class GetCodeSnippetsArgs(BaseArguments):
    input: str
    """Path to the package or file from which to extract code snippets."""
    output: Optional[str] = None
    """Output path for the extracted code snippets. If not provided, defaults to a file with the same name as the input."""
    ban_file_patterns: list[str] = [".venv/*", Path(__file__).relative_to(Path.cwd()).as_posix()]
    """List of file patterns to ignore."""
    glob_patterns: list[str] = ["*.py"]
    """List of glob patterns to include."""
    case_sensitive: bool = False
    """Enable case-sensitive matching for glob patterns."""
    prevent_save_file: bool = False
    """Prevent saving the extracted code snippets to a file."""

    def run(self) -> CodeSnippets:
        if not self.prevent_save_file:
            if not self.output:
                output = Path(__file__).with_suffix(".txt")
            else:
                output = Path(self.output)
        else:
            output = None

        cs = CodeSnippets.from_path_or_pkgname(
            path_or_pkgname=self.input,
            ban_file_patterns=self.ban_file_patterns,
            glob_patterns=self.glob_patterns,
            case_sensitive=self.case_sensitive,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(cs.snippets_text, encoding="utf-8")
            logger.info(f"Extracted code snippets from `{self.input}` and saved to `{output}`.")
        else:
            logger.info(f"Extracted code snippets from `{self.input}`.")
        return cs


def main() -> None:
    GetCodeSnippetsArgs().run()


if __name__ == "__main__":
    main()
