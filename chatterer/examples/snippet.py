import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from spargear import RunnableArguments

from chatterer import CodeSnippets

logger = logging.getLogger(__name__)


class Arguments(RunnableArguments[CodeSnippets]):
    PATH_OR_PACKAGE_NAME: str
    """Path to the package or file from which to extract code snippets."""
    output: Optional[str] = None
    """Output path for the extracted code snippets. If not provided, defaults to a file with the current timestamp."""
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
                output = Path(datetime.now().strftime("%Y%m%d_%H%M%S") + "_snippets.txt")
            else:
                output = Path(self.output)
        else:
            output = None

        cs = CodeSnippets.from_path_or_pkgname(
            path_or_pkgname=self.PATH_OR_PACKAGE_NAME,
            ban_file_patterns=self.ban_file_patterns,
            glob_patterns=self.glob_patterns,
            case_sensitive=self.case_sensitive,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(cs.snippets_text, encoding="utf-8")
            logger.info(f"Extracted code snippets from `{self.PATH_OR_PACKAGE_NAME}` and saved to `{output}`.")
        else:
            logger.info(f"Extracted code snippets from `{self.PATH_OR_PACKAGE_NAME}`.")
        return cs


def main() -> None:
    Arguments().run()


if __name__ == "__main__":
    main()
