import logging
import sys
from pathlib import Path
from typing import Optional

from spargear import RunnableArguments

from chatterer.tools.convert_to_text import pdf_to_text

logger = logging.getLogger(__name__)


class Arguments(RunnableArguments[None]):
    PDF_PATH: Path
    """Path to the PDF file to convert to text."""
    output: Optional[Path]
    """Path to the output text file. If not provided, defaults to the input file with a .txt suffix."""
    page: Optional[str] = None
    """Comma-separated list of zero-based page indices to extract from the PDF. Supports ranges, e.g., '0,2,4-8'."""

    def run(self) -> None:
        input = self.PDF_PATH.resolve()
        out = self.output or input.with_suffix(".txt")
        if not input.is_file():
            sys.exit(1)
        out.write_text(
            pdf_to_text(path_or_file=input, page_indices=self.page),
            encoding="utf-8",
        )
        logger.info(f"Extracted text from `{input}` to `{out}`")


def parse_page_indices(pages_str: str) -> list[int]:
    indices: set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


def main() -> None:
    Arguments().run()


if __name__ == "__main__":
    main()
