import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from chatterer.tools.textify import pdf_to_text


def command(
    pdf_path: Path = typer.Argument(help="Path to the PDF file to convert to text."),
    output: Optional[Path] = typer.Option(None, help="Path to the output text file. If not provided, defaults to the input file with a .txt suffix."),
    page: Optional[str] = typer.Option(None, help="Comma-separated list of zero-based page indices to extract from the PDF. Supports ranges, e.g., '0,2,4-8'."),
) -> None:
    """Extract text from PDF files."""
    input_file = pdf_path.resolve()
    out = output or input_file.with_suffix(".txt")
    if not input_file.is_file():
        sys.exit(1)
    out.write_text(
        pdf_to_text(path_or_file=input_file, page_indices=page),
        encoding="utf-8",
    )
    logger.info(f"Extracted text from `{input_file}` to `{out}`")


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
