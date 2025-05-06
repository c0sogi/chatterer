def resolve_import_path_and_get_logger():
    # ruff: noqa: E402
    import logging
    import sys

    if __name__ == "__main__" and "." not in sys.path:
        sys.path.append(".")

    logger = logging.getLogger(__name__)
    return logger


logger = resolve_import_path_and_get_logger()
import sys
from pathlib import Path
from typing import Optional

from spargear import ArgumentSpec, BaseArguments

from chatterer import Chatterer, PdfToMarkdown


class PdfToMarkdownArgs(BaseArguments):
    in_path: ArgumentSpec[str] = ArgumentSpec(
        ["in-path"], help="Path to the input PDF file or a directory containing PDF files."
    )
    out_path: Optional[str] = None
    """Output path. For a file, path to the output markdown file. For a directory, output directory for .md files."""
    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        default=None,
        help="Chatterer instance for communication.",
        type=Chatterer.from_provider,
        required=True,
    )
    pages: Optional[str] = None
    """Page indices to convert (e.g., '1,3,5-9')."""
    recursive: bool = False
    """If input is a directory, search for PDFs recursively."""

    def run(self) -> list[dict[str, str]]:
        in_path = Path(self.in_path.unwrap()).resolve()
        page_indices = parse_page_indices(self.pages) if self.pages else None
        pdf_files: list[Path] = []
        is_dir = False
        if in_path.is_file():
            if in_path.suffix.lower() != ".pdf":
                sys.exit(1)
            pdf_files.append(in_path)
        elif in_path.is_dir():
            is_dir = True
            pattern = "*.pdf"
            pdf_files = sorted([
                f for f in (in_path.rglob(pattern) if self.recursive else in_path.glob(pattern)) if f.is_file()
            ])
            if not pdf_files:
                sys.exit(0)
        else:
            sys.exit(1)
        if self.out_path:
            out_base = Path(self.out_path).resolve()
        elif is_dir:
            out_base = in_path
        else:
            out_base = in_path.with_suffix(".md")

        if is_dir:
            out_base.mkdir(parents=True, exist_ok=True)
        else:
            out_base.parent.mkdir(parents=True, exist_ok=True)

        converter = PdfToMarkdown(chatterer=self.chatterer.unwrap())
        results: list[dict[str, str]] = []
        for pdf in pdf_files:
            out_path = (out_base / (pdf.stem + ".md")) if is_dir else out_base
            md = converter.convert(str(pdf), page_indices)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
            results.append({"input": pdf.as_posix(), "output": out_path.as_posix(), "result": md})
        logger.info(f"Converted {len(pdf_files)} PDF(s) to markdown and saved to `{out_base}`.")
        return results


def parse_page_indices(pages_str: str) -> list[int] | None:
    if not pages_str:
        return None
    indices: set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                raise ValueError
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    if not indices:
        raise ValueError
    return sorted(indices)


if __name__ == "__main__":
    PdfToMarkdownArgs().run()
