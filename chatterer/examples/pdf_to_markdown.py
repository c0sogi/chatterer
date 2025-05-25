import logging
import sys
from pathlib import Path
from typing import Optional

from spargear import ArgumentSpec, BaseArguments

from chatterer import Chatterer, PdfToMarkdown

logger = logging.getLogger(__name__)


class PdfToMarkdownArgs(BaseArguments):
    input: str
    """Input PDF file or directory containing PDF files to convert to markdown."""
    output: Optional[str] = None
    """Output path. For a file, path to the output markdown file. For a directory, output directory for .md files."""
    """Chatterer instance for communication."""
    pages: Optional[str] = None
    """Page indices to convert (e.g., '1,3,5-9')."""
    recursive: bool = False
    """If input is a directory, search for PDFs recursively."""
    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        default_factory=lambda: Chatterer.from_provider("google:gemini-2.5-flash-preview-05-20"),
        help="Chatterer instance for communication.",
        type=Chatterer.from_provider,
    )

    def run(self) -> list[dict[str, str]]:
        input = Path(self.input).resolve()
        page_indices = parse_page_indices(self.pages) if self.pages else None
        pdf_files: list[Path] = []
        is_dir = False
        if input.is_file():
            if input.suffix.lower() != ".pdf":
                sys.exit(1)
            pdf_files.append(input)
        elif input.is_dir():
            is_dir = True
            pattern = "*.pdf"
            pdf_files = sorted([
                f for f in (input.rglob(pattern) if self.recursive else input.glob(pattern)) if f.is_file()
            ])
            if not pdf_files:
                sys.exit(0)
        else:
            sys.exit(1)
        if self.output:
            out_base = Path(self.output).resolve()
        elif is_dir:
            out_base = input
        else:
            out_base = input.with_suffix(".md")

        if is_dir:
            out_base.mkdir(parents=True, exist_ok=True)
        else:
            out_base.parent.mkdir(parents=True, exist_ok=True)

        converter = PdfToMarkdown(chatterer=self.chatterer.unwrap())
        results: list[dict[str, str]] = []
        for pdf in pdf_files:
            output = (out_base / (pdf.stem + ".md")) if is_dir else out_base
            md = converter.convert(str(pdf), page_indices)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(md, encoding="utf-8")
            results.append({"input": pdf.as_posix(), "output": output.as_posix(), "result": md})
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


def main() -> None:
    PdfToMarkdownArgs().run()


if __name__ == "__main__":
    main()
