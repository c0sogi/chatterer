# pdf_to_text.py
import sys
from pathlib import Path

sys.path.append(".")
from chatterer import ArgumentSpec, BaseArguments
from chatterer.tools.convert_to_text import pdf_to_text


class PdfToTextArgs(BaseArguments):
    input: ArgumentSpec[Path] = ArgumentSpec(["input"], help="Path to the PDF file.")
    pages: ArgumentSpec[str] = ArgumentSpec(["--pages"], default=None, help="Page indices to extract, e.g. '1,3,5-9'.")
    out: ArgumentSpec[Path] = ArgumentSpec(["--out"], default=None, help="Output file path.")


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
    PdfToTextArgs.load()
    input = PdfToTextArgs.input.value_not_none.resolve()
    out = PdfToTextArgs.out.value or input.with_suffix(".txt")
    if not input.is_file():
        sys.exit(1)
    out.write_text(
        pdf_to_text(input, parse_page_indices(pages_arg) if (pages_arg := PdfToTextArgs.pages.value) else None),
        encoding="utf-8",
    )
    print(f"[*] Extracted text from `{input}` to `{out}`")


if __name__ == "__main__":
    main()
