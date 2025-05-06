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

from spargear import ArgumentSpec, BaseArguments

from chatterer.tools.convert_to_text import pdf_to_text


class PdfToTextArgs(BaseArguments):
    in_path: ArgumentSpec[Path] = ArgumentSpec(["in-path"], help="Path to the PDF file.")
    out_path: ArgumentSpec[Path] = ArgumentSpec(["--out-path"], default=None, help="Output file path.")
    pages: ArgumentSpec[str] = ArgumentSpec(["--pages"], default=None, help="Page indices to extract, e.g. '1,3,5-9'.")

    def run(self) -> None:
        input = self.in_path.unwrap().resolve()
        out = self.out_path.value or input.with_suffix(".txt")
        if not input.is_file():
            sys.exit(1)
        out.write_text(
            pdf_to_text(input, parse_page_indices(pages_arg) if (pages_arg := self.pages.value) else None),
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


if __name__ == "__main__":
    PdfToTextArgs().run()
