def resolve_import_path():
    # ruff: noqa: E402
    import sys
    from pathlib import Path

    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.append(str(parent))


resolve_import_path()
import sys
from pathlib import Path

from chatterer import Chatterer, PdfToMarkdown
from spargear import ArgumentSpec, BaseArguments


class PdfToMarkdownArgs(BaseArguments):
    input: ArgumentSpec[Path] = ArgumentSpec(
        ["input"], help="Path to the input PDF file or a directory containing PDF files."
    )
    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        default=None,
        help="Chatterer instance for communication.",
        type=Chatterer.from_provider,
        required=True,
    )
    pages: ArgumentSpec[str] = ArgumentSpec(
        ["--pages"], default=None, help="Page indices to convert (e.g., '1,3,5-9')."
    )
    out: ArgumentSpec[Path] = ArgumentSpec(
        ["--out"],
        default=None,
        help="Output path. For a file, path to the output markdown file. For a directory, output directory for .md files.",
    )
    recursive: ArgumentSpec[bool] = ArgumentSpec(
        ["--recursive"],
        action="store_true",
        default=False,
        help="If input is a directory, search for PDFs recursively.",
    )


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
    PdfToMarkdownArgs.load()
    src_pinputth = PdfToMarkdownArgs.input.unwrap().resolve()
    pages_arg = PdfToMarkdownArgs.pages.value
    page_indices = parse_page_indices(pages_arg) if pages_arg else None
    pdf_files: list[Path] = []
    is_dir = False
    if src_pinputth.is_file():
        if src_pinputth.suffix.lower() != ".pdf":
            sys.exit(1)
        pdf_files.append(src_pinputth)
    elif src_pinputth.is_dir():
        is_dir = True
        pattern = "*.pdf"
        pdf_files = sorted([
            f
            for f in (src_pinputth.rglob(pattern) if PdfToMarkdownArgs.recursive.value else src_pinputth.glob(pattern))
            if f.is_file()
        ])
        if not pdf_files:
            sys.exit(0)
    else:
        sys.exit(1)
    out_base = (
        PdfToMarkdownArgs.out.value
        if PdfToMarkdownArgs.out.value
        else (src_pinputth if is_dir else src_pinputth.with_suffix(".md"))
    )
    if not out_base.exists():
        out_base.mkdir(parents=True, exist_ok=True) if is_dir else out_base.parent.mkdir(parents=True, exist_ok=True)
    converter = PdfToMarkdown(chatterer=PdfToMarkdownArgs.chatterer.unwrap())
    for pdf in pdf_files:
        out_path = (out_base / (pdf.stem + ".md")) if is_dir else out_base
        md = converter.convert(str(pdf), page_indices)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
