import argparse
import sys
from pathlib import Path

sys.path.append(".")

from chatterer import Chatterer, PdfToMarkdownConverter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using Chatterer.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    parser.add_argument(
        "--page_indices",
        type=int,
        nargs="+",
        default=None,
        help="List of page indices to convert (default: all pages).",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path
    md_path = Path(pdf_path).with_suffix(".md")
    page_indices = args.page_indices
    converter = PdfToMarkdownConverter(chatterer=Chatterer.google("gemini-2.0-flash"))
    md = converter.convert(pdf_path, page_indices=page_indices)
    md_path.write_text(md, encoding="utf-8")
    print(f"[*] PDF converted to markdown and saved to {md_path}")
    print("[*] Done!")
    print("[*] Press Enter to exit.")
