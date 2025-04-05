#!/usr/bin/env python3
"""
A CLI app to convert PDFs to Markdown using the multimodal LLM (Chatterer).
It processes PDFs page by page—because even PDFs need a little context—
and outputs a Markdown file for you to marvel at (or just store away).
"""

import argparse
import sys
from pathlib import Path

sys.path.append(".")
from chatterer import Chatterer, PdfToMarkdown


def parse_page_indices(pages_str: str) -> list[int]:
    """
    Parse a string representing page indices, supporting individual pages and ranges.

    For example, '1,3,5-9' returns [1, 3, 5, 6, 7, 8, 9].
    Yes, it's basic, but it's better than letting fate decide.
    """
    indices: set[int] = set()
    parts = pages_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    raise ValueError(f"Range start {start} is greater than end {end}.")
                indices.update(range(start, end + 1))
            except ValueError as ve:
                raise ValueError(f"Invalid range specification '{part}': {ve}")
        else:
            try:
                indices.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid page index '{part}'.")
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using the magical powers of Chatterer.")
    parser.add_argument(
        "source",
        type=str,
        help="Path to the PDF file.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        help="The LLM backend and model to use for filtering the markdown. (e.g. `openai:gpt-4o-mini` or `anthropic:claude-3-7-sonnet-20250219`, `google:gemini-2.0-flash`",
        required=True,
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page indices to convert, e.g. '1,3,5-9'. If omitted, converts all pages.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".md"),
        help="Output Markdown file path.",
    )

    args = parser.parse_args()
    if not Path(args.source).is_file():
        print(f"[!] File not found: {args.source}. Did you even check?")
        sys.exit(1)

    page_indices = None
    if args.pages:
        try:
            page_indices = parse_page_indices(args.pages)
        except ValueError as e:
            print(f"[!] {e}. Use comma-separated numbers and ranges (e.g. 1,3,5-9).")
            sys.exit(1)

    # Create a Chatterer instance.
    chatterer = Chatterer.from_provider(args.llm)
    pdf_converter = PdfToMarkdown(chatterer=chatterer)

    try:
        markdown = pdf_converter.convert(args.source, page_indices)
        if args.out:
            print(
                f"[*] Saving generated Markdown to {args.out}... because even your mediocre PDF deserves a proper home."
            )
            args.out.write_text(markdown, encoding="utf-8")
            print("[*] Done. It's written. Now go impress someone—or at least file it away.")
        else:
            print("[*] Generated Markdown:\n")
            print(markdown)
    except Exception as e:
        print(f"[!] Conversion failed: {e}. Looks like even PDFs can disappoint sometimes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
