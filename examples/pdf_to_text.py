import argparse
import sys
from pathlib import Path

sys.path.append(".")
from chatterer.tools.convert_to_text import pdf_to_text


def parse_page_indices(pages_str: str) -> list[int]:
    """
    Parse a string representing page indices, supporting individual pages and ranges.

    For example, '1,3,5-9' returns [1, 3, 5, 6, 7, 8, 9].
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
    parser = argparse.ArgumentParser(description="Extract text from a PDF and insert page markers.")
    parser.add_argument("source", type=str, help="Path to the PDF file.")
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page indices to extract, e.g. '1,3,5-9'. If not provided, extracts all pages.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".txt"),
        help="Output file path.",
    )

    args = parser.parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        print(f"[!] File not found: {source_path}")
        sys.exit(1)

    page_indices = None
    if args.pages:
        try:
            page_indices = parse_page_indices(args.pages)
        except ValueError as e:
            print(f"[!] {e}. Use comma-separated numbers and ranges (e.g. 1,3,5-9).")
            sys.exit(1)

    try:
        text = pdf_to_text(source_path, page_indices)
        if args.out:
            print(f"[*] Saving extracted text to {args.out}...")
            args.out.write_text(text, encoding="utf-8")
            print("[*] Done. It’s written. Now go do something with it, maybe?")
        else:
            print("[*] Extracted text:\n")
            print(text)
    except FileNotFoundError:
        print(f"[!] Could not open file: {args.source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
