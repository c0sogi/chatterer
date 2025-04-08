#!/usr/bin/env python3
"""
A CLI app to convert PDFs to Markdown using the multimodal LLM (Chatterer).
Processes a single PDF or all PDFs within a specified directory, page by page,
and outputs Markdown files.
"""

import argparse
import sys
from pathlib import Path

# Ensure the script can find the chatterer library relative to its location
# sys.path.append(".") # Uncomment if chatterer is in the current dir or needs specific pathing
try:
    from chatterer import Chatterer, PdfToMarkdown
except ImportError:
    print("[!] Failed to import 'chatterer'. Make sure it's installed and accessible.")
    sys.exit(1)


def parse_page_indices(pages_str: str) -> list[int] | None:
    """
    Parse a string representing page indices, supporting individual pages and ranges.
    Returns None if input is None or empty.

    For example, '1,3,5-9' returns [1, 3, 5, 6, 7, 8, 9].
    Handles empty or None input gracefully.
    """
    if not pages_str:
        return None

    indices: set[int] = set()
    parts = pages_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:  # Skip empty parts that might result from trailing commas etc.
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                # Allow empty start/end to mean beginning/end if needed in future,
                # but for now require both.
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start <= 0 or end <= 0:
                    raise ValueError("Page numbers must be positive.")
                if start > end:
                    raise ValueError(f"Range start {start} cannot be greater than end {end}.")
                indices.update(range(start, end + 1))
            except ValueError as ve:
                # Provide more context in the error message
                raise ValueError(
                    f"Invalid range specification '{part}'. Use format 'start-end' with positive integers. Error: {ve}"
                )
            except Exception as e:  # Catch broader errors during range parsing
                raise ValueError(f"Error parsing range '{part}': {e}")

        else:
            try:
                page_num = int(part)
                if page_num <= 0:
                    raise ValueError("Page number must be positive.")
                indices.add(page_num)
            except ValueError:
                raise ValueError(f"Invalid page index '{part}'. Must be a positive integer.")
            except Exception as e:  # Catch broader errors during single page parsing
                raise ValueError(f"Error parsing page index '{part}': {e}")

    if not indices:
        raise ValueError("No valid page numbers were specified.")

    return sorted(list(indices))


def process_pdf(pdf_path: Path, pdf_converter: PdfToMarkdown, output_path: Path, page_indices: list[int] | None):
    """Processes a single PDF file."""
    print(f"[*] Processing {pdf_path.name}...")
    try:
        markdown = pdf_converter.convert(str(pdf_path), page_indices)
        print(f"[*] Saving generated Markdown to {output_path}...")
        # Ensure parent directory exists for the output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"[*] Successfully converted {pdf_path.name} to {output_path.name}.")
    except FileNotFoundError:
        print(f"[!] Error: Source PDF not found at {pdf_path}")
    except Exception as e:
        print(f"[!] Conversion failed for {pdf_path.name}: {e}. Skipping this file.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF file(s) to Markdown using the Chatterer LLM.",
        formatter_class=argparse.RawTextHelpFormatter,  # Preserve formatting in help text
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to the source PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        help="REQUIRED: The LLM backend and model (e.g., openai:gpt-4o-mini, anthropic:claude-3-opus-20240229, google:gemini-1.5-flash).",
        required=True,
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page indices to convert (e.g., '1,3,5-9'). Applies to all processed PDFs. If omitted, converts all pages.",
    )
    parser.add_argument(
        "--out",
        type=str,  # Changed to str to handle both file and dir cases initially
        default=None,  # Default to None, handle logic below
        help=(
            "Output path.\n"
            "- If source is a FILE: Path to the output Markdown file. Defaults to '<source_name>.md' in the same directory.\n"
            "- If source is a DIRECTORY: Path to the output directory where .md files will be saved. Defaults to the source directory."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",  # Add a flag for recursive search
        help="If source is a directory, search for PDFs recursively in subdirectories.",
    )

    args = parser.parse_args()
    source_path = Path(args.source).resolve()  # Use resolve for absolute path

    if not source_path.exists():
        print(f"[!] Source path not found: {args.source}")
        sys.exit(1)

    page_indices = None
    if args.pages:
        try:
            page_indices = parse_page_indices(args.pages)
            print(f"[*] Applying page filter: {args.pages} (indices: {page_indices})")
        except ValueError as e:
            print(
                f"[!] Invalid page specification: {e}. Use comma-separated positive integers and ranges (e.g., 1,3,5-9)."
            )
            sys.exit(1)

    # --- Determine PDF files to process ---
    pdf_files_to_process: list[Path] = []
    output_is_directory = False

    if source_path.is_file():
        if source_path.suffix.lower() != ".pdf":
            print(f"[!] Source file is not a PDF: {args.source}")
            sys.exit(1)
        pdf_files_to_process.append(source_path)
        output_is_directory = False
        print(f"[*] Target: Single file - {source_path.name}")

    elif source_path.is_dir():
        print(f"[*] Target: Directory - {source_path}")
        output_is_directory = True
        search_pattern = "*.pdf"
        if args.recursive:
            print("[*] Searching for PDF files recursively...")
            pdf_iterator = source_path.rglob(search_pattern)
        else:
            print("[*] Searching for PDF files in the top-level directory...")
            pdf_iterator = source_path.glob(search_pattern)

        pdf_files_to_process = sorted([f for f in pdf_iterator if f.is_file()])  # Ensure they are files

        if not pdf_files_to_process:
            print(f"[!] No PDF files found in {source_path}" + (" (recursively)." if args.recursive else "."))
            sys.exit(0)  # Exit gracefully if no PDFs found

        print(f"[*] Found {len(pdf_files_to_process)} PDF file(s) to process.")

    else:
        print(f"[!] Source path is neither a file nor a directory: {args.source}")
        sys.exit(1)

    # --- Determine Output Path(s) ---
    output_base: Path
    if args.out:
        output_base = Path(args.out)
        if output_is_directory:
            # If source is dir, --out specifies the output directory
            print(f"[*] Output directory specified: {output_base}")
            # Create the output directory if it doesn't exist
            output_base.mkdir(parents=True, exist_ok=True)
        else:
            # If source is file, --out specifies the output file
            print(f"[*] Output file specified: {output_base}")
            # Ensure parent dir exists for the single output file
            output_base.parent.mkdir(parents=True, exist_ok=True)

    else:
        # Default output locations
        if output_is_directory:
            # Default output dir is the source dir itself
            output_base = source_path
            print(f"[*] Output directory not specified, using source directory: {output_base}")
        else:
            # Default output file is source_name.md in the same dir
            output_base = source_path.with_suffix(".md")
            print(f"[*] Output file not specified, defaulting to: {output_base}")

    # --- Initialize Converter ---
    try:
        chatterer = Chatterer.from_provider(args.llm)
        pdf_converter = PdfToMarkdown(chatterer=chatterer)
        print(f"[*] Using LLM: {args.llm}")
    except Exception as e:
        print(f"[!] Failed to initialize Chatterer or PdfToMarkdown: {e}")
        sys.exit(1)

    # --- Process Files ---
    total_files = len(pdf_files_to_process)
    for i, pdf_path in enumerate(pdf_files_to_process, 1):
        print(f"\n--- Processing file {i}/{total_files} ---")
        output_path: Path
        if output_is_directory:
            # Construct output path inside the output directory
            # Ensure .md suffix, even if original had .PDF
            output_filename = pdf_path.stem + ".md"
            output_path = output_base / output_filename
        else:
            # Use the specific output file path determined earlier
            output_path = output_base  # Already set correctly

        process_pdf(pdf_path, pdf_converter, output_path, page_indices)

    print(f"\n[*] Finished processing all {total_files} file(s).")


if __name__ == "__main__":
    main()
