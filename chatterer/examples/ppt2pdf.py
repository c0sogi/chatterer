#!/usr/bin/env python3
"""
PPTX to PDF Converter Example

A simple example demonstrating how to convert PowerPoint presentations to PDF format.
Supports both single file and batch conversion with automatic tool detection.

Example usage:
    python -m chatterer.examples.ppt2pdf /path/to/presentation.pptx
    python -m chatterer.examples.ppt2pdf /path/to/presentations/ --output /path/to/output/
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from chatterer.tools.ppt2pdf import PPTXConverter


def command(
    pptx_or_directory_path: Path = typer.Argument(
        help="Path to the PPTX file or directory containing presentation files to convert."
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Path to the output PDF file or directory. If not provided, defaults to the input file with a .pdf suffix.",
    ),
) -> None:
    """Convert PPTX presentations to PDF format."""
    pptx_path: Path = Path(pptx_or_directory_path).resolve()
    supported_formats = (".pptx", ".ppt", ".odp")

    if not pptx_path.exists():
        logger.error(f"Path does not exist: {pptx_path}")
        sys.exit(1)

    # Initialize converter
    try:
        converter: PPTXConverter = PPTXConverter()
    except RuntimeError as e:
        logger.error(f"{e}")
        sys.exit(1)

    if pptx_path.is_dir():
        # Batch conversion
        files = [p for p in pptx_path.glob("*") if p.is_file() and p.suffix.lower() in supported_formats]

        if not files:
            logger.warning(f"No presentation files found in directory: {pptx_path}")
            sys.exit(0)

        logger.info(f"Found {len(files)} presentation file(s)")
        success_count: int = 0

        for file in files:
            batch_output_path: Path
            if output:
                batch_output_path = Path(output) / file.with_suffix(".pdf").name
            else:
                batch_output_path = file.with_suffix(".pdf")

            try:
                if converter.convert(file, batch_output_path):
                    logger.success(f"Converted: {file.name} -> {batch_output_path}")
                    success_count += 1
                else:
                    logger.error(f"Failed to convert: {file.name}")
            except Exception as e:
                logger.error(f"Error converting {file.name}: {e}")

        logger.info(f"Completed: {success_count}/{len(files)} file(s) converted successfully")

    else:
        # Single file conversion
        if pptx_path.suffix.lower() not in supported_formats:
            logger.error(f"Unsupported file format: {pptx_path.suffix}")
            logger.info(f"Supported formats: {', '.join(supported_formats)}")
            sys.exit(1)

        single_output_path: Path = output or pptx_path.with_suffix(".pdf")

        try:
            if converter.convert(pptx_path, single_output_path):
                logger.success(f"Conversion completed: {single_output_path}")
            else:
                logger.error(f"Conversion failed: {pptx_path}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        typer.run(command)
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
