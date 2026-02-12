#!/usr/bin/env python3
"""
PDF to Markdown Converter CLI

A command-line tool for converting PDF documents to Markdown using multimodal LLMs.
Supports both sequential and parallel processing modes with async capabilities.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Literal, NamedTuple, Optional

import typer
from loguru import logger
from rich.progress import Progress, TaskID

from chatterer import Chatterer
from chatterer.constants import DEFAULT_GOOGLE_MODEL
from chatterer.tools.pdf2md import PdfToMarkdown


class ConversionResult(NamedTuple):
    """Type definition for conversion results."""

    input: str
    output: str
    result: str
    processing_time: float
    characters: int


@dataclass
class Pdf2MdRunner:
    """PDF to Markdown converter runner."""

    pdf_or_directory_path: str
    output: Optional[str] = None
    page: Optional[str] = None
    recursive: bool = False
    max_concurrent: int = 10
    image_zoom: float = 2.0
    image_format: Literal["png", "jpg", "jpeg"] = "png"
    image_quality: int = 95
    context_tail_lines: int = 10
    chatterer: Chatterer = field(default_factory=lambda: Chatterer.from_provider(f"google:{DEFAULT_GOOGLE_MODEL}"))

    def __post_init__(self) -> None:
        """Validate and adjust arguments after initialization."""
        if self.max_concurrent < 1:
            logger.warning("max_concurrent must be >= 1. Setting to 1.")
            self.max_concurrent = 1
        elif self.max_concurrent > 10:
            logger.warning("max_concurrent > 10 may cause rate limiting. Consider reducing.")

    def run(self) -> list[ConversionResult]:
        """Execute the PDF to Markdown conversion."""

        async def _run() -> list[ConversionResult]:
            result: list[ConversionResult] = []
            async for item in self.arun():
                result.append(item)
            return result

        return asyncio.get_event_loop().run_until_complete(_run())

    async def arun(self) -> AsyncIterator[ConversionResult]:
        """Execute asynchronous conversion with parallel processing."""
        pdf_files, output_base, is_dir = self._prepare_files()

        converter: PdfToMarkdown = PdfToMarkdown(
            chatterer=self.chatterer,
            image_zoom=self.image_zoom,
            image_format=self.image_format,
            image_jpg_quality=self.image_quality,
            context_tail_lines=self.context_tail_lines,
        )

        total_start_time: float = time.time()
        logger.info(
            f"Converting {len(pdf_files)} PDF(s) into markdown with {self.max_concurrent} concurrent requests..."
        )

        async def process_pdf(progress: Progress, task: TaskID, pdf: Path) -> ConversionResult | Exception:
            start_time: float = time.time()
            output_path = (output_base / f"{pdf.stem}.md") if is_dir else output_base
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Progress callback for individual PDF
            def progress_callback(current: int, total: int) -> None:
                if progress.tasks[task].total is None:
                    progress.update(task, total=total)
                progress.advance(task)

            try:
                markdown: str = await converter.aconvert(
                    pdf_input=str(pdf),
                    page_indices=self.page,
                    progress_callback=progress_callback,
                    max_concurrent=self.max_concurrent,  # Limit per-PDF concurrency
                )
            except Exception as e:
                logger.error(f"  Failed to process {pdf.name}: {e}")
                return e

            output_path.write_text(markdown, encoding="utf-8")

            elapsed: float = time.time() - start_time
            chars_per_sec: float = len(markdown) / elapsed if elapsed > 0 else float("nan")

            logger.info(f"  {pdf.name} completed in {elapsed:.1f}s ({chars_per_sec:.0f} chars/s)")
            logger.info(f"  Generated {len(markdown):,} characters -> {output_path}")

            return ConversionResult(
                input=pdf.as_posix(),
                output=output_path.as_posix(),
                result=markdown,
                processing_time=elapsed,
                characters=len(markdown),
            )

        total_chars: int = 0
        total_successes: int = 0
        with Progress() as progress:
            for i, pdf in enumerate(pdf_files, 1):
                task = progress.add_task(f"[{i}/{len(pdf_files)}] Processing {pdf.name} ...", total=None)
                result: ConversionResult | Exception = await process_pdf(progress, task, pdf)
                if isinstance(result, ConversionResult):
                    yield result
                    total_chars += len(result.result)
                    total_successes += 1
                else:
                    logger.error(f"  Failed to process {pdf.name}: {result}")
                progress.stop_task(task)

        total_elapsed: float = time.time() - total_start_time
        summary = {
            "Total Time (s)": total_elapsed,
            "Success Rate (%)": total_successes / len(pdf_files) * 100,
            "Total Output (chars)": total_chars,
            "Average Speed (chars/s)": total_chars / total_elapsed,
        }
        logger.info(f"Summary:\n{'\n'.join(f'\t{k}: {v}' for k, v in summary.items())}")

    def _prepare_files(self) -> tuple[list[Path], Path, bool]:
        """Prepare input and output file paths."""
        input_path = Path(self.pdf_or_directory_path).resolve()
        pdf_files: list[Path] = []
        is_dir = False

        # Determine input files
        if input_path.is_file():
            if input_path.suffix.lower() != ".pdf":
                logger.error(f"Input file must be a PDF: {input_path}")
                sys.exit(1)
            pdf_files.append(input_path)
        elif input_path.is_dir():
            is_dir = True
            pattern = "**/*.pdf" if self.recursive else "*.pdf"
            pdf_files = sorted([f for f in input_path.glob(pattern) if f.is_file()])
            if not pdf_files:
                logger.warning(f"No PDF files found in {input_path}")
                sys.exit(0)
        else:
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)

        # Determine output path
        if self.output:
            output_base = Path(self.output).resolve()
        elif is_dir:
            output_base = input_path
        else:
            output_base = input_path.with_suffix(".md")

        # Create output directories
        if is_dir:
            output_base.mkdir(parents=True, exist_ok=True)
        else:
            output_base.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_base}")
        logger.info(f"Found {len(pdf_files)} PDF file(s)")

        return pdf_files, output_base, is_dir


def command(
    pdf_or_directory_path: str = typer.Argument(
        help="Input PDF file or directory containing PDF files to convert to markdown."
    ),
    output: Optional[str] = typer.Option(
        None,
        help="Output path. For a file, path to the output markdown file. For a directory, output directory for .md files.",
    ),
    page: Optional[str] = typer.Option(
        None, help="Zero-based page indices to convert (e.g., '0,2,4-8'). If None, converts all pages."
    ),
    recursive: bool = typer.Option(False, help="If input is a directory, search for PDFs recursively."),
    max_concurrent: int = typer.Option(10, help="Maximum number of concurrent LLM requests when using async mode."),
    image_zoom: float = typer.Option(
        2.0, help="Zoom factor for rendering PDF pages as images (higher zoom = higher resolution)."
    ),
    image_format: Literal["png", "jpg", "jpeg"] = typer.Option(
        "png", help="Image format for PDF page rendering (png, jpg, jpeg)."
    ),
    image_quality: int = typer.Option(95, help="JPEG quality when using jpg/jpeg format (1-100)."),
    context_tail_lines: int = typer.Option(
        10, help="Number of lines from previous page's markdown to use as context (sequential mode only)."
    ),
    chatterer: str = typer.Option(
        f"google:{DEFAULT_GOOGLE_MODEL}",
        help=f"Chatterer instance configuration (e.g., 'google:{DEFAULT_GOOGLE_MODEL}').",
    ),
) -> None:
    """Convert PDF documents to Markdown using multimodal LLMs."""
    runner = Pdf2MdRunner(
        pdf_or_directory_path=pdf_or_directory_path,
        output=output,
        page=page,
        recursive=recursive,
        max_concurrent=max_concurrent,
        image_zoom=image_zoom,
        image_format=image_format,
        image_quality=image_quality,
        context_tail_lines=context_tail_lines,
        chatterer=Chatterer.from_provider(chatterer),
    )
    runner.run()


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        typer.run(command)
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
