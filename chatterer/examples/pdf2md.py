#!/usr/bin/env python3
"""
PDF to Markdown Converter CLI

A command-line tool for converting PDF documents to Markdown using multimodal LLMs.
Supports both sequential and parallel processing modes with async capabilities.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

from spargear import ArgumentSpec, RunnableArguments

from chatterer import Chatterer
from chatterer.tools.convert_pdf_to_markdown import PdfToMarkdown


class ConversionResult(TypedDict, total=False):
    """Type definition for conversion results."""

    input: str
    output: str
    result: str
    processing_time: float
    characters: int
    error: str


# Setup enhanced logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class Arguments(RunnableArguments[List[ConversionResult]]):
    """Command-line arguments for PDF to Markdown conversion."""

    PDF_OR_DIRECTORY_PATH: str
    """Input PDF file or directory containing PDF files to convert to markdown."""

    output: Optional[str] = None
    """Output path. For a file, path to the output markdown file. For a directory, output directory for .md files."""

    page: Optional[str] = None
    """Zero-based page indices to convert (e.g., '0,2,4-8'). If None, converts all pages."""

    recursive: bool = False
    """If input is a directory, search for PDFs recursively."""

    mode: Literal["sequential", "parallel"] = "parallel"
    """Processing mode: 'sequential' for strict continuity, 'parallel' for faster processing."""

    sync: bool = False
    """Enable synchronous processing for sequential mode. If set to True, will run in sync mode."""

    max_concurrent: int = 10
    """Maximum number of concurrent LLM requests when using async mode."""

    image_zoom: float = 2.0
    """Zoom factor for rendering PDF pages as images (higher zoom = higher resolution)."""

    image_format: Literal["png", "jpg", "jpeg"] = "png"
    """Image format for PDF page rendering."""

    image_quality: int = 95
    """JPEG quality when using jpg/jpeg format (1-100)."""

    context_tail_lines: int = 10
    """Number of lines from previous page's markdown to use as context (sequential mode only)."""

    verbose: bool = False
    """Enable verbose logging output."""

    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        default_factory=lambda: Chatterer.from_provider("google:gemini-2.5-flash-preview-05-20"),
        help="Chatterer instance configuration (e.g., 'google:gemini-2.5-flash-preview-05-20').",
        type=Chatterer.from_provider,
    )

    def __post_init__(self) -> None:
        """Validate and adjust arguments after initialization."""
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        if not self.sync and self.mode == "sequential":
            logger.warning("Async mode is only available with parallel mode. Switching to parallel mode.")
            self.mode = "parallel"

        if self.max_concurrent < 1:
            logger.warning("max_concurrent must be >= 1. Setting to 1.")
            self.max_concurrent = 1
        elif self.max_concurrent > 10:
            logger.warning("max_concurrent > 10 may cause rate limiting. Consider reducing.")

    def run(self) -> List[ConversionResult]:
        """Execute the PDF to Markdown conversion."""
        if not self.sync:
            return asyncio.run(self._run_async())
        else:
            return self._run_sync()

    def _run_sync(self) -> List[ConversionResult]:
        """Execute synchronous conversion."""
        pdf_files, output_base, is_dir = self._prepare_files()

        converter = PdfToMarkdown(
            chatterer=self.chatterer.unwrap(),
            image_zoom=self.image_zoom,
            image_format=self.image_format,
            image_jpg_quality=self.image_quality,
            context_tail_lines=self.context_tail_lines,
        )

        results: List[ConversionResult] = []
        total_start_time = time.time()

        logger.info(f"🚀 Starting {self.mode} conversion of {len(pdf_files)} PDF(s)...")

        for i, pdf in enumerate(pdf_files, 1):
            output_path = (output_base / f"{pdf.stem}.md") if is_dir else output_base

            logger.info(f"📄 Processing {i}/{len(pdf_files)}: {pdf.name}")
            start_time = time.time()

            # Progress callback for individual PDF
            def progress_callback(current: int, total: int) -> None:
                progress = (current / total) * 100
                logger.info(f"  └─ Progress: {current}/{total} pages ({progress:.1f}%)")

            try:
                markdown = converter.convert(
                    pdf_input=str(pdf),
                    page_indices=self.page,
                    mode=self.mode,
                    progress_callback=progress_callback,
                )

                # Save result
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(markdown, encoding="utf-8")

                elapsed = time.time() - start_time
                chars_per_sec = len(markdown) / elapsed if elapsed > 0 else 0

                logger.info(f"  ✅ Completed in {elapsed:.1f}s ({chars_per_sec:.0f} chars/s)")
                logger.info(f"  📝 Generated {len(markdown):,} characters → {output_path}")

                results.append({
                    "input": pdf.as_posix(),
                    "output": output_path.as_posix(),
                    "result": markdown,
                    "processing_time": elapsed,
                    "characters": len(markdown),
                })

            except Exception as e:
                logger.error(f"  ❌ Failed to process {pdf.name}: {e}")
                results.append({
                    "input": pdf.as_posix(),
                    "output": "",
                    "result": "",
                    "error": str(e),
                })

        total_elapsed = time.time() - total_start_time
        total_chars = sum(len(r.get("result", "")) for r in results)
        successful_conversions = sum(1 for r in results if "error" not in r)

        logger.info("🎉 Conversion complete!")
        logger.info(f"  📊 Total time: {total_elapsed:.1f}s")
        logger.info(f"  📈 Success rate: {successful_conversions}/{len(pdf_files)} ({(successful_conversions / len(pdf_files) * 100):.1f}%)")
        logger.info(f"  📝 Total output: {total_chars:,} characters")
        logger.info(f"  ⚡ Average speed: {total_chars / total_elapsed:.0f} chars/s")

        return results

    async def _run_async(self) -> List[ConversionResult]:
        """Execute asynchronous conversion with parallel processing."""
        pdf_files, output_base, is_dir = self._prepare_files()

        converter = PdfToMarkdown(
            chatterer=self.chatterer.unwrap(),
            image_zoom=self.image_zoom,
            image_format=self.image_format,
            image_jpg_quality=self.image_quality,
            context_tail_lines=self.context_tail_lines,
        )

        total_start_time = time.time()

        logger.info(f"🚀 Starting ASYNC parallel conversion of {len(pdf_files)} PDF(s)...")
        logger.info(f"⚡ Max concurrent: {self.max_concurrent} LLM requests")

        # Process PDFs concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_pdf(pdf: Path, index: int) -> ConversionResult:
            async with semaphore:
                output_path = (output_base / f"{pdf.stem}.md") if is_dir else output_base

                logger.info(f"📄 Processing {index}/{len(pdf_files)}: {pdf.name}")
                start_time = time.time()

                # Progress callback for individual PDF
                def progress_callback(current: int, total: int) -> None:
                    progress = (current / total) * 100
                    logger.info(f"  └─ {pdf.name}: {current}/{total} pages ({progress:.1f}%)")

                try:
                    markdown = await converter.aconvert(
                        pdf_input=str(pdf),
                        page_indices=self.page,
                        progress_callback=progress_callback,
                        max_concurrent=self.max_concurrent,  # Limit per-PDF concurrency
                    )

                    # Save result
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(markdown, encoding="utf-8")

                    elapsed = time.time() - start_time
                    chars_per_sec = len(markdown) / elapsed if elapsed > 0 else 0

                    logger.info(f"  ✅ {pdf.name} completed in {elapsed:.1f}s ({chars_per_sec:.0f} chars/s)")
                    logger.info(f"  📝 Generated {len(markdown):,} characters → {output_path}")

                    return {
                        "input": pdf.as_posix(),
                        "output": output_path.as_posix(),
                        "result": markdown,
                        "processing_time": elapsed,
                        "characters": len(markdown),
                    }

                except Exception as e:
                    logger.error(f"  ❌ Failed to process {pdf.name}: {e}")
                    return {
                        "input": pdf.as_posix(),
                        "output": "",
                        "result": "",
                        "error": str(e),
                    }

        # Execute all PDF processing tasks
        tasks = [process_pdf(pdf, i) for i, pdf in enumerate(pdf_files, 1)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        final_results: List[ConversionResult] = []
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                final_results.append(ConversionResult(input="", output="", result="", error=str(result)))
            else:
                # Type narrowing: result is ConversionResult after isinstance check
                final_results.append(result)  # type: ignore[arg-type]

        total_elapsed = time.time() - total_start_time
        total_chars = sum(len(r.get("result", "")) for r in final_results)
        successful_conversions = sum(1 for r in final_results if "error" not in r)

        logger.info("🎉 ASYNC conversion complete!")
        logger.info(f"  📊 Total time: {total_elapsed:.1f}s")
        logger.info(f"  📈 Success rate: {successful_conversions}/{len(pdf_files)} ({(successful_conversions / len(pdf_files) * 100):.1f}%)")
        logger.info(f"  📝 Total output: {total_chars:,} characters")
        logger.info(f"  ⚡ Average speed: {total_chars / total_elapsed:.0f} chars/s")
        logger.info(f"  🚀 Speedup: ~{len(pdf_files) / max(1, total_elapsed / 60):.1f}x faster than sequential")

        return final_results

    def _prepare_files(self) -> tuple[List[Path], Path, bool]:
        """Prepare input and output file paths."""
        input_path = Path(self.PDF_OR_DIRECTORY_PATH).resolve()
        pdf_files: List[Path] = []
        is_dir = False

        # Determine input files
        if input_path.is_file():
            if input_path.suffix.lower() != ".pdf":
                logger.error(f"❌ Input file must be a PDF: {input_path}")
                sys.exit(1)
            pdf_files.append(input_path)
        elif input_path.is_dir():
            is_dir = True
            pattern = "**/*.pdf" if self.recursive else "*.pdf"
            pdf_files = sorted([f for f in input_path.glob(pattern) if f.is_file()])
            if not pdf_files:
                logger.warning(f"⚠️  No PDF files found in {input_path}")
                sys.exit(0)
        else:
            logger.error(f"❌ Input path does not exist: {input_path}")
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

        logger.info(f"📂 Input: {input_path}")
        logger.info(f"📁 Output: {output_base}")
        logger.info(f"📄 Found {len(pdf_files)} PDF file(s)")

        return pdf_files, output_base, is_dir


def main() -> None:
    """Main entry point for the CLI application."""
    args = None
    try:
        args = Arguments()
        args.run()
    except KeyboardInterrupt:
        logger.info("🛑 Conversion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args and hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
