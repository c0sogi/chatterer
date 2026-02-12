#!/usr/bin/env python3
"""
Cross-platform PPTX to PDF Converter
Supports Windows/Mac/Linux

Priority:
1. PowerPoint (best quality)
2. LibreOffice (fallback)

Requirements:
- Python 3.8+
- Windows: Microsoft PowerPoint or LibreOffice
- Mac: Microsoft PowerPoint or LibreOffice
- Linux: LibreOffice
- loguru: pip install loguru
"""

import os
import subprocess
import sys
from dataclasses import InitVar, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, Optional

import typer
from loguru import logger

ConverterType = Literal["powerpoint", "libreoffice"]


@dataclass
class PPTXConverter:
    """PPTX to PDF converter with cross-platform support"""

    preferred_converter: InitVar[Optional[ConverterType]] = None
    """Preferred converter to use. If not provided, the best available converter will be used."""
    converter: ConverterType = field(init=False)
    """Converter to use. If not provided, the best available converter will be used."""

    def __post_init__(self, preferred_converter: Optional[ConverterType]) -> None:
        """Detect available conversion tools"""
        if preferred_converter is not None:
            self.converter = preferred_converter
            return

        logger.info("Detecting conversion tools...")

        # Priority 1: PowerPoint
        if _find_powerpoint() is not None:
            logger.success(f"Microsoft PowerPoint detected ({sys.platform})")
            self.converter = "powerpoint"
            return

        # Priority 2: LibreOffice
        if _find_libreoffice() is not None:
            logger.success(f"LibreOffice detected ({sys.platform})")
            self.converter = "libreoffice"
            return

        # No converter found
        logger.error("No conversion tool found")
        logger.info("\n" + "=" * 60)
        logger.info("Installation Guide")
        logger.info("=" * 60)

        match sys.platform:
            case "win32":
                logger.info("\nRecommended: Microsoft PowerPoint")
                logger.info("  - Purchase and install Microsoft Office")
                logger.info("  - Or subscribe to Microsoft 365")
                logger.info("\nAlternative: LibreOffice (Free)")
                logger.info("  - https://www.libreoffice.org/download")
                logger.info("  - Or via winget: winget install LibreOffice.LibreOffice")

            case "darwin":  # Mac
                logger.info("\nRecommended: Microsoft PowerPoint")
                logger.info("  - Purchase and install Microsoft Office")
                logger.info("  - Or subscribe to Microsoft 365")
                logger.info("\nAlternative: LibreOffice (Free)")
                logger.info("  - brew install --cask libreoffice")

            case _:  # Linux
                logger.info("\nLibreOffice Installation (Free)")
                logger.info("  - Ubuntu/Debian: sudo apt install libreoffice")
                logger.info("  - Fedora: sudo dnf install libreoffice")
                logger.info("  - Arch: sudo pacman -S libreoffice-fresh")
                logger.info("  - Snap: sudo snap install libreoffice")

        logger.info("=" * 60 + "\n")
        raise RuntimeError("No conversion tool found")

    def convert(self, pptx_path: str | Path, pdf_path: Optional[str | Path] = None) -> bool:
        pptx_path_obj: Path = Path(pptx_path).resolve()

        # Set PDF path
        pdf_path_obj: Path
        if pdf_path is None:
            pdf_path_obj = pptx_path_obj.with_suffix(".pdf")
        else:
            pdf_path_obj = Path(pdf_path).resolve()

        pdf_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Execute conversion
        logger.info(f"Converting: {pptx_path_obj.name}")
        logger.info(f"Converter: {self.converter}")

        match self.converter:
            case "powerpoint":
                return _convert_with_powerpoint(pptx_path_obj, pdf_path_obj)
            case "libreoffice":
                return _convert_with_libreoffice(pptx_path_obj, pdf_path_obj)
            case _:  # pyright: ignore[reportUnnecessaryComparison]
                raise RuntimeError(f"Unsupported converter: {self.converter}")

    @classmethod
    def batch_convert(cls, directory: str | Path, output_dir: Optional[str | Path] = None) -> None:
        """
        Batch convert presentations in a directory

        Args:
            directory: Directory containing presentation files
            output_dir: Output directory for PDF files (optional)
        """
        directory_path: Path = Path(directory)

        if not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return

        # Find presentation files
        patterns: list[str] = ["*.pptx", "*.ppt", "*.odp"]
        files: list[Path] = []
        for pattern in patterns:
            files.extend(directory_path.glob(pattern))

        if not files:
            logger.warning(f"No presentation files found in directory: {directory_path}")
            return

        logger.info(f"\nConverting {len(files)} file(s)...")

        converter = cls()
        if not converter.converter:
            return

        success_count: int = 0
        for pptx_file in files:
            pdf_file_path: Path
            if output_dir:
                pdf_file_path = Path(output_dir) / pptx_file.with_suffix(".pdf").name
            else:
                pdf_file_path = pptx_file.with_suffix(".pdf")

            if converter.convert(pptx_file, pdf_file_path):
                success_count += 1
            logger.info("")

        logger.info(f"Completed: {success_count}/{len(files)} file(s) converted successfully")


def _convert_with_powerpoint(pptx_path: Path, pdf_path: Path) -> bool:
    """Convert using PowerPoint"""
    try:
        if sys.platform == "win32":
            import win32com.client

            powerpoint: Optional[Any] = None
            presentation: Optional[Any] = None

            try:
                # Launch PowerPoint
                powerpoint = win32com.client.Dispatch("PowerPoint.Application")
                powerpoint.Visible = 1

                # Open presentation
                presentation = powerpoint.Presentations.Open(str(pptx_path), WithWindow=False)
                if presentation is None:
                    raise RuntimeError(f"Cannot open powerpoint with path: {pptx_path}")

                # Save as PDF (32 = ppSaveAsPDF)
                presentation.SaveAs(str(pdf_path), 32)

                logger.success(f"Conversion completed: {pdf_path}")
                return True

            except Exception as e:
                logger.error(f"Error: {e}")
                return False

            finally:
                # Cleanup
                if presentation:
                    presentation.Close()
                if powerpoint:
                    powerpoint.Quit()
        elif sys.platform == "darwin":
            return _convert_powerpoint_mac(pptx_path, pdf_path)
        else:
            return False
    except Exception as e:
        logger.error(f"PowerPoint conversion failed: {e}")
        logger.info("Retrying with LibreOffice...")
        if _find_libreoffice() is not None:
            return _convert_with_libreoffice(pptx_path, pdf_path)
        return False


def _convert_powerpoint_mac(pptx_path: Path, pdf_path: Path) -> bool:
    """Convert using Mac PowerPoint (AppleScript)"""
    applescript: str = f'''
    set pptxFile to POSIX file "{pptx_path}" as alias
    set pdfFile to POSIX file "{pdf_path}"
    
    tell application "Microsoft PowerPoint"
        activate
        open pptxFile
        save active presentation in pdfFile as save as PDF
        close active presentation saving no
    end tell
    '''

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=120
        )

        if result.returncode == 0:
            logger.success(f"Conversion completed: {pdf_path}")
            return True
        else:
            logger.error(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Conversion timeout")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def _convert_with_libreoffice(pptx_path: Path, pdf_path: Path) -> bool:
    """Convert using LibreOffice"""
    libreoffice_path: Optional[str] = _find_libreoffice()

    if not libreoffice_path:
        logger.error("LibreOffice not found")
        return False

    try:
        # Convert to temporary directory
        temp_output: Path = pptx_path.parent

        cmd: list[str] = [
            libreoffice_path,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(temp_output),
            str(pptx_path),
        ]

        result: subprocess.CompletedProcess[str] = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            # Move generated PDF
            generated_pdf: Path = temp_output / pptx_path.with_suffix(".pdf").name

            if generated_pdf.exists():
                if generated_pdf != pdf_path:
                    if pdf_path.exists():
                        pdf_path.unlink()
                    generated_pdf.rename(pdf_path)
                logger.success(f"Conversion completed: {pdf_path}")
                return True
            else:
                logger.error("PDF file was not generated")
                return False
        else:
            logger.error("Conversion failed")
            if result.stderr:
                logger.error(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error("Conversion timeout")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


@cache
def _find_powerpoint(
    windows_powerpoint_class_name: str = "PowerPoint.Application",
    mac_powerpoint_app_path: str = "/Applications/Microsoft PowerPoint.app",
) -> Optional[str]:
    """Find PowerPoint executable"""
    try:
        if sys.platform == "win32":
            # Windows: Check COM object
            import win32com.client

            try:
                win32com.client.Dispatch(windows_powerpoint_class_name)
                return windows_powerpoint_class_name
            except Exception:
                return None

        elif sys.platform == "darwin":  # Mac
            # Mac: Check application existence
            return mac_powerpoint_app_path if os.path.exists(mac_powerpoint_app_path) else None

        else:  # Linux
            # PowerPoint not supported on Linux
            return None

    except ImportError:
        # win32com not available
        return None
    except Exception:
        return None


@cache
def _find_libreoffice() -> Optional[str]:
    """Find LibreOffice executable"""
    possible_paths: list[str]

    if sys.platform == "win32":
        possible_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\LibreOffice\program\soffice.exe"),
        ]
    elif sys.platform == "darwin":
        possible_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            os.path.expanduser("~/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        ]
    else:  # Linux
        possible_paths = [
            "/usr/bin/soffice",
            "/usr/bin/libreoffice",
            "/snap/bin/libreoffice",
        ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Try to find in PATH
    try:
        cmd: list[str] = ["soffice", "--version"] if sys.platform != "win32" else ["soffice.exe", "--version"]
        result: subprocess.CompletedProcess[bytes] = subprocess.run(
            cmd,
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "soffice" if sys.platform != "win32" else "soffice.exe"
    except Exception:
        pass

    return None


def command(
    file_or_directory: Path = typer.Argument(help="Path to the directory/PPTX file to convert to PDF."),
    out: Optional[Path] = typer.Option(
        None,
        help="Path to the output directory/PDF file. If not provided, defaults to the input file with a .pdf suffix.",
    ),
) -> None:
    """Convert PPTX/PPT/ODP files to PDF."""
    file_or_directory_path: Path = Path(file_or_directory)
    supported_formats = (".pptx", ".ppt", ".odp")

    def export(file: Path) -> bool:
        output = out or file.with_suffix(".pdf")
        converter: PPTXConverter = PPTXConverter()
        if not converter.converter:
            logger.error(f"No conversion tool available: {file}")
            return False
        if converter.convert(file, output):
            logger.success(f"Conversion completed: {output}")
            return True
        else:
            logger.error(f"Conversion failed: {file}")
            return False

    if file_or_directory_path.is_dir():
        files = [p for p in file_or_directory_path.glob("*") if p.is_file() and p.suffix.lower() in supported_formats]
        if not files:
            logger.warning(f"No presentation files found in directory: {file_or_directory_path}")
            return
        for file in files:
            try:
                export(file)
            except Exception as e:
                logger.error(f"Error converting {file}: {e}")
        logger.info(f"Completed: {len(files)} file(s) converted successfully")
        return
    else:
        if not export(file_or_directory_path):
            return
        logger.info(f"Conversion completed: {file_or_directory_path.with_suffix('.pdf')}")
        return
