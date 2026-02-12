from pathlib import Path
from typing import Literal, Optional

import typer
from langchain_core.documents.base import Blob
from loguru import logger

from chatterer import Chatterer
from chatterer.tools.upstage import (
    DEFAULT_IMAGE_DIR,
    DOCUMENT_PARSE_BASE_URL,
    DOCUMENT_PARSE_DEFAULT_MODEL,
    OCR,
    Category,
    OutputFormat,
    UpstageDocumentParseParser,
)


def command(
    input_path: Path = typer.Argument(help="Input file to parse. Can be a PDF, image, or other supported formats."),
    output: Optional[Path] = typer.Option(
        None, help="Output file path for the parsed content. Defaults to input file with .md suffix if not provided."
    ),
    api_key: Optional[str] = typer.Option(None, help="API key for the Upstage API."),
    base_url: str = typer.Option(DOCUMENT_PARSE_BASE_URL, help="Base URL for the Upstage API."),
    model: str = typer.Option(DOCUMENT_PARSE_DEFAULT_MODEL, help="Model to use for parsing."),
    split: Literal["none", "page", "element"] = typer.Option("none", help="Split type for the parsed content."),
    ocr: OCR = typer.Option("auto", help="OCR type for parsing."),
    output_format: OutputFormat = typer.Option("markdown", help="Output format for the parsed content."),
    coordinates: bool = typer.Option(False, help="Whether to include coordinates in the output."),
    base64_encoding: Optional[list[Category]] = typer.Option(
        None, help="Base64 encoding for specific categories in the parsed content."
    ),
    image_description_instruction: str = typer.Option(
        "Describe the image in detail.", help="Instruction for generating image descriptions."
    ),
    image_dir: str = typer.Option(DEFAULT_IMAGE_DIR, help="Directory to save images extracted from the document."),
    chatterer: Optional[str] = typer.Option(None, help="Chatterer instance for communication."),
) -> None:
    """Parse documents using Upstage API."""
    input_resolved = input_path.resolve()
    out = output or input_resolved.with_suffix(".md")
    chatterer_obj = Chatterer.from_provider(chatterer) if chatterer else None
    base64_categories: list[Category] = base64_encoding or ["figure"]

    parser = UpstageDocumentParseParser(
        api_key=api_key,
        base_url=base_url,
        model=model,
        split=split,
        ocr=ocr,
        output_format=output_format,
        coordinates=coordinates,
        base64_encoding=base64_categories,
        image_description_instruction=image_description_instruction,
        image_dir=image_dir,
        chatterer=chatterer_obj,
    )
    docs = parser.parse(Blob.from_path(input_resolved))  # pyright: ignore[reportUnknownMemberType]

    if image_dir:
        for path_str, image in parser.image_data.items():
            (p := Path(path_str)).parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(image)
            logger.info(f"Saved image to `{p}`")

    markdown: str = "\n\n".join(f"<!--- page {i} -->\n{doc.page_content}" for i, doc in enumerate(docs, 1))
    out.write_text(markdown, encoding="utf-8")
    logger.info(f"Parsed `{input_resolved}` to `{out}`")
