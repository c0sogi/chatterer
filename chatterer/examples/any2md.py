from pathlib import Path
from typing import Optional, TypedDict

import openai
import typer
from loguru import logger

from chatterer.tools.textify import anything_to_markdown


class AnythingToMarkdownReturns(TypedDict):
    input: str
    output: Optional[str]
    out_text: str


def command(
    source: str = typer.Argument(help="Input file to convert to markdown. Can be a file path or a URL."),
    output: Optional[str] = typer.Option(None, help="Output path for the converted markdown file. If not provided, the input file's suffix is replaced with .md"),
    model: Optional[str] = typer.Option(None, help="OpenAI Model to use for conversion"),
    api_key: Optional[str] = typer.Option(None, help="API key for OpenAI API"),
    base_url: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    style_map: Optional[str] = typer.Option(None, help="Output style map"),
    exiftool_path: Optional[str] = typer.Option(None, help="Path to exiftool for metadata extraction"),
    docintel_endpoint: Optional[str] = typer.Option(None, help="Document Intelligence API endpoint"),
    prevent_save_file: bool = typer.Option(False, help="Prevent saving the converted file to disk."),
    encoding: str = typer.Option("utf-8", help="Encoding for the output file."),
) -> AnythingToMarkdownReturns:
    """Convert various file types to markdown."""
    output_path: Path | None
    if not prevent_save_file:
        if not output:
            output_path = Path(source).with_suffix(".md")
        else:
            output_path = Path(output)
    else:
        output_path = None

    if model:
        llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        llm_model = model
    else:
        llm_client = None
        llm_model = None

    text: str = anything_to_markdown(
        source,
        llm_client=llm_client,
        llm_model=llm_model,
        style_map=style_map,
        exiftool_path=exiftool_path,
        docintel_endpoint=docintel_endpoint,
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding=encoding)
        logger.info(f"Converted `{source}` to markdown and saved to `{output_path}`.")
    else:
        logger.info(f"Converted `{source}` to markdown.")
    return {
        "input": source,
        "output": str(output_path) if output_path is not None else None,
        "out_text": text,
    }
