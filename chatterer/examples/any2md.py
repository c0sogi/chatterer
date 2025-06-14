import logging
from pathlib import Path
from typing import Optional, TypedDict

import openai
from spargear import RunnableArguments

from chatterer import anything_to_markdown

logger = logging.getLogger(__name__)


class AnythingToMarkdownReturns(TypedDict):
    input: str
    output: Optional[str]
    out_text: str


class Arguments(RunnableArguments[AnythingToMarkdownReturns]):
    """Command line arguments for converting various file types to markdown."""

    SOURCE: str
    """Input file to convert to markdown. Can be a file path or a URL."""
    output: Optional[str] = None
    """Output path for the converted markdown file. If not provided, the input file's suffix is replaced with .md"""
    model: Optional[str] = None
    """OpenAI Model to use for conversion"""
    api_key: Optional[str] = None
    """API key for OpenAI API"""
    base_url: Optional[str] = None
    """Base URL for OpenAI API"""
    style_map: Optional[str] = None
    """Output style map"""
    exiftool_path: Optional[str] = None
    """"Path to exiftool for metadata extraction"""
    docintel_endpoint: Optional[str] = None
    "Document Intelligence API endpoint"
    prevent_save_file: bool = False
    """Prevent saving the converted file to disk."""
    encoding: str = "utf-8"
    """Encoding for the output file."""

    def run(self) -> AnythingToMarkdownReturns:
        if not self.prevent_save_file:
            if not self.output:
                output = Path(self.SOURCE).with_suffix(".md")
            else:
                output = Path(self.output)
        else:
            output = None

        if self.model:
            llm_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            llm_model = self.model
        else:
            llm_client = None
            llm_model = None

        text: str = anything_to_markdown(
            self.SOURCE,
            llm_client=llm_client,
            llm_model=llm_model,
            style_map=self.style_map,
            exiftool_path=self.exiftool_path,
            docintel_endpoint=self.docintel_endpoint,
        )
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding=self.encoding)
            logger.info(f"Converted `{self.SOURCE}` to markdown and saved to `{output}`.")
        else:
            logger.info(f"Converted `{self.SOURCE}` to markdown.")
        return {
            "input": self.SOURCE,
            "output": str(output) if output is not None else None,
            "out_text": text,
        }


def main() -> None:
    Arguments().run()


if __name__ == "__main__":
    main()
