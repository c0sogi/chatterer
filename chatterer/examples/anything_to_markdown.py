def resolve_import_path_and_get_logger():
    # ruff: noqa: E402
    import logging
    import sys

    if __name__ == "__main__" and "." not in sys.path:
        sys.path.append(".")

    logger = logging.getLogger(__name__)
    return logger


logger = resolve_import_path_and_get_logger()
from pathlib import Path
from typing import Optional, TypedDict

import openai
from spargear import ArgumentSpec, BaseArguments

from chatterer import anything_to_markdown


class AnythingToMarkdownReturns(TypedDict):
    in_path: str
    out_path: Optional[str]
    out_text: str


class AnythingToMarkdownArguments(BaseArguments):
    """Command line arguments for converting various file types to markdown."""

    in_path: ArgumentSpec[str] = ArgumentSpec(["in-path"], help="Input file to convert to markdown")
    out_path: Optional[str] = None
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
        in_path = self.in_path.unwrap()
        if not self.prevent_save_file:
            if not self.out_path:
                out_path = Path(in_path).with_suffix(".md")
            else:
                out_path = Path(self.out_path)
        else:
            out_path = None

        if self.model:
            llm_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            llm_model = self.model
        else:
            llm_client = None
            llm_model = None

        text: str = anything_to_markdown(
            in_path,
            llm_client=llm_client,
            llm_model=llm_model,
            style_map=self.style_map,
            exiftool_path=self.exiftool_path,
            docintel_endpoint=self.docintel_endpoint,
        )
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding=self.encoding)
            logger.info(f"Converted `{in_path}` to markdown and saved to `{out_path}`.")
        else:
            logger.info(f"Converted `{in_path}` to markdown.")
        return {
            "in_path": in_path,
            "out_path": str(out_path) if out_path is not None else None,
            "out_text": text,
        }


if __name__ == "__main__":
    AnythingToMarkdownArguments().run()
