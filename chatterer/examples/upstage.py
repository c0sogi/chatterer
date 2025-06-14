import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents.base import Blob
from spargear import ArgumentSpec, BaseArguments

from chatterer import Chatterer, UpstageDocumentParseParser
from chatterer.tools.upstage_document_parser import (
    DEFAULT_IMAGE_DIR,
    DOCUMENT_PARSE_BASE_URL,
    DOCUMENT_PARSE_DEFAULT_MODEL,
    OCR,
    Category,
    OutputFormat,
    SplitType,
)

logger = logging.getLogger(__name__)


class Arguments(BaseArguments):
    INPUT_PATH: Path
    """Input file to parse. Can be a PDF, image, or other supported formats."""
    output: Optional[Path] = None
    """Output file path for the parsed content. Defaults to input file with .md suffix if not provided."""
    api_key: Optional[str] = None
    """API key for the Upstage API."""
    base_url: str = DOCUMENT_PARSE_BASE_URL
    """Base URL for the Upstage API."""
    model: str = DOCUMENT_PARSE_DEFAULT_MODEL
    """Model to use for parsing."""
    split: SplitType = "none"
    """Split type for the parsed content."""
    ocr: OCR = "auto"
    """OCR type for parsing."""
    output_format: OutputFormat = "markdown"
    """Output format for the parsed content."""
    coordinates: bool = False
    """Whether to include coordinates in the output."""
    base64_encoding: list[Category] = ["figure"]
    """Base64 encoding for specific categories in the parsed content."""
    image_description_instruction: str = "Describe the image in detail."
    """Instruction for generating image descriptions."""
    image_dir: str = DEFAULT_IMAGE_DIR
    """Directory to save images extracted from the document."""
    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        default=None,
        help="Chatterer instance for communication.",
        type=Chatterer.from_provider,
    )

    def run(self) -> None:
        input = self.INPUT_PATH.resolve()
        out = self.output or input.with_suffix(".md")

        parser = UpstageDocumentParseParser(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            split=self.split,
            ocr=self.ocr,
            output_format=self.output_format,
            coordinates=self.coordinates,
            base64_encoding=self.base64_encoding,
            image_description_instruction=self.image_description_instruction,
            image_dir=self.image_dir,
            chatterer=self.chatterer.value,
        )
        docs = parser.parse(Blob.from_path(input))  # pyright: ignore[reportUnknownMemberType]

        if self.image_dir:
            for path, image in parser.image_data.items():
                (path := Path(path)).parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image)
                logger.info(f"Saved image to `{path}`")

        markdown: str = "\n\n".join(f"<!--- page {i} -->\n{doc.page_content}" for i, doc in enumerate(docs, 1))
        out.write_text(markdown, encoding="utf-8")
        logger.info(f"Parsed `{input}` to `{out}`")


def main() -> None:
    Arguments().run()


if __name__ == "__main__":
    main()
