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


class UpstageParserArguments(BaseArguments):
    in_path: ArgumentSpec[Path] = ArgumentSpec(["in-path"], help="Path to the input file.")
    out_path: ArgumentSpec[Path] = ArgumentSpec(["--out-path"], default=None, help="Output file path.")
    api_key: ArgumentSpec[str] = ArgumentSpec(["--api-key"], default=None, help="API key for the Upstage API.")
    base_url: ArgumentSpec[str] = ArgumentSpec(
        ["--base-url"], default=DOCUMENT_PARSE_BASE_URL, help="Base URL for the Upstage API."
    )
    model: ArgumentSpec[str] = ArgumentSpec(
        ["--model"], default=DOCUMENT_PARSE_DEFAULT_MODEL, help="Model to use for parsing."
    )
    split: ArgumentSpec[SplitType] = ArgumentSpec(["--split"], default="none", help="Split type for parsing.")
    ocr: ArgumentSpec[OCR] = ArgumentSpec(["--ocr"], default="auto", help="OCR type for parsing.")
    output_format: ArgumentSpec[OutputFormat] = ArgumentSpec(
        ["--output-format"], default="markdown", help="Output format."
    )
    coordinates: ArgumentSpec[bool] = ArgumentSpec(["--coordinates"], action="store_true", help="Include coordinates.")
    base64_encoding: ArgumentSpec[list[Category]] = ArgumentSpec(
        ["--base64-encoding"], default=["figure"], help="Base64 encoding for specific categories."
    )
    image_description_instruction: ArgumentSpec[str] = ArgumentSpec(
        ["--image-description-instruction"],
        default="Describe the image in detail.",
        help="Instruction for image description.",
    )
    image_dir: ArgumentSpec[str] = ArgumentSpec(
        ["--image-dir"],
        default=DEFAULT_IMAGE_DIR,
        help="Directory for image paths.",
    )
    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        default=None,
        help="Chatterer instance for communication.",
        type=Chatterer.from_provider,
    )

    def run(self) -> None:
        UpstageParserArguments.load()
        input = UpstageParserArguments.in_path.unwrap().resolve()
        out = UpstageParserArguments.out_path.value or input.with_suffix(".md")

        parser = UpstageDocumentParseParser(
            api_key=UpstageParserArguments.api_key.value,
            base_url=UpstageParserArguments.base_url.unwrap(),
            model=UpstageParserArguments.model.unwrap(),
            split=UpstageParserArguments.split.unwrap(),
            ocr=UpstageParserArguments.ocr.unwrap(),
            output_format=UpstageParserArguments.output_format.unwrap(),
            coordinates=UpstageParserArguments.coordinates.unwrap(),
            base64_encoding=UpstageParserArguments.base64_encoding.unwrap(),
            image_description_instruction=UpstageParserArguments.image_description_instruction.unwrap(),
            image_dir=UpstageParserArguments.image_dir.value,
            chatterer=UpstageParserArguments.chatterer.value,
        )

        docs = parser.parse(Blob.from_path(input))  # pyright: ignore[reportUnknownMemberType]

        if UpstageParserArguments.image_dir.value:
            for path, image in parser.image_data.items():
                (path := Path(path)).parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image)
                logger.info(f"Saved image to `{path}`")

        markdown: str = "\n\n".join(f"<!--- page {i} -->\n{doc.page_content}" for i, doc in enumerate(docs, 1))
        out.write_text(markdown, encoding="utf-8")
        logger.info(f"Parsed `{input}` to `{out}`")


if __name__ == "__main__":
    UpstageParserArguments().run()
