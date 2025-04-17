# anything_to_markdown.py
import sys
from pathlib import Path

import openai
from spargear import ArgumentSpec, BaseArguments

sys.path.append(".")
from chatterer import anything_to_markdown


class AnythingToMarkdownArgs(BaseArguments):
    input: ArgumentSpec[Path] = ArgumentSpec(["input"], help="Path to the file to convert")
    model: ArgumentSpec[str] = ArgumentSpec(
        ["--model"], default=None, help="Model to use for conversion (default: 'gpt-4o-mini')."
    )
    api_key: ArgumentSpec[str] = ArgumentSpec(["--api-key"], default=None, help="API key for OpenAI (default: None).")
    base_url: ArgumentSpec[str] = ArgumentSpec(
        ["--base-url"], default=None, help="Base URL for OpenAI API (default: None)."
    )
    out: ArgumentSpec[Path] = ArgumentSpec(
        ["--out"],
        default=None,
        help="Output path for the converted markdown file. If not provided, the input file’s suffix is replaced with .md.",
    )
    style_map: ArgumentSpec[str] = ArgumentSpec(["--style-map"], default=None, help="Output style map (default: None).")
    exiftool_path: ArgumentSpec[str] = ArgumentSpec(
        ["--exiftool-path"], default=None, help="Path to exiftool for metadata extraction (default: None)."
    )
    docintel_endpoint: ArgumentSpec[str] = ArgumentSpec(
        ["--docintel-endpoint"], default=None, help="Document Intelligence API endpoint (default: None)."
    )


def main() -> None:
    AnythingToMarkdownArgs.load()
    input = AnythingToMarkdownArgs.input.unwrap()
    out = AnythingToMarkdownArgs.out.value or Path(input).with_suffix(".md")
    result = (
        anything_to_markdown(
            input,
            llm_client=openai.OpenAI(
                api_key=AnythingToMarkdownArgs.api_key.value, base_url=AnythingToMarkdownArgs.base_url.value
            ),
            llm_model=AnythingToMarkdownArgs.model.value,
            style_map=AnythingToMarkdownArgs.style_map.value,
            exiftool_path=AnythingToMarkdownArgs.exiftool_path.value,
            docintel_endpoint=AnythingToMarkdownArgs.docintel_endpoint.value,
        )
        if AnythingToMarkdownArgs.model.value
        else anything_to_markdown(
            input,
            style_map=AnythingToMarkdownArgs.style_map.value,
            exiftool_path=AnythingToMarkdownArgs.exiftool_path.value,
            docintel_endpoint=AnythingToMarkdownArgs.docintel_endpoint.value,
        )
    )
    out.write_text(result, encoding="utf-8")
    print(f"[*] Converted `{input}` to markdown and saved to `{out}`.")


if __name__ == "__main__":
    main()
