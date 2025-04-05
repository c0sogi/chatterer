import argparse
import sys
from pathlib import Path

sys.path.append(".")
from chatterer import anything_to_markdown

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a file to markdown using Chatterer.")
    parser.add_argument("source", type=str, help="Path to the file to convert")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for conversion (default: 'gpt-4o-mini').",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI (default: None).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI API (default: None).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".md"),
        help="out path for the converted markdown file",
    )
    parser.add_argument(
        "--style-map",
        type=str,
        default=None,
        help="out style map (default: None).",
    )
    parser.add_argument(
        "--exiftool-path",
        type=str,
        default=None,
        help="Path to exiftool for metadata extraction (default: None).",
    )
    parser.add_argument(
        "--docintel-endpoint",
        type=str,
        default=None,
        help="Document Intelligence API endpoint (default: None).",
    )
    args = parser.parse_args()
    model = args.model
    if model:
        import openai

        client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
        result = anything_to_markdown(
            args.source,
            llm_client=client,
            llm_model=model,
            style_map=args.style_map,
            exiftool_path=args.exiftool_path,
            docintel_endpoint=args.docintel_endpoint,
        )
    else:
        result = anything_to_markdown(
            args.source,
            style_map=args.style_map,
            exiftool_path=args.exiftool_path,
            docintel_endpoint=args.docintel_endpoint,
        )
    if args.out:
        if args.out.exists():
            print(f"[*] File {args.out} already exists. Overwriting...")
        else:
            print(f"[*] Saving to {args.out}...")
        args.out.write_text(result, encoding="utf-8")
        print(f"[*] Saved to {args.out}")
    else:
        print(f"[*] Converted markdown from {args.source}:")
        print(result)
