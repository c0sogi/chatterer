import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.append(".")

from chatterer import Chatterer, MarkdownLink, PlayWrightBot


def truncate_string(string: str) -> str:
    return string[:50] + "..." if len(string) > 50 else string


def main_sync(
    url: str, out: Path, chatterer: Optional[Chatterer], engine: Literal["chromium", "firefox", "webkit"]
) -> None:
    url = url.strip()
    with PlayWrightBot(chatterer=chatterer) as bot:
        md = bot.url_to_md(url)
        out.write_text(md, encoding="utf-8")
        print(f"[*] Website converted to markdown and saved to {out}")

        if chatterer is None:
            print("[!] No LLM provided. Skipping LLM filtering.")
            return

        md_llm = bot.url_to_md_with_llm(url)
        out.write_text(md_llm, encoding="utf-8")
        print(f"[*] Markdown filtered with LLM and saved to {out}")

        print("[*] Found links:")
        links: list[MarkdownLink] = MarkdownLink.from_markdown(md, referer_url=url)
        for link in links:
            if link.type == "link":
                print(
                    f"- [{truncate_string(link.url)}] {truncate_string(link.inline_text)} ({truncate_string(link.inline_title)})"
                )
            elif link.type == "image":
                print(f"- ![{truncate_string(link.url)}] ({truncate_string(link.inline_text)})")


async def main_async(
    url: str, out: Path, chatterer: Optional[Chatterer], engine: Literal["chromium", "firefox", "webkit"]
) -> None:
    url = url.strip()
    async with PlayWrightBot(chatterer=chatterer) as bot:
        md = await bot.aurl_to_md(url)
        out.write_text(md, encoding="utf-8")
        print(f"[*] Website converted to markdown and saved to {out}")

        if chatterer is None:
            print("[!] No LLM provided. Skipping LLM filtering.")
            return

        md_llm = await bot.aurl_to_md_with_llm(url)
        out.write_text(md_llm, encoding="utf-8")
        print(f"[*] Markdown filtered with LLM and saved to {out}")

        print("[*] Found links:")
        links: list[MarkdownLink] = MarkdownLink.from_markdown(md, referer_url=url)
        for link in links:
            if link.type == "link":
                print(
                    f"- [{truncate_string(link.url)}] {truncate_string(link.inline_text)} ({truncate_string(link.inline_title)})"
                )
            elif link.type == "image":
                print(f"- ![{truncate_string(link.url)}] ({truncate_string(link.inline_text)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a webpage to markdown using PlayWright.")
    parser.add_argument("url", type=str, help="The URL to crawl.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".md"),
        help="The out file path.",
    )
    parser.add_argument("--sync", action="store_true", help="Run the script synchronously.", default=False)
    parser.add_argument(
        "--llm",
        type=str,
        help="The LLM backend and model to use for filtering the markdown. (e.g. `openai:gpt-4o-mini` or `anthropic:claude-3-7-sonnet-20250219`, `google:gemini-2.0-flash`",
        default=None,
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["firefox", "chromium", "webkit"],
        default="firefox",
        help="The browser engine to use.",
    )

    args = parser.parse_args()
    chatterer: Optional[Chatterer] = None
    if args.llm:
        chatterer = Chatterer.from_provider(args.llm)

    if args.sync:
        print("[*] Running script synchronously...")
        main_sync(args.url, args.out, chatterer, args.engine)
    else:
        print("[*] Running script asynchronously...")
        asyncio.run(main_async(args.url, args.out, chatterer, args.engine))
    print("[*] Done.")
