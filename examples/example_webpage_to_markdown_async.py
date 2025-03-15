import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(".")

from chatterer.tools.webpage_to_markdown import MarkdownLink, PlayWrightBot


async def main(url: str, output: str) -> None:
    url = url.strip()
    out_path = Path(output)
    out_llm_path = out_path.with_suffix(f".llm{out_path.suffix}")

    def truncate_string(string: str) -> str:
        return string[:50] + "..." if len(string) > 50 else string

    async with PlayWrightBot() as bot:
        md = await bot.aurl_to_md(url)
        out_path.write_text(md, encoding="utf-8")
        print(f"[*] Website converted to markdown and saved to {out_path}")

        if input("Do you want to filter the markdown with LLM? (y/n): ").strip().lower() != "y":
            return

        md_llm = await bot.aurl_to_md_with_llm(url)
        out_llm_path.write_text(md_llm, encoding="utf-8")
        print(f"[*] Markdown filtered with LLM and saved to {out_llm_path}")

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
    parser.add_argument("--output", type=str, default=".tmp_webpage.md", help="The output file path.")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.output))
