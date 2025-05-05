def resolve_import_path():
    # ruff: noqa: E402
    import sys
    from pathlib import Path

    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.append(str(parent))


resolve_import_path()
import asyncio
import sys
from pathlib import Path
from typing import Literal, Optional

from chatterer import Chatterer, MarkdownLink, PlayWrightBot
from spargear import ArgumentSpec, BaseArguments


class WebpageToMarkdownArgs(BaseArguments):
    url: ArgumentSpec[str] = ArgumentSpec(["url"], help="The URL to crawl.")
    out: ArgumentSpec[Path] = ArgumentSpec(["--out"], default=None, help="The output file path.")
    sync: ArgumentSpec[bool] = ArgumentSpec(
        ["--sync"], action="store_true", default=False, help="Run the script synchronously."
    )
    llm: ArgumentSpec[str] = ArgumentSpec(
        ["--llm"], default=None, help="The LLM backend and model to use for filtering the markdown."
    )
    engine: ArgumentSpec[Literal["firefox", "chromium", "webkit"]] = ArgumentSpec(
        ["--engine"], default="firefox", help="The browser engine to use.", choices=["firefox", "chromium", "webkit"]
    )


def truncate_string(s: str) -> str:
    return s[:50] + "..." if len(s) > 50 else s


def main_sync(url: str, out: Path, ch: Optional[Chatterer], engine: Literal["chromium", "firefox", "webkit"]) -> None:
    with PlayWrightBot(chatterer=ch, engine=engine) as bot:
        md = bot.url_to_md(url.strip())
        out.write_text(md, encoding="utf-8")
        if ch:
            md_llm = bot.url_to_md_with_llm(url.strip())
            out.write_text(md_llm, encoding="utf-8")
        links = MarkdownLink.from_markdown(md, referer_url=url)
        for link in links:
            if link.type == "link":
                print(
                    f"- [{truncate_string(link.url)}] {truncate_string(link.inline_text)} ({truncate_string(link.inline_title)})"
                )
            elif link.type == "image":
                print(f"- ![{truncate_string(link.url)}] ({truncate_string(link.inline_text)})")


async def main_async(
    url: str, out: Path, ch: Optional[Chatterer], engine: Literal["chromium", "firefox", "webkit"]
) -> None:
    async with PlayWrightBot(chatterer=ch, engine=engine) as bot:
        md = await bot.aurl_to_md(url.strip())
        out.write_text(md, encoding="utf-8")
        if ch:
            md_llm = await bot.aurl_to_md_with_llm(url.strip())
            out.write_text(md_llm, encoding="utf-8")
        links = MarkdownLink.from_markdown(md, referer_url=url)
        for link in links:
            if link.type == "link":
                print(
                    f"- [{truncate_string(link.url)}] {truncate_string(link.inline_text)} ({truncate_string(link.inline_title)})"
                )
            elif link.type == "image":
                print(f"- ![{truncate_string(link.url)}] ({truncate_string(link.inline_text)})")


def main() -> None:
    WebpageToMarkdownArgs.load()
    out_path = WebpageToMarkdownArgs.out.value or Path(__file__).with_suffix(".md")
    ch = Chatterer.from_provider(WebpageToMarkdownArgs.llm.value) if WebpageToMarkdownArgs.llm.value else None
    if WebpageToMarkdownArgs.sync.value:
        main_sync(WebpageToMarkdownArgs.url.unwrap(), out_path, ch, WebpageToMarkdownArgs.engine.unwrap())
    else:
        asyncio.run(main_async(WebpageToMarkdownArgs.url.unwrap(), out_path, ch, WebpageToMarkdownArgs.engine.unwrap()))
    sys.exit(0)


if __name__ == "__main__":
    main()
