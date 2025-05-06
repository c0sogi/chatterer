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
from typing import Literal

from spargear import ArgumentSpec, BaseArguments

from chatterer import Chatterer, MarkdownLink, PlayWrightBot


class WebpageToMarkdownArgs(BaseArguments):
    url: ArgumentSpec[str] = ArgumentSpec(["url"], help="The URL to crawl.")
    out_path: str = Path(__file__).with_suffix(".md").as_posix()
    """The output file path for the markdown file."""
    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--llm"],
        default=None,
        type=Chatterer.from_provider,
        help="The Chatterer backend and model to use for filtering the markdown.",
    )
    engine: Literal["firefox", "chromium", "webkit"] = "firefox"
    """The browser engine to use."""

    def run(self) -> None:
        chatterer = self.chatterer.value
        url: str = self.url.unwrap().strip()
        out_path: Path = Path(self.out_path).resolve()
        with PlayWrightBot(chatterer=chatterer, engine=self.engine) as bot:
            md = bot.url_to_md(url)
            out_path.write_text(md, encoding="utf-8")
            if chatterer is not None:
                md_llm = bot.url_to_md_with_llm(url.strip())
                out_path.write_text(md_llm, encoding="utf-8")
            links = MarkdownLink.from_markdown(md, referer_url=url)
            for link in links:
                if link.type == "link":
                    print(
                        f"- [{truncate_string(link.url)}] {truncate_string(link.inline_text)} ({truncate_string(link.inline_title)})"
                    )
                elif link.type == "image":
                    print(f"- ![{truncate_string(link.url)}] ({truncate_string(link.inline_text)})")

    async def arun(self) -> None:
        chatterer = self.chatterer.value
        url: str = self.url.unwrap().strip()
        out_path: Path = Path(self.out_path).resolve()
        async with PlayWrightBot(chatterer=chatterer, engine=self.engine) as bot:
            md = await bot.aurl_to_md(url)
            out_path.write_text(md, encoding="utf-8")
            if chatterer is not None:
                md_llm = await bot.aurl_to_md_with_llm(url.strip())
                out_path.write_text(md_llm, encoding="utf-8")
            links = MarkdownLink.from_markdown(md, referer_url=url)
            for link in links:
                if link.type == "link":
                    print(
                        f"- [{truncate_string(link.url)}] {truncate_string(link.inline_text)} ({truncate_string(link.inline_title)})"
                    )
                elif link.type == "image":
                    print(f"- ![{truncate_string(link.url)}] ({truncate_string(link.inline_text)})")


def truncate_string(s: str) -> str:
    return s[:50] + "..." if len(s) > 50 else s


if __name__ == "__main__":
    WebpageToMarkdownArgs().run()
