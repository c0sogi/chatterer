from pathlib import Path
from typing import Literal

from spargear import ArgumentSpec, BaseArguments

from chatterer import Chatterer, MarkdownLink, PlayWrightBot


class WebpageToMarkdownArgs(BaseArguments):
    url: str
    """The URL to crawl."""
    output: str = Path(__file__).with_suffix(".md").as_posix()
    """The output file path for the markdown file."""
    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        help="The Chatterer backend and model to use for filtering the markdown.",
        type=Chatterer.from_provider,
    )
    engine: Literal["firefox", "chromium", "webkit"] = "firefox"
    """The browser engine to use."""

    def run(self) -> None:
        chatterer = self.chatterer.value
        url: str = self.url.strip()
        output: Path = Path(self.output).resolve()
        with PlayWrightBot(chatterer=chatterer, engine=self.engine) as bot:
            md = bot.url_to_md(url)
            output.write_text(md, encoding="utf-8")
            if chatterer is not None:
                md_llm = bot.url_to_md_with_llm(url.strip())
                output.write_text(md_llm, encoding="utf-8")
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
        url: str = self.url.strip()
        output: Path = Path(self.output).resolve()
        async with PlayWrightBot(chatterer=chatterer, engine=self.engine) as bot:
            md = await bot.aurl_to_md(url)
            output.write_text(md, encoding="utf-8")
            if chatterer is not None:
                md_llm = await bot.aurl_to_md_with_llm(url.strip())
                output.write_text(md_llm, encoding="utf-8")
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


def main() -> None:
    WebpageToMarkdownArgs().run()


if __name__ == "__main__":
    main()
