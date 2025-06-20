from datetime import datetime
from pathlib import Path
from typing import Literal

from spargear import ArgumentSpec, RunnableArguments

from chatterer import Chatterer, MarkdownLink, PlayWrightBot


def ouput_path_factory() -> Path:
    """Factory function to generate a default output path for the markdown file."""
    return Path(datetime.now().strftime("%Y%m%d_%H%M%S") + "_web2md.md").resolve()


class Arguments(RunnableArguments[None]):
    URL: str
    """The URL to crawl."""
    output: ArgumentSpec[Path] = ArgumentSpec(
        ["--output", "-o"],
        default_factory=ouput_path_factory,
        help="The output file path for the markdown file.",
    )
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
        url: str = self.URL.strip()
        output: Path = self.output.unwrap().resolve()
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
        url: str = self.URL.strip()
        output: Path = self.output.unwrap().resolve()
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
    Arguments().run()


if __name__ == "__main__":
    main()
