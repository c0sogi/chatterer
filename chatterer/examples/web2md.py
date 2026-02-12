from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import typer

from chatterer import Chatterer
from chatterer.tools.img2txt import MarkdownLink
from chatterer.tools.web2md import PlayWrightBot


def ouput_path_factory() -> Path:
    """Factory function to generate a default output path for the markdown file."""
    return Path(datetime.now().strftime("%Y%m%d_%H%M%S") + "_web2md.md").resolve()


def command(
    url: str = typer.Argument(help="The URL to crawl."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="The output file path for the markdown file."),
    chatterer: Optional[str] = typer.Option(
        None, help="The Chatterer backend and model to use for filtering the markdown."
    ),
    engine: Literal["firefox", "chromium", "webkit"] = typer.Option(
        "firefox", help="The browser engine to use (firefox, chromium, webkit)."
    ),
) -> None:
    """Convert web pages to markdown."""
    chatterer_obj = Chatterer.from_provider(chatterer) if chatterer else None
    url_str: str = url.strip()
    output_path: Path = (output or ouput_path_factory()).resolve()
    with PlayWrightBot(chatterer=chatterer_obj, engine=engine) as bot:
        md = bot.url_to_md(url_str)
        output_path.write_text(md, encoding="utf-8")
        if chatterer_obj is not None:
            md_llm = bot.url_to_md_with_llm(url_str)
            output_path.write_text(md_llm, encoding="utf-8")
        links = MarkdownLink.from_markdown(md, referer_url=url_str)
        for link in links:
            if link.type == "link":
                print(
                    f"- [{truncate_string(link.url)}] {truncate_string(link.inline_text)} ({truncate_string(link.inline_title)})"
                )
            elif link.type == "image":
                print(f"- ![{truncate_string(link.url)}] ({truncate_string(link.inline_text)})")


def truncate_string(s: str) -> str:
    return s[:50] + "..." if len(s) > 50 else s
