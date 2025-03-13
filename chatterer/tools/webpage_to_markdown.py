"""
PlaywrightBot

This module provides a single class that uses Playwright to:
  - Fetch and render HTML pages (with JavaScript execution),
  - Optionally scroll down or reload pages,
  - Convert rendered HTML into Markdown,
  - Extract specific elements using CSS selectors,
  - Filter key information from a page via integration with a language model (Chatterer).

Both synchronous and asynchronous methods are available in this unified class.
Use the synchronous methods (without the "a" prefix) in a normal context manager,
or use the asynchronous methods (prefixed with "a") within an async context manager.
"""

from __future__ import annotations

import asyncio
import os.path
import re
from base64 import b64encode
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from traceback import format_exception_only, print_exc
from types import TracebackType
from typing import (
    Awaitable,
    Callable,
    ClassVar,
    Literal,
    NamedTuple,
    NewType,
    NotRequired,
    Optional,
    Self,
    Sequence,
    Type,
    TypeAlias,
    TypedDict,
    TypeGuard,
    Union,
    cast,
)
from urllib.parse import urljoin, urlparse

import mistune
import requests
from aiohttp import ClientSession
from bs4 import Tag
from markdownify import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    ASTERISK,
    SPACES,
    STRIP,
    UNDERLINED,
    markdownify,  # pyright: ignore[reportUnknownVariableType]
)
from PIL.Image import Resampling
from PIL.Image import open as image_open
from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import BrowserContext as AsyncBrowserContext
from playwright.async_api import Playwright as AsyncPlaywright
from playwright.async_api import async_playwright
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, ProxySettings, sync_playwright
from pydantic import BaseModel, Field

from ..language_model import DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION, Chatterer

CodeLanguageCallback: TypeAlias = Callable[[Tag], Optional[str]]
ImageDataAndReferences = dict[Optional[str], list["LinkInfo"]]
ImageDescriptionAndReferences = NewType("ImageDescriptionAndReferences", ImageDataAndReferences)
WaitUntil: TypeAlias = Literal["commit", "domcontentloaded", "load", "networkidle"]

DEFAULT_UA: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)


# Define a Pydantic model for the selected line ranges returned by the LLM.
class SelectedLineRanges(BaseModel):
    line_ranges: list[str] = Field(description="List of inclusive line ranges, e.g., ['1-3', '5-5', '7-10']")


class PlaywrightLaunchOptions(TypedDict):
    executable_path: NotRequired[str | Path]
    channel: NotRequired[str]
    args: NotRequired[Sequence[str]]
    ignore_default_args: NotRequired[bool | Sequence[str]]
    handle_sigint: NotRequired[bool]
    handle_sigterm: NotRequired[bool]
    handle_sighup: NotRequired[bool]
    timeout: NotRequired[float]
    env: NotRequired[dict[str, str | float | bool]]
    headless: NotRequired[bool]
    devtools: NotRequired[bool]
    proxy: NotRequired[ProxySettings]
    downloads_path: NotRequired[str | Path]
    slow_mo: NotRequired[float]
    traces_dir: NotRequired[str | Path]
    chromium_sandbox: NotRequired[bool]
    firefox_user_prefs: NotRequired[dict[str, str | float | bool]]


class PlaywrightOptions(PlaywrightLaunchOptions):
    user_data_dir: NotRequired[str | Path]


def get_default_playwright_options() -> PlaywrightOptions:
    return {"headless": True}


class HtmlToMarkdownOptions(TypedDict):
    autolinks: NotRequired[bool]
    bullets: NotRequired[str]
    code_language: NotRequired[str]
    code_language_callback: NotRequired[CodeLanguageCallback]
    convert: NotRequired[Sequence[str]]
    default_title: NotRequired[bool]
    escape_asterisks: NotRequired[bool]
    escape_underscores: NotRequired[bool]
    escape_misc: NotRequired[bool]
    heading_style: NotRequired[str]
    keep_inline_images_in: NotRequired[Sequence[str]]
    newline_style: NotRequired[str]
    strip: NotRequired[Sequence[str]]
    strip_document: NotRequired[str]
    strong_em_symbol: NotRequired[str]
    sub_symbol: NotRequired[str]
    sup_symbol: NotRequired[str]
    table_infer_header: NotRequired[bool]
    wrap: NotRequired[bool]
    wrap_width: NotRequired[int]


def get_default_html_to_markdown_options() -> HtmlToMarkdownOptions:
    return {
        "autolinks": True,
        "bullets": "*+-",  # An iterable of bullet types.
        "code_language": "",
        "default_title": False,
        "escape_asterisks": True,
        "escape_underscores": True,
        "escape_misc": False,
        "heading_style": UNDERLINED,
        "keep_inline_images_in": [],
        "newline_style": SPACES,
        "strip_document": STRIP,
        "strong_em_symbol": ASTERISK,
        "sub_symbol": "",
        "sup_symbol": "",
        "table_infer_header": False,
        "wrap": False,
        "wrap_width": 80,
    }


class ImageProcessingConfig(TypedDict):
    """
    이미지 필터링/변환 시 사용할 설정.
      - formats: (Sequence[str]) 허용할 이미지 포맷(소문자, 예: ["jpeg", "png", "webp"]).
      - max_size_mb: (float) 이미지 용량 상한(MB). 초과 시 제외.
      - min_largest_side: (int) 가로나 세로 중 가장 큰 변의 최소 크기. 미만 시 제외.
      - resize_if_min_side_exceeds: (int) 가로나 세로 중 작은 변이 이 값 이상이면 리스케일.
      - resize_target_for_min_side: (int) 리스케일시, '가장 작은 변'을 이 값으로 줄임(비율 유지는 Lanczos).
    """

    formats: Sequence[str]
    max_size_mb: NotRequired[float]
    min_largest_side: NotRequired[int]
    resize_if_min_side_exceeds: NotRequired[int]
    resize_target_for_min_side: NotRequired[int]


def get_default_image_processing_config() -> ImageProcessingConfig:
    return {
        "max_size_mb": 5,
        "min_largest_side": 200,
        "resize_if_min_side_exceeds": 2000,
        "resize_target_for_min_side": 1000,
        "formats": ["png", "jpg", "jpeg", "gif", "bmp", "webp"],
    }


@dataclass
class PlayWrightBot:
    """
    A unified bot that leverages Playwright to render web pages, convert them to Markdown,
    extract elements, and filter key information using a language model.

    This class exposes both synchronous and asynchronous methods.

    Synchronous usage:
      with UnifiedPlaywrightBot() as bot:
          md = bot.url_to_md("https://example.com")
          headings = bot.select_and_extract("https://example.com", "h2")
          filtered_md = bot.url_to_md_with_llm("https://example.com")

    Asynchronous usage:
      async with UnifiedPlaywrightBot() as bot:
          md = await bot.aurl_to_md("https://example.com")
          headings = await bot.aselect_and_extract("https://example.com", "h2")
          filtered_md = await bot.aurl_to_md_with_llm("https://example.com")

    Attributes:
        headless (bool): Whether to run the browser in headless mode (default True).
        chatterer (Chatterer): An instance of the language model interface for processing text.
    """

    chatterer: Chatterer = field(default_factory=Chatterer.openai)
    playwright_options: PlaywrightOptions = field(default_factory=get_default_playwright_options)
    html_to_markdown_options: HtmlToMarkdownOptions = field(default_factory=get_default_html_to_markdown_options)
    image_processing_config: ImageProcessingConfig = field(default_factory=get_default_image_processing_config)
    markdown_filtering_instruction: str = """You are a web parser bot, an AI agent that filters out redundant fields from a webpage.

You excel at the following tasks:
1. Identifying the main article content of a webpage.
2. Filtering out ads, navigation links, and other irrelevant information.
3. Selecting the line number ranges that correspond to the article content.
4. Providing these inclusive ranges in the format 'start-end' or 'single_line_number'.

However, there are a few rules you must follow:
1. Do not remove the title of the article, if present.
2. Do not remove the author's name or the publication date, if present.
3. Include only images that are part of the article.

Now, return a valid JSON object, for example: {'line_ranges': ['1-3', '5-5', '7-10']}.

Markdown-formatted webpage content is provided below for your reference:
---
""".strip()
    description_format: str = (
        "<details><summary>{image_summary}</summary><img src='{url}' alt='{inline_text}'></details>"
    )
    image_description_instruction: str = DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION

    _sync_playwright: Optional[Playwright] = field(default=None, init=False)
    _sync_browser: Optional[Browser | BrowserContext] = field(default=None, init=False)
    _async_playwright: Optional[AsyncPlaywright] = field(default=None, init=False)
    _async_browser: Optional[AsyncBrowser | AsyncBrowserContext] = field(default=None, init=False)

    @property
    def playwright_launch_options(self) -> PlaywrightLaunchOptions:
        options = self.playwright_options.copy()
        options.pop("user_data_dir", None)
        return options

    # =======================
    # Synchronous Context Management
    # =======================
    def __enter__(self) -> PlayWrightBot:
        """
        Enter the synchronous context.

        Starts Playwright synchronously and launches a Chromium browser.

        Returns:
            UnifiedPlaywrightBot: The bot instance.
        """

        self._sync_playwright = sync_playwright().start()
        if "user_data_dir" in self.playwright_options:
            self._sync_browser = self._sync_playwright.chromium.launch_persistent_context(**self.playwright_options)
        else:
            self._sync_browser = self._sync_playwright.chromium.launch(**self.playwright_launch_options)
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit the synchronous context.

        Closes the browser and stops Playwright.
        """
        if self._sync_browser:
            self._sync_browser.close()
        if self._sync_playwright:
            self._sync_playwright.stop()

    # =======================
    # Asynchronous Context Management
    # =======================
    async def __aenter__(self) -> PlayWrightBot:
        """
        Asynchronously enter the context.

        Starts Playwright asynchronously and launches a Chromium browser.

        Returns:
            UnifiedPlaywrightBot: The bot instance.
        """
        self._async_playwright = await async_playwright().start()
        if "user_data_dir" in self.playwright_options:
            self._async_browser = await self._async_playwright.chromium.launch_persistent_context(
                **self.playwright_options
            )
        else:
            self._async_browser = await self._async_playwright.chromium.launch(**self.playwright_launch_options)
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Asynchronously exit the context.

        Closes the asynchronous browser and stops Playwright.
        """
        if self._async_browser:
            await self._async_browser.close()
        if self._async_playwright:
            await self._async_playwright.stop()

    # =======================
    # Synchronous Methods
    # =======================
    def get_page(
        self,
        url: str,
        timeout: float = 10.0,
        wait_until: Optional[WaitUntil] = "domcontentloaded",
        referer: Optional[str] = None,
    ) -> Page:
        """
        Create a new page and navigate to the given URL synchronously.

        Args:
            url (str): URL to navigate to.
            timeout (float): Maximum navigation time in seconds.
            wait_until (str): Load state to wait for (e.g., "domcontentloaded").
            referer (Optional[str]): Referer URL to set.

        Returns:
            Page: The Playwright page object.
        """
        page = self._sync_browser.new_page()  # type: ignore
        page.goto(url, timeout=int(timeout * 1000), wait_until=wait_until, referer=referer)
        return page

    def url_to_md(
        self,
        url: str,
        wait: float = 0.2,
        scrolldown: bool = False,
        sleep: int = 0,
        reload: bool = True,
        timeout: Union[float, int] = 8,
        keep_page: bool = False,
        referer: Optional[str] = None,
    ) -> str:
        """
        Navigate to a URL, optionally wait, scroll, or reload the page, and convert the rendered HTML to Markdown.

        Args:
            url (str): URL of the page.
            wait (float): Time to wait after navigation (in seconds).
            scrolldown (bool): If True, scroll to the bottom of the page.
            sleep (int): Time to wait after scrolling (in seconds).
            reload (bool): If True, reload the page.
            timeout (float | int): Navigation timeout in seconds.
            keep_page (bool): If True, do not close the page after processing.
            referer (Optional[str]): Referer URL to set.

        Returns:
            str: The page content converted to Markdown.
        """
        page = self.get_page(url, timeout=timeout, referer=referer)
        if wait:
            page.wait_for_timeout(wait * 1000)
        if scrolldown:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            if sleep:
                page.wait_for_timeout(sleep * 1000)
        if reload:
            page.reload(timeout=int(timeout * 1000))
        html = page.content()
        md = html_to_markdown(html=html, options=self.html_to_markdown_options)
        if not keep_page:
            page.close()
        return md

    def select_and_extract(
        self,
        url: str,
        css_selector: str,
        wait: float = 0.2,
        scrolldown: bool = False,
        sleep: int = 0,
        reload: bool = True,
        timeout: Union[float, int] = 8,
        keep_page: bool = False,
        referer: Optional[str] = None,
    ) -> list[str]:
        """
        Navigate to a URL, render the page, and extract text from elements matching the given CSS selector.

        Args:
            url (str): URL of the page.
            css_selector (str): CSS selector to locate elements.
            wait (float): Time to wait after navigation (in seconds).
            scrolldown (bool): If True, scroll the page.
            sleep (int): Time to wait after scrolling (in seconds).
            reload (bool): If True, reload the page.
            timeout (float | int): Maximum navigation time (in seconds).
            keep_page (bool): If True, do not close the page after processing.
            referer (Optional[str]): Referer URL to set.

        Returns:
            List[str]: A list of text contents from the matching elements.
        """
        page = self.get_page(url, timeout=timeout, referer=referer)
        if wait:
            page.wait_for_timeout(wait * 1000)
        if scrolldown:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            if sleep:
                page.wait_for_timeout(sleep * 1000)
        if reload:
            page.reload(timeout=int(timeout * 1000))
        elements = page.query_selector_all(css_selector)
        texts = [element.inner_text() for element in elements]
        if not keep_page:
            page.close()
        return texts

    def url_to_md_with_llm(
        self,
        url: str,
        chunk_size: Optional[int] = None,
        wait: float = 0.2,
        scrolldown: bool = False,
        sleep: int = 0,
        reload: bool = True,
        timeout: Union[float, int] = 8,
        keep_page: bool = False,
        referer: Optional[str] = None,
    ) -> str:
        """
        Convert a URL's page to Markdown and use a language model (Chatterer) to filter out unimportant lines.

        The method splits the Markdown text into chunks, prepends line numbers, and prompts the LLM
        to select the important line ranges. It then reconstructs the filtered Markdown.

        Args:
            url (str): URL of the page.
            chunk_size (Optional[int]): Number of lines per chunk. Defaults to the full content.
            wait (float): Time to wait after navigation (in seconds).
            scrolldown (bool): If True, scroll down the page.
            sleep (int): Time to wait after scrolling (in seconds).
            reload (bool): If True, reload the page.
            timeout (float | int): Navigation timeout (in seconds).
            keep_page (bool): If True, do not close the page after processing.
            referer (Optional[str]): Referer URL to set.

        Returns:
            str: Filtered Markdown containing only the important lines.
        """
        markdown_content = self.url_to_md(
            url,
            wait=wait,
            scrolldown=scrolldown,
            sleep=sleep,
            reload=reload,
            timeout=timeout,
            keep_page=keep_page,
            referer=referer,
        )
        markdown_content = self.describe_images(markdown_text=markdown_content, referer_url=url)
        lines = markdown_content.split("\n")
        line_length = len(lines)
        important_lines: set[int] = set()

        def _into_safe_range(value: int) -> int:
            """Ensure the line index stays within bounds."""
            return min(max(value, 0), line_length - 1)

        if chunk_size is None:
            chunk_size = line_length

        # Process the markdown in chunks.
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i : i + chunk_size]
            # Prepend line numbers to each line.
            numbered_markdown = "\n".join(f"[Ln {line_no}] {line}" for line_no, line in enumerate(chunk_lines, start=1))
            # Use the language model synchronously to get the line ranges.
            result: SelectedLineRanges = self.chatterer.generate_pydantic(
                response_model=SelectedLineRanges,
                messages=f"{self.markdown_filtering_instruction}\n{numbered_markdown}",
            )
            for range_str in result.line_ranges:
                if "-" in range_str:
                    start, end = map(int, range_str.split("-"))
                    important_lines.update(range(_into_safe_range(start + i - 1), _into_safe_range(end + i)))
                else:
                    important_lines.add(_into_safe_range(int(range_str) + i - 1))
        # Reconstruct the filtered markdown.
        return "\n".join(lines[line_no] for line_no in sorted(important_lines))

    # =======================
    # Asynchronous Methods
    # =======================
    async def aget_page(
        self,
        url: str,
        timeout: float = 8,
        wait_until: Optional[WaitUntil] = "domcontentloaded",
        referer: Optional[str] = None,
    ):
        """
        Create a new page and navigate to the given URL asynchronously.

        Args:
            url (str): URL to navigate to.
            timeout (float): Maximum navigation time in seconds.
            wait_until (str): Load state to wait for.
            referer (Optional[str]): Referer URL to set.

        Returns:
            AsyncPage: The Playwright asynchronous page object.
        """
        page = await self._async_browser.new_page()  # type: ignore
        await page.goto(url, timeout=int(timeout * 1000), wait_until=wait_until, referer=referer)
        return page

    async def aurl_to_md(
        self,
        url: str,
        wait: float = 0.2,
        scrolldown: bool = False,
        sleep: int = 0,
        reload: bool = True,
        timeout: Union[float, int] = 8,
        keep_page: bool = False,
        referer: Optional[str] = None,
    ) -> str:
        """
        Asynchronously navigate to a URL, wait, scroll or reload if specified,
        and convert the rendered HTML to Markdown.

        Args:
            url (str): URL of the page.
            wait (float): Time to wait after navigation (in seconds).
            scrolldown (bool): If True, scroll the page.
            sleep (int): Time to wait after scrolling (in seconds).
            reload (bool): If True, reload the page.
            timeout (float | int): Navigation timeout (in seconds).
            keep_page (bool): If True, do not close the page after processing.
            referer (Optional[str]): Referer URL to set.

        Returns:
            str: The page content converted to Markdown.
        """
        page = await self.aget_page(url, timeout=timeout, referer=referer)
        if wait:
            await page.wait_for_timeout(wait * 1000)
        if scrolldown:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            if sleep:
                await page.wait_for_timeout(sleep * 1000)
        if reload:
            await page.reload(timeout=int(timeout * 1000))
        html = await page.content()
        md = html_to_markdown(html=html, options=self.html_to_markdown_options)
        if not keep_page:
            await page.close()
        return md

    async def aselect_and_extract(
        self,
        url: str,
        css_selector: str,
        wait: float = 0.2,
        scrolldown: bool = False,
        sleep: int = 0,
        reload: bool = True,
        timeout: Union[float, int] = 8,
        keep_page: bool = False,
        referer: Optional[str] = None,
    ) -> list[str]:
        """
        Asynchronously navigate to a URL, render the page, and extract text from elements matching the CSS selector.

        Args:
            url (str): URL of the page.
            css_selector (str): CSS selector to locate elements.
            wait (float): Time to wait after navigation (in seconds).
            scrolldown (bool): If True, scroll the page.
            sleep (int): Time to wait after scrolling (in seconds).
            reload (bool): If True, reload the page.
            timeout (float | int): Navigation timeout (in seconds).
            keep_page (bool): If True, do not close the page after processing.
            referer (Optional[str]): Referer URL to set.

        Returns:
            List[str]: A list of text contents from the matching elements.
        """
        page = await self.aget_page(url, timeout=timeout, referer=referer)
        if wait:
            await page.wait_for_timeout(wait * 1000)
        if scrolldown:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            if sleep:
                await page.wait_for_timeout(sleep * 1000)
        if reload:
            await page.reload(timeout=int(timeout * 1000))
        elements = await page.query_selector_all(css_selector)
        texts: list[str] = []
        for element in elements:
            text = await element.inner_text()
            texts.append(text)
        if not keep_page:
            await page.close()
        return texts

    async def aurl_to_md_with_llm(
        self,
        url: str,
        chunk_size: Optional[int] = None,
        wait: float = 0.2,
        scrolldown: bool = False,
        sleep: int = 0,
        reload: bool = True,
        timeout: Union[float, int] = 8,
        keep_page: bool = False,
        referer: Optional[str] = None,
    ) -> str:
        """
        Asynchronously convert a URL's page to Markdown and use the language model (Chatterer)
        to filter out unimportant lines.

        The method splits the Markdown text into chunks, prepends line numbers, and prompts the LLM
        to select the important line ranges. It then reconstructs the filtered Markdown.

        Args:
            url (str): URL of the page.
            chunk_size (Optional[int]): Number of lines per chunk; defaults to the full content.
            wait (float): Time to wait after navigation (in seconds).
            scrolldown (bool): If True, scroll the page.
            sleep (int): Time to wait after scrolling (in seconds).
            reload (bool): If True, reload the page.
            timeout (float | int): Navigation timeout (in seconds).
            keep_page (bool): If True, do not close the page after processing.
            referer (Optional[str]): Referer URL to set.

        Returns:
            str: Filtered Markdown containing only the important lines.
        """
        markdown_content = await self.aurl_to_md(
            url,
            wait=wait,
            scrolldown=scrolldown,
            sleep=sleep,
            reload=reload,
            timeout=timeout,
            keep_page=keep_page,
            referer=referer,
        )
        markdown_content = await self.adescribe_images(markdown_text=markdown_content, referer_url=url)
        lines = markdown_content.split("\n")
        line_length = len(lines)
        important_lines: set[int] = set()

        def _into_safe_range(value: int) -> int:
            """Ensure the line index is within valid bounds."""
            return min(max(value, 0), line_length - 1)

        if chunk_size is None:
            chunk_size = line_length

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i : i + chunk_size]
            numbered_markdown = "\n".join(f"[Ln {line_no}] {line}" for line_no, line in enumerate(chunk_lines, start=1))
            # Use the asynchronous language model method.
            result: SelectedLineRanges = await self.chatterer.agenerate_pydantic(
                response_model=SelectedLineRanges,
                messages=f"{self.markdown_filtering_instruction}\n{numbered_markdown}",
            )
            for range_str in result.line_ranges:
                if "-" in range_str:
                    start, end = map(int, range_str.split("-"))
                    important_lines.update(range(_into_safe_range(start + i - 1), _into_safe_range(end + i)))
                else:
                    important_lines.add(_into_safe_range(int(range_str) + i - 1))
        return "\n".join(lines[line_no] for line_no in sorted(important_lines))

    async def adescribe_images(self, markdown_text: str, referer_url: str) -> str:
        """
        Replace image URLs in Markdown text with their alt text and generate descriptions using a language model.
        """
        image_url_and_link_infos: dict[Optional[str], list[LinkInfo]] = await _aget_image_url_and_link_infos(
            markdown_text=markdown_text,
            referer_url=referer_url,
            config=self.image_processing_config,
        )

        async def dummy() -> None:
            pass

        def _handle_exception(e: Optional[str | BaseException]) -> TypeGuard[Optional[str]]:
            if isinstance(e, BaseException):
                print(format_exception_only(type(e), e))
                return False
            return True

        coros: list[Awaitable[Optional[str]]] = [
            self.chatterer.adescribe_image(image_url=image_url, instruction=self.image_description_instruction)
            if image_url is not None
            else dummy()
            for image_url in image_url_and_link_infos.keys()
        ]

        return _replace_images(
            markdown_text=markdown_text,
            image_description_and_references=ImageDescriptionAndReferences({
                image_summary: link_infos
                for link_infos, image_summary in zip(
                    image_url_and_link_infos.values(), await asyncio.gather(*coros, return_exceptions=True)
                )
                if _handle_exception(image_summary)
            }),
            description_format=self.description_format,
        )

    def describe_images(self, markdown_text: str, referer_url: str) -> str:
        """
        Replace image URLs in Markdown text with their alt text and generate descriptions using a language model.
        """
        image_url_and_link_infos: dict[Optional[str], list[LinkInfo]] = _get_image_url_and_link_infos(
            markdown_text=markdown_text,
            referer_url=referer_url,
            config=self.image_processing_config,
        )

        image_description_and_references: ImageDescriptionAndReferences = ImageDescriptionAndReferences({})
        for image_url, link_infos in image_url_and_link_infos.items():
            if image_url is not None:
                try:
                    image_summary: str = self.chatterer.describe_image(
                        image_url=image_url,
                        instruction=self.image_description_instruction,
                    )
                except Exception:
                    print_exc()
                    continue
                image_description_and_references[image_summary] = link_infos
            else:
                image_description_and_references[None] = link_infos

        return _replace_images(
            markdown_text=markdown_text,
            image_description_and_references=image_description_and_references,
            description_format=self.description_format,
        )


class LinkInfo(NamedTuple):
    type: Literal["link", "image"]
    url: str
    text: str
    title: Optional[str]
    pos: int
    end_pos: int

    @classmethod
    def from_markdown(cls, markdown_text: str, referer_url: Optional[str]) -> list[Self]:
        """
        The main function that returns the list of LinkInfo for the input text.
        For simplicity, we do a "pure inline parse" of the entire text
        instead of letting the block parser break it up. That ensures that
        link tokens cover the global positions of the entire input.
        """
        md = mistune.Markdown(inline=_TrackingInlineParser())
        # Create an inline state that references the full text.
        state = _TrackingInlineState({})
        state.src = markdown_text

        # Instead of calling md.parse, we can directly run the inline parser on
        # the entire text, so that positions match the entire input:
        md.inline.parse(state)

        # Now gather all the link info from the tokens.
        return cls._extract_links(tokens=state.tokens, referer_url=referer_url)

    @property
    def inline_text(self) -> str:
        return self.text.replace("\n", " ").strip()

    @property
    def inline_title(self) -> str:
        return self.title.replace("\n", " ").strip() if self.title else ""

    @property
    def link_markdown(self) -> str:
        if self.title:
            return f'[{self.inline_text}]({self.url} "{self.inline_title}")'
        return f"[{self.inline_text}]({self.url})"

    @classmethod
    def replace(cls, text: str, replacements: list[tuple[Self, str]]) -> str:
        for self, replacement in sorted(replacements, key=lambda x: x[0].pos, reverse=True):
            text = text[: self.pos] + replacement + text[self.end_pos :]
        return text

    @classmethod
    def _extract_links(cls, tokens: list[dict[str, object]], referer_url: Optional[str]) -> list[Self]:
        results: list[Self] = []
        for token in tokens:
            if (
                (type := token.get("type")) in ("link", "image")
                and "global_pos" in token
                and "attrs" in token
                and _attrs_typeguard(attrs := token["attrs"])
                and "url" in attrs
                and _url_typeguard(url := attrs["url"])
                and _global_pos_typeguard(global_pos := token["global_pos"])
            ):
                if referer_url:
                    url = _to_absolute_path(path=url, referer=referer_url)
                children: object | None = token.get("children")
                if _children_typeguard(children):
                    text = _extract_text(children)
                else:
                    text = ""

                if "title" in attrs:
                    title = str(attrs["title"])
                else:
                    title = None

                start, end = global_pos
                results.append(cls(type, url, text, title, start, end))
            if "children" in token and _children_typeguard(children := token["children"]):
                results.extend(cls._extract_links(children, referer_url))

        return results


class _TrackingInlineState(mistune.InlineState):
    meta_offset: int = 0  # Where in the original text does self.src start?

    def copy(self) -> Self:
        new_state = self.__class__(self.env)
        new_state.src = self.src
        new_state.tokens = []
        new_state.in_image = self.in_image
        new_state.in_link = self.in_link
        new_state.in_emphasis = self.in_emphasis
        new_state.in_strong = self.in_strong
        new_state.meta_offset = self.meta_offset
        return new_state


class _TrackingInlineParser(mistune.InlineParser):
    state_cls: ClassVar = _TrackingInlineState

    def parse_link(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, m: re.Match[str], state: _TrackingInlineState
    ) -> Optional[int]:
        """
        Mistune calls parse_link with a match object for the link syntax
        and the current inline state. If we successfully parse the link,
        super().parse_link(...) returns the new position *within self.src*.
        We add that to state.meta_offset for the global position.

        Because parse_link in mistune might return None or an int, we only
        record positions if we get an int back (meaning success).
        """
        offset = state.meta_offset
        new_pos: int | None = super().parse_link(m, state)
        if new_pos is not None:
            # We have successfully parsed a link.
            # The link token we just added should be the last token in state.tokens:
            if state.tokens:
                token = state.tokens[-1]
                # The local end is new_pos in the substring.
                # So the global start/end in the *original* text is offset + local positions.
                token["global_pos"] = (offset + m.start(), offset + new_pos)
        return new_pos


# --------------------------------------------------------------------
# Type Guards & Helper to gather plain text from nested tokens (for the link text).
# --------------------------------------------------------------------
def _children_typeguard(obj: object) -> TypeGuard[list[dict[str, object]]]:
    if not isinstance(obj, list):
        return False
    return all(isinstance(i, dict) for i in cast(list[object], obj))


def _attrs_typeguard(obj: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(obj, dict):
        return False
    return all(isinstance(k, str) for k in cast(dict[object, object], obj))


def _global_pos_typeguard(obj: object) -> TypeGuard[tuple[int, int]]:
    if not isinstance(obj, tuple):
        return False
    obj = cast(tuple[object, ...], obj)
    if len(obj) != 2:
        return False
    return all(isinstance(i, int) for i in obj)


def _url_typeguard(obj: object) -> TypeGuard[str]:
    return isinstance(obj, str)


def _extract_text(tokens: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for t in tokens:
        if t.get("type") == "text":
            parts.append(str(t.get("raw", "")))
        elif "children" in t:
            children: object = t["children"]
            if not _children_typeguard(children):
                continue
            parts.append(_extract_text(children))
    return "".join(parts)


def _is_url(path: str) -> bool:
    """
    path가 절대 URL 형태인지 여부를 bool로 반환
    (scheme과 netloc이 모두 존재하면 URL로 간주)
    """
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)


def _to_absolute_path(path: str, referer: str) -> str:
    """
    path     : 변환할 경로(상대/절대 경로 혹은 URL일 수도 있음)
    referer  : 기준이 되는 절대경로(혹은 URL)
    """
    # referer가 URL인지 파일 경로인지 먼저 판별
    ref_parsed = urlparse(referer)
    is_referer_url = bool(ref_parsed.scheme and ref_parsed.netloc)

    if is_referer_url:
        # referer가 URL이라면,
        # 1) path 자체가 이미 절대 URL인지 확인
        parsed = urlparse(path)
        if parsed.scheme and parsed.netloc:
            # path가 이미 완전한 URL (예: http://, https:// 등)
            return path
        else:
            # 그렇지 않다면(슬래시로 시작 포함), urljoin을 써서 referer + path 로 합침
            return urljoin(referer, path)
    else:
        # referer가 로컬 경로라면,
        # path가 로컬 파일 시스템에서의 절대경로인지 판단
        if os.path.isabs(path):
            return path
        else:
            # 파일이면 referer의 디렉토리만 추출
            if not os.path.isdir(referer):
                referer_dir = os.path.dirname(referer)
            else:
                referer_dir = referer

            combined = os.path.join(referer_dir, path)
            return os.path.abspath(combined)


# =======================


def html_to_markdown(html: str, options: HtmlToMarkdownOptions) -> str:
    """
    Convert HTML content to Markdown using the provided options.

    Args:
        html (str): HTML content to convert.
        options (HtmlToMarkdownOptions): Options for the conversion.

    Returns:
        str: The Markdown content.
    """
    return str(markdownify(html, **options))  # pyright: ignore[reportUnknownArgumentType]


# =======================


def _get_image_bytes(image_url: str, referer_url: str) -> Optional[bytes]:
    try:
        with requests.Session() as session:
            response = session.get(image_url, headers={"Referer": referer_url, "User-Agent": DEFAULT_UA})
            if not response.ok:
                return
            return bytes(response.content or b"")
    except Exception:
        return


async def _aget_image_bytes(image_url: str, referer_url: str) -> Optional[bytes]:
    try:
        async with ClientSession() as session:
            async with session.get(image_url, headers={"Referer": referer_url, "User-Agent": DEFAULT_UA}) as response:
                if not response.ok:
                    return
                return await response.read()
    except Exception:
        return


# =======================


def _fetch_remote_image(url: str, referer_url: str, config: ImageProcessingConfig) -> Optional[str]:
    image_bytes = _get_image_bytes(image_url=url.strip(), referer_url=referer_url)
    if not image_bytes:
        return None
    return _convert_image_into_base64(image_bytes, config)


async def _afetch_remote_image(url: str, referer_url: str, config: ImageProcessingConfig) -> Optional[str]:
    image_bytes = await _aget_image_bytes(image_url=url.strip(), referer_url=referer_url)
    if not image_bytes:
        return None
    return _convert_image_into_base64(image_bytes, config)


# =======================


def _process_markdown_image(link_info: LinkInfo, referer_url: str, config: ImageProcessingConfig) -> Optional[str]:
    """마크다운 이미지 패턴에 매칭된 하나의 이미지를 처리해 Base64 URL을 반환(동기)."""
    if link_info.type != "image":
        return
    url: str = link_info.url
    if url.startswith("data:image/"):
        return url
    elif _is_url(url):
        return _fetch_remote_image(url, referer_url, config)
    return _process_local_image(Path(url), config)


async def _aprocess_markdown_image(
    link_info: LinkInfo, referer_url: str, config: ImageProcessingConfig
) -> Optional[str]:
    """마크다운 이미지 패턴에 매칭된 하나의 이미지를 처리해 Base64 URL을 반환(비동기)."""
    if link_info.type != "image":
        return
    url: str = link_info.url
    if url.startswith("data:image/"):
        return url
    elif _is_url(url):
        return await _afetch_remote_image(url, referer_url, config)
    return _process_local_image(Path(url), config)


# =======================


def _get_image_url_and_link_infos(
    markdown_text: str, referer_url: str, config: ImageProcessingConfig
) -> dict[Optional[str], list[LinkInfo]]:
    image_matches: dict[Optional[str], list[LinkInfo]] = {}
    for link_info in LinkInfo.from_markdown(markdown_text, referer_url):
        if link_info.type == "link":
            image_matches.setdefault(None, []).append(link_info)
            continue
        image_data = _process_markdown_image(link_info, referer_url, config)
        if not image_data:
            continue
        image_matches.setdefault(image_data, []).append(link_info)
    return image_matches


async def _aget_image_url_and_link_infos(
    markdown_text: str, referer_url: str, config: ImageProcessingConfig
) -> dict[Optional[str], list[LinkInfo]]:
    image_matches: dict[Optional[str], list[LinkInfo]] = {}
    for link_info in LinkInfo.from_markdown(markdown_text, referer_url):
        if link_info.type == "link":
            image_matches.setdefault(None, []).append(link_info)
            continue
        image_data = await _aprocess_markdown_image(link_info, referer_url, config)
        if not image_data:
            continue
        image_matches.setdefault(image_data, []).append(link_info)
    return image_matches


# =======================


def _simple_base64_encode(image_data: bytes) -> Optional[str]:
    """
    Retrieve an image URL and return a base64-encoded data URL.
    """
    image_type = _detect_image_type(image_data)
    if not image_type:
        return
    encoded_data = b64encode(image_data).decode("utf-8")
    return f"data:image/{image_type};base64,{encoded_data}"


def _convert_image_into_base64(image_data: bytes, config: Optional[ImageProcessingConfig]) -> Optional[str]:
    """
    Retrieve an image in bytes and return a base64-encoded data URL,
    applying dynamic rules from 'config'.
    """
    if not config:
        # config 없으면 그냥 기존 헤더만 보고 돌려주는 간단 로직
        return _simple_base64_encode(image_data)

    # 1) 용량 검사
    max_size_mb = config.get("max_size_mb", float("inf"))
    image_size_mb = len(image_data) / (1024 * 1024)
    if image_size_mb > max_size_mb:
        print(f"Image too large: {image_size_mb:.2f} MB > {max_size_mb} MB")
        return None

    # 2) Pillow로 이미지 열기
    try:
        with image_open(BytesIO(image_data)) as im:
            w, h = im.size
            # 가장 큰 변
            largest_side = max(w, h)
            # 가장 작은 변
            smallest_side = min(w, h)

            # min_largest_side 기준
            min_largest_side = config.get("min_largest_side", 1)
            if largest_side < min_largest_side:
                print(f"Image too small: {largest_side} < {min_largest_side}")
                return None

            # resize 로직
            resize_if_min_side_exceeds = config.get("resize_if_min_side_exceeds", float("inf"))
            if smallest_side >= resize_if_min_side_exceeds:
                # resize_target_for_min_side 로 축소
                resize_target = config.get("resize_target_for_min_side", 1000)
                ratio = resize_target / float(smallest_side)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                im = im.resize((new_w, new_h), Resampling.LANCZOS)

            # 포맷 제한
            # PIL이 인식한 포맷이 대문자(JPEG)일 수 있으므로 소문자로
            pil_format = (im.format or "").lower()
            allowed_formats = config.get("formats", [])
            if pil_format not in allowed_formats:
                print(f"Invalid format: {pil_format} not in {allowed_formats}")
                return None

            # JPG -> JPEG 로 포맷명 정리
            if pil_format == "jpg":
                pil_format = "jpeg"

            # 다시 bytes 로 저장
            output_buffer = BytesIO()
            im.save(output_buffer, format=pil_format.upper())  # PIL에 맞춰서 대문자로
            output_buffer.seek(0)
            final_bytes = output_buffer.read()

    except Exception:
        print_exc()
        return None

    # 최종 base64 인코딩
    encoded_data = b64encode(final_bytes).decode("utf-8")
    return f"data:image/{pil_format};base64,{encoded_data}"


def _detect_image_type(image_data: bytes) -> Optional[str]:
    """
    Detect the image format based on the image binary signature (header).
    Only JPEG, PNG, GIF, WEBP, and BMP are handled as examples.
    If the format is not recognized, return None.
    """
    # JPEG: 시작 바이트가 FF D8 FF
    if image_data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    # PNG: 시작 바이트가 89 50 4E 47 0D 0A 1A 0A
    elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    # GIF: 시작 바이트가 GIF87a 또는 GIF89a
    elif image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
        return "gif"
    # WEBP: 시작 바이트가 RIFF....WEBP
    elif image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "webp"
    # BMP: 시작 바이트가 BM
    elif image_data.startswith(b"BM"):
        return "bmp"


def _process_local_image(path: Path, config: ImageProcessingConfig) -> Optional[str]:
    """로컬 파일이 존재하고 유효한 이미지 포맷이면 Base64 데이터 URL을 반환, 아니면 None."""
    if not path.is_file():
        return None
    lowered_suffix = path.suffix.lower()
    if not lowered_suffix or (lowered_suffix_without_dot := lowered_suffix[1:]) not in config["formats"]:
        return None
    return f"data:image/{lowered_suffix_without_dot};base64,{path.read_bytes().hex()}"


def _replace_images(
    markdown_text: str, image_description_and_references: ImageDescriptionAndReferences, description_format: str
) -> str:
    replacements: list[tuple[LinkInfo, str]] = []
    for image_description, link_infos in image_description_and_references.items():
        for link_info in link_infos:
            if image_description is None:
                replacements.append((link_info, link_info.link_markdown))
            else:
                replacements.append((
                    link_info,
                    description_format.format(
                        image_summary=image_description.replace("\n", " "),
                        inline_text=link_info.inline_text,
                        **link_info._asdict(),
                    ),
                ))

    return LinkInfo.replace(markdown_text, replacements)


# =======================

# =======================
# Example Usage
# =======================
if __name__ == "__main__":
    from pathlib import Path

    # Synchronous example:
    # sample_url = input("Enter the URL to crawl (sync): ").strip()
    # output_md_path = Path(".tmp.webpage_to_markdown.md")
    # output_llm_md_path = output_md_path.with_suffix(".llm.md")

    # with PlayWrightBot() as bot:
    #     md = bot.url_to_md(sample_url)
    #     print("[Sync Markdown result]\n", md[:200], "...")
    #     output_md_path.write_text(md, encoding="utf-8")

    #     headings = bot.select_and_extract(sample_url, "h2")
    #     print("\n[Sync h2 Tag Texts]")
    #     for idx, text in enumerate(headings, start=1):
    #         print(f"{idx}. {text}")

    #     md_llm = bot.url_to_md_with_llm(sample_url)
    #     print("\n[Sync LLM result]\n", md_llm[:200], "...")
    #     output_llm_md_path.write_text(md_llm, encoding="utf-8")
    #     print(f"\nResults saved to {output_md_path} and {output_llm_md_path}")

    # To run the asynchronous methods, use an async context:

    async def amain():
        sample_url = input("Enter the URL to crawl (async): ").strip()
        output_md_path = Path(".tmp.webpage_to_markdown.md")
        output_llm_md_path = output_md_path.with_suffix(".llm.md")

        async with PlayWrightBot() as bot:
            # md = await bot.aurl_to_md(sample_url)
            # print("[Async Markdown result]\n", md[:200], "...")
            # output_md_path.write_text(md, encoding="utf-8")

            md_llm = await bot.aurl_to_md_with_llm(sample_url)
            print("\n[Async LLM result]\n", md_llm[:200], "...")
            output_llm_md_path.write_text(md_llm, encoding="utf-8")
            print(f"\nResults saved to {output_md_path} and {output_llm_md_path}")

    asyncio.run(amain())
