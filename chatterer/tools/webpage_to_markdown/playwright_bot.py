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

from dataclasses import dataclass, field
from types import TracebackType
from typing import (
    Awaitable,
    Callable,
    Optional,
    Self,
    Type,
    Union,
)
from uuid import uuid4

import playwright.async_api
import playwright.sync_api

from ...language_model import DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION, Chatterer
from ..convert_to_text import HtmlToMarkdownOptions, get_default_html_to_markdown_options, html_to_markdown
from .utils import (
    DEFAULT_UA,
    Base64Image,
    ImageProcessingConfig,
    MarkdownLink,
    PlaywrightLaunchOptions,
    PlaywrightPersistencyOptions,
    Replacable,
    ReplacementAndLinks,
    SelectedLineRanges,
    WaitUntil,
    aprocess_image_and_links,
    get_default_image_processing_config,
    get_default_playwright_launch_options,
    process_image_and_links,
    replace_image_or_links,
)


@dataclass
class PlayWrightBot:
    """
    A unified bot that leverages Playwright to render web pages, convert them to Markdown,
    extract elements, and filter key information using a language model.

    This class exposes both synchronous and asynchronous methods.

    Synchronous usage:
      with UnifiedPlaywrightBot() as bot:
          md = bot.url_to_markdown("https://example.com")
          headings = bot.select_and_extract("https://example.com", "h2")
          filtered_markdown = bot.url_to_markdown_with_llm("https://example.com")

    Asynchronous usage:
      async with UnifiedPlaywrightBot() as bot:
          md = await bot.aurl_to_markdown("https://example.com")
          headings = await bot.aselect_and_extract("https://example.com", "h2")
          filtered_markdown = await bot.aurl_to_markdown_with_llm("https://example.com")

    Attributes:
        headless (bool): Whether to run the browser in headless mode (default True).
        chatterer (Chatterer): An instance of the language model interface for processing text.
    """

    chatterer: Chatterer = field(default_factory=Chatterer.openai)
    playwright_launch_options: PlaywrightLaunchOptions = field(default_factory=get_default_playwright_launch_options)
    playwright_persistency_options: PlaywrightPersistencyOptions = field(default_factory=PlaywrightPersistencyOptions)
    html_to_markdown_options: HtmlToMarkdownOptions = field(default_factory=get_default_html_to_markdown_options)
    image_processing_config: ImageProcessingConfig = field(default_factory=get_default_image_processing_config)
    headers: dict[str, str] = field(default_factory=lambda: {"User-Agent": DEFAULT_UA})
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
    image_description_instruction: str = DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION
    replacer: Callable[[Replacable, MarkdownLink], str] = (
        lambda replacement,
        markdown_link: "<details><summary>{image_summary}</summary><img src='{url}' alt='{inline_text}'></details>".format(
            image_summary=str(replacement).replace("\n", " "),
            inline_text=markdown_link.inline_text,
            **markdown_link._asdict(),
        )
    )

    sync_playwright: Optional[playwright.sync_api.Playwright] = None
    sync_browser_context: Optional[playwright.sync_api.BrowserContext] = None
    async_playwright: Optional[playwright.async_api.Playwright] = None
    async_browser_context: Optional[playwright.async_api.BrowserContext] = None

    def get_sync_playwright(self) -> playwright.sync_api.Playwright:
        if self.sync_playwright is None:
            self.sync_playwright = playwright.sync_api.sync_playwright().start()
        return self.sync_playwright

    async def get_async_playwright(self) -> playwright.async_api.Playwright:
        if self.async_playwright is None:
            self.async_playwright = await playwright.async_api.async_playwright().start()
        return self.async_playwright

    def get_sync_browser(self) -> playwright.sync_api.BrowserContext:
        if self.sync_browser_context is not None:
            return self.sync_browser_context

        user_data_dir = self.playwright_persistency_options.get("user_data_dir")
        if user_data_dir:
            # Use persistent context if user_data_dir is provided
            self.sync_browser_context = self.get_sync_playwright().chromium.launch_persistent_context(
                user_data_dir=user_data_dir, **self.playwright_launch_options
            )
            return self.sync_browser_context

        # Otherwise, launch a new context
        browser = self.get_sync_playwright().chromium.launch(**self.playwright_launch_options)
        storage_state = self.playwright_persistency_options.get("storage_state")
        if storage_state:
            self.sync_browser_context = browser.new_context(storage_state=storage_state)
        else:
            self.sync_browser_context = browser.new_context()
        return self.sync_browser_context

    async def get_async_browser(self) -> playwright.async_api.BrowserContext:
        if self.async_browser_context is not None:
            return self.async_browser_context

        user_data_dir = self.playwright_persistency_options.get("user_data_dir")
        if user_data_dir:
            # Use persistent context if user_data_dir is provided
            self.async_browser_context = await (await self.get_async_playwright()).chromium.launch_persistent_context(
                user_data_dir=user_data_dir, **self.playwright_launch_options
            )
            return self.async_browser_context

        # Otherwise, launch a new context
        browser = await (await self.get_async_playwright()).chromium.launch(**self.playwright_launch_options)
        storage_state = self.playwright_persistency_options.get("storage_state")
        if storage_state:
            self.async_browser_context = await browser.new_context(storage_state=storage_state)
        else:
            self.async_browser_context = await browser.new_context()
        return self.async_browser_context

    def get_page(
        self,
        url: str,
        timeout: float = 10.0,
        wait_until: Optional[WaitUntil] = "domcontentloaded",
        referer: Optional[str] = None,
    ) -> playwright.sync_api.Page:
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
        page = self.get_sync_browser().new_page()
        page.goto(url, timeout=int(timeout * 1000), wait_until=wait_until, referer=referer)
        return page

    async def aget_page(
        self,
        url: str,
        timeout: float = 8,
        wait_until: Optional[WaitUntil] = "domcontentloaded",
        referer: Optional[str] = None,
    ) -> playwright.async_api.Page:
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
        page = await (await self.get_async_browser()).new_page()
        await page.goto(url, timeout=int(timeout * 1000), wait_until=wait_until, referer=referer)
        return page

    def url_to_markdown(
        self,
        url: str,
        wait: float = 0.2,
        scrolldown: bool = False,
        sleep: int = 0,
        reload: bool = True,
        timeout: Union[float, int] = 8,
        keep_page: bool = False,
        referer: Optional[str] = None,
        replace_base64_images: bool = False,
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
            replace_base64_images (bool): If True, replace base64 images with local file path(s).

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

    async def aurl_to_markdown(
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

    def url_to_markdown_with_llm(
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
        describe_images: bool = True,
        filter: bool = True,
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
            describe_images (bool): If True, describe images in the Markdown text.
            filter (bool): If True, filter the important lines using the language model.

        Returns:
            str: Filtered Markdown containing only the important lines.
        """
        markdown_content = self.url_to_markdown(
            url,
            wait=wait,
            scrolldown=scrolldown,
            sleep=sleep,
            reload=reload,
            timeout=timeout,
            keep_page=keep_page,
            referer=referer,
        )
        markdown_content = self.format_markdown_links_with_llm(
            markdown_text=markdown_content, referer_url=url, describe_images=describe_images
        )
        if not filter:
            return markdown_content
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

    async def aurl_to_markdown_with_llm(
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
        describe_images: bool = True,
        filter: bool = True,
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
            describe_images (bool): If True, describe images in the Markdown text.
            filter (bool): If True, filter the important lines using the language model.

        Returns:
            str: Filtered Markdown containing only the important lines.
        """
        markdown_content = await self.aurl_to_markdown(
            url,
            wait=wait,
            scrolldown=scrolldown,
            sleep=sleep,
            reload=reload,
            timeout=timeout,
            keep_page=keep_page,
            referer=referer,
        )
        markdown_content = await self.aformat_markdown_links_with_llm(
            markdown_text=markdown_content, referer_url=url, describe_images=describe_images
        )
        if not filter:
            return markdown_content
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

    def format_markdown_links_with_llm(self, markdown_text: str, referer_url: str, describe_images: bool) -> str:
        """
        Replace image URLs in Markdown text with their alt text and generate descriptions using a language model.
        """
        return replace_image_or_links(
            markdown_text=markdown_text,
            replacement_and_links=self.get_replacement_and_links(
                markdown_text=markdown_text, referer_url=referer_url, describe_images=describe_images
            ),
            replacer=self.replacer,
        )

    async def aformat_markdown_links_with_llm(self, markdown_text: str, referer_url: str, describe_images: bool) -> str:
        """
        Replace image URLs in Markdown text with their alt text and generate descriptions using a language model.
        """

        return replace_image_or_links(
            markdown_text=markdown_text,
            replacement_and_links=await self.aget_replacement_and_links(
                markdown_text=markdown_text, referer_url=referer_url, describe_images=describe_images
            ),
            replacer=self.replacer,
        )

    def get_replacement_and_links(
        self, markdown_text: str, referer_url: str, describe_images: bool
    ) -> ReplacementAndLinks:
        def _image_data_processor(image_url: Base64Image) -> str:
            return self.chatterer.describe_image(
                image_url=image_url.to_string(),
                instruction=self.image_description_instruction,
            )

        return ReplacementAndLinks(
            process_image_and_links(
                markdown_text=markdown_text,
                headers=self.headers | {"Referer": referer_url},
                config=self.image_processing_config,
                image_data_processor=_image_data_processor,
            )
        )

    async def aget_replacement_and_links(
        self, markdown_text: str, referer_url: str, describe_images: bool
    ) -> ReplacementAndLinks:
        image_data_processor: Callable[[MarkdownLink, Base64Image], Awaitable[str]]
        if describe_images:

            async def _image_description_processor(image_url: Base64Image) -> str:
                return self.chatterer.describe_image(
                    image_url=image_url.to_string(),
                    instruction=self.image_description_instruction,
                )

            image_data_processor = _image_description_processor

        else:

            async def _local_file_mapping_processor(image_url: Base64Image) -> str:
                return f"!{}"

        return ReplacementAndLinks(
            await aprocess_image_and_links(
                markdown_text=markdown_text,
                headers=self.headers | {"Referer": referer_url},
                config=self.image_processing_config,
                image_data_processor=image_data_processor,
            )
        )

    def __enter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit the synchronous context.

        Closes the browser and stops Playwright.
        """
        if self.sync_browser_context is not None:
            self.sync_browser_context.close()
            self.sync_browser_context = None
        if self.sync_playwright:
            self.sync_playwright.stop()
            self.sync_playwright = None

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Asynchronously exit the context.

        Closes the asynchronous browser and stops Playwright.
        """
        if self.async_browser_context is not None:
            await self.async_browser_context.close()
            self.async_browser_context = None
        if self.async_playwright:
            await self.async_playwright.stop()
            self.async_playwright = None
