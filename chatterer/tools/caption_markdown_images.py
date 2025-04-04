import os.path
import re
from asyncio import gather
from traceback import format_exception_only, print_exc
from typing import (
    Awaitable,
    Callable,
    ClassVar,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Self,
    TypeGuard,
    cast,
)
from urllib.parse import urljoin, urlparse

from chatterer.language_model import Chatterer

from ..utils.base64_image import Base64Image, ImageProcessingConfig


class MarkdownLink(NamedTuple):
    type: Literal["link", "image"]
    url: str
    text: str
    title: Optional[str]
    pos: int
    end_pos: int

    @classmethod
    def from_markdown(cls, markdown_text: str, referer_url: Optional[str]) -> list[Self]:
        """
        The main function that returns the list of MarkdownLink for the input text.
        For simplicity, we do a "pure inline parse" of the entire text
        instead of letting the block parser break it up. That ensures that
        link tokens cover the global positions of the entire input.
        """

        from mistune import InlineParser, InlineState, Markdown

        class _TrackingInlineState(InlineState):
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

        class _TrackingInlineParser(InlineParser):
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

        md = Markdown(inline=_TrackingInlineParser())
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


ImageDataAndReferences = dict[Optional[str], list[MarkdownLink]]
ImageDescriptionAndReferences = NewType("ImageDescriptionAndReferences", ImageDataAndReferences)


def caption_markdown_images(
    markdown_text: str,
    headers: dict[str, str],
    image_processing_config: ImageProcessingConfig,
    description_format: str,
    image_description_instruction: str,
    chatterer: Chatterer,
    img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], bytes]] = None,
) -> str:
    """
    Replace image URLs in Markdown text with their alt text and generate descriptions using a language model.
    """
    image_url_and_markdown_links: dict[Optional[Base64Image], list[MarkdownLink]] = _get_image_url_and_markdown_links(
        markdown_text=markdown_text,
        headers=headers,
        config=image_processing_config,
        img_bytes_fetcher=img_bytes_fetcher,
    )

    image_description_and_references: ImageDescriptionAndReferences = ImageDescriptionAndReferences({})
    for image_url, markdown_links in image_url_and_markdown_links.items():
        if image_url is not None:
            try:
                image_summary: str = chatterer.describe_image(
                    image_url=image_url.data_uri,
                    instruction=image_description_instruction,
                )
            except Exception:
                print_exc()
                continue
            image_description_and_references[image_summary] = markdown_links
        else:
            image_description_and_references[None] = markdown_links

    return _replace_images(
        markdown_text=markdown_text,
        image_description_and_references=image_description_and_references,
        description_format=description_format,
    )


async def acaption_markdown_images(
    markdown_text: str,
    headers: dict[str, str],
    image_processing_config: ImageProcessingConfig,
    description_format: str,
    image_description_instruction: str,
    chatterer: Chatterer,
    img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], Awaitable[bytes]]] = None,
) -> str:
    """
    Replace image URLs in Markdown text with their alt text and generate descriptions using a language model.
    """
    image_url_and_markdown_links: dict[
        Optional[Base64Image], list[MarkdownLink]
    ] = await _aget_image_url_and_markdown_links(
        markdown_text=markdown_text,
        headers=headers,
        config=image_processing_config,
        img_bytes_fetcher=img_bytes_fetcher,
    )

    async def dummy() -> None:
        pass

    def _handle_exception(e: Optional[str | BaseException]) -> TypeGuard[Optional[str]]:
        if isinstance(e, BaseException):
            print(format_exception_only(type(e), e))
            return False
        return True

    coros: list[Awaitable[Optional[str]]] = [
        chatterer.adescribe_image(image_url=image_url.data_uri, instruction=image_description_instruction)
        if image_url is not None
        else dummy()
        for image_url in image_url_and_markdown_links.keys()
    ]

    return _replace_images(
        markdown_text=markdown_text,
        image_description_and_references=ImageDescriptionAndReferences({
            image_summary: markdown_links
            for markdown_links, image_summary in zip(
                image_url_and_markdown_links.values(), await gather(*coros, return_exceptions=True)
            )
            if _handle_exception(image_summary)
        }),
        description_format=description_format,
    )


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


def _get_image_url_and_markdown_links(
    markdown_text: str,
    headers: dict[str, str],
    config: ImageProcessingConfig,
    img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], bytes]] = None,
) -> dict[Optional[Base64Image], list[MarkdownLink]]:
    image_matches: dict[Optional[Base64Image], list[MarkdownLink]] = {}
    for markdown_link in MarkdownLink.from_markdown(markdown_text=markdown_text, referer_url=headers.get("Referer")):
        if markdown_link.type == "link":
            image_matches.setdefault(None, []).append(markdown_link)
            continue

        image_data = Base64Image.from_url_or_path(
            markdown_link.url, headers=headers, config=config, img_bytes_fetcher=img_bytes_fetcher
        )
        if not image_data:
            image_matches.setdefault(None, []).append(markdown_link)
            continue
        image_matches.setdefault(image_data, []).append(markdown_link)
    return image_matches


async def _aget_image_url_and_markdown_links(
    markdown_text: str,
    headers: dict[str, str],
    config: ImageProcessingConfig,
    img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], Awaitable[bytes]]] = None,
) -> dict[Optional[Base64Image], list[MarkdownLink]]:
    image_matches: dict[Optional[Base64Image], list[MarkdownLink]] = {}
    for markdown_link in MarkdownLink.from_markdown(markdown_text=markdown_text, referer_url=headers.get("Referer")):
        if markdown_link.type == "link":
            image_matches.setdefault(None, []).append(markdown_link)
            continue
        image_data = await Base64Image.afrom_url_or_path(
            markdown_link.url, headers=headers, config=config, img_bytes_fetcher=img_bytes_fetcher
        )
        if not image_data:
            image_matches.setdefault(None, []).append(markdown_link)
            continue
        image_matches.setdefault(image_data, []).append(markdown_link)
    return image_matches


def _replace_images(
    markdown_text: str, image_description_and_references: ImageDescriptionAndReferences, description_format: str
) -> str:
    replacements: list[tuple[MarkdownLink, str]] = []
    for image_description, markdown_links in image_description_and_references.items():
        for markdown_link in markdown_links:
            if image_description is None:
                if markdown_link.type == "link":
                    replacements.append((markdown_link, markdown_link.link_markdown))
                elif markdown_link.type == "image":
                    replacements.append((markdown_link, f"![{markdown_link.inline_text}](...)"))
            else:
                replacements.append((
                    markdown_link,
                    description_format.format(
                        image_summary=image_description.replace("\n", " "),
                        inline_text=markdown_link.inline_text,
                        **markdown_link._asdict(),
                    ),
                ))

    return MarkdownLink.replace(markdown_text, replacements)
