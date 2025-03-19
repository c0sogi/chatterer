from __future__ import annotations

import os.path
import re
from asyncio import gather
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from traceback import print_exc
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
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
)
from urllib.parse import urljoin, urlparse

import mistune
import playwright.sync_api
import requests
from aiohttp import ClientSession
from PIL.Image import Resampling
from PIL.Image import open as image_open
from pydantic import BaseModel, Field

ImageType: TypeAlias = Literal["jpeg", "jpg", "png", "gif", "webp", "bmp"]
IMAGE_TYPES: set[str] = {str(t) for t in get_args(ImageType)}
IMAGE_PATTERN: re.Pattern[str] = re.compile(rf"data:image/({'|'.join(IMAGE_TYPES)});base64,[A-Za-z0-9+/]+={0, 2}$")


class Base64Image(BaseModel):
    ext: ImageType
    data: str

    def model_post_init(self, __context: object) -> None:
        if self.ext == "jpg":
            self.ext = "jpeg"

    def to_string(self) -> str:
        return f"data:image/{self.ext.replace('jpg', 'jpeg')};base64,{self.data}"

    @classmethod
    def from_string(cls, data: str) -> Optional[Base64Image]:
        match = IMAGE_PATTERN.fullmatch(data)
        if not match:
            return None
        return cls(ext=cast(ImageType, match.group(1)), data=match.group(2))

    @staticmethod
    def verify_ext(ext: str, allowed_types: Sequence[ImageType]) -> TypeGuard[ImageType]:
        return ext in allowed_types


T = TypeVar("T")


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
    proxy: NotRequired[playwright.sync_api.ProxySettings]
    downloads_path: NotRequired[str | Path]
    slow_mo: NotRequired[float]
    traces_dir: NotRequired[str | Path]
    chromium_sandbox: NotRequired[bool]
    firefox_user_prefs: NotRequired[dict[str, str | float | bool]]


class PlaywrightPersistencyOptions(TypedDict):
    user_data_dir: NotRequired[str | Path]
    storage_state: NotRequired[playwright.sync_api.StorageState]


class PlaywrightOptions(PlaywrightLaunchOptions, PlaywrightPersistencyOptions): ...


def get_default_playwright_launch_options() -> PlaywrightLaunchOptions:
    return {"headless": True}


class ImageProcessingConfig(TypedDict):
    """
    이미지 필터링/변환 시 사용할 설정.
      - formats: (Sequence[str]) 허용할 이미지 포맷(소문자, 예: ["jpeg", "png", "webp"]).
      - max_size_mb: (float) 이미지 용량 상한(MB). 초과 시 제외.
      - min_largest_side: (int) 가로나 세로 중 가장 큰 변의 최소 크기. 미만 시 제외.
      - resize_if_min_side_exceeds: (int) 가로나 세로 중 작은 변이 이 값 이상이면 리스케일.
      - resize_target_for_min_side: (int) 리스케일시, '가장 작은 변'을 이 값으로 줄임(비율 유지는 Lanczos).
    """

    formats: Sequence[ImageType]
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
        "formats": ["png", "jpeg", "gif", "bmp", "webp"],
    }


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


def _get_image_bytes(image_url: str, headers: dict[str, str]) -> Optional[bytes]:
    try:
        with requests.Session() as session:
            response = session.get(image_url, headers={k: str(v) for k, v in headers.items()})
            if not response.ok:
                return
            return bytes(response.content or b"")
    except Exception:
        return


async def _aget_image_bytes(image_url: str, headers: dict[str, str]) -> Optional[bytes]:
    try:
        async with ClientSession() as session:
            async with session.get(image_url, headers={k: str(v) for k, v in headers.items()}) as response:
                if not response.ok:
                    return
                return await response.read()
    except Exception:
        return


# =======================


def _fetch_remote_image(url: str, headers: dict[str, str], config: ImageProcessingConfig) -> Optional[Base64Image]:
    image_bytes = _get_image_bytes(image_url=url.strip(), headers=headers)
    if not image_bytes:
        return None
    return _convert_image_into_base64(image_bytes, config)


async def _afetch_remote_image(
    url: str, headers: dict[str, str], config: ImageProcessingConfig
) -> Optional[Base64Image]:
    image_bytes = await _aget_image_bytes(image_url=url.strip(), headers=headers)
    if not image_bytes:
        return None
    return _convert_image_into_base64(image_bytes, config)


# =======================


def _process_markdown_image(
    markdown_link: MarkdownLink, headers: dict[str, str], config: ImageProcessingConfig
) -> Optional[Base64Image]:
    """마크다운 이미지 패턴에 매칭된 하나의 이미지를 처리해 Base64 URL을 반환(동기)."""
    if markdown_link.type != "image":
        return
    url: str = markdown_link.url
    if maybe_base64 := Base64Image.from_string(url):
        return maybe_base64
    elif _is_url(url):
        return _fetch_remote_image(url, headers, config)
    return _process_local_image(Path(url), config)


async def _aprocess_markdown_image(
    markdown_link: MarkdownLink, headers: dict[str, str], config: ImageProcessingConfig
) -> Optional[Base64Image]:
    """마크다운 이미지 패턴에 매칭된 하나의 이미지를 처리해 Base64 URL을 반환(비동기)."""
    if markdown_link.type != "image":
        return
    url: str = markdown_link.url
    if maybe_base64 := Base64Image.from_string(url):
        return maybe_base64
    elif _is_url(url):
        return await _afetch_remote_image(url, headers, config)
    return _process_local_image(Path(url), config)


# =======================


def process_image_and_links(
    markdown_text: str,
    headers: dict[str, str],
    config: ImageProcessingConfig,
    image_data_processor: Callable[[MarkdownLink, Base64Image], T],
) -> dict[Optional[T], list[MarkdownLink]]:
    result: dict[Optional[T], list[MarkdownLink]] = {}
    for markdown_link in MarkdownLink.from_markdown(markdown_text=markdown_text, referer_url=headers.get("Referer")):
        if markdown_link.type == "link":
            result.setdefault(None, []).append(markdown_link)
            continue
        image_data = _process_markdown_image(markdown_link, headers, config)
        if not image_data:
            continue
        result.setdefault(image_data_processor(markdown_link, image_data), []).append(markdown_link)
    return result


async def aprocess_image_and_links(
    markdown_text: str,
    headers: dict[str, str],
    config: ImageProcessingConfig,
    image_data_processor: Callable[[MarkdownLink, Base64Image], Awaitable[T]],
) -> dict[Optional[T], list[MarkdownLink]]:
    async def _process_link(markdown_link: MarkdownLink) -> tuple[Optional[T], MarkdownLink]:
        if markdown_link.type == "link":
            return (None, markdown_link)
        image_data = await _aprocess_markdown_image(markdown_link, headers, config)
        if not image_data:
            raise ValueError("Failed to process image data")
        return (await image_data_processor(markdown_link, image_data), markdown_link)

    coro_result: list[tuple[Optional[T], MarkdownLink] | BaseException] = await gather(
        *(
            _process_link(markdown_link)
            for markdown_link in MarkdownLink.from_markdown(markdown_text, headers.get("Referer"))
        ),
        return_exceptions=True,
    )
    result: dict[Optional[T], list[MarkdownLink]] = {}
    for item in coro_result:
        if isinstance(item, BaseException):
            continue
        data, markdown_link = item
        result.setdefault(data, []).append(markdown_link)
    return result


# =======================


def _simple_base64_encode(image_data: bytes) -> Optional[Base64Image]:
    """
    Retrieve an image URL and return a base64-encoded data URL.
    """
    ext = _detect_image_type(image_data)
    if not ext:
        return
    return Base64Image(ext=ext, data=b64encode(image_data).decode("utf-8"))


def _convert_image_into_base64(image_data: bytes, config: Optional[ImageProcessingConfig]) -> Optional[Base64Image]:
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
            pil_format: str = (im.format or "").lower()
            allowed_formats: Sequence[ImageType] = config.get("formats", [])
            if not Base64Image.verify_ext(pil_format, allowed_formats):
                print(f"Invalid format: {pil_format} not in {allowed_formats}")
                return None

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
    return Base64Image(ext=pil_format, data=encoded_data)


def _detect_image_type(image_data: bytes) -> Optional[ImageType]:
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


def _process_local_image(path: Path, config: ImageProcessingConfig) -> Optional[Base64Image]:
    """로컬 파일이 존재하고 유효한 이미지 포맷이면 Base64 데이터 URL을 반환, 아니면 None."""
    if not path.is_file():
        return None
    ext = path.suffix.lower().removeprefix(".")
    if not Base64Image.verify_ext(ext, config["formats"]):
        return None
    return Base64Image(ext=ext, data=b64encode(path.read_bytes()).decode("ascii"))


def replace_image_or_links(
    markdown_text: str, replacement_and_links: ReplacementAndLinks, replacer: Callable[[str, MarkdownLink], str]
) -> str:
    return MarkdownLink.replace(
        text=markdown_text,
        replacements=[
            (markdown_link, markdown_link.link_markdown)
            if replacement is None
            else (markdown_link, replacer(replacement, markdown_link))
            for replacement, links in replacement_and_links.items()
            for markdown_link in links
        ],
    )


Replacable: TypeAlias = str | tuple[str, str]
ImageDataAndLinks = NewType("ImageDataAndLinks", dict[Optional[str], list[MarkdownLink]])
ReplacementAndLinks = NewType("ReplacementAndLinks", dict[Optional[str], list[MarkdownLink]])
WaitUntil: TypeAlias = Literal["commit", "domcontentloaded", "load", "networkidle"]

DEFAULT_UA: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
