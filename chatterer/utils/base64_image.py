import os
import re
from base64 import b64decode, b64encode
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    ClassVar,
    Literal,
    NotRequired,
    Optional,
    Self,
    Sequence,
    Type,
    TypeAlias,
    TypedDict,
    TypeGuard,
    get_args,
)
from urllib.parse import urlparse

import requests
from aiohttp import ClientSession
from loguru import logger
from PIL.Image import Resampling
from PIL.Image import open as image_open
from pydantic import BaseModel

from ..messages import HumanMessage, MessageType
from .imghdr import what

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_content_part_image_param import ChatCompletionContentPartImageParam

ImageFormat: TypeAlias = Literal["jpeg", "png", "gif", "webp", "bmp"]
ExtendedImageFormat: TypeAlias = ImageFormat | Literal["jpg", "JPG"] | Literal["JPEG", "PNG", "GIF", "WEBP", "BMP"]

ALLOWED_IMAGE_FORMATS: tuple[ImageFormat, ...] = get_args(ImageFormat)


class ImageProcessingConfig(TypedDict):
    """
    이미지 필터링/변환 시 사용할 설정.
      - formats: (Sequence[str]) 허용할 이미지 포맷(소문자, 예: ["jpeg", "png", "webp"]).
      - max_size_mb: (float) 이미지 용량 상한(MB). 초과 시 제외.
      - min_largest_side: (int) 가로나 세로 중 가장 큰 변의 최소 크기. 미만 시 제외.
      - resize_if_min_side_exceeds: (int) 가로나 세로 중 작은 변이 이 값 이상이면 리스케일.
      - resize_target_for_min_side: (int) 리스케일시, '가장 작은 변'을 이 값으로 줄임(비율 유지는 Lanczos).
    """

    formats: Sequence[ImageFormat]
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


class Base64Image(BaseModel):
    ext: ImageFormat
    data: str

    IMAGE_TYPES: ClassVar[tuple[str, ...]] = ALLOWED_IMAGE_FORMATS
    IMAGE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"data:image/(" + "|".join(IMAGE_TYPES) + r");base64,([A-Za-z0-9+/]+={0,2})"
    )

    def __hash__(self) -> int:
        return hash((self.ext, self.data))

    @classmethod
    def new(
        cls,
        url_or_path_or_bytes: str | bytes,
        *,
        headers: dict[str, str] = {},
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], bytes]] = None,
    ) -> Self:
        if isinstance(url_or_path_or_bytes, bytes):
            ext = what(url_or_path_or_bytes)
            if ext is None:
                raise ValueError(f"Invalid image format: {url_or_path_or_bytes[:8]} ...")
            return cls.from_bytes(url_or_path_or_bytes, ext=_to_image_format(ext))
        elif maybe_base64 := cls.from_base64(url_or_path_or_bytes):
            return maybe_base64
        elif maybe_url_or_path := cls.from_url_or_path(
            url_or_path_or_bytes, headers=headers, img_bytes_fetcher=img_bytes_fetcher
        ):
            return maybe_url_or_path
        else:
            raise ValueError(f"Invalid image format: {url_or_path_or_bytes}")

    @classmethod
    async def anew(
        cls,
        url_or_path_or_bytes: str | bytes,
        *,
        headers: dict[str, str] = {},
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], Awaitable[bytes]]] = None,
    ) -> Self:
        if isinstance(url_or_path_or_bytes, bytes):
            ext = what(url_or_path_or_bytes)
            if ext is None:
                raise ValueError(f"Invalid image format: {url_or_path_or_bytes[:8]} ...")
            return cls.from_bytes(url_or_path_or_bytes, ext=_to_image_format(ext))
        elif maybe_base64 := cls.from_base64(url_or_path_or_bytes):
            return maybe_base64
        elif maybe_url_or_path := await cls.afrom_url_or_path(
            url_or_path_or_bytes, headers=headers, img_bytes_fetcher=img_bytes_fetcher
        ):
            return maybe_url_or_path
        else:
            raise ValueError(f"Invalid image format: {url_or_path_or_bytes}")

    @classmethod
    def from_base64(cls, data: str) -> Optional[Self]:
        match = cls.IMAGE_PATTERN.fullmatch(data)
        if match:
            return cls(ext=_to_image_format(match.group(1)), data=match.group(2))

        ext = what(data)
        if ext is None:
            return None
        return cls(
            ext=_to_image_format(ext), data=data
        )  # Assume data is already base64 encoded, since it passed `what`

    @classmethod
    def from_bytes(cls, data: bytes, ext: ExtendedImageFormat | None = None) -> Self:
        if ext is None:
            maybe_ext = what(data)
            if maybe_ext is None:
                raise ValueError(f"Invalid image format: {data[:8]} ...")
            ext = _to_image_format(maybe_ext)
        else:
            ext = _to_image_format(ext)
        return cls(ext=ext, data=b64encode(data).decode("utf-8"))

    @classmethod
    def from_path(cls, path: os.PathLike[str] | str, *, ext: ExtendedImageFormat | None = None) -> Optional[Self]:
        if isinstance(path, str):
            path = parse_path(path)
        else:
            path = Path(path)
        return cls.from_bytes(path.read_bytes(), ext=ext)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        headers: dict[str, str] = {},
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], bytes]] = None,
    ) -> Optional[Self]:
        fetcher = img_bytes_fetcher or cls._fetch_remote_image
        img_bytes = fetcher(url, headers)
        if not img_bytes:
            return None
        return cls.from_bytes(img_bytes)

    @classmethod
    async def afrom_url(
        cls,
        url: str,
        *,
        headers: dict[str, str] = {},
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], Awaitable[bytes]]] = None,
    ) -> Optional[Self]:
        fetcher = img_bytes_fetcher or cls._afetch_remote_image
        img_bytes = await fetcher(url, headers)
        if not img_bytes:
            return None
        return cls.from_bytes(img_bytes)

    @classmethod
    def from_url_or_path(
        cls,
        url_or_path: str,
        *,
        headers: dict[str, str] = {},
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], bytes]] = None,
    ) -> Optional[Self]:
        """Return a Base64Image instance from a URL or local file path."""
        if is_remote_url(url_or_path):
            return cls.from_url(url_or_path, headers=headers, img_bytes_fetcher=img_bytes_fetcher)
        return cls.from_path(url_or_path)

    @classmethod
    async def afrom_url_or_path(
        cls,
        url_or_path: str,
        *,
        headers: dict[str, str] = {},
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], Awaitable[bytes]]] = None,
    ) -> Optional[Self]:
        """Return a Base64Image instance from a URL or local file path."""
        if is_remote_url(url_or_path):
            return await cls.afrom_url(url_or_path, headers=headers, img_bytes_fetcher=img_bytes_fetcher)
        return cls.from_path(url_or_path)

    def to_message(
        self,
        *texts: str,
        message_type: Type[MessageType] = HumanMessage,
        text_position: Literal["before", "after"] = "after",
    ) -> MessageType:
        contents: list[str | dict[str, object]] = [self.data_uri_content_dict]
        text_contents: list[dict[str, object]] = [{"type": "text", "text": text} for text in texts]
        if text_position == "before":
            return message_type(content=text_contents + contents)
        else:
            return message_type(content=contents + text_contents)

    @property
    def data_uri(self) -> str:
        return f"data:image/{self.ext.replace('jpg', 'jpeg')};base64,{self.data}"

    @property
    def data_uri_content(self) -> "ChatCompletionContentPartImageParam":
        return {"type": "image_url", "image_url": {"url": self.data_uri}}

    @property
    def data_uri_content_dict(self) -> dict[str, object]:
        return {"type": "image_url", "image_url": {"url": self.data_uri}}

    @classmethod
    def _fetch_remote_image(cls, url: str, headers: dict[str, str]) -> bytes:
        try:
            with requests.Session() as session:
                response = session.get(url.strip(), headers={k: str(v) for k, v in headers.items()})
                response.raise_for_status()
                image_bytes = bytes(response.content or b"")
                if not image_bytes:
                    return b""
                return image_bytes
        except Exception:
            return b""

    @classmethod
    async def _afetch_remote_image(cls, url: str, headers: dict[str, str]) -> bytes:
        try:
            async with ClientSession() as session:
                async with session.get(url.strip(), headers={k: str(v) for k, v in headers.items()}) as response:
                    response.raise_for_status()
                    return await response.read()
        except Exception:
            return b""

    def check(self, config: ImageProcessingConfig, *, verbose: bool = False) -> Optional[Self]:
        """
        Retrieve an image in bytes and return a base64-encoded data URL,
        applying dynamic rules from 'config'.
        """

        image_data: bytes = b64decode(self.data)

        # Check image size
        max_size_mb = config.get("max_size_mb", float("inf"))
        image_size_mb = len(image_data) / (1024 * 1024)
        if image_size_mb > max_size_mb:
            if verbose:
                logger.error(f"Image too large: {image_size_mb:.2f} MB > {max_size_mb} MB")
            return None

        # Open image with Pillow
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
                    if verbose:
                        logger.error(f"Image too small: {largest_side} < {min_largest_side}")
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
                allowed_formats: Sequence[ImageFormat] = config.get("formats", [])
                if not _verify_ext(pil_format, allowed_formats):
                    if verbose:
                        logger.error(f"Invalid format: {pil_format} not in {allowed_formats}")
                    return None

                return self.__class__(ext=pil_format, data=b64encode(im.tobytes()).decode("utf-8"))
        except Exception as e:
            if verbose:
                logger.error(f"Error processing image: {e}")
            return None


def _to_image_format(ext: str) -> ImageFormat:
    lowered = ext.lower()
    if lowered in ALLOWED_IMAGE_FORMATS:
        return lowered
    elif lowered == "jpg":
        return "jpeg"  # jpg -> jpeg
    else:
        raise ValueError(f"Invalid image format: {ext}")


def parse_path(path_string: str) -> Path:
    """경로 문자열을 적절한 Path 객체로 변환"""

    if not path_string:
        path_string = "."

    # Convert file:// URI to a platform-specific path
    if path_string.startswith("file://"):
        parsed = urlparse(path_string)

        # Ignore localhost
        netloc = parsed.netloc
        if netloc == "localhost":
            netloc = ""

        if netloc:
            # Check if netloc is a drive letter (C:, D:, etc.)
            if len(netloc) == 2 and netloc[1] == ":":
                # file://C:/Users → C:/Users
                path_string = f"{netloc}{parsed.path}"
            else:
                # file://server/share → //server/share (UNC)
                path_string = f"//{netloc}{parsed.path}"
        else:
            # file:///path or file://localhost/path
            path_string = parsed.path
            # file:///C:/... → C:/...
            if len(path_string) > 2 and path_string[0] == "/" and path_string[2] == ":":
                path_string = path_string[1:]

    # Normalize backslashes to forward slashes
    path_string = path_string.replace("\\", "/")

    # /C:/... → C:/...
    if len(path_string) > 2 and path_string[0] == "/" and path_string[2] == ":":
        path_string = path_string[1:]

    return Path(path_string)


def _verify_ext(ext: str, allowed_types: Sequence[ImageFormat]) -> TypeGuard[ImageFormat]:
    return ext in allowed_types


def is_remote_url(path: str) -> bool:
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)


if __name__ == "__main__":
    import platform

    current_os = platform.system()

    test_cases = [
        # 기본 케이스
        ("/home/user/data.json", "/home/user/data.json"),
        ("C:\\Users\\A\\x.txt", "C:/Users/A/x.txt"),
        # file:// 변형
        ("file:///home/user/data.json", "/home/user/data.json"),
        ("file:///C:/Users/A/x.txt", "C:/Users/A/x.txt"),
        ("file://C:/Users/A/x.txt", "C:/Users/A/x.txt"),
        ("file://server/share/dir/file.txt", "//server/share/dir/file.txt"),
        # localhost
        ("file://localhost/C:/path/file.txt", "C:/path/file.txt"),
        ("file://localhost/home/user/file.txt", "/home/user/file.txt"),
        # 특수 케이스
        ("//server/share/file.txt", "//server/share/file.txt"),
        ("C:/Users/A/x.txt", "C:/Users/A/x.txt"),
        ("/C:/Users/A/x.txt", "C:/Users/A/x.txt"),
        # 상대 경로
        ("relative/path/file.txt", "relative/path/file.txt"),
        ("./relative/path.txt", "relative/path.txt"),
        ("../parent/path.txt", "../parent/path.txt"),
        # 빈 경로
        ("", "."),
        (".", "."),
    ]

    print(f"Platform: {platform.system()}\n")

    passed = 0
    failed = 0

    for test_input, expected_posix in test_cases:
        try:
            result = parse_path(test_input)
            actual_posix = result.as_posix()

            if actual_posix == expected_posix:
                status = "✓"
                passed += 1
            else:
                status = f"✗ (expected '{expected_posix}')"
                failed += 1

            print(f"{status} '{test_input}' → '{actual_posix}'")
        except Exception as e:
            print(f"✗ '{test_input}' → Error: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
