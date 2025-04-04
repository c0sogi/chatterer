from __future__ import annotations

import re
from base64 import b64encode
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import (
    Awaitable,
    Callable,
    ClassVar,
    Literal,
    NotRequired,
    Optional,
    Self,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeGuard,
    cast,
    get_args,
)
from urllib.parse import urlparse

import requests
from aiohttp import ClientSession
from PIL.Image import Resampling
from PIL.Image import open as image_open
from pydantic import BaseModel

logger = getLogger(__name__)
ImageType: TypeAlias = Literal["jpeg", "jpg", "png", "gif", "webp", "bmp"]


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
        "formats": ["png", "jpeg", "jpg", "gif", "bmp", "webp"],
    }


# image_url: str, headers: dict[str, str]) -> Optional[bytes]:
class Base64Image(BaseModel):
    ext: ImageType
    data: str

    IMAGE_TYPES: ClassVar[tuple[str, ...]] = tuple(map(str, get_args(ImageType)))
    IMAGE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"data:image/(" + "|".join(IMAGE_TYPES) + r");base64,([A-Za-z0-9+/]+={0,2})"
    )

    def __hash__(self) -> int:
        return hash((self.ext, self.data))

    def model_post_init(self, __context: object) -> None:
        if self.ext == "jpg":
            self.ext = "jpeg"

    @classmethod
    def from_string(cls, data: str) -> Optional[Self]:
        match = cls.IMAGE_PATTERN.fullmatch(data)
        if not match:
            return None
        return cls(ext=cast(ImageType, match.group(1)), data=match.group(2))

    @classmethod
    def from_bytes(cls, data: bytes, ext: ImageType) -> Self:
        return cls(ext=ext, data=b64encode(data).decode("utf-8"))

    @classmethod
    def from_url_or_path(
        cls,
        url_or_path: str,
        *,
        headers: dict[str, str] = {},
        config: ImageProcessingConfig = get_default_image_processing_config(),
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], bytes]] = None,
    ) -> Optional[Self]:
        """Return a Base64Image instance from a URL or local file path."""
        if maybe_base64 := cls.from_string(url_or_path):
            return maybe_base64
        elif is_remote_url(url_or_path):
            if img_bytes_fetcher:
                img_bytes = img_bytes_fetcher(url_or_path, headers)
            else:
                img_bytes = cls._fetch_remote_image(url_or_path, headers)
            if not img_bytes:
                return None
            return cls._convert_image_into_base64(img_bytes, config)
        try:
            return cls._process_local_image(Path(url_or_path), config)
        except Exception:
            return None

    @classmethod
    async def afrom_url_or_path(
        cls,
        url_or_path: str,
        *,
        headers: dict[str, str] = {},
        config: ImageProcessingConfig = get_default_image_processing_config(),
        img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], Awaitable[bytes]]] = None,
    ) -> Optional[Self]:
        """Return a Base64Image instance from a URL or local file path."""
        if maybe_base64 := cls.from_string(url_or_path):
            return maybe_base64
        elif is_remote_url(url_or_path):
            if img_bytes_fetcher:
                img_bytes = await img_bytes_fetcher(url_or_path, headers)
            else:
                img_bytes = await cls._afetch_remote_image(url_or_path, headers)
            if not img_bytes:
                return None
            return cls._convert_image_into_base64(img_bytes, config)
        try:
            return cls._process_local_image(Path(url_or_path), config)
        except Exception:
            return None

    @property
    def data_uri(self) -> str:
        return f"data:image/{self.ext.replace('jpg', 'jpeg')};base64,{self.data}"

    @property
    def data_uri_content(self) -> dict[Literal["type", "image_url"], Literal["image_url"] | dict[Literal["url"], str]]:
        return {"type": "image_url", "image_url": {"url": self.data_uri}}

    @staticmethod
    def _verify_ext(ext: str, allowed_types: Sequence[ImageType]) -> TypeGuard[ImageType]:
        return ext in allowed_types

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

    @classmethod
    def _convert_image_into_base64(cls, image_data: bytes, config: Optional[ImageProcessingConfig]) -> Optional[Self]:
        """
        Retrieve an image in bytes and return a base64-encoded data URL,
        applying dynamic rules from 'config'.
        """

        if not config:
            # config 없으면 그냥 기존 헤더만 보고 돌려주는 간단 로직
            return cls._simple_base64_encode(image_data)

        # 1) 용량 검사
        max_size_mb = config.get("max_size_mb", float("inf"))
        image_size_mb = len(image_data) / (1024 * 1024)
        if image_size_mb > max_size_mb:
            logger.error(f"Image too large: {image_size_mb:.2f} MB > {max_size_mb} MB")
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
                allowed_formats: Sequence[ImageType] = config.get("formats", [])
                if not cls._verify_ext(pil_format, allowed_formats):
                    logger.error(f"Invalid format: {pil_format} not in {allowed_formats}")
                    return None

                # 다시 bytes 로 저장
                output_buffer = BytesIO()
                im.save(output_buffer, format=pil_format.upper())  # PIL에 맞춰서 대문자로
                output_buffer.seek(0)
                final_bytes = output_buffer.read()

        except Exception:
            return None

        # 최종 base64 인코딩
        encoded_data = b64encode(final_bytes).decode("utf-8")
        return cls(ext=pil_format, data=encoded_data)

    @classmethod
    def _simple_base64_encode(cls, image_data: bytes) -> Optional[Self]:
        """
        Retrieve an image URL and return a base64-encoded data URL.
        """
        ext = detect_image_type(image_data)
        if not ext:
            return
        return cls(ext=ext, data=b64encode(image_data).decode("utf-8"))

    @classmethod
    def _process_local_image(cls, path: Path, config: ImageProcessingConfig) -> Optional[Self]:
        """로컬 파일이 존재하고 유효한 이미지 포맷이면 Base64 데이터 URL을 반환, 아니면 None."""
        if not path.is_file():
            return None
        ext = path.suffix.lower().removeprefix(".")
        if not cls._verify_ext(ext, config["formats"]):
            return None
        return cls(ext=ext, data=b64encode(path.read_bytes()).decode("ascii"))


def is_remote_url(path: str) -> bool:
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)


def detect_image_type(image_data: bytes) -> Optional[ImageType]:
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
