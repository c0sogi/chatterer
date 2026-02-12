"""Image processing utilities for filtering and transforming images."""

from typing import (
    Awaitable,
    Callable,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
)
from urllib.parse import urlparse

from b64image import Base64Image, ImageType
from loguru import logger


class ImageProcessingConfig(TypedDict):
    """
    Settings for image filtering/transformation.
      - formats: (Sequence[str]) Allowed image formats (lowercase, e.g. ["jpeg", "png", "webp"]).
      - max_size_mb: (float) Maximum image size in MB. Exclude if exceeds.
      - min_largest_side: (int) Minimum size of the largest side of the image. Exclude if less than this.
      - resize_if_min_side_exceeds: (int) Resize if the smaller side exceeds this value.
      - resize_target_for_min_side: (int) Resize target for the smaller side. Keep ratio using Lanczos.
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


def is_remote_url(path: str) -> bool:
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)


def check_image(image: Base64Image, config: ImageProcessingConfig, *, verbose: bool = False) -> Optional[Base64Image]:
    """
    Validate and optionally resize a Base64Image according to config rules.
    Returns None if the image doesn't meet the criteria.
    """
    # Check image size
    max_size_mb = config.get("max_size_mb", float("inf"))
    if image.size_mb > max_size_mb:
        if verbose:
            logger.error(f"Image too large: {image.size_mb:.2f} MB > {max_size_mb} MB")
        return None

    # Check format
    allowed_formats: Sequence[ImageType] = config.get("formats", [])
    if image.ext not in allowed_formats:
        if verbose:
            logger.error(f"Invalid format: {image.ext} not in {allowed_formats}")
        return None

    try:
        w, h = image.dimensions
        largest_side = max(w, h)
        smallest_side = min(w, h)

        # Check minimum dimensions
        min_largest_side = config.get("min_largest_side", 1)
        if largest_side < min_largest_side:
            if verbose:
                logger.error(f"Image too small: {largest_side} < {min_largest_side}")
            return None

        # Resize if needed
        resize_if_min_side_exceeds = config.get("resize_if_min_side_exceeds", float("inf"))
        if smallest_side >= resize_if_min_side_exceeds:
            resize_target = config.get("resize_target_for_min_side", 1000)
            ratio = resize_target / float(smallest_side)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            return image.resize(new_w, new_h)

        return image
    except Exception as e:
        if verbose:
            logger.error(f"Error processing image: {e}")
        return None


def load_image(
    url_or_path: str,
    *,
    headers: dict[str, str] = {},
    img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], bytes]] = None,
) -> Optional[Base64Image]:
    """Load a Base64Image from a URL or local file path, with optional custom fetcher."""
    if is_remote_url(url_or_path):
        fetcher = img_bytes_fetcher or _default_fetch
        img_bytes = fetcher(url_or_path, headers)
        if not img_bytes:
            return None
        try:
            return Base64Image.from_bytes(img_bytes)
        except ValueError:
            return None
    try:
        return Base64Image.from_path(url_or_path)
    except (FileNotFoundError, ValueError):
        return None


async def aload_image(
    url_or_path: str,
    *,
    headers: dict[str, str] = {},
    img_bytes_fetcher: Optional[Callable[[str, dict[str, str]], Awaitable[bytes]]] = None,
) -> Optional[Base64Image]:
    """Async load a Base64Image from a URL or local file path, with optional custom fetcher."""
    if is_remote_url(url_or_path):
        fetcher = img_bytes_fetcher or _default_afetch
        img_bytes = await fetcher(url_or_path, headers)
        if not img_bytes:
            return None
        try:
            return Base64Image.from_bytes(img_bytes)
        except ValueError:
            return None
    try:
        return Base64Image.from_path(url_or_path)
    except (FileNotFoundError, ValueError):
        return None


def _default_fetch(url: str, headers: dict[str, str]) -> bytes:
    import requests

    try:
        with requests.Session() as session:
            response = session.get(url.strip(), headers={k: str(v) for k, v in headers.items()})
            response.raise_for_status()
            return bytes(response.content or b"")
    except Exception:
        return b""


async def _default_afetch(url: str, headers: dict[str, str]) -> bytes:
    from aiohttp import ClientSession

    try:
        async with ClientSession() as session:
            async with session.get(url.strip(), headers={k: str(v) for k, v in headers.items()}) as response:
                response.raise_for_status()
                return await response.read()
    except Exception:
        return b""
