"""
Recognize image file formats based on their first few bytes (base64-encoded).
Originally derived from Python's imghdr, modified for base64 inputs.
"""

import base64
import math
from typing import Callable, List, Literal, Optional

ImageType = Literal["jpeg", "png", "gif", "tiff", "rgb", "pbm", "pgm", "ppm", "rast", "xbm", "bmp", "webp", "exr"]

tests: List[Callable[[bytes], Optional[ImageType]]] = []


def register_test(func: Callable[[bytes], Optional[ImageType]]) -> Callable[[bytes], Optional[ImageType]]:
    tests.append(func)
    return func


def decode_prefix(b64_data: str, prefix_bytes: int = 32) -> bytes:
    needed_chars = math.ceil(prefix_bytes * 4 / 3)
    truncated_data = b64_data[:needed_chars]

    try:
        return base64.b64decode(truncated_data)
    except Exception:
        return base64.b64decode(b64_data)


def what(b64_data: str) -> Optional[ImageType]:
    """
    base64 인코딩된 문자열에 포함된 이미지의 타입을 반환한다.

    :param b64_data: 이미지 데이터를 담은 base64 문자열.
    :return: 이미지 포맷 문자열 (예: "jpeg", "png", "gif", 등) 또는 인식되지 않으면 None.
    """
    h: bytes = decode_prefix(b64_data, prefix_bytes=32)

    for tf in tests:
        res = tf(h)
        if res:
            return res
    return None


# --- 테스트 함수들 --- #


@register_test
def test_jpeg(h: bytes) -> Optional[ImageType]:
    if len(h) >= 10 and h[6:10] in (b"JFIF", b"Exif"):
        return "jpeg"
    elif h.startswith(b"\xff\xd8\xff\xdb"):
        return "jpeg"
    return None


@register_test
def test_png(h: bytes) -> Optional[ImageType]:
    if h.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return None


@register_test
def test_gif(h: bytes) -> Optional[ImageType]:
    if h.startswith(b"GIF87a") or h.startswith(b"GIF89a"):
        return "gif"
    return None


@register_test
def test_tiff(h: bytes) -> Optional[ImageType]:
    if h[:2] in (b"MM", b"II"):
        return "tiff"
    return None


@register_test
def test_rgb(h: bytes) -> Optional[ImageType]:
    if h.startswith(b"\x01\xda"):
        return "rgb"
    return None


@register_test
def test_pbm(h: bytes) -> Optional[ImageType]:
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"14" and h[2] in b" \t\n\r":
        return "pbm"
    return None


@register_test
def test_pgm(h: bytes) -> Optional[ImageType]:
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"25" and h[2] in b" \t\n\r":
        return "pgm"
    return None


@register_test
def test_ppm(h: bytes) -> Optional[ImageType]:
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"36" and h[2] in b" \t\n\r":
        return "ppm"
    return None


@register_test
def test_rast(h: bytes) -> Optional[ImageType]:
    if h.startswith(b"\x59\xa6\x6a\x95"):
        return "rast"
    return None


@register_test
def test_xbm(h: bytes) -> Optional[ImageType]:
    if h.startswith(b"#define "):
        return "xbm"
    return None


@register_test
def test_bmp(h: bytes) -> Optional[ImageType]:
    if h.startswith(b"BM"):
        return "bmp"
    return None


@register_test
def test_webp(h: bytes) -> Optional[ImageType]:
    if len(h) >= 12 and h.startswith(b"RIFF") and h[8:12] == b"WEBP":
        return "webp"
    return None


@register_test
def test_exr(h: bytes) -> Optional[ImageType]:
    if h.startswith(b"\x76\x2f\x31\x01"):
        return "exr"
    return None


if __name__ == "__main__":
    example_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/5+BAQAE/wH+U6az4wAAAABJRU5ErkJggg=="
    )

    fmt = what(example_png_base64)
    print(f"Detected format: {fmt}")  # Expected: png
