"""Tests for b64image integration and chatterer.utils.image_processing."""

import asyncio
import os
import struct
import tempfile
import unittest
import zlib

from b64image import Base64Image, what

from chatterer.utils.image_processing import (
    ImageProcessingConfig,
    aload_image,
    check_image,
    get_default_image_processing_config,
    is_remote_url,
    load_image,
)


def _make_png(width: int, height: int, r: int = 255, g: int = 0, b: int = 0) -> bytes:
    """Create a minimal valid PNG image."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + bytes([r, g, b]) * width
    compressed = zlib.compress(raw_rows)
    idat_crc = zlib.crc32(b"IDAT" + compressed)
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND")
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return sig + ihdr + idat + iend


class TestBase64Image(unittest.TestCase):
    def setUp(self) -> None:
        self.png_bytes = _make_png(10, 5)
        self.img = Base64Image.from_bytes(self.png_bytes)

    def test_from_bytes(self) -> None:
        self.assertEqual(self.img.ext, "png")
        self.assertGreater(len(self.img.data), 0)

    def test_data_uri(self) -> None:
        self.assertTrue(self.img.data_uri.startswith("data:image/png;base64,"))

    def test_from_base64_roundtrip(self) -> None:
        img2 = Base64Image.from_base64(self.img.data_uri)
        self.assertEqual(img2.ext, self.img.ext)
        self.assertEqual(img2.data, self.img.data)

    def test_dimensions(self) -> None:
        self.assertEqual(self.img.dimensions, (10, 5))

    def test_width(self) -> None:
        self.assertEqual(self.img.width, 10)

    def test_height(self) -> None:
        self.assertEqual(self.img.height, 5)

    def test_size_bytes(self) -> None:
        self.assertGreater(self.img.size_bytes, 0)

    def test_size_mb(self) -> None:
        self.assertAlmostEqual(self.img.size_mb, self.img.size_bytes / (1024 * 1024))

    def test_resize(self) -> None:
        resized = self.img.resize(20, 10)
        self.assertIsNot(resized, self.img)
        self.assertEqual(resized.dimensions, (20, 10))
        self.assertEqual(resized.ext, "png")

    def test_resize_roundtrip(self) -> None:
        resized = self.img.resize(20, 10)
        rt = Base64Image.from_bytes(resized.to_bytes())
        self.assertEqual(rt.dimensions, (20, 10))

    def test_resize_down(self) -> None:
        small = self.img.resize(3, 2)
        self.assertEqual(small.dimensions, (3, 2))

    def test_hash_consistency(self) -> None:
        same = Base64Image(ext=self.img.ext, data=self.img.data)
        self.assertEqual(hash(self.img), hash(same))

    def test_hash_differs(self) -> None:
        resized = self.img.resize(20, 10)
        self.assertNotEqual(hash(self.img), hash(resized))

    def test_what_bytes(self) -> None:
        self.assertEqual(what(self.png_bytes), "png")

    def test_what_base64(self) -> None:
        self.assertEqual(what(self.img.data), "png")

    def test_from_bytes_invalid(self) -> None:
        with self.assertRaises(ValueError):
            Base64Image.from_bytes(b"not an image")

    def test_from_base64_invalid(self) -> None:
        with self.assertRaises(ValueError):
            Base64Image.from_base64("not_valid_base64!!!")


class TestIsRemoteUrl(unittest.TestCase):
    def test_https(self) -> None:
        self.assertTrue(is_remote_url("https://example.com/img.png"))

    def test_http(self) -> None:
        self.assertTrue(is_remote_url("http://example.com/img.png"))

    def test_local_absolute(self) -> None:
        self.assertFalse(is_remote_url("/local/path/img.png"))

    def test_local_relative(self) -> None:
        self.assertFalse(is_remote_url("relative/path.png"))

    def test_file_uri_is_not_remote(self) -> None:
        self.assertFalse(is_remote_url("file:///path/to/file.png"))


class TestDefaultConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = get_default_image_processing_config()

    def test_formats(self) -> None:
        self.assertIn("png", self.cfg["formats"])
        self.assertIn("jpeg", self.cfg["formats"])

    def test_max_size_mb(self) -> None:
        self.assertEqual(self.cfg.get("max_size_mb"), 5)

    def test_min_largest_side(self) -> None:
        self.assertEqual(self.cfg.get("min_largest_side"), 200)


class TestCheckImage(unittest.TestCase):
    def setUp(self) -> None:
        self.img_300x250 = Base64Image.from_bytes(_make_png(300, 250))
        self.default_cfg = get_default_image_processing_config()

    def test_passes_valid_image(self) -> None:
        result = check_image(self.img_300x250, self.default_cfg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.ext, "png")

    def test_no_resize_under_threshold(self) -> None:
        result = check_image(self.img_300x250, self.default_cfg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.data, self.img_300x250.data)

    def test_rejects_too_small(self) -> None:
        tiny = Base64Image.from_bytes(_make_png(5, 3))
        self.assertIsNone(check_image(tiny, self.default_cfg))

    def test_rejects_wrong_format(self) -> None:
        cfg: ImageProcessingConfig = {"formats": ["jpeg"]}
        self.assertIsNone(check_image(self.img_300x250, cfg))

    def test_resize_triggers(self) -> None:
        cfg: ImageProcessingConfig = {
            "formats": ["png"],
            "resize_if_min_side_exceeds": 200,
            "resize_target_for_min_side": 100,
        }
        result = check_image(self.img_300x250, cfg)
        self.assertIsNotNone(result)
        assert result is not None
        w, h = result.dimensions
        self.assertEqual(h, 100)
        self.assertEqual(w, 120)

    def test_resize_produces_valid_image(self) -> None:
        cfg: ImageProcessingConfig = {
            "formats": ["png"],
            "resize_if_min_side_exceeds": 200,
            "resize_target_for_min_side": 100,
        }
        result = check_image(self.img_300x250, cfg)
        self.assertIsNotNone(result)
        assert result is not None
        rt = Base64Image.from_bytes(result.to_bytes())
        self.assertEqual(rt.ext, "png")
        self.assertEqual(rt.dimensions, (120, 100))

    def test_no_resize_when_under_threshold(self) -> None:
        cfg: ImageProcessingConfig = {
            "formats": ["png"],
            "resize_if_min_side_exceeds": 500,
        }
        result = check_image(self.img_300x250, cfg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.data, self.img_300x250.data)

    def test_verbose_mode(self) -> None:
        tiny = Base64Image.from_bytes(_make_png(5, 3))
        self.assertIsNone(check_image(tiny, self.default_cfg, verbose=True))

    def test_rejects_oversized(self) -> None:
        img = Base64Image.from_bytes(_make_png(10, 10))
        cfg: ImageProcessingConfig = {"formats": ["png"], "max_size_mb": 0.000001}
        self.assertIsNone(check_image(img, cfg))


class TestLoadImage(unittest.TestCase):
    def test_local_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(50, 50))
            tmp_path = f.name
        try:
            loaded = load_image(tmp_path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.dimensions, (50, 50))
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_returns_none(self) -> None:
        self.assertIsNone(load_image("/nonexistent/file.png"))


class TestAloadImage(unittest.TestCase):
    def test_local_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(50, 50))
            tmp_path = f.name
        try:
            loaded = asyncio.run(aload_image(tmp_path))
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.dimensions, (50, 50))
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_returns_none(self) -> None:
        self.assertIsNone(asyncio.run(aload_image("/nonexistent/file.png")))


class TestReexports(unittest.TestCase):
    def test_base64image(self) -> None:
        from chatterer import Base64Image as ChBase64Image

        self.assertIs(ChBase64Image, Base64Image)

    def test_what(self) -> None:
        from chatterer import what as ch_what

        self.assertIs(ch_what, what)


if __name__ == "__main__":
    unittest.main()
