import os
import tempfile

from src.core.preprocessing import load_img, validate_image_size
import cv2
import numpy as np
import pytest

from src.core.preprocessing import load_img_from_bytes, encode_img_to_bytes

# ---------------------------------------------------------------------------
# Helpers – synthetic image generators
# ---------------------------------------------------------------------------
def make_solid_bgr(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    """Return a solid-color BGR image."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def make_gradient_bgr(h: int, w: int) -> np.ndarray:
    """Return an image with a horizontal gradient (0-255) across all channels."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.stack([np.tile(row, (h, 1))] * 3, axis=-1)
    return img


def make_random_bgr(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Return a random-noise BGR image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_checkerboard_bgr(h: int, w: int, tile: int = 64) -> np.ndarray:
    """Return a black-and-white checkerboard BGR image."""
    xs = np.arange(w) // tile
    ys = np.arange(h) // tile
    grid = ((xs[None, :] + ys[:, None]) % 2 == 0).astype(np.uint8) * 255
    return np.stack([grid, grid, grid], axis=-1)


def make_channel_test_bgr(h: int, w: int) -> np.ndarray:
    """Return an image where each colour channel ramps independently."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))  # B
    img[:, :, 1] = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))  # G
    img[:, :, 2] = 128  # R constant
    return img


def save_bgr_to_tmp(img: np.ndarray, suffix: str = ".png") -> str:
    """Write a BGR ndarray to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    cv2.imwrite(path, img)

    return path


# ---------------------------------------------------------------------------
# Tests for load_img
# ---------------------------------------------------------------------------
class TestLoadImg:
    def test_load_solid_black_512(self):
        img = make_solid_bgr(512, 512, (0, 0, 0))
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert result.shape == (512, 512, 3)
            assert result.dtype == np.uint8
            assert np.all(result == 0)
        finally:
            os.unlink(path)

    def test_load_solid_white_512(self):
        img = make_solid_bgr(512, 512, (255, 255, 255))
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert result.shape == (512, 512, 3)
            assert np.all(result == 255)
        finally:
            os.unlink(path)

    def test_load_solid_red_512(self):
        """Red in BGR is (0, 0, 255)."""
        img = make_solid_bgr(512, 512, (0, 0, 255))
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert result.shape == (512, 512, 3)
            assert np.all(result[:, :, 0] == 0)  # B
            assert np.all(result[:, :, 1] == 0)  # G
            assert np.all(result[:, :, 2] == 255)  # R
        finally:
            os.unlink(path)

    def test_load_solid_green_512(self):
        """Green in BGR is (0, 255, 0)."""
        img = make_solid_bgr(512, 512, (0, 255, 0))
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert result.shape == (512, 512, 3)
            assert np.all(result[:, :, 1] == 255)
        finally:
            os.unlink(path)

    def test_load_solid_blue_512(self):
        """Blue in BGR is (255, 0, 0)."""
        img = make_solid_bgr(512, 512, (255, 0, 0))
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert result.shape == (512, 512, 3)
            assert np.all(result[:, :, 0] == 255)
        finally:
            os.unlink(path)

    def test_load_gradient_512(self):
        img = make_gradient_bgr(512, 512)
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert np.array_equal(result, img)
            assert result.shape == (512, 512, 3)
        finally:
            os.unlink(path)

    def test_load_random_noise_512(self):
        img = make_random_bgr(512, 512, seed=0)
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert np.array_equal(result, img)
            assert result.shape == (512, 512, 3)
        finally:
            os.unlink(path)

    def test_load_checkerboard_512(self):
        img = make_checkerboard_bgr(512, 512)
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert np.array_equal(result, img)
            assert result.shape == (512, 512, 3)
        finally:
            os.unlink(path)

    def test_load_channel_ramp_512(self):
        img = make_channel_test_bgr(512, 512)
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert np.array_equal(result, img)
            assert result.shape == (512, 512, 3)
        finally:
            os.unlink(path)

    # --- Different file formats ---

    def test_load_jpeg_image(self):
        img = make_random_bgr(512, 512, seed=7)
        path = save_bgr_to_tmp(img, suffix=".jpg")
        try:
            result = load_img(path)

            # JPEG is lossy so we only check shape / type
            assert result.shape == (512, 512, 3)
            assert result.dtype == np.uint8
        finally:
            os.unlink(path)

    def test_load_bmp_image(self):
        img = make_solid_bgr(512, 512, (10, 20, 30))
        path = save_bgr_to_tmp(img, suffix=".bmp")
        try:
            result = load_img(path)
            assert result.shape == (512, 512, 3)
        finally:
            os.unlink(path)

    # --- Return type checks ---

    def test_return_type_is_ndarray(self):
        img = make_solid_bgr(512, 512, (100, 100, 100))
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert isinstance(result, np.ndarray)
        finally:
            os.unlink(path)

    def test_return_dtype_uint8(self):
        img = make_random_bgr(512, 512)
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert result.dtype == np.uint8
        finally:
            os.unlink(path)

    def test_return_ndim_is_3(self):
        img = make_solid_bgr(512, 512, (0, 128, 255))
        path = save_bgr_to_tmp(img)
        try:
            result = load_img(path)
            assert result.ndim == 3
        finally:
            os.unlink(path)

    # --- Error cases ---

    def test_raises_file_not_found_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_img("/nonexistent/path/image.png")

    def test_raises_file_not_found_for_empty_string(self):
        with pytest.raises(FileNotFoundError):
            load_img("")

    def test_raises_file_not_found_for_nonexistent_extension(self):
        with pytest.raises(FileNotFoundError):
            load_img("/tmp/does_not_exist_xyz_abc.png")

    def test_file_not_found_message_contains_path(self):
        bad_path = "/no/such/file.png"
        with pytest.raises(FileNotFoundError, match=bad_path):
            load_img(bad_path)


# ---------------------------------------------------------------------------
# Tests for validate_image_size
# ---------------------------------------------------------------------------
class TestValidateImageSize:
    def test_exact_512x512_match(self):
        gray = np.zeros((512, 512), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is True

    def test_solid_gray_512x512(self):
        gray = np.full((512, 512), 128, dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is True

    def test_random_noise_512x512(self):
        rng = np.random.default_rng(99)
        gray = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is True

    def test_gradient_512x512(self):
        gray = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
        assert validate_image_size(gray, (512, 512)) is True

    def test_float_gray_512x512(self):
        gray = np.random.rand(512, 512).astype(np.float32)
        assert validate_image_size(gray, (512, 512)) is True

    def test_non_512_match_256x256(self):
        gray = np.zeros((256, 256), dtype=np.uint8)
        assert validate_image_size(gray, (256, 256)) is True

    def test_non_square_match(self):
        gray = np.zeros((256, 512), dtype=np.uint8)
        assert validate_image_size(gray, (256, 512)) is True

    def test_1x1_match(self):
        gray = np.zeros((1, 1), dtype=np.uint8)
        assert validate_image_size(gray, (1, 1)) is True

    def test_large_image_match(self):
        gray = np.zeros((1024, 1024), dtype=np.uint8)
        assert validate_image_size(gray, (1024, 1024)) is True

    # --- Valid non-matching cases (should return False) ---

    def test_height_mismatch_512x512_vs_256x512(self):
        gray = np.zeros((256, 512), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is False

    def test_width_mismatch_512x512_vs_512x256(self):
        gray = np.zeros((512, 256), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is False

    def test_both_dimensions_mismatch(self):
        gray = np.zeros((256, 256), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is False

    def test_wrong_height_only(self):
        gray = np.zeros((100, 512), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is False

    def test_wrong_width_only(self):
        gray = np.zeros((512, 100), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is False

    def test_swapped_dimensions(self):
        """A 256x512 image should NOT match (512, 256)."""
        gray = np.zeros((256, 512), dtype=np.uint8)
        assert validate_image_size(gray, (512, 256)) is False

    def test_off_by_one_height(self):
        gray = np.zeros((511, 512), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is False

    def test_off_by_one_width(self):
        gray = np.zeros((512, 511), dtype=np.uint8)
        assert validate_image_size(gray, (512, 512)) is False

    # --- Error / edge cases ---

    def test_raises_on_3d_input(self):
        """BGR/RGB image (3-channel) must raise ValueError."""
        bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2 dimensions"):
            validate_image_size(bgr, (512, 512))

    def test_raises_on_1d_input(self):
        arr = np.zeros(512, dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image_size(arr, (512, 512))

    def test_raises_on_4d_input(self):
        arr = np.zeros((512, 512, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image_size(arr, (512, 512))

    def test_raises_on_expected_size_length_1(self):
        gray = np.zeros((512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="expected_size"):
            validate_image_size(gray, (512,))

    def test_raises_on_expected_size_length_3(self):
        gray = np.zeros((512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="expected_size"):
            validate_image_size(gray, (512, 512, 3))

    def test_raises_on_expected_size_empty(self):
        gray = np.zeros((512, 512), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image_size(gray, ())

    # --- Integration: load then validate ---

    def test_load_then_validate_correct_size(self):
        img_bgr = make_solid_bgr(512, 512, (50, 100, 150))
        path = save_bgr_to_tmp(img_bgr)
        try:
            loaded = load_img(path)
            gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
            assert validate_image_size(gray, (512, 512)) is True
        finally:
            os.unlink(path)

    def test_load_then_validate_wrong_size(self):
        img_bgr = make_solid_bgr(256, 512, (50, 100, 150))
        path = save_bgr_to_tmp(img_bgr)
        try:
            loaded = load_img(path)
            gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
            assert validate_image_size(gray, (512, 512)) is False
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests for load_img_from_bytes
# ---------------------------------------------------------------------------
class TestLoadImgFromBytes:

    # --- Basic loading ---

    def test_load_solid_black_png_bytes(self):
        img = make_solid_bgr(512, 512, (0, 0, 0))
        _, buf = cv2.imencode(".png", img)
        result = load_img_from_bytes(buf.tobytes(), "image.png")
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8
        assert np.all(result == 0)

    def test_load_solid_white_png_bytes(self):
        img = make_solid_bgr(512, 512, (255, 255, 255))
        _, buf = cv2.imencode(".png", img)
        result = load_img_from_bytes(buf.tobytes(), "image.png")
        assert result.shape == (512, 512, 3)
        assert np.all(result == 255)

    def test_load_solid_red_png_bytes(self):
        """Red in BGR is (0, 0, 255)."""
        img = make_solid_bgr(512, 512, (0, 0, 255))
        _, buf = cv2.imencode(".png", img)
        result = load_img_from_bytes(buf.tobytes(), "image.png")
        assert np.all(result[:, :, 0] == 0)    # B
        assert np.all(result[:, :, 1] == 0)    # G
        assert np.all(result[:, :, 2] == 255)  # R

    def test_load_gradient_roundtrip(self):
        img = make_gradient_bgr(512, 512)
        _, buf = cv2.imencode(".png", img)
        result = load_img_from_bytes(buf.tobytes(), "image.png")
        assert np.array_equal(result, img)

    def test_load_random_noise_roundtrip(self):
        img = make_random_bgr(512, 512, seed=3)
        _, buf = cv2.imencode(".png", img)
        result = load_img_from_bytes(buf.tobytes(), "image.png")
        assert np.array_equal(result, img)

    def test_load_checkerboard_roundtrip(self):
        img = make_checkerboard_bgr(512, 512)
        _, buf = cv2.imencode(".png", img)
        result = load_img_from_bytes(buf.tobytes(), "image.png")
        assert np.array_equal(result, img)

    def test_load_channel_ramp_roundtrip(self):
        img = make_channel_test_bgr(512, 512)
        _, buf = cv2.imencode(".png", img)
        result = load_img_from_bytes(buf.tobytes(), "image.png")
        assert np.array_equal(result, img)

    # --- Supported formats ---

    def test_load_bmp_bytes(self):
        img = make_solid_bgr(256, 256, (10, 20, 30))
        _, buf = cv2.imencode(".bmp", img)
        result = load_img_from_bytes(buf.tobytes(), "image.bmp")
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_load_tiff_bytes(self):
        img = make_solid_bgr(256, 256, (50, 100, 150))
        _, buf = cv2.imencode(".tiff", img)
        result = load_img_from_bytes(buf.tobytes(), "image.tiff")
        assert result.shape == (256, 256, 3)

    def test_load_tif_extension(self):
        img = make_solid_bgr(256, 256, (50, 100, 150))
        _, buf = cv2.imencode(".tiff", img)
        result = load_img_from_bytes(buf.tobytes(), "image.tif")
        assert result.shape == (256, 256, 3)

    # --- Unsupported extensions ---

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".gif"])
    def test_reject_unsupported_extension(self, ext):
        img = make_solid_bgr(128, 128, (0, 0, 0))
        _, buf = cv2.imencode(".png", img)
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_img_from_bytes(buf.tobytes(), f"image{ext}")

    def test_extension_check_is_case_insensitive(self):
        img = make_solid_bgr(128, 128, (0, 0, 0))
        _, buf = cv2.imencode(".png", img)
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_img_from_bytes(buf.tobytes(), "image.JPG")

    # --- Corrupt / invalid bytes ---

    def test_raises_on_empty_bytes(self):
        with pytest.raises((ValueError, cv2.error)):
            load_img_from_bytes(b"")

    def test_raises_on_garbage_bytes(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            load_img_from_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")

    def test_raises_on_truncated_png(self):
        img = make_solid_bgr(128, 128, (100, 100, 100))
        _, buf = cv2.imencode(".png", img)
        with pytest.raises(ValueError, match="Failed to decode"):
            load_img_from_bytes(buf.tobytes()[:50], "image.png")

    # --- Return type ---

    def test_return_type_is_ndarray(self):
        img = make_solid_bgr(128, 128, (50, 50, 50))
        _, buf = cv2.imencode(".png", img)
        assert isinstance(load_img_from_bytes(buf.tobytes(), "image.png"), np.ndarray)

    def test_return_dtype_uint8(self):
        img = make_random_bgr(128, 128)
        _, buf = cv2.imencode(".png", img)
        assert load_img_from_bytes(buf.tobytes(), "image.png").dtype == np.uint8

    def test_return_ndim_is_3(self):
        img = make_solid_bgr(128, 128, (0, 128, 255))
        _, buf = cv2.imencode(".png", img)
        assert load_img_from_bytes(buf.tobytes(), "image.png").ndim == 3

    def test_return_has_3_channels(self):
        img = make_solid_bgr(128, 128, (0, 128, 255))
        _, buf = cv2.imencode(".png", img)
        assert load_img_from_bytes(buf.tobytes(), "image.png").shape[2] == 3


# ---------------------------------------------------------------------------
# Tests for encode_img_to_bytes
# ---------------------------------------------------------------------------
class TestEncodeImgToBytes:

    # --- Basic encoding ---

    def test_encode_returns_bytes(self):
        img = make_solid_bgr(512, 512, (0, 0, 0))
        assert isinstance(encode_img_to_bytes(img, ".png"), bytes)

    def test_encoded_png_starts_with_png_signature(self):
        img = make_solid_bgr(128, 128, (100, 150, 200))
        assert encode_img_to_bytes(img, ".png")[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encode_solid_black_png_roundtrip(self):
        img = make_solid_bgr(512, 512, (0, 0, 0))
        encoded = encode_img_to_bytes(img, ".png")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    def test_encode_random_noise_png_roundtrip(self):
        img = make_random_bgr(512, 512, seed=5)
        encoded = encode_img_to_bytes(img, ".png")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    def test_encode_checkerboard_png_roundtrip(self):
        img = make_checkerboard_bgr(512, 512)
        encoded = encode_img_to_bytes(img, ".png")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    def test_encode_channel_ramp_png_roundtrip(self):
        img = make_channel_test_bgr(512, 512)
        encoded = encode_img_to_bytes(img, ".png")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    # --- Non-PNG supported formats ---

    def test_encode_bmp_roundtrip(self):
        img = make_random_bgr(256, 256, seed=5)
        encoded = encode_img_to_bytes(img, ".bmp")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    def test_encode_tiff_roundtrip(self):
        img = make_random_bgr(256, 256, seed=5)
        encoded = encode_img_to_bytes(img, ".tiff")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    def test_encode_tif_roundtrip(self):
        img = make_random_bgr(256, 256, seed=5)
        encoded = encode_img_to_bytes(img, ".tif")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    def test_encode_ext_is_case_insensitive(self):
        img = make_solid_bgr(128, 128, (0, 128, 255))
        encoded = encode_img_to_bytes(img, ".PNG")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    # --- Unsupported extensions ---

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".gif", ".webp"])
    def test_reject_unsupported_extension(self, ext):
        img = make_solid_bgr(128, 128, (0, 0, 0))
        with pytest.raises(ValueError, match="Unsupported file type"):
            encode_img_to_bytes(img, ext)

    # --- Different sizes ---

    def test_encode_non_square_image(self):
        img = make_solid_bgr(256, 512, (10, 20, 30))
        encoded = encode_img_to_bytes(img, ".png")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (256, 512, 3)
        assert np.array_equal(decoded, img)

    def test_encode_large_image(self):
        img = make_random_bgr(1024, 1024, seed=7)
        encoded = encode_img_to_bytes(img, ".png")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, img)

    def test_double_encode_decode_png_is_lossless(self):
        img = make_random_bgr(128, 128, seed=13)
        once = encode_img_to_bytes(img, ".png")
        decoded_once = cv2.imdecode(np.frombuffer(once, dtype=np.uint8), cv2.IMREAD_COLOR)
        twice = encode_img_to_bytes(decoded_once, ".png")
        decoded_twice = cv2.imdecode(np.frombuffer(twice, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded_twice, img)

    # --- Cross-function: encode_img_to_bytes -> load_img_from_bytes ---

    def test_encode_then_load_img_from_bytes_roundtrip(self):
        img = make_random_bgr(256, 256, seed=11)
        encoded = encode_img_to_bytes(img, ".png")
        reloaded = load_img_from_bytes(encoded, "image.png")
        assert np.array_equal(reloaded, img)