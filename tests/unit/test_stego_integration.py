import os
import cv2
import pytest
import tempfile
import numpy as np
from pathlib import Path

from src.core.preprocessing import load_img
from src.core.pseudorandom import password_to_seed, generate_pixel_coordinates
from src.core.embedding import embed_message
from src.core.lbp import compute_lbp_classification
from src.core.extraction import extract_message


cover_path = Path("./data/cover")


# ---------------------------------------------------------------------------
# Helpers – image generators and I/O
# ---------------------------------------------------------------------------
def make_solid_bgr(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:

    return np.full((h, w, 3), color, dtype=np.uint8)


def make_gradient_bgr(h: int, w: int) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)

    return np.stack([np.tile(row, (h, 1))] * 3, axis=-1)


def make_random_bgr(h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_checkerboard_bgr(h: int, w: int, tile: int = 64) -> np.ndarray:
    xs = np.arange(w) // tile
    ys = np.arange(h) // tile
    grid = ((xs[None, :] + ys[:, None]) % 2 == 0).astype(np.uint8) * 255

    return np.stack([grid, grid, grid], axis=-1)


def make_channel_test_bgr(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img[:, :, 1] = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
    img[:, :, 2] = 128

    return img


def save_img_to_tmp(img: np.ndarray, suffix: str = ".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if not cv2.imwrite(path, img):
        os.unlink(path)
        raise ValueError("Failed to write image")

    return path


def get_cover_image_paths() -> list[Path]:
    if not cover_path.exists():
        return []

    return [p for p in cover_path.iterdir() if p.is_file()]


def run_roundtrip(
    cover_img: np.ndarray, message: str, password: str
) -> tuple[str, bool]:
    """
    Full pipeline: embed → save → reload → LBP check → extract.
    Returns (extracted_message, lbp_preserved).
    """
    h, w, _ = cover_img.shape
    cmap = compute_lbp_classification(cover_img)
    seed = password_to_seed(password)
    coords = generate_pixel_coordinates(h, w, seed)

    stego_in_memory = embed_message(
        rgb_img=cover_img,
        secret_message=message,
        classification_map=cmap,
        pixel_coords=coords,
    )

    path = save_img_to_tmp(stego_in_memory, ".png")

    try:
        stego_img = load_img(path)
        stego_cmap = compute_lbp_classification(stego_img)
        lbp_preserved = np.array_equal(cmap, stego_cmap)
        extracted = extract_message(stego_img, stego_cmap, coords)
    finally:
        os.unlink(path)

    return extracted, lbp_preserved


# ---------------------------------------------------------------------------
# Tests for embed_message + extract_message integration
# ---------------------------------------------------------------------------
class TestExtractMessageFromCoverImages:
    """Roundtrip tests using real images loaded from ./data/cover."""

    @pytest.mark.skipif(not get_cover_image_paths(), reason="No cover images found")
    def test_roundtrip_all_cover_images(self):
        """Every image in cover_path should embed and extract cleanly."""
        message = "Lorem ipsum dolor sit amet consectetur adipiscing elit. Dolor sit amet consectetur adipiscing elit quisque faucibus."
        password = "jonathan"

        for img_path in get_cover_image_paths():
            cover = load_img(str(img_path))
            extracted, lbp_preserved = run_roundtrip(cover, message, password)

            assert lbp_preserved, f"LBP changed for {img_path.name}"
            assert extracted == message, f"Extraction failed for {img_path.name}"

    @pytest.mark.skipif(not get_cover_image_paths(), reason="No cover images found")
    def test_lbp_invariant_all_cover_images(self):
        """LBP classification map must be identical before and after embedding."""
        for img_path in get_cover_image_paths():
            cover = load_img(str(img_path))
            _, lbp_preserved = run_roundtrip(cover, "LBP check", "PASSWORD")

            assert lbp_preserved, f"LBP not preserved for {img_path.name}"

    @pytest.mark.skipif(not get_cover_image_paths(), reason="No cover images found")
    def test_different_passwords_same_cover_image(self):
        """Same cover + different passwords should both extract correctly."""
        img_path = get_cover_image_paths()[0]
        cover = load_img(str(img_path))

        passwords = ["alpha", "beta", "gamma123", "!@#secret"]

        for password in passwords:
            extracted, _ = run_roundtrip(cover, "password test", password)

            assert extracted == "password test"

    @pytest.mark.skipif(not get_cover_image_paths(), reason="No cover images found")
    def test_different_messages_same_cover_image(self):
        """Different messages embedded into same cover should all extract correctly."""
        img_path = get_cover_image_paths()[0]
        cover = load_img(str(img_path))

        messages = [
            "short",
            "Hello, World!",
            "Special chars: !@#$%^&*()",
            "0123456789",
            "A longer message to test multi-pixel embedding across the image.",
            "Lorem ipsum dolor sit amet consectetur adipiscing elit. Dolor sit amet consectetur adipiscing elit quisque faucibus.",
        ]

        for msg in messages:
            extracted, _ = run_roundtrip(cover, msg, "testpassword")

            assert extracted == msg


class TestExtractMessageFromSyntheticImages:
    """Roundtrip tests using programmatically generated 512x512 images."""

    MESSAGE = "steganography"
    PASSWORD = "synthetic"

    def test_roundtrip_solid_black(self):
        img = make_solid_bgr(512, 512, (0, 0, 0))
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_roundtrip_solid_white(self):
        img = make_solid_bgr(512, 512, (255, 255, 255))
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_roundtrip_solid_midgray(self):
        img = make_solid_bgr(512, 512, (128, 128, 128))
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_roundtrip_solid_red(self):
        img = make_solid_bgr(512, 512, (0, 0, 255))
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_roundtrip_gradient(self):
        img = make_gradient_bgr(512, 512)
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_roundtrip_random_noise(self):
        img = make_random_bgr(512, 512, seed=0)
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_roundtrip_checkerboard(self):
        img = make_checkerboard_bgr(512, 512)
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_roundtrip_channel_ramp(self):
        img = make_channel_test_bgr(512, 512)
        extracted, lbp_preserved = run_roundtrip(img, self.MESSAGE, self.PASSWORD)

        assert lbp_preserved
        assert extracted == self.MESSAGE

    def test_lbp_invariant_random_noise(self):
        """Explicit LBP assertion separate from message extraction."""
        img = make_random_bgr(512, 512, seed=7)
        _, lbp_preserved = run_roundtrip(img, "lbp test", "lbppass")

        assert lbp_preserved, (
            "LBP classification changed after embedding random noise image"
        )

    def test_lbp_invariant_checkerboard(self):
        img = make_checkerboard_bgr(512, 512)
        _, lbp_preserved = run_roundtrip(img, "lbp test", "lbppass")

        assert lbp_preserved, (
            "LBP classification changed after embedding checkerboard image"
        )


class TestExtractMessageEdgeCases:
    """Edge cases: short messages, special characters, long messages, same seed."""

    def _img(self) -> np.ndarray:
        return make_random_bgr(512, 512, seed=99)

    def test_single_character_message(self):
        extracted, lbp = run_roundtrip(self._img(), "A", "pw")

        assert lbp
        assert extracted == "A"

    def test_single_word_message(self):
        extracted, lbp = run_roundtrip(self._img(), "secret", "pw")

        assert lbp
        assert extracted == "secret"

    def test_sentence_with_spaces(self):
        msg = "the quick brown fox"
        extracted, lbp = run_roundtrip(self._img(), msg, "pw")

        assert lbp
        assert extracted == msg

    def test_special_ascii_characters(self):
        msg = "!@#$%^&*()-_=+[]{}|;:',.<>?/`~"
        extracted, lbp = run_roundtrip(self._img(), msg, "pw")

        assert lbp
        assert extracted == msg

    def test_numeric_string(self):
        msg = "1234567890"
        extracted, lbp = run_roundtrip(self._img(), msg, "pw")

        assert lbp
        assert extracted == msg

    def test_mixed_case_message(self):
        msg = "Hello World"
        extracted, lbp = run_roundtrip(self._img(), msg, "pw")

        assert lbp
        assert extracted == msg

    def test_long_message(self):
        msg = "a" * 100
        extracted, lbp = run_roundtrip(self._img(), msg, "pw")

        assert lbp
        assert extracted == msg

    def test_same_password_different_messages_extract_correctly(self):
        img = make_random_bgr(512, 512, seed=55)

        for msg in ["alpha", "beta", "gamma"]:
            extracted, _ = run_roundtrip(img, msg, "samepassword")

            assert extracted == msg

    def test_same_message_different_passwords_extract_correctly(self):
        img = make_random_bgr(512, 512, seed=66)

        for pw in ["pw1", "pw2", "pw3"]:
            extracted, _ = run_roundtrip(img, "same message", pw)

            assert extracted == "same message"

    def test_stego_green_channel_unchanged_after_reload(self):
        """After save/reload, G channel of stego must equal G channel of cover."""
        cover = make_random_bgr(512, 512, seed=77)
        h, w, _ = cover.shape
        cmap = compute_lbp_classification(cover)
        seed = password_to_seed("greencheck")
        coords = generate_pixel_coordinates(h, w, seed)
        stego_mem = embed_message(cover, "green test", cmap, coords)
        path = save_img_to_tmp(stego_mem, ".png")

        try:
            stego_disk = load_img(path)
        finally:
            os.unlink(path)

        assert np.array_equal(cover[:, :, 1], stego_disk[:, :, 1]), (
            "Green channel was modified during embedding"
        )

    def test_deterministic_extraction_same_inputs(self):
        """Extracting twice from the same stego image yields identical results."""
        cover = make_random_bgr(512, 512, seed=88)
        h, w, _ = cover.shape
        cmap = compute_lbp_classification(cover)
        seed = password_to_seed("determinism")
        coords = generate_pixel_coordinates(h, w, seed)
        stego = embed_message(cover, "repeat", cmap, coords)
        path = save_img_to_tmp(stego, ".png")

        try:
            stego_img = load_img(path)
            stego_cmap = compute_lbp_classification(stego_img)
            r1 = extract_message(stego_img, stego_cmap, coords)
            r2 = extract_message(stego_img, stego_cmap, coords)
        finally:
            os.unlink(path)

        assert r1 == r2
