from src.core.preprocessing import load_img, img_to_grayscale, validate_image_size
import numpy as np
import pytest
import cv2


# --- Tests for load_img ---
@pytest.fixture
def dummy_image_path(tmp_path):
    """Create a temporary 10x10 red BGR image for testing."""

    img_path = str(tmp_path / "test_image.png")

    # In BGR, Red is [0, 0, 255]
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 255]
    cv2.imwrite(img_path, img)

    return img_path


def test_load_img_returns_numpy_array(dummy_image_path):
    """Tests that the returned object is a NumPy array."""

    img = load_img(dummy_image_path)
    assert isinstance(img, np.ndarray)


def test_load_img_shape(dummy_image_path):
    """Tests that the image has height, width, and exactly three color channels."""

    img = load_img(dummy_image_path)
    assert len(img.shape) == 3
    assert img.shape[2] == 3


def test_load_img_bgr_order(dummy_image_path):
    """Tests that the color channels are in BGR order (red is at index 2)."""

    img = load_img(dummy_image_path)

    # The dummy data was red: B=0, G=0, R=255
    assert img[0, 0, 0] == 0  # Blue
    assert img[0, 0, 1] == 0  # Green
    assert img[0, 0, 2] == 255  # Red


def test_load_img_not_found():
    """
    Ensures that a FileNotFoundError is raised if the file does not exist.
    This validates that the function properly handles missing file paths.
    """

    with pytest.raises(FileNotFoundError):
        load_img("non_existent_file.png")


def test_load_img_invalid_file(tmp_path):
    """
    Ensures that a FileNotFoundError is raised for files that are not valid images.
    This validates that the function correctly rejects unsupported or corrupted file types.
    """

    text_file = tmp_path / "not_an_image.txt"
    text_file.write_text("This is just text.")

    with pytest.raises(FileNotFoundError):
        load_img(str(text_file))


# --- Tests for img_to_grayscale ---
def test_img_to_grayscale_shape():
    """Tests that a (H, W, 3) image is converted to (H, W)."""

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = img_to_grayscale(img)

    assert gray.shape == (10, 10)


def test_img_to_grayscale_dtype():
    """Tests that the output is specifically 8-bit unsigned integer."""

    img = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
    gray = img_to_grayscale(img)

    assert gray.dtype == np.uint8


def test_img_to_grayscale_math():
    """Tests that weights (0.11B, 0.59G, 0.30R) are applied correctly."""

    # Create a single pixel: Blue=100, Green=0, Red=0
    img = np.array([[[100, 0, 0]]], dtype=np.uint8)
    gray = img_to_grayscale(img)

    # 100 * 0.11 = 11
    assert gray[0, 0] == 11


def test_img_to_grayscale_constant_values():
    """Ensure a grey pixel remains the same value after conversion."""

    # 128*0.11 + 128*0.59 + 128*0.30 = 128
    img = np.full((2, 2, 3), 128, dtype=np.uint8)
    gray = img_to_grayscale(img)

    assert np.all(gray == 128)


# --- Tests for validate_image_size ---
def test_validate_image_size_match():
    """
    Ensures that a ValueError is raised when the input image does not have exactly two dimensions.
    This validates that the function correctly rejects images that are not 2D arrays.
    """

    img = np.zeros((100, 200), dtype=np.uint8)
    assert validate_image_size(img, (100, 200)) is True


def test_validate_image_size_mismatch():
    """
    Ensures that a ValueError is raised when the expected_size argument is not a (height, width) tuple.
    This validates that the function correctly enforces the required tuple structure.
    """

    img = np.zeros((100, 200), dtype=np.uint8)
    assert validate_image_size(img, (50, 50)) is False
