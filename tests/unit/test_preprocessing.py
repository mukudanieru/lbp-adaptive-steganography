import pytest
import numpy as np
import cv2
from src.core.preprocessing import load_img


@pytest.fixture
def dummy_image_path(tmp_path):
    """Create a temporary 10x10 red BGR image for testing."""

    img_path = str(tmp_path / "./data/airplane.png")

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
