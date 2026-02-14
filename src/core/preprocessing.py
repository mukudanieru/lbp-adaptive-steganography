"""
Image preprocessing module
Handles image loading, conversion, and bit manipulation
"""

import cv2
import numpy as np


def load_img(file: str) -> np.ndarray:
    """
    Load an image from file as a NumPy array containing RGB values stored in BGR order.

    Args:
        file: Path to the image file

    Returns:
        An image as a NumPy array of shape (height, width, 3) in BGR format

    Raises:
        FileNotFoundError: If the image file cannot be loaded
    """
    img = cv2.imread(file, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"image file not found: {file}")

    return img


def rgb_to_grayscale(rgb_img):
    """
    Convert RGB image to grayscale using luminance formula.

    Formula: 0.30 × R + 0.59 × G + 0.11 × B

    Args:
        rgb_image: RGB image (height, width, 3)

    Returns:
        Grayscale image (height, width)
    """
    ...


def extract_3msb(pixel_value: int):
    """
    Extract 3 most significant bits from an 8-bit pixel value.

    Example:
        162 (10100010) → 5 (101)
        255 (11111111) → 7 (111)

    Args:
        pixel_value: 8-bit pixel intensity (0-255)

    Returns:
        3-MSB value (0-7)
    """
    ...


def validate_image_size(img):
    """
    Validate that image has expected dimensions.

    Args:
        image: Input image
        expected_size: Expected (height, width)

    Returns:
        True if valid, False otherwise
    """
    ...
