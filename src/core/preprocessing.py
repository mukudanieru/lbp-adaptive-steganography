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


def validate_image_size(gray_img: np.ndarray, expected_size: tuple[int, int]) -> bool:
    """
    Validate that a grayscale image has expected (height, width).
    """
    if gray_img.ndim != 2:
        raise ValueError("input must have at least 2 dimensions")

    if len(expected_size) != 2:
        raise ValueError("expected_size must be (height, width)")

    h, w = gray_img.shape
    expected_h, expected_w = expected_size

    return (h == expected_h) and (w == expected_w)
