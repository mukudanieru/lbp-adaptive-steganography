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
        raise FileNotFoundError(f"Image file not found: {file}")

    return img


def img_to_grayscale(rgb_img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminance formula:
    gray = 0.11 * B + 0.59 * G + 0.30 * R

    Args:
        rgb_img: An image as a NumPy array of shape (height, width, 3) representing a BGR image

    Returns:
        A grayscale image as a NumPy array of shape (height, width)
    """
    weights = np.array([0.11, 0.59, 0.30])  # In BGR Order

    # np.dot applies the weights to the last dimension (the 3 color channels)
    # For each pixel [B, G, R], it computes:
    #     B*0.11 + G*0.59 + R*0.30
    floating_gray = np.dot(rgb_img, weights)

    return floating_gray.astype(np.uint8)


def extract_3msb(pixel_value: np.uint8):
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


def validate_image_size(gray_img: np.ndarray, expected_size: tuple[int, int]) -> bool:
    """
    Validate that a grayscale image has expected (height, width).
    """
    if gray_img.ndim != 2:
        raise ValueError("Input must have at least 2 dimensions")

    if len(expected_size) != 2:
        raise ValueError("Expected_size must be (height, width)")

    h, w = gray_img.shape
    expected_h, expected_w = expected_size

    return (h == expected_h) and (w == expected_w)
