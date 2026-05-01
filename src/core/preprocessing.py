"""
Image preprocessing module
Handles image loading, conversion, and bit manipulation
"""

import cv2
import numpy as np
from pathlib import Path

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


def load_img_from_bytes(data: bytes, filename: str = "") -> np.ndarray:
    """
    Decode an image from raw bytes (e.g. from an UploadFile).
 
    Args:
        data: Raw image bytes
        filename: Original filename, used for extension validation
 
    Returns:
        NumPy array of shape (H, W, 3), dtype uint8, in BGR format
 
    Raises:
        ValueError: If file type is unsupported or decoding fails
    """

    if not data:
        raise ValueError("Failed to decode image. File may be corrupt or unsupported.")

    if filename:
        allowed_ext = {".png", ".bmp", ".tiff", ".tif"}
        ext = Path(filename).suffix.lower()
        if ext not in allowed_ext:
            raise ValueError(f"Unsupported file type: {ext}. Allowed: png, bmp, tiff")
 
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
 
    if img is None:
        raise ValueError("Failed to decode image. File may be corrupt or unsupported.")
 
    return img
 
 
def encode_img_to_bytes(img: np.ndarray, ext: str) -> bytes:
    """
    Encode a NumPy image array to bytes in the specified format.
 
    Args:
        img: NumPy array of shape (H, W, 3), dtype uint8
        ext: File extension including dot (e.g. ".png", ".bmp", ".tiff")
 
    Returns:
        Encoded image bytes
 
    Raises:
        ValueError: If the extension is unsupported or encoding fails
    """
    ext = ext.lower()
    allowed_ext = {".png", ".bmp", ".tiff", ".tif"}
 
    if ext not in allowed_ext:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: png, bmp, tiff")
 
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"Failed to encode image as {ext}")
    return buffer.tobytes()
 