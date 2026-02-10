"""
Image preprocessing module
Handles image loading, conversion, and bit manipulation
"""


def load_img(file: str):
    """
    Load an image from file.

    Args:
        filepath: Path to the image file

    Returns:
        RGB image as numpy array (height, width, 3)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image is not 512x512
    """
    ...


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
