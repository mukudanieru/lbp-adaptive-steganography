"""
Message extraction module
Handles extracting hidden messages from stego-images
"""

from typing import Sequence, Tuple
import numpy as np


def extract_bits_from_pixel(
    pixel_rgb: Sequence[int],
    num_bits: int,
) -> str:
    """
    Extract LSB bits from a single RGB pixel.

    Args:
        pixel_rgb: RGB values [R, G, B]
        num_bits: Number of bits to extract per channel (1 or 2)

    Returns:
        Binary string of extracted bits

    Example:
        pixel=[227, 136, 125], num_bits=1
        → "101" (LSB of each channel)
    """
    if num_bits not in (1, 2):
        raise ValueError("num_bits must be 1 or 2")

    if len(pixel_rgb) != 3:
        raise ValueError("pixel_rgb must contain 3 channels")

    bits = ""

    mask = (1 << num_bits) - 1  # 1 → 0b1, 2 → 0b11

    for channel_value in pixel_rgb:
        extracted_value = int(channel_value) & mask
        bits += format(extracted_value, f"0{num_bits}b")

    return bits


def extract_message_length(
    stego_image: np.ndarray,
    classification_map: np.ndarray,
    pixel_coords: Sequence[Tuple[int, int]],
) -> int:
    """
    Extract message length from header.

    Args:
        stego_image: RGB stego-image
        classification_map: Texture map
        pixel_coords: Pixel coordinates

    Returns:
        Message length in bits
    """

    if not isinstance(stego_image, np.ndarray):
        raise TypeError("stego_image must be a NumPy array")

    if classification_map.shape != stego_image.shape[:2]:
        raise ValueError("classification_map must match image dimensions")

    header_bits: str = ""
    height: int
    width: int
    height, width, _ = stego_image.shape

    for y, x in pixel_coords:

        if not (0 <= y < height and 0 <= x < width):
            raise ValueError("Pixel coordinate out of bounds")

        texture_type: int = int(classification_map[y, x])
        bits_per_channel: int = 1 if texture_type == 0 else 2

        pixel_bits: str = extract_bits_from_pixel(
            stego_image[y, x],
            bits_per_channel,
        )

        header_bits += pixel_bits

        if len(header_bits) >= 32:
            break

    if len(header_bits) < 32:
        raise ValueError("Insufficient data to extract header")

    return int(header_bits[:32], 2)


def binary_to_text(binary_string: str) -> str:
    """
    Convert binary string to text.
    """

    if binary_string == "":
        return ""

    if len(binary_string) % 8 != 0:
        raise ValueError("Binary string length must be multiple of 8")

    chars: list[str] = []

    for i in range(0, len(binary_string), 8):
        byte: str = binary_string[i:i + 8]
        chars.append(chr(int(byte, 2)))

    return "".join(chars)


def extract_message(
    stego_image: np.ndarray,
    password: str,
    classification_map: np.ndarray,
    pixel_coords: Sequence[Tuple[int, int]],
) -> str:
    """
    Main extraction function.

    Extracts hidden message from stego-image.

    Args:
        stego_image: RGB stego-image
        password: Password used during embedding
        classification_map: Texture classification map (recomputed)
        pixel_coords: Pseudorandom pixel coordinates (regenerated)

    Returns:
        Extracted secret message
    """
    if not isinstance(stego_image, np.ndarray):
        raise TypeError("stego_image must be a NumPy array")

    if not isinstance(password, str):
        raise TypeError("password must be a string")

    if classification_map.shape != stego_image.shape[:2]:
        raise ValueError("classification_map must match image dimensions")

    message_length: int = extract_message_length(
        stego_image,
        classification_map,
        pixel_coords,
    )

    total_required_bits: int = 32 + message_length
    extracted_bits: str = ""

    height: int
    width: int
    height, width, _ = stego_image.shape

    for y, x in pixel_coords:

        if not (0 <= y < height and 0 <= x < width):
            raise ValueError("Pixel coordinate out of bounds")

        texture_type: int = int(classification_map[y, x])
        bits_per_channel: int = 1 if texture_type == 0 else 2

        pixel_bits: str = extract_bits_from_pixel(
            stego_image[y, x],
            bits_per_channel,
        )

        extracted_bits += pixel_bits

        if len(extracted_bits) >= total_required_bits:
            break

    if len(extracted_bits) < total_required_bits:
        raise ValueError("Insufficient data to extract full message")

    message_bits: str = extracted_bits[32:32 + message_length]

    return binary_to_text(message_bits)
