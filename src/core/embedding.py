"""
Adaptive LSB steganography embedding module
Handles secret message embedding into cover images
"""

import numpy as np


def text_to_binary(text: str) -> str:
    """
    Convert text message to binary string.

    Args:
        text: Secret message

    Returns:
        Binary string (e.g., "010010100101...")
    """

    if not isinstance(text, str):
        raise TypeError("")

    return "".join(format(ord(char), "08b") for char in text)


def get_binary_header(binary_message: str) -> str:
    """
    Get a 32-bit header representing the length of the binary message.

    Args:
        binary_message: Message in binary string

    Returns:
        32-bit binary string representing message length
    """
    if not isinstance(binary_message, str):
        raise TypeError("binary_message must be a string")

    message_length = len(binary_message)
    return format(message_length & 0xFFFFFFFF, "032b")


def calculate_capacity(classification_map: np.ndarray, num_channels: int = 3) -> int:
    """
    Calculate embedding capacity based on texture classification.

    Capacity = num_channels * (num_smooth * 1 + num_rough * 2)

    Args:
        classification_map: Linear array of texture classification (0=smooth, 1=rough)
        num_channels: Number of color channels (default 3 for RGB)

    Returns:
        Total capacity in bits

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If classification_map contains values other than 0 or 1.
    """

    if not isinstance(classification_map, np.ndarray):
        raise TypeError("classification_map must be a numpy.ndarray.")

    if not isinstance(num_channels, int):
        raise TypeError("num_channels must be an integer.")

    if num_channels <= 0:
        raise ValueError("num_channels must be a positive integer.")

    num_rough = np.count_nonzero(classification_map == 1)
    num_smooth = np.count_nonzero(classification_map == 0)

    return int(num_channels * (num_smooth * 1 + num_rough * 2))


def embed_bits_in_pixel(rgb_img: np.ndarray, bits: str, num_bits: int) -> np.ndarray:
    """
    Embed bits into a single RGB pixel using LSB substitution.

    Args:
        pixel_rgb: RGB values [R, G, B]
        bits: Binary string to embed
        num_bits: Number of bits to embed per channel (1 or 2)

    Returns:
        Modified RGB values

    Example:
        pixel=[226, 137, 125], bits="101", num_bits=1
        → [227, 136, 125]  (only LSB changed)
    """
    
    pixel = rgb_img.copy()
    bit_index = 0

    for channel in range(3):
        if bit_index >= len(bits):
            break

        bits_to_embed = bits[bit_index:bit_index + num_bits]
        bit_index += num_bits

        if len(bits_to_embed) == 0:
            break

        embed_value = int(bits_to_embed, 2)

        mask = 0xFF << num_bits & 0xFF
        pixel[channel] = (pixel[channel] & mask) | embed_value
    
    return pixel


def embed_message(
    rgb_img: np.ndarray,
    secret_message: str,
    password: str,
    classification_map: np.ndarray,
    pixel_coords: list[tuple[int, int]],
) -> np.ndarray:
    """
    Main embedding function.

    Embeds secret message into cover image using texture-adaptive LSB.

    Args:
        cover_image: RGB cover image
        secret_message: Text to hide
        password: Password for pixel selection
        classification_map: Texture classification map
        pixel_coords: Pseudorandom pixel coordinates

    Returns:
        Stego-image with embedded message

    Raises:
        ValueError: If message exceeds capacity
    """
    ...

