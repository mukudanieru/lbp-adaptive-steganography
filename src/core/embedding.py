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
    password: str,  # retained for consistency
    classification_map: np.ndarray,
    pixel_coords: list[tuple[int, int]],
) -> np.ndarray:
    """
    Embed a secret message into an RGB image using texture-adaptive LSB.

    Embedding strategy:
        - Convert message to binary
        - Prepend 32-bit header (message length in bits)
        - Smooth pixel  (0) → 1 LSB per channel
        - Rough pixel   (1) → 2 LSBs per channel

    Args:
        rgb_img: Cover image (H, W, 3), dtype uint8
        secret_message: Text message to hide
        password: Password (used externally for coordinate generation)
        classification_map: (H, W) array with values {0,1}
        pixel_coords: List of (y, x) coordinates in embedding order

    Returns:
        Stego-image (NumPy array)

    Raises:
        TypeError, ValueError
    """

    # -------------------------
    # Validation
    # -------------------------
    if not isinstance(rgb_img, np.ndarray):
        raise TypeError("rgb_img must be a NumPy array")

    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("rgb_img must have shape (H, W, 3)")

    if rgb_img.dtype != np.uint8:
        raise ValueError("rgb_img must be dtype uint8")

    if not isinstance(secret_message, str):
        raise TypeError("secret_message must be a string")

    if not isinstance(classification_map, np.ndarray):
        raise TypeError("classification_map must be a NumPy array")

    if classification_map.shape != rgb_img.shape[:2]:
        raise ValueError("classification_map must match image dimensions")

    if not isinstance(pixel_coords, list):
        raise TypeError("pixel_coords must be a list of (y, x) tuples")

    # -------------------------
    # Prepare message bits
    # -------------------------
    binary_message = text_to_binary(secret_message)
    header = get_binary_header(binary_message)
    payload = header + binary_message
    total_bits = len(payload)

    # -------------------------
    # Capacity check
    # -------------------------
    capacity = calculate_capacity(classification_map)

    if total_bits > capacity:
        raise ValueError(
            f"Message too large. Required {total_bits} bits, "
            f"capacity is {capacity} bits."
        )

    # -------------------------
    # Embedding process
    # -------------------------
    stego_img = rgb_img.copy()
    bit_pointer = 0

    height, width, _ = stego_img.shape

    for (y, x) in pixel_coords:

        if bit_pointer >= total_bits:
            break

        if not (0 <= y < height and 0 <= x < width):
            raise ValueError("Pixel coordinate out of bounds")

        texture = classification_map[y, x]

        # Determine embedding strength
        bits_per_channel = 1 if texture == 0 else 2
        bits_per_pixel = bits_per_channel * 3

        # Extract chunk for this pixel
        chunk = payload[bit_pointer: bit_pointer + bits_per_pixel]

        # Embed into pixel
        modified_pixel = embed_bits_in_pixel(
            stego_img[y, x],
            chunk,
            bits_per_channel
        )

        stego_img[y, x] = modified_pixel

        # Move pointer by actual bits embedded
        bit_pointer += len(chunk)

    return stego_img
