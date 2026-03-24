"""
Message extraction module
Handles extracting hidden messages from stego-images
"""

import numpy as np


def extract_bits_from_pixel(
    rgb_pixel: np.ndarray,
    num_bits: int,
) -> str:
    """
    Extract LSB bits from R and B channels only (green channel skipped).

    Args:
        rgb_pixel: NumPy array of shape (3,) representing [R, G, B] values
        num_bits: Number of least significant bits to extract per used channel (1 or 2),
              determined by texture classification (e.g., smooth = 1, rough = 2)

    Returns:
        Binary string of extracted bits

    Raises:
        TypeError: If `rgb_pixel` is not a numpy.ndarray.
        TypeError: If `num_bits` is not an integer.
        ValueError: If `num_bits` is not 1 or 2.
        ValueError: If `rgb_pixel` does not contain exactly 3 values (R, G, B).
        ValueError: If `rgb_pixel` contains values outside the range 0 to 255.
    """
    if not isinstance(rgb_pixel, np.ndarray):
        raise TypeError("rgb_pixel must be a numpy ndarray")

    if not isinstance(num_bits, int):
        raise TypeError("num_bits must be an integer")

    if num_bits not in (1, 2):
        raise ValueError("num_bits must be 1 or 2")

    if len(rgb_pixel) != 3:
        raise ValueError("rgb_pixel must contain exactly 3 values (R, G, B)")

    if any((v < 0 or v > 255) for v in rgb_pixel):
        raise ValueError("RGB values must be in range 0-255")

    bits = ""

    mask = (1 << num_bits) - 1  # 1 → 0b1, 2 → 0b11

    # Extract from R and B channels only (indices 0 and 2)
    for channel_idx in [0, 2]:  # Red and Blue only
        channel_value = rgb_pixel[channel_idx]
        extracted_value = int(channel_value) & mask
        bits += format(extracted_value, f"0{num_bits}b")

    return bits


def extract_message_length(
    stego_image: np.ndarray,
    classification_map: np.ndarray,
    pixel_coords: list[tuple[int, int]],
) -> int:
    """
    Extract the embedded message length from the 32-bit header.

    Args:
        stego_image: NumPy array of shape (H, W, 3), dtype uint8,
                     representing the stego-image.
        classification_map: NumPy array of shape (H, W) with values {0, 1},
                            indicating smooth (0) or rough (1) pixels.
        pixel_coords: List of (y, x) tuples specifying extraction order.

    Returns:
        Message length in bits (integer decoded from 32-bit header).

    Raises:
        TypeError: If `stego_image` is not a NumPy array.
        TypeError: If `classification_map` is not a NumPy array.
        TypeError: If `pixel_coords` is not a list of (y, x) tuples.
        ValueError: If `stego_image` does not have shape (H, W, 3).
        ValueError: If `stego_image` is not of dtype uint8.
        ValueError: If `classification_map` shape does not match image dimensions.
        ValueError: If `classification_map` contains values other than 0 or 1.
        ValueError: If any pixel coordinate is out of bounds.
        ValueError: If insufficient bits are available to extract the 32-bit header.
    """
    if not isinstance(stego_image, np.ndarray):
        raise TypeError("stego_image must be a NumPy array")

    if stego_image.ndim != 3 or stego_image.shape[2] != 3:
        raise ValueError("stego_image must have shape (H, W, 3)")

    if stego_image.dtype != np.uint8:
        raise ValueError("stego_image must be dtype uint8")

    if not isinstance(classification_map, np.ndarray):
        raise TypeError("classification_map must be a NumPy array")

    if classification_map.shape != stego_image.shape[:2]:
        raise ValueError("classification_map must match image dimensions")

    if not np.issubdtype(classification_map.dtype, np.integer):
        raise TypeError("classification_map must contain integers")

    if not np.isin(classification_map, [0, 1]).all():
        raise ValueError("classification_map must contain only 0 or 1")

    if not isinstance(pixel_coords, list):
        raise TypeError("pixel_coords must be a list of (y, x) tuples")

    for coord in pixel_coords:
        if not isinstance(coord, tuple) or len(coord) != 2:
            raise TypeError("pixel_coords must contain (y, x) tuples")

    header_bits: str = ""
    height: int
    width: int
    height, width, _ = stego_image.shape

    for y, x in pixel_coords:
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError("pixel coordinate out of bounds")

        texture_type: int = int(classification_map[y, x])
        bits_per_channel: int = 1 if texture_type == 0 else 2

        # Use R&B only extraction
        pixel_bits: str = extract_bits_from_pixel(
            stego_image[y, x],
            bits_per_channel,
        )

        header_bits += pixel_bits

        if len(header_bits) >= 32:
            break

    if len(header_bits) < 32:
        raise ValueError("insufficient data to extract header")

    return int(header_bits[:32], 2)


def binary_to_text(binary_string: str) -> str:
    """
    Convert a binary string into an ASCII text message.

    The input is expected to be a continuous binary string where every
    8 bits represent a single ASCII character.

    Args:
        binary_string: Binary string (e.g., "010010100101...")

    Returns:
        Decoded ASCII text string

    Raises:
        TypeError: If `binary_string` is not a string.
        ValueError: If `binary_string` contains characters other than '0' or '1'.
        ValueError: If `binary_string` length is not a multiple of 8.
    """
    if not isinstance(binary_string, str):
        raise TypeError("binary_string must be a string")

    if binary_string == "":
        return ""

    if not all(bit in "01" for bit in binary_string):
        raise ValueError("binary_string must contain only '0' and '1'")

    if len(binary_string) % 8 != 0:
        raise ValueError("binary string length must be multiple of 8")

    chars: list[str] = []

    for i in range(0, len(binary_string), 8):
        byte: str = binary_string[i : i + 8]
        chars.append(chr(int(byte, 2)))

    return "".join(chars)


def extract_message(
    stego_image: np.ndarray,
    classification_map: np.ndarray,
    pixel_coords: list[tuple[int, int]],
) -> str:
    """
    Extract a secret message from an RGB stego-image using texture-adaptive LSB extraction.

    Extraction strategy:
        - The first 32 bits represent the message length (header).
        - For each pixel, the texture classification determines extraction strength:
            - Smooth pixel (0) → 1 LSB per used channel
            - Rough pixel  (1) → 2 LSBs per used channel
        - Only the Red and Blue channels are used; Green is ignored.
        - Bits are read in the same order defined by `pixel_coords`.

    Args:
        stego_image: NumPy array of shape (H, W, 3), dtype uint8, representing the stego-image.
        classification_map: NumPy array of shape (H, W) with values {0, 1}, indicating
                            smooth (0) or rough (1) pixels.
        pixel_coords: List of (y, x) tuples specifying extraction order.

    Returns:
        Extracted secret message as a string.

    Raises:
        TypeError: If `stego_image` is not a NumPy array.
        ValueError: If `stego_image` does not have shape (H, W, 3) or dtype uint8.
        TypeError: If `classification_map` is not a NumPy array.
        ValueError: If `classification_map` shape does not match `stego_image`.
        ValueError: If `classification_map` contains values other than 0 or 1.
        TypeError: If `pixel_coords` is not a list of tuples.
        ValueError: If any pixel coordinate is out of image bounds.
        ValueError: If insufficient data is available to extract the full message.
    """
    # -------------------------
    # Validation
    # -------------------------
    if not isinstance(stego_image, np.ndarray):
        raise TypeError("stego_image must be a NumPy array")

    if stego_image.ndim != 3 or stego_image.shape[2] != 3:
        raise ValueError("stego_image must have shape (H, W, 3)")

    if stego_image.dtype != np.uint8:
        raise ValueError("stego_image must be dtype uint8")

    if not isinstance(classification_map, np.ndarray):
        raise TypeError("classification_map must be a NumPy array")

    if classification_map.shape != stego_image.shape[:2]:
        raise ValueError("classification_map must match image dimensions")

    if not np.issubdtype(classification_map.dtype, np.integer):
        raise TypeError("classification_map must contain integers")

    if not np.isin(classification_map, [0, 1]).all():
        raise ValueError("classification_map must contain only 0 or 1")

    if not isinstance(pixel_coords, list):
        raise TypeError("pixel_coords must be a list of (y, x) tuples")

    for coord in pixel_coords:
        if not isinstance(coord, tuple) or len(coord) != 2:
            raise TypeError("pixel_coords must be a list of (y, x) tuples")

    # -------------------------
    # Extract message length (header)
    # -------------------------
    message_length: int = extract_message_length(
        stego_image,
        classification_map,
        pixel_coords,
    )

    total_required_bits: int = 32 + message_length

    # -------------------------
    # Extract message length (header)
    # -------------------------
    extracted_bits: str = ""

    height: int
    width: int
    height, width, _ = stego_image.shape

    for y, x in pixel_coords:
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError("pixel coordinate out of bounds")

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
        raise ValueError("insufficient data to extract full message")

    # -------------------------
    # Extract message length (header)
    # -------------------------
    message_bits: str = extracted_bits[32 : 32 + message_length]

    return binary_to_text(message_bits)
