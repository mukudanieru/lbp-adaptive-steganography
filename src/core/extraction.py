"""
Message extraction module
Handles extracting hidden messages from stego-images
"""


def extract_bits_from_pixel(pixel_rgb, num_bits):
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
    ...


def extract_message(
    stego_image,
    password: str,
    classification_map,
    pixel_coords,
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
    ...


def binary_to_text(binary_string: str) -> str:
    """
    Convert binary string to text.

    Args:
        binary_string: Binary message

    Returns:
        Decoded text message
    """
    ...


def extract_message_length(stego_image, classification_map, pixel_coords) -> int:
    """
    Extract message length from header.

    Args:
        stego_image: RGB stego-image
        classification_map: Texture map
        pixel_coords: Pixel coordinates

    Returns:
        Message length in bits
    """
    ...
