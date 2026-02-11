"""
Pseudorandom number generation module
Handles password-based seed generation and pixel selection
"""

import hashlib


def password_to_seed(password: str) -> int:
    """
    Generate a deterministic 64-bit integer seed from a password string.
    """

    if not isinstance(password, str):
        raise TypeError("password must be a string")

    if password == "":
        raise ValueError("password cannot be empty")

    sha_256: str = hashlib.sha256(password.encode("utf-8")).hexdigest()
    first_8_bytes_hex = sha_256[:16]

    return int(first_8_bytes_hex, 16)


def generate_pixel_coordinates(height: int, width: int, seed: int):
    """
    Generate pseudorandom sequence of pixel coordinates.

    Uses Mersenne Twister PRNG for reproducible shuffling.

    Args:
        height: Image height
        width: Image width
        seed: Integer seed from password

    Returns:
        List of (y, x) coordinates in pseudorandom order

    Example:
        For 3x3 image:
        [(0,0), (0,1), (0,2), (1,0), ..., (2,2)] → shuffled
    """
    ...
