"""
Pseudorandom number generation module
Handles password-based seed generation and pixel selection
"""

import hashlib
import random


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


def generate_pixel_coordinates(
    height: int, width: int, seed: int
) -> list[tuple[int, int]]:
    """
    Generate pseudorandom sequence of pixel coordinates.
    Coordinates are returned in (y, x) order (row, column).

    Args:
        height: Image height (number of rows). Must be >= 0.
        width: Image width (number of columns). Must be >= 0.
        seed: Seed for the PRNG to ensure reproducibility.

    Returns:
        A shuffled list of (y, x) pixel coordinates.

    Raises:
        TypeError: If inputs are not integers.
        ValueError: If height or width is negative.
    """

    if (
        not isinstance(height, int)
        or not isinstance(width, int)
        or not isinstance(seed, int)
    ):
        raise TypeError("height, width, and seed must all be integers")

    if height < 0 or width < 0:
        raise ValueError("height and width must be non-negative")

    rng = random.Random(seed)
    coords: list[tuple[int, int]] = [
        (y, x) for y in range(height) for x in range(width)
    ]

    rng.shuffle(coords)
    return coords
