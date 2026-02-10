"""
Pseudorandom number generation module
Handles password-based seed generation and pixel selection
"""


def password_to_seed(password: str) -> int:
    """
    Convert password string to deterministic integer seed using SHA-256.

    Process:
        1. Hash password with SHA-256
        2. Take first 8 bytes of hash
        3. Convert to 64-bit unsigned integer

    Args:
        password: User-provided password string

    Returns:
        64-bit integer seed

    Example:
        "mypassword" → 9934964102539539065
    """
    ...


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
