"""
Local Binary Pattern (LBP) module
Handles texture analysis and pixel classification
"""

from typing import List, Tuple, Literal
import numpy as np


def get_neighbors(image: np.ndarray, x: int, y: int) -> List[Tuple[int, int]]:
    """
    Get valid neighbor coordinates for a pixel.

    Args:
        image: Input image
        x: X coordinate
        y: Y coordinate

    Returns:
        List of valid (y, x) neighbor coordinates

    Note:
        - Interior pixels: 8 neighbors
        - Edge pixels: 5 neighbors
        - Corner pixels: 3 neighbors
    """
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty or None")

    if len(image.shape) < 2:
        raise ValueError("image must have at least 2 dimensions")

    if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
        raise ValueError("coordinates must be integers")

    height, width = image.shape

    if not (0 <= x < width):
        raise ValueError(f"X coordinate must be between 0 and {width - 1}")

    if not (0 <= y < height):
        raise ValueError(f"Y coordinate must be between 0 and {height - 1}")

    neighbors: List[Tuple[int, int]] = []

    directions: List[Tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]

    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            neighbors.append((ny, nx))

    return neighbors


def compare_neighbors(center_value: int, neighbor_values: List[int]) -> List[int]:
    """
    Compare neighbor values with center pixel value.

    Args:
        center_value: 3-MSB value of center pixel
        neighbor_values: List of 3-MSB values of neighbors

    Returns:
        Binary list: 1 if neighbor >= center, 0 otherwise
    """
    if not isinstance(center_value, int):
        raise TypeError("center_value must be an integer")

    if not isinstance(neighbor_values, list):
        raise TypeError("neighbor_values must be a list of integers")

    if any(not isinstance(n, int) for n in neighbor_values):
        raise ValueError("All neighbor values must be integers")

    return [1 if n >= center_value else 0 for n in neighbor_values]


def count_transitions(binary_pattern: List[int]) -> int:
    """
    Count transitions in a circular binary pattern.

    A transition is a change from 0→1 or 1→0.

    Example:
        [1, 1, 0, 0] → 2 transitions (1→0, 0→1 when wrapping)
        [1, 0, 1, 0] → 4 transitions
        [1, 1, 1, 1] → 0 transitions

    Args:
        binary_pattern: Binary sequence (circular)

    Returns:
        Number of transitions (0 to 2*len)
    """
    if not isinstance(binary_pattern, list):
        raise TypeError("binary_pattern must be a list of integers")

    if any(bit not in (0, 1) for bit in binary_pattern):
        raise ValueError("binary_pattern must contain only 0 or 1")

    if len(binary_pattern) == 0:
        return 0

    transitions = 0
    length = len(binary_pattern)

    for i in range(length):
        if binary_pattern[i] != binary_pattern[(i + 1) % length]:
            transitions += 1

    return transitions


def classify_texture(transition_count: int) -> Literal[0, 1]:
    """
    Classify pixel as smooth or rough based on transition count.

    Args:
        transition_count: Number of LBP transitions

    Returns:
        0 for smooth (≤2 transitions), 1 for rough (>2 transitions)
    """
    if not isinstance(transition_count, int):
        raise TypeError("center_value must be an integer")

    return 0 if transition_count <= 2 else 1


def compute_lbp_for_pixel(msb_image: np.ndarray, x: int, y: int) -> int:
    """
    Compute LBP-based texture classification for a single pixel.

    Args:
        msb_image: 2D NumPy array (values 0-7), already reduced to 3-MSB
        x: X coordinate
        y: Y coordinate

    Returns:
        0 for smooth (≤2 transitions)
        1 for rough (>2 transitions)

    Raises:
        TypeError: If msb_image is not a NumPy array
        ValueError: If coordinates are out of bounds
    """
    if not isinstance(msb_image, np.ndarray):
        raise TypeError("msb_image must be a NumPy array")

    if msb_image.ndim != 2:
        raise ValueError("msb_image must be a 2D array")

    height, width = msb_image.shape

    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Pixel coordinates out of bounds")

    # -------------------------
    # Core LBP Logic
    # -------------------------
    center_value: int = int(msb_image[y, x])

    neighbors_coords: List[Tuple[int, int]] = get_neighbors(msb_image, x, y)
    neighbor_values: List[int] = [int(msb_image[ny, nx]) for ny, nx in neighbors_coords]

    binary_pattern: List[int] = compare_neighbors(center_value, neighbor_values)

    transition_count: int = count_transitions(binary_pattern)

    return classify_texture(transition_count)


def compute_lbp_classification(rgb_img: np.ndarray) -> np.ndarray:
    """
    Compute texture classification for an image using 3-MSB LBP
    applied to the green channel.

    Args:
        rgb_img: NumPy array (H, W, 3), dtype uint8

    Returns:
        Classification map: NumPy array (H, W), dtype uint8
        0 → smooth
        1 → rough

    Raises:
        TypeError: If input is not a NumPy array
        ValueError: If input is not 2D uint8 image
    """
    if not isinstance(rgb_img, np.ndarray):
        raise TypeError("rgb_img must be a NumPy array")

    if rgb_img.dtype != np.uint8:
        raise ValueError("rgb_img must be dtype uint8")

    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("rgb_img must have shape (H, W, 3)")

    # Extract green channel (OpenCV uses BGR, green index = 1)
    green_channel = rgb_img[:, :, 1]
    height, width = green_channel.shape

    # -------------------------
    # Use preprocessing module
    # -------------------------
    classification_map = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            classification_map[y, x] = compute_lbp_for_pixel(green_channel, x, y)

    return classification_map
