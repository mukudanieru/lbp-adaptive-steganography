"""
Unit tests for the Local Binary Pattern (LBP) module.

This test suite validates:

- Neighbor extraction logic (interior, edge, corner cases)
- Binary comparison correctness
- Circular transition counting
- Texture classification threshold behavior
- Pixel-level LBP classification
- Full image classification behavior

Framework:
    pytest

Run with:
    pytest
"""

import numpy as np
import pytest

from src.core.lbp import (
    get_neighbors,
    compare_neighbors,
    count_transitions,
    classify_texture,
    compute_lbp_for_pixel,
    compute_lbp_classification,
)


# -----------------------------
# Tests for get_neighbors
# -----------------------------
def test_get_neighbors_interior():
    """
    Interior pixel should have exactly 8 neighbors.
    """
    image = np.zeros((5, 5), dtype=np.uint8)
    neighbors = get_neighbors(image, x=2, y=2)

    assert len(neighbors) == 8


def test_get_neighbors_corner():
    """
    Corner pixel should have exactly 3 neighbors.
    """
    image = np.zeros((5, 5), dtype=np.uint8)
    neighbors = get_neighbors(image, x=0, y=0)

    assert len(neighbors) == 3


def test_get_neighbors_edge():
    """
    Edge pixel (non-corner) should have exactly 5 neighbors.
    """
    image = np.zeros((5, 5), dtype=np.uint8)
    neighbors = get_neighbors(image, x=2, y=0)

    assert len(neighbors) == 5

# -----------------------------
# compare_neighbors
# -----------------------------


def test_compare_neighbors():
    test_cases = [
        # Interior pixel (8 neighbors)
        {
            "center": 4,
            "neighbors": [5, 3, 4, 6, 2, 4, 3, 5],
            "expected": [1, 0, 1, 1, 0, 1, 0, 1],
            "description": "interior"
        },
        # Edge pixel (5 neighbors)
        {
            "center": 4,
            "neighbors": [5, 4, 3, 6, 4],
            "expected": [1, 1, 0, 1, 1],
            "description": "edge"
        },
        # Corner pixel (3 neighbors)
        {
            "center": 4,
            "neighbors": [5, 3, 4],
            "expected": [1, 0, 1],
            "description": "corner"
        },
    ]

    for case in test_cases:
        result = compare_neighbors(case["center"], case["neighbors"])
        assert result == case["expected"], f"Failed for {case['description']} pixel"


# -----------------------------
# count_transitions
# -----------------------------

def test_count_transitions_interior_uniform():
    """
    Uniform pattern should produce 0 transitions.
    """
    pattern = [1, 1, 1, 1, 1, 1, 1, 1]
    assert count_transitions(pattern) == 0


def test_count_transitions_interior_alternating():
    """
    Interior alternating pattern (8 neighbors)
    should produce 8 transitions.
    """
    pattern = [1, 0, 1, 0, 1, 0, 1, 0]
    assert count_transitions(pattern) == 8


def test_count_transitions_edge_case():
    """
    Edge pixel pattern (5 neighbors).
    Should correctly count circular transitions.
    Example:
        [1, 1, 0, 0, 1]
        Transitions:
        1→1 (0)
        1→0 (1)
        0→0 (0)
        0→1 (1)
        1→1 wrap (0)
        Total = 2
    """
    pattern = [1, 1, 0, 0, 1]
    assert count_transitions(pattern) == 2


def test_count_transitions_corner_case():
    """
    Corner pixel pattern (3 neighbors).
    Example:
        [1, 0, 1]
        Transitions:
        1→0 (1)
        0→1 (1)
        1→1 wrap (0)
        Total = 2
    """
    pattern = [1, 0, 1]
    assert count_transitions(pattern) == 2


def test_count_transitions_empty():
    """
    Empty pattern should return 0 transitions.
    """
    assert count_transitions([]) == 0


# -----------------------------
# Error Handling
# -----------------------------

def test_count_transitions_invalid_type():
    """
    Non-list input should raise TypeError.
    """
    with pytest.raises(TypeError):
        count_transitions("1010")  # type: ignore


def test_count_transitions_invalid_values():
    """
    Pattern containing values other than 0 or 1
    should raise ValueError.
    """
    with pytest.raises(ValueError):
        count_transitions([1, 2, 0, 1])


def test_count_transitions_negative_values():
    """
    Negative or non-binary values should raise ValueError.
    """
    with pytest.raises(ValueError):
        count_transitions([1, -1, 0])


# -----------------------------
# classify_texture
# -----------------------------

def test_classify_texture_smooth():
    assert classify_texture(0) == 0
    assert classify_texture(2) == 0


def test_classify_texture_rough():
    assert classify_texture(3) == 1
    assert classify_texture(6) == 1

# -----------------------------
# compute_lbp_for_pixel
# -----------------------------


def test_compute_lbp_for_pixel_smooth():
    # All pixels same → smooth
    msb_image = np.full((3, 3), 4, dtype=np.uint8)

    result = compute_lbp_for_pixel(msb_image, x=1, y=1)

    assert result == 0  # smooth


def test_compute_lbp_for_pixel_rough():
    # Checkerboard pattern → high transitions
    msb_image = np.array([
        [0, 7, 0],
        [7, 4, 7],
        [0, 7, 0],
    ], dtype=np.uint8)

    result = compute_lbp_for_pixel(msb_image, x=1, y=1)

    assert result == 1  # rough


# -----------------------------
# compute_lbp_classification
# -----------------------------

def test_compute_lbp_classification_shape():
    image = np.zeros((4, 6), dtype=np.uint8)

    result = compute_lbp_classification(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_compute_lbp_classification_uniform_image():
    image = np.full((5, 5), 128, dtype=np.uint8)

    result = compute_lbp_classification(image)

    # Entire image should be smooth
    assert np.all(result == 0)
