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
