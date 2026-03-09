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
# Tests for compare_neighbors
# -----------------------------
def test_compare_neighbors():
    test_cases = [
        # Interior pixel (8 neighbors)
        {
            "center": 4,
            "neighbors": [5, 3, 4, 6, 2, 4, 3, 5],
            "expected": [1, 0, 1, 1, 0, 1, 0, 1],
            "description": "interior",
        },
        # Edge pixel (5 neighbors)
        {
            "center": 4,
            "neighbors": [5, 4, 3, 6, 4],
            "expected": [1, 1, 0, 1, 1],
            "description": "edge",
        },
        # Corner pixel (3 neighbors)
        {
            "center": 4,
            "neighbors": [5, 3, 4],
            "expected": [1, 0, 1],
            "description": "corner",
        },
    ]

    for case in test_cases:
        result = compare_neighbors(case["center"], case["neighbors"])
        assert result == case["expected"], f"Failed for {case['description']} pixel"


# -----------------------------
# Tests for count_transitions
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
# Tests for Error Handling
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
# Tests for classify_texture
# -----------------------------
def test_classify_texture_smooth():
    assert classify_texture(0) == 0
    assert classify_texture(2) == 0


def test_classify_texture_rough():
    assert classify_texture(3) == 1
    assert classify_texture(6) == 1


# -----------------------------
# Tests for compute_lbp_for_pixel
# -----------------------------
def test_compute_lbp_uniform_neighborhood():
    """Test LBP classification for uniform neighborhood (all same values)."""
    msb_image = np.full((10, 10), 5, dtype=np.uint8)  # All pixels = 5

    # Interior pixel with all identical neighbors
    result = compute_lbp_for_pixel(msb_image, 5, 5)

    # All neighbors same = 0 transitions = smooth
    assert result == 0


def test_compute_lbp_smooth_gradient():
    """Test LBP classification for smooth gradient (≤2 transitions)."""
    msb_image = np.array([[4, 4, 4], [4, 5, 5], [5, 5, 5]], dtype=np.uint8)

    # Center pixel (1, 1) has smooth gradient around it
    result = compute_lbp_for_pixel(msb_image, 1, 1)

    # Should have ≤2 transitions = smooth
    assert result == 0


def test_compute_lbp_rough_texture():
    """Test LBP classification for rough/textured area (>2 transitions)."""
    msb_image = np.array([[1, 7, 2], [6, 4, 3], [2, 7, 1]], dtype=np.uint8)

    # Center pixel (1, 1) has high variation around it
    result = compute_lbp_for_pixel(msb_image, 1, 1)

    # Should have >2 transitions = rough
    assert result == 1


def test_compute_lbp_checkerboard_pattern():
    """Test LBP with alternating checkerboard pattern (many transitions)."""
    msb_image = np.array([[0, 7, 0], [7, 4, 7], [0, 7, 0]], dtype=np.uint8)

    # Center pixel has alternating neighbors = many transitions
    result = compute_lbp_for_pixel(msb_image, 1, 1)

    # Checkerboard = many transitions = rough
    assert result == 1


def test_compute_lbp_corner_pixel_top_left():
    """Test LBP for corner pixel (top-left, only 3 neighbors)."""
    msb_image = np.full((5, 5), 4, dtype=np.uint8)
    msb_image[0, 0] = 3  # Different center value

    # Corner pixel (0, 0) has only 3 neighbors
    result = compute_lbp_for_pixel(msb_image, 0, 0)

    # Should still compute valid classification
    assert result in [0, 1]


def test_compute_lbp_corner_pixel_bottom_right():
    """Test LBP for corner pixel (bottom-right, only 3 neighbors)."""
    msb_image = np.full((5, 5), 4, dtype=np.uint8)

    # Corner pixel (4, 4)
    result = compute_lbp_for_pixel(msb_image, 4, 4)

    # Uniform neighbors = smooth
    assert result == 0


def test_compute_lbp_edge_pixel_top():
    """Test LBP for edge pixel (top edge, 5 neighbors)."""
    msb_image = np.array(
        [[3, 4, 5, 6, 7], [3, 4, 5, 6, 7], [3, 4, 5, 6, 7]], dtype=np.uint8
    )

    # Top edge pixel (2, 0) has 5 neighbors
    result = compute_lbp_for_pixel(msb_image, 2, 0)

    # Should compute valid classification
    assert result in [0, 1]


def test_compute_lbp_edge_pixel_left():
    """Test LBP for edge pixel (left edge, 5 neighbors)."""
    msb_image = np.full((5, 5), 5, dtype=np.uint8)

    # Left edge pixel (0, 2)
    result = compute_lbp_for_pixel(msb_image, 0, 2)

    # Uniform = smooth
    assert result == 0


def test_compute_lbp_interior_pixel():
    """Test LBP for interior pixel (8 neighbors)."""
    msb_image = np.random.randint(0, 8, (10, 10), dtype=np.uint8)

    # Interior pixel (5, 5) has all 8 neighbors
    result = compute_lbp_for_pixel(msb_image, 5, 5)

    # Should return valid classification
    assert result in [0, 1]


def test_compute_lbp_exactly_two_transitions():
    """Test LBP with exactly 2 transitions (boundary case for smooth)."""
    # Create pattern with exactly 2 transitions
    msb_image = np.array([[3, 3, 3], [3, 4, 5], [5, 5, 5]], dtype=np.uint8)

    # Should classify as smooth (≤2 transitions)
    result = compute_lbp_for_pixel(msb_image, 1, 1)

    assert result == 0


def test_compute_lbp_five_transitions():
    """Test LBP with exactly 5 transitions (boundary case for rough)."""
    # Create pattern with exactly 5 transitions
    msb_image = np.array([[5, 2, 5], [2, 4, 5], [2, 5, 2]], dtype=np.uint8)

    # Should classify as rough (>2 transitions)
    result = compute_lbp_for_pixel(msb_image, 1, 1)

    assert result == 1


def test_compute_lbp_all_msb_values():
    """Test LBP works with all valid MSB values (0-7)."""
    msb_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0]], dtype=np.uint8)

    # Should handle all MSB values 0-7
    result = compute_lbp_for_pixel(msb_image, 1, 1)

    assert result in [0, 1]


def test_compute_lbp_returns_binary():
    """Test that LBP always returns 0 or 1."""
    msb_image = np.random.randint(0, 8, (20, 20), dtype=np.uint8)

    # Test multiple random positions
    for y in range(1, 19):
        for x in range(1, 19):
            result = compute_lbp_for_pixel(msb_image, x, y)
            assert result in [0, 1], f"Invalid result {result} at ({x}, {y})"


def test_compute_lbp_consistent_results():
    """Test that same input gives same output (deterministic)."""
    msb_image = np.array([[3, 5, 2], [7, 4, 1], [6, 2, 5]], dtype=np.uint8)

    # Multiple calls should give same result
    result1 = compute_lbp_for_pixel(msb_image, 1, 1)
    result2 = compute_lbp_for_pixel(msb_image, 1, 1)
    result3 = compute_lbp_for_pixel(msb_image, 1, 1)

    assert result1 == result2 == result3


def test_compute_lbp_single_pixel_image():
    """Test LBP for 1x1 image (edge case, 0 neighbors)."""
    msb_image = np.array([[4]], dtype=np.uint8)

    # Single pixel has no neighbors
    result = compute_lbp_for_pixel(msb_image, 0, 0)

    # Should handle gracefully (0 transitions = smooth)
    assert result == 0


# -----------------------------
# Tests for compute_lbp_classification
# -----------------------------
def test_compute_lbp_classification_uniform_image():
    """Test LBP classification on completely uniform image."""
    grayscale_image = np.full((512, 512), 128, dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    # All pixels should be smooth (0 transitions everywhere)
    assert classification_map.shape == (512, 512)
    assert classification_map.dtype == np.uint8
    assert np.all(classification_map == 0)


def test_compute_lbp_classification_checkerboard():
    """Test LBP classification on checkerboard pattern."""
    grayscale_image = np.zeros((512, 512), dtype=np.uint8)
    grayscale_image[::2, ::2] = 255  # White squares
    grayscale_image[1::2, 1::2] = 255  # Alternating pattern

    classification_map = compute_lbp_classification(grayscale_image)

    # Should have SOME rough pixels (not necessarily majority with 3-MSB)
    assert classification_map.shape == (512, 512)
    assert classification_map.dtype == np.uint8
    num_rough = np.sum(classification_map == 1)
    num_smooth = np.sum(classification_map == 0)
    # Both types should exist
    assert num_rough > 0
    assert num_smooth > 0


def test_compute_lbp_classification_smooth_gradient():
    """Test LBP classification on smooth gradient."""
    # Create horizontal gradient
    gradient = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))

    classification_map = compute_lbp_classification(gradient)

    # Smooth gradient should have mostly smooth pixels
    assert classification_map.shape == (512, 512)
    assert classification_map.dtype == np.uint8
    smooth_ratio = np.sum(classification_map == 0) / classification_map.size
    assert smooth_ratio > 0.5  # Majority should be smooth


def test_compute_lbp_classification_output_shape():
    """Test that output shape matches input shape."""
    grayscale_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    assert classification_map.shape == grayscale_image.shape


def test_compute_lbp_classification_output_dtype():
    """Test that output dtype is uint8."""
    grayscale_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    assert classification_map.dtype == np.uint8


def test_compute_lbp_classification_binary_values():
    """Test that all output values are 0 or 1."""
    grayscale_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    unique_values = np.unique(classification_map)
    assert len(unique_values) <= 2
    assert all(val in [0, 1] for val in unique_values)


def test_compute_lbp_classification_small_image():
    """Test LBP classification on smaller image."""
    grayscale_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    assert classification_map.shape == (50, 50)
    assert classification_map.dtype == np.uint8
    assert np.all((classification_map == 0) | (classification_map == 1))


def test_compute_lbp_classification_horizontal_stripes():
    """Test LBP on horizontal stripe pattern."""
    grayscale_image = np.zeros((512, 512), dtype=np.uint8)
    grayscale_image[::10, :] = 255  # Horizontal stripes every 10 rows

    classification_map = compute_lbp_classification(grayscale_image)

    # Should have mix of smooth and rough
    assert classification_map.shape == (512, 512)
    num_smooth = np.sum(classification_map == 0)
    num_rough = np.sum(classification_map == 1)
    assert num_smooth > 0
    assert num_rough > 0


def test_compute_lbp_classification_vertical_stripes():
    """Test LBP on vertical stripe pattern."""
    grayscale_image = np.zeros((512, 512), dtype=np.uint8)
    grayscale_image[:, ::10] = 255  # Vertical stripes every 10 columns

    classification_map = compute_lbp_classification(grayscale_image)

    # Should have mix of smooth and rough
    assert classification_map.shape == (512, 512)
    num_smooth = np.sum(classification_map == 0)
    num_rough = np.sum(classification_map == 1)
    assert num_smooth > 0
    assert num_rough > 0


def test_compute_lbp_classification_random_noise():
    """Test LBP on random noise."""
    np.random.seed(42)  # For reproducibility
    grayscale_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    # Random noise should produce mixed results
    # With 3-MSB, variation is reduced, so we just check both exist
    num_rough = np.sum(classification_map == 1)
    num_smooth = np.sum(classification_map == 0)
    assert num_rough > 0
    assert num_smooth > 0


def test_compute_lbp_classification_uniform_vs_noise():
    """Test that uniform image is smoother than noisy image."""
    uniform_image = np.full((512, 512), 128, dtype=np.uint8)
    noisy_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    uniform_map = compute_lbp_classification(uniform_image)
    noisy_map = compute_lbp_classification(noisy_image)

    # Uniform should have more smooth pixels than noisy
    uniform_smooth = np.sum(uniform_map == 0)
    noisy_smooth = np.sum(noisy_map == 0)

    assert uniform_smooth > noisy_smooth


def test_compute_lbp_classification_edges_handled():
    """Test that edge and corner pixels are properly classified."""
    grayscale_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    # Check corners are classified
    assert classification_map[0, 0] in [0, 1]
    assert classification_map[0, 511] in [0, 1]
    assert classification_map[511, 0] in [0, 1]
    assert classification_map[511, 511] in [0, 1]

    # Check edges are classified
    assert np.all((classification_map[0, :] == 0) | (classification_map[0, :] == 1))
    assert np.all((classification_map[:, 0] == 0) | (classification_map[:, 0] == 1))
    assert np.all((classification_map[511, :] == 0) | (classification_map[511, :] == 1))
    assert np.all((classification_map[:, 511] == 0) | (classification_map[:, 511] == 1))


def test_compute_lbp_classification_deterministic():
    """Test that same input produces same output (deterministic)."""
    np.random.seed(123)
    grayscale_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    classification_map1 = compute_lbp_classification(grayscale_image)
    classification_map2 = compute_lbp_classification(grayscale_image)

    np.testing.assert_array_equal(classification_map1, classification_map2)


def test_compute_lbp_classification_all_black():
    """Test LBP on all-black image."""
    grayscale_image = np.zeros((512, 512), dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    # All black = uniform = all smooth
    assert np.all(classification_map == 0)


def test_compute_lbp_classification_all_white():
    """Test LBP on all-white image."""
    grayscale_image = np.full((512, 512), 255, dtype=np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    # All white = uniform = all smooth
    assert np.all(classification_map == 0)


def test_compute_lbp_classification_boundary_values():
    """Test LBP with boundary pixel values (0 and 255)."""
    np.random.seed(456)
    grayscale_image = np.random.choice([0, 255], size=(512, 512)).astype(np.uint8)

    classification_map = compute_lbp_classification(grayscale_image)

    # Should complete without error
    assert classification_map.shape == (512, 512)
    assert classification_map.dtype == np.uint8
    assert np.all((classification_map == 0) | (classification_map == 1))
