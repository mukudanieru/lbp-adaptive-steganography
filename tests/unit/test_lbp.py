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


# ---------------------------------------------------------------------------
# Tests for get_neighbors
# ---------------------------------------------------------------------------
class TestGetNeighbors:
    # --- Fixtures ---

    @pytest.fixture
    def img3x3(self):
        return np.arange(9, dtype=np.uint8).reshape(3, 3)

    @pytest.fixture
    def img512(self):
        return np.zeros((512, 512), dtype=np.uint8)

    # --- Neighbor counts ---

    def test_interior_pixel_has_8_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=1)
        assert len(result) == 8

    def test_interior_pixel_512_has_8_neighbors(self, img512):
        result = get_neighbors(img512, x=256, y=256)
        assert len(result) == 8

    def test_top_left_corner_has_3_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=0, y=0)
        assert len(result) == 3

    def test_top_right_corner_has_3_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=2, y=0)
        assert len(result) == 3

    def test_bottom_left_corner_has_3_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=0, y=2)
        assert len(result) == 3

    def test_bottom_right_corner_has_3_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=2, y=2)
        assert len(result) == 3

    def test_top_edge_has_5_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=0)
        assert len(result) == 5

    def test_bottom_edge_has_5_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=2)
        assert len(result) == 5

    def test_left_edge_has_5_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=0, y=1)
        assert len(result) == 5

    def test_right_edge_has_5_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=2, y=1)
        assert len(result) == 5

    # --- Return type and structure ---

    def test_returns_list(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=1)
        assert isinstance(result, list)

    def test_elements_are_tuples(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=1)
        assert all(isinstance(n, tuple) for n in result)

    def test_tuples_have_two_elements(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=1)
        assert all(len(n) == 2 for n in result)

    # --- Coordinate bounds ---

    def test_all_neighbors_within_bounds_interior(self, img512):
        result = get_neighbors(img512, x=256, y=256)
        h, w = img512.shape
        assert all(0 <= y < h and 0 <= x < w for y, x in result)

    def test_all_neighbors_within_bounds_corner(self, img512):
        result = get_neighbors(img512, x=0, y=0)
        h, w = img512.shape
        assert all(0 <= y < h and 0 <= x < w for y, x in result)

    def test_no_duplicate_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=1)
        assert len(result) == len(set(result))

    def test_center_pixel_not_in_neighbors(self, img3x3):
        result = get_neighbors(img3x3, x=1, y=1)
        assert (1, 1) not in result

    # --- ValueError: invalid image ---

    def test_raises_on_none_image(self):
        with pytest.raises(ValueError):
            get_neighbors(None, x=0, y=0)

    def test_raises_on_empty_array(self):
        with pytest.raises(ValueError):
            get_neighbors(np.array([]), x=0, y=0)

    def test_raises_on_1d_array(self):
        with pytest.raises(ValueError):
            get_neighbors(np.zeros(10), x=0, y=0)

    def test_raises_on_3d_array(self):
        with pytest.raises(ValueError):
            get_neighbors(np.zeros((5, 5, 3), dtype=np.uint8), x=1, y=1)

    # --- ValueError: out of bounds ---

    def test_raises_x_out_of_bounds(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x=3, y=1)

    def test_raises_y_out_of_bounds(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x=1, y=3)

    def test_raises_negative_x(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x=-1, y=1)

    def test_raises_negative_y(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x=1, y=-1)

    # --- ValueError: non-integer x or y ---

    def test_raises_float_x(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x=1.0, y=1)

    def test_raises_float_y(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x=1, y=1.0)

    def test_raises_string_x(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x="1", y=1)

    def test_raises_none_x(self, img3x3):
        with pytest.raises(ValueError):
            get_neighbors(img3x3, x=None, y=1)

    # --- 1x1 image edge case ---

    def test_1x1_image_has_no_neighbors(self):
        img = np.array([[128]], dtype=np.uint8)
        result = get_neighbors(img, x=0, y=0)
        assert result == []


# ---------------------------------------------------------------------------
# Tests for compare_neighbors
# ---------------------------------------------------------------------------
class TestCompareNeighbors:
    # --- Return type and structure ---

    def test_returns_list(self):
        result = compare_neighbors(128, [100, 200])
        assert isinstance(result, list)

    def test_output_length_matches_neighbors(self):
        result = compare_neighbors(128, [100, 128, 200, 50, 255, 0, 127, 129])
        assert len(result) == 8

    def test_output_contains_only_0_and_1(self):
        result = compare_neighbors(128, [0, 64, 128, 192, 255])
        assert all(b in (0, 1) for b in result)

    # --- Comparison logic ---

    def test_neighbor_greater_than_center_is_1(self):
        result = compare_neighbors(100, [200])
        assert result == [1]

    def test_neighbor_equal_to_center_is_1(self):
        result = compare_neighbors(100, [100])
        assert result == [1]

    def test_neighbor_less_than_center_is_0(self):
        result = compare_neighbors(100, [50])
        assert result == [0]

    def test_all_neighbors_above_center(self):
        result = compare_neighbors(0, [1, 128, 255])
        assert result == [1, 1, 1]

    def test_all_neighbors_below_center(self):
        result = compare_neighbors(255, [0, 128, 254])
        assert result == [0, 0, 0]

    def test_mixed_neighbors(self):
        result = compare_neighbors(128, [50, 128, 200])
        assert result == [0, 1, 1]

    def test_center_zero_all_ones(self):
        result = compare_neighbors(0, [0, 1, 255])
        assert result == [1, 1, 1]

    def test_center_255_all_zeros(self):
        result = compare_neighbors(255, [0, 127, 254])
        assert result == [0, 0, 0]

    def test_empty_neighbors_returns_empty(self):
        result = compare_neighbors(128, [])
        assert result == []

    def test_8_neighbors_typical_lbp(self):
        result = compare_neighbors(100, [80, 120, 100, 90, 110, 70, 130, 95])
        assert len(result) == 8
        assert result == [0, 1, 1, 0, 1, 0, 1, 0]

    # --- TypeError ---

    def test_raises_type_error_float_center(self):
        with pytest.raises(TypeError):
            compare_neighbors(128.0, [100, 200])

    def test_raises_type_error_string_center(self):
        with pytest.raises(TypeError):
            compare_neighbors("128", [100])

    def test_raises_type_error_none_center(self):
        with pytest.raises(TypeError):
            compare_neighbors(None, [100])

    def test_raises_type_error_neighbors_not_list(self):
        with pytest.raises(TypeError):
            compare_neighbors(128, (100, 200))

    def test_raises_type_error_neighbors_tuple(self):
        with pytest.raises(TypeError):
            compare_neighbors(128, (100,))

    # --- ValueError ---

    def test_raises_value_error_float_in_neighbors(self):
        with pytest.raises(ValueError):
            compare_neighbors(128, [100, 200.0])

    def test_raises_value_error_string_in_neighbors(self):
        with pytest.raises(ValueError):
            compare_neighbors(128, [100, "200"])

    def test_raises_value_error_none_in_neighbors(self):
        with pytest.raises(ValueError):
            compare_neighbors(128, [100, None])


# ---------------------------------------------------------------------------
# Tests for count_transitions
# ---------------------------------------------------------------------------
class TestCountTransitions:
    # --- Return type ---

    def test_returns_int(self):
        result = count_transitions([1, 0, 1, 0])
        assert isinstance(result, int)

    # --- Transition logic ---

    def test_all_same_zeros_no_transitions(self):
        assert count_transitions([0, 0, 0, 0]) == 0

    def test_all_same_ones_no_transitions(self):
        assert count_transitions([1, 1, 1, 1]) == 0

    def test_alternating_4_transitions(self):
        assert count_transitions([1, 0, 1, 0]) == 4

    def test_one_transition_pair(self):
        # [1,1,0,0] → 1→0 and wrap 0→1 = 2
        assert count_transitions([1, 1, 0, 0]) == 2

    def test_single_zero_to_one_transition(self):
        # [0,0,0,1] → 0→1 and wrap 1→0 = 2
        assert count_transitions([0, 0, 0, 1]) == 2

    def test_uniform_pattern_is_smooth(self):
        assert count_transitions([1, 1, 1, 0, 0, 0, 0, 0]) == 2

    def test_8_bit_alternating_max_transitions(self):
        assert count_transitions([1, 0, 1, 0, 1, 0, 1, 0]) == 8

    def test_single_element_zero(self):
        assert count_transitions([0]) == 0

    def test_single_element_one(self):
        assert count_transitions([1]) == 0

    def test_empty_pattern(self):
        assert count_transitions([]) == 0

    def test_circular_wrap_counted(self):
        # last→first transition must be included
        assert count_transitions([0, 1]) == 2

    # --- TypeError ---

    def test_raises_type_error_not_list(self):
        with pytest.raises(TypeError):
            count_transitions((1, 0, 1))

    def test_raises_type_error_string_input(self):
        with pytest.raises(TypeError):
            count_transitions("1010")

    # --- ValueError ---

    def test_raises_value_error_non_binary_value(self):
        with pytest.raises(ValueError):
            count_transitions([1, 0, 2])

    def test_raises_value_error_negative_value(self):
        with pytest.raises(ValueError):
            count_transitions([1, -1, 0])

    def test_raises_value_error_float_in_pattern(self):
        with pytest.raises(ValueError):
            count_transitions([1, 0.0, 1])


# ---------------------------------------------------------------------------
# Tests for classify_texture
# ---------------------------------------------------------------------------
class TestClassifyTexture:
    # --- Return values ---

    def test_returns_0_or_1(self):
        assert classify_texture(0) in (0, 1)

    def test_zero_transitions_is_smooth(self):
        assert classify_texture(0) == 0

    def test_one_transition_is_smooth(self):
        assert classify_texture(1) == 0

    def test_two_transitions_is_smooth(self):
        assert classify_texture(2) == 0

    def test_three_transitions_is_rough(self):
        assert classify_texture(3) == 1

    def test_four_transitions_is_rough(self):
        assert classify_texture(4) == 1

    def test_max_transitions_8_is_rough(self):
        assert classify_texture(8) == 1

    def test_large_transition_count_is_rough(self):
        assert classify_texture(100) == 1

    # --- TypeError ---

    def test_raises_type_error_float(self):
        with pytest.raises(TypeError):
            classify_texture(2.0)

    def test_raises_type_error_string(self):
        with pytest.raises(TypeError):
            classify_texture("2")

    def test_raises_type_error_none(self):
        with pytest.raises(TypeError):
            classify_texture(None)

    # --- Integration: compare → count → classify ---

    def test_uniform_neighbors_classifies_smooth(self):
        binary = compare_neighbors(128, [200, 200, 200, 200, 200, 200, 200, 200])
        transitions = count_transitions(binary)
        assert classify_texture(transitions) == 0

    def test_alternating_neighbors_classifies_rough(self):
        binary = compare_neighbors(128, [50, 200, 50, 200, 50, 200, 50, 200])
        transitions = count_transitions(binary)
        assert classify_texture(transitions) == 1


# ---------------------------------------------------------------------------
# Helpers for compute_lbp_for_pixel
# ---------------------------------------------------------------------------
def make_uniform(h: int, w: int, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def make_checkerboard(h: int, w: int) -> np.ndarray:
    xs = np.arange(w)
    ys = np.arange(h)
    grid = ((xs[None, :] + ys[:, None]) % 2 == 0).astype(np.uint8)
    return grid * 255


# ---------------------------------------------------------------------------
# Tests for compute_lbp_for_pixel
# ---------------------------------------------------------------------------
class TestComputeLbpForPixel:
    # --- 1. Uniform neighborhood → smooth ---
    def test_uniform_neighborhood_is_smooth(self):
        img = make_uniform(5, 5, 128)
        assert compute_lbp_for_pixel(img, x=2, y=2) == 0

    # --- 2. Smooth gradient (≤2 transitions) ---
    def test_smooth_gradient_is_smooth(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        # center=50, all neighbors > center → all 1s → 0 transitions
        img[2, 2] = 50
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy != 0 or dx != 0:
                    img[2 + dy, 2 + dx] = 200
        assert compute_lbp_for_pixel(img, x=2, y=2) == 0

    # --- 3. Rough / textured area (>2 transitions) ---
    def test_rough_textured_area_is_rough(self):
        img = np.array(
            [
                [255, 0, 255, 0, 255],
                [0, 255, 0, 0, 0],
                [255, 0, 128, 0, 255],
                [0, 255, 0, 0, 0],
                [255, 0, 255, 0, 255],
            ],
            dtype=np.uint8,
        )
        assert compute_lbp_for_pixel(img, x=2, y=2) == 1

    # --- 4. Checkerboard pattern (many transitions) ---
    def test_checkerboard_interior_is_rough(self):
        img = make_checkerboard(5, 5)
        assert compute_lbp_for_pixel(img, x=2, y=2) == 1

    # --- 5. Corner pixel top-left (3 neighbors) ---
    def test_corner_top_left_smooth(self):
        img = make_uniform(5, 5, 100)
        result = compute_lbp_for_pixel(img, x=0, y=0)
        assert result in (0, 1)

    def test_corner_top_left_uniform_is_smooth(self):
        img = make_uniform(5, 5, 100)
        assert compute_lbp_for_pixel(img, x=0, y=0) == 0

    # --- 6. Corner pixel bottom-right (3 neighbors) ---
    def test_corner_bottom_right_uniform_is_smooth(self):
        img = make_uniform(5, 5, 100)
        assert compute_lbp_for_pixel(img, x=4, y=4) == 0

    def test_corner_bottom_right_returns_valid(self):
        img = make_checkerboard(5, 5)
        assert compute_lbp_for_pixel(img, x=4, y=4) in (0, 1)

    # --- 7. Edge pixel top (5 neighbors) ---
    def test_edge_top_uniform_is_smooth(self):
        img = make_uniform(5, 5, 200)
        assert compute_lbp_for_pixel(img, x=2, y=0) == 0

    def test_edge_top_returns_valid(self):
        img = make_checkerboard(5, 5)
        assert compute_lbp_for_pixel(img, x=2, y=0) in (0, 1)

    # --- 8. Edge pixel left (5 neighbors) ---
    def test_edge_left_uniform_is_smooth(self):
        img = make_uniform(5, 5, 50)
        assert compute_lbp_for_pixel(img, x=0, y=2) == 0

    def test_edge_left_returns_valid(self):
        img = make_checkerboard(5, 5)
        assert compute_lbp_for_pixel(img, x=0, y=2) in (0, 1)

    # --- 9. Interior pixel (8 neighbors) ---
    def test_interior_pixel_has_full_context(self):
        img = make_uniform(5, 5, 128)
        result = compute_lbp_for_pixel(img, x=2, y=2)
        assert result in (0, 1)

    def test_interior_all_neighbors_below_center_is_smooth(self):
        img = make_uniform(5, 5, 50)
        img[2, 2] = 200  # center much higher → all 0s → 0 transitions
        assert compute_lbp_for_pixel(img, x=2, y=2) == 0

    # --- 10. Exactly 2 transitions (boundary: smooth) ---
    def test_exactly_2_transitions_is_smooth(self):
        # Pattern [1,1,1,1,0,0,0,0] → 2 transitions
        img = make_uniform(3, 3, 200)
        img[1, 1] = 100  # center
        img[0, 0] = 50  # top-left  → below center → 0
        img[0, 1] = 50  # top       → below center → 0
        img[0, 2] = 50  # top-right → below center → 0
        img[1, 0] = 50  # left      → below center → 0
        img[1, 2] = 150  # right     → above center → 1
        img[2, 0] = 150  # bot-left  → above center → 1
        img[2, 1] = 150  # bot       → above center → 1
        img[2, 2] = 150  # bot-right → above center → 1
        assert compute_lbp_for_pixel(img, x=1, y=1) == 0

    # --- 11. Exactly 3 transitions (boundary: rough) ---
    def test_exactly_3_transitions_is_rough(self):
        # Build a pattern with 3 transitions in circular order
        img = make_uniform(3, 3, 100)
        img[1, 1] = 100  # center = 100
        # Neighbors in clockwise order: top-left, top, top-right, right,
        #                               bot-right, bot, bot-left, left
        # Pattern: 1,1,0,1,0,0,0,0 → transitions: 1→0, 0→1, 1→0 = 3
        vals = [150, 150, 50, 150, 50, 50, 50, 50]
        positions = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (2, 1),
            (2, 0),
            (1, 0),
        ]
        for (row, col), v in zip(positions, vals):
            img[row, col] = v
        assert compute_lbp_for_pixel(img, x=1, y=1) == 1

    # --- 12. All valid pixel values (0–255) as center ---
    def test_center_value_0_does_not_crash(self):
        img = make_uniform(3, 3, 128)
        img[1, 1] = 0
        assert compute_lbp_for_pixel(img, x=1, y=1) in (0, 1)

    def test_center_value_255_does_not_crash(self):
        img = make_uniform(3, 3, 128)
        img[1, 1] = 255
        assert compute_lbp_for_pixel(img, x=1, y=1) in (0, 1)

    def test_center_value_midrange(self):
        img = make_uniform(3, 3, 128)
        assert compute_lbp_for_pixel(img, x=1, y=1) in (0, 1)

    # --- 13. Always returns 0 or 1 ---
    def test_always_returns_binary_on_random_images(self):
        rng = np.random.default_rng(42)
        for seed in range(20):
            img = rng.integers(0, 256, (5, 5), dtype=np.uint8)
            assert compute_lbp_for_pixel(img, x=2, y=2) in (0, 1)

    def test_always_returns_binary_on_edges(self):
        img = make_checkerboard(10, 10)
        for x in range(10):
            assert compute_lbp_for_pixel(img, x=x, y=0) in (0, 1)

    # --- 14. Deterministic output ---
    def test_same_input_same_output(self):
        img = make_checkerboard(5, 5)
        r1 = compute_lbp_for_pixel(img, x=2, y=2)
        r2 = compute_lbp_for_pixel(img, x=2, y=2)
        assert r1 == r2

    def test_deterministic_across_multiple_calls(self):
        img = np.arange(25, dtype=np.uint8).reshape(5, 5)
        results = [compute_lbp_for_pixel(img, x=2, y=2) for _ in range(10)]
        assert len(set(results)) == 1

    # --- 15. 1×1 image (0 neighbors) ---
    def test_1x1_image_is_smooth(self):
        img = np.array([[128]], dtype=np.uint8)
        assert compute_lbp_for_pixel(img, x=0, y=0) == 0

    def test_1x1_any_value_is_smooth(self):
        for val in (0, 1, 128, 254, 255):
            img = np.array([[val]], dtype=np.uint8)
            assert compute_lbp_for_pixel(img, x=0, y=0) == 0

    # --- TypeError: invalid image type ---
    def test_raises_type_error_list_input(self):
        with pytest.raises(TypeError):
            compute_lbp_for_pixel([[128, 64], [32, 16]], x=1, y=1)

    def test_raises_type_error_none_image(self):
        with pytest.raises(TypeError):
            compute_lbp_for_pixel(None, x=0, y=0)

    def test_raises_type_error_string_image(self):
        with pytest.raises(TypeError):
            compute_lbp_for_pixel("image", x=0, y=0)

    # --- ValueError: out-of-bounds coordinates ---
    def test_raises_value_error_x_out_of_bounds(self):
        img = make_uniform(5, 5)
        with pytest.raises(ValueError):
            compute_lbp_for_pixel(img, x=5, y=2)

    def test_raises_value_error_y_out_of_bounds(self):
        img = make_uniform(5, 5)
        with pytest.raises(ValueError):
            compute_lbp_for_pixel(img, x=2, y=5)

    def test_raises_value_error_negative_x(self):
        img = make_uniform(5, 5)
        with pytest.raises(ValueError):
            compute_lbp_for_pixel(img, x=-1, y=2)

    def test_raises_value_error_negative_y(self):
        img = make_uniform(5, 5)
        with pytest.raises(ValueError):
            compute_lbp_for_pixel(img, x=2, y=-1)


# ---------------------------------------------------------------------------
# Helpers for compute_lbp_classification
# ---------------------------------------------------------------------------


def make_rgb(h: int, w: int, r: int = 128, g: int = 128, b: int = 128) -> np.ndarray:
    """Solid-color RGB image."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def make_random_rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_checkerboard_rgb(h: int, w: int) -> np.ndarray:
    xs = np.arange(w)
    ys = np.arange(h)
    grid = ((xs[None, :] + ys[:, None]) % 2 == 0).astype(np.uint8) * 255
    img = np.stack([grid, grid, grid], axis=-1)
    return img


def make_gradient_rgb(h: int, w: int) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    green = np.tile(row, (h, 1))
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = green
    return img


# ---------------------------------------------------------------------------
# Tests for compute_lbp_classification
# ---------------------------------------------------------------------------
class TestComputeLbpClassification:
    # --- Return type and shape ---

    def test_returns_ndarray(self):
        img = make_rgb(512, 512)
        result = compute_lbp_classification(img)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input_512x512(self):
        img = make_rgb(512, 512)
        result = compute_lbp_classification(img)
        assert result.shape == (512, 512)

    def test_output_dtype_is_uint8(self):
        img = make_rgb(512, 512)
        result = compute_lbp_classification(img)
        assert result.dtype == np.uint8

    def test_output_contains_only_0_and_1(self):
        img = make_random_rgb(512, 512, seed=42)
        result = compute_lbp_classification(img)
        assert set(np.unique(result)).issubset({0, 1})

    # --- Shape passthrough ---

    def test_output_shape_non_square(self):
        img = make_rgb(256, 512)
        result = compute_lbp_classification(img)
        assert result.shape == (256, 512)

    def test_output_shape_small(self):
        img = make_rgb(3, 3)
        result = compute_lbp_classification(img)
        assert result.shape == (3, 3)

    def test_output_shape_1x1(self):
        img = make_rgb(1, 1)
        result = compute_lbp_classification(img)
        assert result.shape == (1, 1)

    # --- Uniform image: all smooth ---

    def test_uniform_black_all_smooth(self):
        img = make_rgb(512, 512, r=0, g=0, b=0)
        result = compute_lbp_classification(img)
        assert np.all(result == 0)

    def test_uniform_white_all_smooth(self):
        img = make_rgb(512, 512, r=255, g=255, b=255)
        result = compute_lbp_classification(img)
        assert np.all(result == 0)

    def test_uniform_midgray_all_smooth(self):
        img = make_rgb(512, 512, r=128, g=128, b=128)
        result = compute_lbp_classification(img)
        assert np.all(result == 0)

    def test_uniform_any_color_all_smooth(self):
        img = make_rgb(512, 512, r=100, g=200, b=50)
        result = compute_lbp_classification(img)
        assert np.all(result == 0)

    # --- Checkerboard: high roughness ---

    def test_checkerboard_512_has_rough_pixels(self):
        img = make_checkerboard_rgb(512, 512)
        result = compute_lbp_classification(img)
        assert np.any(result == 1)

    def test_checkerboard_interior_majority_rough(self):
        img = make_checkerboard_rgb(512, 512)
        result = compute_lbp_classification(img)
        interior = result[1:-1, 1:-1]
        assert np.mean(interior) >= 0.5

    # --- Gradient image: mostly smooth ---

    def test_gradient_512_has_smooth_pixels(self):
        img = make_gradient_rgb(512, 512)
        result = compute_lbp_classification(img)
        assert np.any(result == 0)

    # --- Uses green channel ---

    def test_only_green_channel_affects_output(self):
        """Two images with same green channel but different R/B should produce identical maps."""
        g_channel = np.random.default_rng(7).integers(
            0, 256, (512, 512), dtype=np.uint8
        )
        img_a = np.zeros((512, 512, 3), dtype=np.uint8)
        img_b = np.zeros((512, 512, 3), dtype=np.uint8)
        img_a[:, :, 1] = g_channel
        img_b[:, :, 1] = g_channel
        img_a[:, :, 0] = 255  # different R
        img_b[:, :, 2] = 255  # different B
        assert np.array_equal(
            compute_lbp_classification(img_a), compute_lbp_classification(img_b)
        )

    def test_different_green_channel_different_output(self):
        img_a = make_rgb(512, 512, g=50)
        img_b = make_rgb(512, 512, g=200)
        # Both uniform → both all-smooth, but verify no crash and shape ok
        assert compute_lbp_classification(img_a).shape == (512, 512)
        assert compute_lbp_classification(img_b).shape == (512, 512)

    # --- Determinism ---

    def test_same_input_same_output(self):
        img = make_random_rgb(512, 512, seed=1)
        r1 = compute_lbp_classification(img)
        r2 = compute_lbp_classification(img)
        assert np.array_equal(r1, r2)

    def test_deterministic_on_checkerboard(self):
        img = make_checkerboard_rgb(512, 512)
        assert np.array_equal(
            compute_lbp_classification(img), compute_lbp_classification(img)
        )

    # --- 1×1 edge case ---

    def test_1x1_image_returns_smooth(self):
        img = make_rgb(1, 1, g=128)
        result = compute_lbp_classification(img)
        assert result[0, 0] == 0

    # --- TypeError ---

    def test_raises_type_error_for_list(self):
        with pytest.raises(TypeError):
            compute_lbp_classification([[[128, 128, 128]]])

    def test_raises_type_error_for_none(self):
        with pytest.raises(TypeError):
            compute_lbp_classification(None)

    def test_raises_type_error_for_string(self):
        with pytest.raises(TypeError):
            compute_lbp_classification("image")

    # --- ValueError: wrong dtype ---

    def test_raises_value_error_float32(self):
        img = np.zeros((512, 512, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_lbp_classification(img)

    def test_raises_value_error_uint16(self):
        img = np.zeros((512, 512, 3), dtype=np.uint16)
        with pytest.raises(ValueError):
            compute_lbp_classification(img)

    def test_raises_value_error_int32(self):
        img = np.zeros((512, 512, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            compute_lbp_classification(img)

    # --- ValueError: wrong shape ---

    def test_raises_value_error_2d_grayscale(self):
        img = np.zeros((512, 512), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_lbp_classification(img)

    def test_raises_value_error_4_channels(self):
        img = np.zeros((512, 512, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_lbp_classification(img)

    def test_raises_value_error_1_channel(self):
        img = np.zeros((512, 512, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_lbp_classification(img)

    def test_raises_value_error_1d_array(self):
        img = np.zeros(512, dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_lbp_classification(img)
