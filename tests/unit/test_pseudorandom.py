from src.core.pseudorandom import password_to_seed, generate_pixel_coordinates
import pytest


# ---------------------------------------------------------------------------
# Tests for password_to_seed
# ---------------------------------------------------------------------------
class TestPasswordToSeed:
    # --- Return type and range ---

    def test_returns_integer(self):
        result = password_to_seed("hello")
        assert isinstance(result, int)

    def test_returns_64bit_range(self):
        result = password_to_seed("hello")
        assert 0 <= result <= 0xFFFFFFFFFFFFFFFF

    # --- Determinism ---

    def test_same_password_same_seed(self):
        assert password_to_seed("password123") == password_to_seed("password123")

    def test_same_password_repeated_calls(self):
        pw = "steganography"
        seeds = [password_to_seed(pw) for _ in range(5)]
        assert len(set(seeds)) == 1

    # --- Different passwords produce different seeds ---

    def test_different_passwords_different_seeds(self):
        assert password_to_seed("abc") != password_to_seed("def")

    def test_case_sensitive(self):
        assert password_to_seed("Secret") != password_to_seed("secret")

    def test_trailing_space_differs(self):
        assert password_to_seed("pass") != password_to_seed("pass ")

    def test_leading_space_differs(self):
        assert password_to_seed("pass") != password_to_seed(" pass")

    # --- Edge cases: special inputs ---

    def test_single_character(self):
        result = password_to_seed("a")
        assert isinstance(result, int)

    def test_single_space(self):
        result = password_to_seed(" ")
        assert isinstance(result, int)

    def test_numeric_string(self):
        result = password_to_seed("1234567890")
        assert isinstance(result, int)

    def test_special_characters(self):
        result = password_to_seed("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert isinstance(result, int)

    def test_unicode_password(self):
        result = password_to_seed("пароль")
        assert isinstance(result, int)

    def test_emoji_password(self):
        result = password_to_seed("🔑🔒")
        assert isinstance(result, int)

    def test_long_password(self):
        result = password_to_seed("a" * 10_000)
        assert isinstance(result, int)

    def test_whitespace_only(self):
        result = password_to_seed("   ")
        assert isinstance(result, int)

    # --- Avalanche: small change causes large seed difference ---

    def test_one_char_difference_changes_seed(self):
        s1 = password_to_seed("password1")
        s2 = password_to_seed("password2")
        assert s1 != s2

    def test_reversed_string_differs(self):
        assert password_to_seed("abcdef") != password_to_seed("fedcba")

    # --- Error / edge cases ---

    def test_raises_type_error_non_string(self):
        with pytest.raises(TypeError, match="password must be a string"):
            password_to_seed(123)

    def test_raises_value_error_empty_string(self):
        with pytest.raises(ValueError, match="password cannot be empty"):
            password_to_seed("")


# ---------------------------------------------------------------------------
# Tests for generate_pixel_coordinates
# ---------------------------------------------------------------------------
class TestGeneratePixelCoordinates:
    # --- Return type and structure ---

    def test_returns_list(self):
        result = generate_pixel_coordinates(4, 4, seed=0)
        assert isinstance(result, list)

    def test_elements_are_tuples(self):
        result = generate_pixel_coordinates(4, 4, seed=0)
        assert all(isinstance(coord, tuple) for coord in result)

    def test_tuples_have_two_elements(self):
        result = generate_pixel_coordinates(4, 4, seed=0)
        assert all(len(coord) == 2 for coord in result)

    # --- Correct total count ---

    def test_count_512x512(self):
        result = generate_pixel_coordinates(512, 512, seed=42)
        assert len(result) == 512 * 512

    def test_count_4x4(self):
        result = generate_pixel_coordinates(4, 4, seed=0)
        assert len(result) == 16

    def test_count_1x1(self):
        result = generate_pixel_coordinates(1, 1, seed=0)
        assert len(result) == 1

    def test_count_non_square(self):
        result = generate_pixel_coordinates(256, 512, seed=0)
        assert len(result) == 256 * 512

    def test_count_single_row(self):
        result = generate_pixel_coordinates(1, 10, seed=0)
        assert len(result) == 10

    def test_count_single_column(self):
        result = generate_pixel_coordinates(10, 1, seed=0)
        assert len(result) == 10

    # --- Coordinate bounds ---

    def test_y_within_height_512x512(self):
        result = generate_pixel_coordinates(512, 512, seed=1)
        assert all(0 <= y < 512 for y, _ in result)

    def test_x_within_width_512x512(self):
        result = generate_pixel_coordinates(512, 512, seed=1)
        assert all(0 <= x < 512 for _, x in result)

    def test_bounds_non_square(self):
        h, w = 100, 200
        result = generate_pixel_coordinates(h, w, seed=7)
        assert all(0 <= y < h and 0 <= x < w for y, x in result)

    def test_single_pixel_coordinate(self):
        result = generate_pixel_coordinates(1, 1, seed=0)
        assert result == [(0, 0)]

    # --- No duplicates (full coverage) ---

    def test_no_duplicate_coordinates_4x4(self):
        result = generate_pixel_coordinates(4, 4, seed=0)
        assert len(result) == len(set(result))

    def test_no_duplicate_coordinates_512x512(self):
        result = generate_pixel_coordinates(512, 512, seed=99)
        assert len(result) == len(set(result))

    def test_no_duplicate_coordinates_non_square(self):
        result = generate_pixel_coordinates(64, 128, seed=5)
        assert len(result) == len(set(result))

    # --- Reproducibility ---

    def test_same_seed_same_output(self):
        r1 = generate_pixel_coordinates(512, 512, seed=42)
        r2 = generate_pixel_coordinates(512, 512, seed=42)
        assert r1 == r2

    def test_different_seeds_different_order(self):
        r1 = generate_pixel_coordinates(512, 512, seed=1)
        r2 = generate_pixel_coordinates(512, 512, seed=2)
        assert r1 != r2

    def test_seed_zero_is_deterministic(self):
        r1 = generate_pixel_coordinates(4, 4, seed=0)
        r2 = generate_pixel_coordinates(4, 4, seed=0)
        assert r1 == r2

    def test_seed_from_password_is_reproducible(self):
        seed = password_to_seed("mysecret")
        r1 = generate_pixel_coordinates(512, 512, seed=seed)
        r2 = generate_pixel_coordinates(512, 512, seed=seed)
        assert r1 == r2

    # --- Shuffled (not sequential) ---

    def test_output_is_not_row_major_order(self):
        """Coordinates should not simply be [(0,0),(0,1),...] in order."""
        result = generate_pixel_coordinates(512, 512, seed=42)
        sequential = [(y, x) for y in range(512) for x in range(512)]
        assert result != sequential

    def test_small_grid_is_shuffled(self):
        result = generate_pixel_coordinates(4, 4, seed=123)
        sequential = [(y, x) for y in range(4) for x in range(4)]
        assert result != sequential

    # --- Zero dimensions ---

    def test_zero_height_returns_empty(self):
        result = generate_pixel_coordinates(0, 512, seed=0)
        assert result == []

    def test_zero_width_returns_empty(self):
        result = generate_pixel_coordinates(512, 0, seed=0)
        assert result == []

    def test_zero_both_returns_empty(self):
        result = generate_pixel_coordinates(0, 0, seed=0)
        assert result == []

    # --- TypeError ---

    def test_raises_type_error_float_height(self):
        with pytest.raises(TypeError):
            generate_pixel_coordinates(512.0, 512, seed=0)

    def test_raises_type_error_float_width(self):
        with pytest.raises(TypeError):
            generate_pixel_coordinates(512, 512.0, seed=0)

    def test_raises_type_error_float_seed(self):
        with pytest.raises(TypeError):
            generate_pixel_coordinates(512, 512, seed=0.5)

    def test_raises_type_error_string_height(self):
        with pytest.raises(TypeError):
            generate_pixel_coordinates("512", 512, seed=0)

    def test_raises_type_error_none_height(self):
        with pytest.raises(TypeError):
            generate_pixel_coordinates(None, 512, seed=0)

    def test_raises_type_error_none_width(self):
        with pytest.raises(TypeError):
            generate_pixel_coordinates(512, None, seed=0)

    # --- ValueError ---

    def test_raises_value_error_negative_height(self):
        with pytest.raises(ValueError):
            generate_pixel_coordinates(-1, 512, seed=0)

    def test_raises_value_error_negative_width(self):
        with pytest.raises(ValueError):
            generate_pixel_coordinates(512, -1, seed=0)

    def test_raises_value_error_both_negative(self):
        with pytest.raises(ValueError):
            generate_pixel_coordinates(-512, -512, seed=0)

    # --- Integration with password_to_seed ---

    def test_integration_password_seed_bounds(self):
        seed = password_to_seed("framework_test")
        result = generate_pixel_coordinates(512, 512, seed=seed)
        assert len(result) == 512 * 512
        assert all(0 <= y < 512 and 0 <= x < 512 for y, x in result)

    def test_integration_different_passwords_different_sequences(self):
        seed_a = password_to_seed("alpha")
        seed_b = password_to_seed("beta")
        r_a = generate_pixel_coordinates(512, 512, seed=seed_a)
        r_b = generate_pixel_coordinates(512, 512, seed=seed_b)
        assert r_a != r_b
