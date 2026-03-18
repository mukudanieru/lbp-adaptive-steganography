from src.core.embedding import (
    text_to_binary,
    get_binary_header,
    calculate_capacity,
    embed_bits_in_pixel,
    embed_message,
)
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests for text_to_binary
# ---------------------------------------------------------------------------
class TestTextToBinary:
    # --- Return type and structure ---

    def test_returns_string(self):
        assert isinstance(text_to_binary("A"), str)

    def test_output_contains_only_0_and_1(self):
        result = text_to_binary("Hello")
        assert all(c in "01" for c in result)

    def test_output_length_is_8_bits_per_char(self):
        text = "Hello"
        assert len(text_to_binary(text)) == len(text) * 8

    def test_single_char_is_8_bits(self):
        assert len(text_to_binary("A")) == 8

    # --- Correctness ---

    def test_capital_a_encoding(self):
        # 'A' = ASCII 65 = 01000001
        assert text_to_binary("A") == "01000001"

    def test_capital_b_encoding(self):
        # 'B' = ASCII 66 = 01000010
        assert text_to_binary("B") == "01000010"

    def test_lowercase_a_encoding(self):
        # 'a' = ASCII 97 = 01100001
        assert text_to_binary("a") == "01100001"

    def test_space_encoding(self):
        # ' ' = ASCII 32 = 00100000
        assert text_to_binary(" ") == "00100000"

    def test_null_character_encoding(self):
        # '\x00' = ASCII 0 = 00000000
        assert text_to_binary("\x00") == "00000000"

    def test_del_character_encoding(self):
        # '\x7f' = ASCII 127 = 01111111
        assert text_to_binary("\x7f") == "01111111"

    def test_multi_char_output_is_concatenated(self):
        result = text_to_binary("AB")
        assert result == "0100000101000010"

    def test_numeric_string(self):
        result = text_to_binary("123")
        assert len(result) == 24
        assert all(c in "01" for c in result)

    def test_special_chars(self):
        result = text_to_binary("!@#")
        assert len(result) == 24

    def test_full_sentence(self):
        text = "Hello, World!"
        result = text_to_binary(text)
        assert len(result) == len(text) * 8

    # --- Edge cases ---

    def test_empty_string_returns_empty(self):
        assert text_to_binary("") == ""

    def test_single_space(self):
        assert len(text_to_binary(" ")) == 8

    def test_long_string(self):
        text = "a" * 1000
        assert len(text_to_binary(text)) == 8000

    # --- TypeError ---

    def test_raises_type_error_for_int(self):
        with pytest.raises(TypeError):
            text_to_binary(123)

    def test_raises_type_error_for_none(self):
        with pytest.raises(TypeError):
            text_to_binary(None)

    def test_raises_type_error_for_list(self):
        with pytest.raises(TypeError):
            text_to_binary(["hello"])

    def test_raises_type_error_for_bytes(self):
        with pytest.raises(TypeError):
            text_to_binary(b"hello")

    # --- ValueError: non-ASCII ---

    def test_raises_value_error_for_emoji(self):
        with pytest.raises(ValueError):
            text_to_binary("hello 🔑")

    def test_raises_value_error_for_accented_letter(self):
        with pytest.raises(ValueError):
            text_to_binary("café")

    def test_raises_value_error_for_unicode_char(self):
        with pytest.raises(ValueError):
            text_to_binary("пароль")

    def test_raises_value_error_for_cjk_char(self):
        with pytest.raises(ValueError):
            text_to_binary("你好")


# ---------------------------------------------------------------------------
# Tests for get_binary_header
# ---------------------------------------------------------------------------
class TestGetBinaryHeader:
    # --- Return type ---

    def test_returns_string(self):
        assert isinstance(get_binary_header("01010101"), str)

    def test_output_is_32_bits(self):
        assert len(get_binary_header("01010101")) == 32

    def test_output_contains_only_0_and_1(self):
        result = get_binary_header("01010101")
        assert all(c in "01" for c in result)

    # --- Correctness ---

    def test_empty_message_header_is_zero(self):
        # length 0 → 32-bit zero
        assert get_binary_header("") == "00000000000000000000000000000000"

    def test_8_bit_message_header(self):
        # length 8 → 00000000000000000000000000001000
        assert get_binary_header("01000001") == "00000000000000000000000000001000"

    def test_header_encodes_message_length(self):
        msg = "0" * 64
        result = get_binary_header(msg)
        assert int(result, 2) == 64

    def test_header_length_matches_for_various_messages(self):
        for n in [8, 16, 32, 64, 128, 256]:
            msg = "1" * n
            result = get_binary_header(msg)
            assert int(result, 2) == n

    def test_large_message_length_fits_32_bits(self):
        # max 32-bit value = 2^32 - 1; use a message of length 1000*8 = 8000 bits
        msg = "0" * 8000
        result = get_binary_header(msg)
        assert len(result) == 32
        assert int(result, 2) == 8000

    def test_header_always_32_bits_regardless_of_content(self):
        for msg in ["0", "01", "0" * 100, "1" * 255]:
            assert len(get_binary_header(msg)) == 32

    # --- TypeError ---

    def test_raises_type_error_for_int(self):
        with pytest.raises(TypeError):
            get_binary_header(123)

    def test_raises_type_error_for_none(self):
        with pytest.raises(TypeError):
            get_binary_header(None)

    def test_raises_type_error_for_list(self):
        with pytest.raises(TypeError):
            get_binary_header(["01010101"])

    def test_raises_type_error_for_bytes(self):
        with pytest.raises(TypeError):
            get_binary_header(b"01010101")


# ---------------------------------------------------------------------------
# Tests for calculate_capacity
# ---------------------------------------------------------------------------
class TestCalculateCapacity:
    # --- Helpers ---

    def all_smooth(self, n: int) -> np.ndarray:
        return np.zeros(n, dtype=np.uint8)

    def all_rough(self, n: int) -> np.ndarray:
        return np.ones(n, dtype=np.uint8)

    def mixed(self, n_smooth: int, n_rough: int) -> np.ndarray:
        return np.array([0] * n_smooth + [1] * n_rough, dtype=np.uint8)

    # --- Return type ---

    def test_returns_int(self):
        result = calculate_capacity(self.all_smooth(10))
        assert isinstance(result, int)

    # --- Capacity formula: num_channels * (smooth*1 + rough*2) ---

    def test_all_smooth_default_channels(self):
        # 10 smooth, 2 channels → 2 * (10*1 + 0*2) = 20
        assert calculate_capacity(self.all_smooth(10)) == 20

    def test_all_rough_default_channels(self):
        # 10 rough, 2 channels → 2 * (0*1 + 10*2) = 40
        assert calculate_capacity(self.all_rough(10)) == 40

    def test_mixed_default_channels(self):
        # 6 smooth + 4 rough, 2 channels → 2 * (6 + 8) = 28
        assert calculate_capacity(self.mixed(6, 4)) == 28

    def test_all_smooth_single_channel(self):
        # 10 smooth, 1 channel → 1 * 10 = 10
        assert calculate_capacity(self.all_smooth(10), num_channels=1) == 10

    def test_all_rough_single_channel(self):
        # 10 rough, 1 channel → 1 * 20 = 20
        assert calculate_capacity(self.all_rough(10), num_channels=1) == 20

    def test_mixed_three_channels(self):
        # 4 smooth + 4 rough, 3 channels → 3 * (4 + 8) = 36
        assert calculate_capacity(self.mixed(4, 4), num_channels=3) == 36

    def test_512x512_all_smooth(self):
        # 512*512 = 262144 smooth, 2 channels → 524288
        cmap = self.all_smooth(512 * 512)
        assert calculate_capacity(cmap) == 2 * 262144

    def test_512x512_all_rough(self):
        # 512*512 rough, 2 channels → 2 * 2 * 262144 = 1048576
        cmap = self.all_rough(512 * 512)
        assert calculate_capacity(cmap) == 4 * 262144

    def test_empty_map_returns_zero(self):
        cmap = np.array([], dtype=np.uint8)
        assert calculate_capacity(cmap) == 0

    def test_rough_embeds_more_than_smooth(self):
        assert calculate_capacity(self.all_rough(100)) > calculate_capacity(
            self.all_smooth(100)
        )

    def test_capacity_scales_with_num_channels(self):
        cmap = self.mixed(50, 50)
        c1 = calculate_capacity(cmap, num_channels=1)
        c2 = calculate_capacity(cmap, num_channels=2)
        assert c2 == 2 * c1

    # --- 2D classification map (real use case: 512x512) ---

    def test_2d_map_flat_smooth(self):
        cmap = np.zeros((512, 512), dtype=np.uint8)
        result = calculate_capacity(cmap.flatten())
        assert result == 2 * 512 * 512

    def test_2d_map_flat_rough(self):
        cmap = np.ones((512, 512), dtype=np.uint8)
        result = calculate_capacity(cmap.flatten())
        assert result == 4 * 512 * 512

    # --- TypeError: invalid classification_map ---

    def test_raises_type_error_list_map(self):
        with pytest.raises(TypeError):
            calculate_capacity([0, 1, 0, 1])

    def test_raises_type_error_none_map(self):
        with pytest.raises(TypeError):
            calculate_capacity(None)

    def test_raises_type_error_float_map(self):
        with pytest.raises(TypeError):
            calculate_capacity(np.array([0.0, 1.0]))

    def test_raises_type_error_float_channels(self):
        with pytest.raises(TypeError):
            calculate_capacity(self.all_smooth(10), num_channels=2.0)

    def test_raises_type_error_string_channels(self):
        with pytest.raises(TypeError):
            calculate_capacity(self.all_smooth(10), num_channels="2")

    def test_raises_type_error_none_channels(self):
        with pytest.raises(TypeError):
            calculate_capacity(self.all_smooth(10), num_channels=None)

    # --- ValueError: invalid values in map ---

    def test_raises_value_error_value_2_in_map(self):
        with pytest.raises(ValueError):
            calculate_capacity(np.array([0, 1, 2], dtype=np.uint8))

    def test_raises_value_error_negative_value_in_map(self):
        with pytest.raises(ValueError):
            calculate_capacity(np.array([0, -1, 1], dtype=np.int8))

    def test_raises_value_error_zero_channels(self):
        with pytest.raises(ValueError):
            calculate_capacity(self.all_smooth(10), num_channels=0)

    def test_raises_value_error_negative_channels(self):
        with pytest.raises(ValueError):
            calculate_capacity(self.all_smooth(10), num_channels=-1)


# ---------------------------------------------------------------------------
# Tests for embed_bits_in_pixel
# ---------------------------------------------------------------------------
class TestEmbedBitsInPixel:
    # --- Helpers ---

    def pixel(self, r: int = 200, g: int = 100, b: int = 150) -> np.ndarray:
        return np.array([r, g, b], dtype=np.uint8)

    # --- Return type and shape ---

    def test_returns_ndarray(self):
        result = embed_bits_in_pixel(self.pixel(), "10", num_bits=1)
        assert isinstance(result, np.ndarray)

    def test_output_shape_is_3(self):
        result = embed_bits_in_pixel(self.pixel(), "10", num_bits=1)
        assert result.shape == (3,)

    def test_output_dtype_is_uint8(self):
        result = embed_bits_in_pixel(self.pixel(), "10", num_bits=1)
        assert result.dtype == np.uint8

    # --- Green channel is never modified ---

    def test_green_channel_unchanged_num_bits_1(self):
        p = self.pixel(r=200, g=100, b=150)
        result = embed_bits_in_pixel(p, "10", num_bits=1)
        assert result[1] == 100

    def test_green_channel_unchanged_num_bits_2(self):
        p = self.pixel(r=200, g=100, b=150)
        result = embed_bits_in_pixel(p, "1011", num_bits=2)
        assert result[1] == 100

    def test_green_channel_unchanged_various_bits(self):
        p = self.pixel(g=77)
        for bits in ["0", "1", "00", "11", "01", "10"]:
            result = embed_bits_in_pixel(p, bits, num_bits=1)
            assert result[1] == 77

    # --- LSB substitution correctness: num_bits = 1 ---

    def test_embed_1_bit_into_r_sets_lsb(self):
        # R=200 (11001000), embed '1' → 11001001 = 201
        p = self.pixel(r=200)
        result = embed_bits_in_pixel(p, "10", num_bits=1)
        assert result[0] & 1 == 1

    def test_embed_1_bit_clears_lsb_when_0(self):
        # R=201 (11001001), embed '0' → 11001000 = 200
        p = self.pixel(r=201)
        result = embed_bits_in_pixel(p, "00", num_bits=1)
        assert result[0] & 1 == 0

    def test_embed_1_bit_r_and_b_channels(self):
        # num_bits=1, 2 bits total: 1 bit into R, 1 bit into B
        p = np.array([200, 100, 150], dtype=np.uint8)
        result = embed_bits_in_pixel(p, "10", num_bits=1)
        assert result[0] & 1 == 1  # R LSB = 1
        assert result[2] & 1 == 0  # B LSB = 0

    def test_embed_1_bit_preserves_upper_bits_of_r(self):
        p = self.pixel(r=200)
        result = embed_bits_in_pixel(p, "10", num_bits=1)
        assert result[0] >> 1 == 200 >> 1

    def test_embed_1_bit_preserves_upper_bits_of_b(self):
        p = self.pixel(b=150)
        result = embed_bits_in_pixel(p, "10", num_bits=1)
        assert result[2] >> 1 == 150 >> 1

    # --- LSB substitution correctness: num_bits = 2 ---

    def test_embed_2_bits_into_r_sets_2_lsb(self):
        # R=200 (11001000), embed '11' → lower 2 bits = 11 → 11001011 = 203
        p = self.pixel(r=200)
        result = embed_bits_in_pixel(p, "1100", num_bits=2)
        assert result[0] & 0b11 == 0b11

    def test_embed_2_bits_clears_2_lsb_when_00(self):
        p = self.pixel(r=255)
        result = embed_bits_in_pixel(p, "0000", num_bits=2)
        assert result[0] & 0b11 == 0b00

    def test_embed_2_bits_r_and_b_channels(self):
        # num_bits=2, 4 bits total: 2 into R, 2 into B
        p = np.array([200, 100, 150], dtype=np.uint8)
        result = embed_bits_in_pixel(p, "1001", num_bits=2)
        assert result[0] & 0b11 == 0b10
        assert result[2] & 0b11 == 0b01

    def test_embed_2_bits_preserves_upper_6_bits_of_r(self):
        p = self.pixel(r=200)
        result = embed_bits_in_pixel(p, "1011", num_bits=2)
        assert result[0] >> 2 == 200 >> 2

    def test_embed_2_bits_preserves_upper_6_bits_of_b(self):
        p = self.pixel(b=150)
        result = embed_bits_in_pixel(p, "1011", num_bits=2)
        assert result[2] >> 2 == 150 >> 2

    # --- Boundary pixel values ---

    def test_embed_into_pixel_all_zeros(self):
        p = np.array([0, 0, 0], dtype=np.uint8)
        result = embed_bits_in_pixel(p, "11", num_bits=1)
        assert result.shape == (3,)
        assert result[1] == 0

    def test_embed_into_pixel_all_255(self):
        p = np.array([255, 255, 255], dtype=np.uint8)
        result = embed_bits_in_pixel(p, "00", num_bits=1)
        assert result.shape == (3,)
        assert result[1] == 255

    # --- Partial bits (fewer than max capacity) ---

    def test_embed_empty_bits_returns_unchanged_pixel(self):
        p = self.pixel(r=200, g=100, b=150)
        result = embed_bits_in_pixel(p, "", num_bits=1)
        assert np.array_equal(result, p)

    def test_embed_1_bit_only_modifies_r(self):
        p = self.pixel(r=200, g=100, b=150)
        result = embed_bits_in_pixel(p, "1", num_bits=1)
        assert result[0] & 1 == 1
        assert result[2] == 150  # B untouched

    # --- TypeError ---

    def test_raises_type_error_list_pixel(self):
        with pytest.raises(TypeError):
            embed_bits_in_pixel([200, 100, 150], "10", num_bits=1)

    def test_raises_type_error_none_pixel(self):
        with pytest.raises(TypeError):
            embed_bits_in_pixel(None, "10", num_bits=1)

    def test_raises_type_error_int_bits(self):
        with pytest.raises(TypeError):
            embed_bits_in_pixel(self.pixel(), 10, num_bits=1)

    def test_raises_type_error_none_bits(self):
        with pytest.raises(TypeError):
            embed_bits_in_pixel(self.pixel(), None, num_bits=1)

    def test_raises_type_error_float_num_bits(self):
        with pytest.raises(TypeError):
            embed_bits_in_pixel(self.pixel(), "10", num_bits=1.0)

    def test_raises_type_error_string_num_bits(self):
        with pytest.raises(TypeError):
            embed_bits_in_pixel(self.pixel(), "10", num_bits="1")

    # --- ValueError: num_bits not 1 or 2 ---

    def test_raises_value_error_num_bits_0(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(self.pixel(), "10", num_bits=0)

    def test_raises_value_error_num_bits_3(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(self.pixel(), "10", num_bits=3)

    def test_raises_value_error_num_bits_negative(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(self.pixel(), "10", num_bits=-1)

    # --- ValueError: wrong pixel shape ---

    def test_raises_value_error_pixel_length_2(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(np.array([100, 200], dtype=np.uint8), "10", num_bits=1)

    def test_raises_value_error_pixel_length_4(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(
                np.array([100, 200, 150, 255], dtype=np.uint8), "10", num_bits=1
            )

    def test_raises_value_error_pixel_2d(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(
                np.array([[100, 200, 150]], dtype=np.uint8), "10", num_bits=1
            )

    # --- ValueError: pixel values out of range ---

    def test_raises_value_error_pixel_value_above_255(self):
        p = np.array([256, 100, 150], dtype=np.int32)
        with pytest.raises(ValueError):
            embed_bits_in_pixel(p, "10", num_bits=1)

    def test_raises_value_error_pixel_value_negative(self):
        p = np.array([-1, 100, 150], dtype=np.int32)
        with pytest.raises(ValueError):
            embed_bits_in_pixel(p, "10", num_bits=1)

    # --- ValueError: non-binary characters in bits ---

    def test_raises_value_error_bits_with_2(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(self.pixel(), "12", num_bits=1)

    def test_raises_value_error_bits_with_letter(self):
        with pytest.raises(ValueError):
            embed_bits_in_pixel(self.pixel(), "1a", num_bits=1)

    # --- ValueError: bits exceed max capacity ---

    def test_raises_value_error_bits_too_long_num_bits_1(self):
        # max = 2 * 1 = 2 bits; passing 3 should fail
        with pytest.raises(ValueError):
            embed_bits_in_pixel(self.pixel(), "101", num_bits=1)

    def test_raises_value_error_bits_too_long_num_bits_2(self):
        # max = 2 * 2 = 4 bits; passing 5 should fail
        with pytest.raises(ValueError):
            embed_bits_in_pixel(self.pixel(), "10110", num_bits=2)


# ---------------------------------------------------------------------------
# Helpers for embed_message
# ---------------------------------------------------------------------------
def solid_rgb(h: int, w: int, r: int = 100, g: int = 150, b: int = 200) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def random_rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def all_smooth(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def all_rough(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def sequential_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(y, x) for y in range(h) for x in range(w)]


# ---------------------------------------------------------------------------
# Tests for embed_message
# ---------------------------------------------------------------------------
class TestEmbedMessage:
    # --- Output contract ---

    def test_returns_ndarray(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Hi", cmap, coords)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Hi", cmap, coords)
        assert result.shape == (512, 512, 3)

    def test_output_dtype_is_uint8(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Hi", cmap, coords)
        assert result.dtype == np.uint8

    def test_does_not_mutate_original_image(self):
        img = solid_rgb(512, 512)
        original = img.copy()
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        embed_message(img, "Hi", cmap, coords)
        assert np.array_equal(img, original)

    def test_output_is_different_array_from_input(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Hi", cmap, coords)
        assert result is not img

    # --- Green channel invariant ---

    def test_green_channel_unchanged_all_smooth(self):
        img = random_rgb(512, 512, seed=1)
        original_green = img[:, :, 1].copy()
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Hello World", cmap, coords)
        assert np.array_equal(result[:, :, 1], original_green)

    def test_green_channel_unchanged_all_rough(self):
        img = random_rgb(512, 512, seed=2)
        original_green = img[:, :, 1].copy()
        cmap = all_rough(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Hello World", cmap, coords)
        assert np.array_equal(result[:, :, 1], original_green)

    def test_green_channel_unchanged_mixed_map(self):
        img = random_rgb(512, 512, seed=3)
        original_green = img[:, :, 1].copy()
        rng = np.random.default_rng(3)
        cmap = rng.integers(0, 2, (512, 512), dtype=np.uint8)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Secret", cmap, coords)
        assert np.array_equal(result[:, :, 1], original_green)

    # --- Pixel modification locality ---

    def test_unvisited_pixels_are_unchanged(self):
        """Only pixels in pixel_coords should be touched."""
        img = random_rgb(10, 10, seed=5)
        cmap = all_smooth(10, 10)
        # Only use a small subset of coords
        coords = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
        ]
        result = embed_message(img, "A", cmap, coords)
        # All rows from row 2 onwards must be untouched
        assert np.array_equal(result[2:, :, :], img[2:, :, :])

    def test_visited_pixel_r_or_b_may_change(self):
        """Pixels in coords should have R or B potentially modified."""
        img = solid_rgb(10, 10, r=200, g=100, b=150)
        cmap = all_smooth(10, 10)
        coords = sequential_coords(10, 10)
        result = embed_message(img, "Hello", cmap, coords)
        # At least some R or B values among visited pixels differ
        rb_changed = not (
            np.array_equal(result[:, :, 0], img[:, :, 0])
            and np.array_equal(result[:, :, 2], img[:, :, 2])
        )
        assert rb_changed

    # --- Stego differs from cover ---

    def test_stego_differs_from_cover_image(self):
        img = random_rgb(512, 512, seed=10)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "This is a secret message.", cmap, coords)
        assert not np.array_equal(result, img)

    def test_stego_differs_on_rough_map(self):
        img = random_rgb(512, 512, seed=11)
        cmap = all_rough(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Rough embedding test.", cmap, coords)
        assert not np.array_equal(result, img)

    # --- Determinism ---

    def test_same_inputs_same_stego(self):
        img = random_rgb(512, 512, seed=20)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        r1 = embed_message(img, "determinism", cmap, coords)
        r2 = embed_message(img, "determinism", cmap, coords)
        assert np.array_equal(r1, r2)

    def test_different_messages_different_stego(self):
        img = random_rgb(512, 512, seed=21)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        r1 = embed_message(img, "messageA", cmap, coords)
        r2 = embed_message(img, "messageB", cmap, coords)
        assert not np.array_equal(r1, r2)

    def test_pixel_value_range_stays_0_to_255(self):
        img = random_rgb(512, 512, seed=30)
        cmap = all_rough(512, 512)
        coords = sequential_coords(512, 512)
        result = embed_message(img, "Range check test!", cmap, coords)
        assert result.min() >= 0
        assert result.max() <= 255

    # --- TypeError: rgb_img ---

    def test_raises_type_error_list_image(self):
        with pytest.raises(TypeError):
            embed_message([[[0, 0, 0]]], "Hi", all_smooth(1, 1), [(0, 0)])

    def test_raises_type_error_none_image(self):
        with pytest.raises(TypeError):
            embed_message(None, "Hi", all_smooth(1, 1), [(0, 0)])

    # --- ValueError: rgb_img ---

    def test_raises_value_error_wrong_shape_2d(self):
        with pytest.raises(ValueError):
            embed_message(
                np.zeros((512, 512), dtype=np.uint8),
                "Hi",
                all_smooth(512, 512),
                [(0, 0)],
            )

    def test_raises_value_error_wrong_dtype(self):
        with pytest.raises(ValueError):
            embed_message(
                np.zeros((512, 512, 3), dtype=np.float32),
                "Hi",
                all_smooth(512, 512),
                [(0, 0)],
            )

    def test_raises_value_error_4_channels(self):
        with pytest.raises(ValueError):
            embed_message(
                np.zeros((512, 512, 4), dtype=np.uint8),
                "Hi",
                all_smooth(512, 512),
                [(0, 0)],
            )

    # --- TypeError: secret_message ---

    def test_raises_type_error_int_message(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            embed_message(img, 123, all_smooth(512, 512), sequential_coords(512, 512))

    def test_raises_type_error_none_message(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            embed_message(img, None, all_smooth(512, 512), sequential_coords(512, 512))

    # --- TypeError: classification_map ---

    def test_raises_type_error_list_classification_map(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            embed_message(img, "Hi", [[0, 1], [1, 0]], [(0, 0)])

    def test_raises_type_error_none_classification_map(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            embed_message(img, "Hi", None, [(0, 0)])

    # --- ValueError: classification_map shape mismatch ---

    def test_raises_value_error_classification_map_shape_mismatch(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(256, 256)  # wrong shape
        with pytest.raises(ValueError):
            embed_message(img, "Hi", cmap, sequential_coords(512, 512))

    # --- TypeError: pixel_coords ---

    def test_raises_type_error_coords_not_list(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            embed_message(img, "Hi", all_smooth(512, 512), {(0, 0), (0, 1)})

    def test_raises_type_error_coords_list_of_lists(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            embed_message(img, "Hi", all_smooth(512, 512), [[0, 0], [0, 1]])

    # --- ValueError: out-of-bounds coordinates ---

    def test_raises_value_error_coord_out_of_bounds(self):
        img = solid_rgb(5, 5)
        with pytest.raises(ValueError):
            embed_message(img, "A", all_smooth(5, 5), [(0, 0), (10, 10)])

    def test_raises_value_error_negative_coord(self):
        img = solid_rgb(5, 5)
        with pytest.raises(ValueError):
            embed_message(img, "A", all_smooth(5, 5), [(-1, 0)])

    # --- ValueError: message exceeds capacity ---

    def test_raises_value_error_message_too_long(self):
        # 1x1 image, all smooth → capacity = 2 * 1 = 2 bits (way too small for any message)
        img = solid_rgb(1, 1)
        with pytest.raises(ValueError):
            embed_message(img, "This is way too long", all_smooth(1, 1), [(0, 0)])

    def test_raises_value_error_message_exceeds_rough_capacity(self):
        # 2x2 image, all rough → capacity = 2 * (4 * 2) = 16 bits = 2 chars max (minus 32-bit header)
        img = solid_rgb(2, 2)
        with pytest.raises(ValueError):
            embed_message(
                img,
                "Too long message for this tiny image",
                all_rough(2, 2),
                sequential_coords(2, 2),
            )
