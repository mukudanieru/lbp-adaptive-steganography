import numpy as np
import pytest

from src.core.embedding import embed_message, embed_bits_in_pixel, text_to_binary
from src.core.lbp import compute_lbp_classification
from src.core.extraction import (
    extract_bits_from_pixel,
    binary_to_text,
    extract_message_length,
    extract_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pixel(r: int = 200, g: int = 100, b: int = 150) -> np.ndarray:
    return np.array([r, g, b], dtype=np.uint8)


def solid_rgb(h: int, w: int, r: int = 100, g: int = 150, b: int = 200) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def sequential_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(y, x) for y in range(h) for x in range(w)]


def all_smooth(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def all_rough(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests for extract_bits_from_pixel
# ---------------------------------------------------------------------------
class TestExtractBitsFromPixel:
    # --- Return type and structure ---

    def test_returns_string(self):
        assert isinstance(extract_bits_from_pixel(pixel(), num_bits=1), str)

    def test_output_contains_only_0_and_1(self):
        result = extract_bits_from_pixel(pixel(), num_bits=1)
        assert all(c in "01" for c in result)

    def test_output_length_num_bits_1(self):
        # 1 bit from R + 1 bit from B = 2 bits
        result = extract_bits_from_pixel(pixel(), num_bits=1)
        assert len(result) == 2

    def test_output_length_num_bits_2(self):
        # 2 bits from R + 2 bits from B = 4 bits
        result = extract_bits_from_pixel(pixel(), num_bits=2)
        assert len(result) == 4

    # --- Green channel is skipped ---

    def test_green_channel_not_in_output_num_bits_1(self):
        # pixel where only G carries a signal — output should not reflect G
        p = np.array([0b11111110, 0b00000001, 0b11111110], dtype=np.uint8)
        result = extract_bits_from_pixel(p, num_bits=1)
        # R LSB=0, B LSB=0 → "00"
        assert result == "00"

    def test_green_channel_ignored_regardless_of_value(self):
        p1 = np.array([200, 0, 150], dtype=np.uint8)
        p2 = np.array([200, 255, 150], dtype=np.uint8)
        assert extract_bits_from_pixel(p1, num_bits=1) == extract_bits_from_pixel(
            p2, num_bits=1
        )

    # --- Correctness: num_bits=1 ---

    def test_num_bits_1_extracts_r_lsb_then_b_lsb(self):
        # R=201 (LSB=1), B=150 (LSB=0) → "10"
        p = np.array([201, 100, 150], dtype=np.uint8)
        assert extract_bits_from_pixel(p, num_bits=1) == "10"

    def test_num_bits_1_both_lsb_zero(self):
        p = np.array([200, 100, 150], dtype=np.uint8)  # R LSB=0, B LSB=0
        assert extract_bits_from_pixel(p, num_bits=1) == "00"

    def test_num_bits_1_both_lsb_one(self):
        p = np.array([201, 100, 151], dtype=np.uint8)  # R LSB=1, B LSB=1
        assert extract_bits_from_pixel(p, num_bits=1) == "11"

    # --- Correctness: num_bits=2 ---

    def test_num_bits_2_extracts_2_lsb_from_r_then_b(self):
        # R=0b11001010 → 2 LSB = 10, B=0b10110001 → 2 LSB = 01 → "1001"
        p = np.array([0b11001010, 100, 0b10110001], dtype=np.uint8)
        assert extract_bits_from_pixel(p, num_bits=2) == "1001"

    def test_num_bits_2_all_zeros(self):
        p = np.array([0b11111100, 100, 0b11111100], dtype=np.uint8)
        assert extract_bits_from_pixel(p, num_bits=2) == "0000"

    def test_num_bits_2_all_ones(self):
        p = np.array([0b11111111, 100, 0b11111111], dtype=np.uint8)
        assert extract_bits_from_pixel(p, num_bits=2) == "1111"

    # --- Boundary pixel values ---

    def test_pixel_value_0_extracts_zero_bits(self):
        p = np.array([0, 0, 0], dtype=np.uint8)
        assert extract_bits_from_pixel(p, num_bits=1) == "00"

    def test_pixel_value_255_extracts_one_bits(self):
        p = np.array([255, 255, 255], dtype=np.uint8)
        assert extract_bits_from_pixel(p, num_bits=1) == "11"

    # --- Embed-extract roundtrip consistency ---

    def test_extracted_bits_match_embedded_bits_num_bits_1(self):
        original = np.array([200, 100, 150], dtype=np.uint8)
        embedded = embed_bits_in_pixel(original, "10", num_bits=1)
        assert extract_bits_from_pixel(embedded, num_bits=1) == "10"

    def test_extracted_bits_match_embedded_bits_num_bits_2(self):
        original = np.array([200, 100, 150], dtype=np.uint8)
        embedded = embed_bits_in_pixel(original, "1011", num_bits=2)
        assert extract_bits_from_pixel(embedded, num_bits=2) == "1011"

    # --- TypeError ---

    def test_raises_type_error_list_pixel(self):
        with pytest.raises(TypeError):
            extract_bits_from_pixel([200, 100, 150], num_bits=1)

    def test_raises_type_error_none_pixel(self):
        with pytest.raises(TypeError):
            extract_bits_from_pixel(None, num_bits=1)

    def test_raises_type_error_float_num_bits(self):
        with pytest.raises(TypeError):
            extract_bits_from_pixel(pixel(), num_bits=1.0)

    def test_raises_type_error_string_num_bits(self):
        with pytest.raises(TypeError):
            extract_bits_from_pixel(pixel(), num_bits="1")

    # --- ValueError ---

    def test_raises_value_error_num_bits_0(self):
        with pytest.raises(ValueError):
            extract_bits_from_pixel(pixel(), num_bits=0)

    def test_raises_value_error_num_bits_3(self):
        with pytest.raises(ValueError):
            extract_bits_from_pixel(pixel(), num_bits=3)

    def test_raises_value_error_wrong_pixel_length_2(self):
        with pytest.raises(ValueError):
            extract_bits_from_pixel(np.array([100, 200], dtype=np.uint8), num_bits=1)

    def test_raises_value_error_wrong_pixel_length_4(self):
        with pytest.raises(ValueError):
            extract_bits_from_pixel(
                np.array([100, 200, 150, 255], dtype=np.uint8), num_bits=1
            )

    def test_raises_value_error_pixel_value_above_255(self):
        p = np.array([256, 100, 150], dtype=np.int32)
        with pytest.raises(ValueError):
            extract_bits_from_pixel(p, num_bits=1)

    def test_raises_value_error_pixel_value_negative(self):
        p = np.array([-1, 100, 150], dtype=np.int32)
        with pytest.raises(ValueError):
            extract_bits_from_pixel(p, num_bits=1)


# ---------------------------------------------------------------------------
# Tests for extract_message_length
# ---------------------------------------------------------------------------
class TestExtractMessageLength:
    # --- Return type ---

    def test_returns_int(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        stego = embed_message(img, "Hi", cmap, coords)
        result = extract_message_length(stego, cmap, coords)
        assert isinstance(result, int)

    # --- Correctness: extracted length matches embedded message bit length ---

    def test_length_matches_short_message_smooth(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        msg = "Hi"
        stego = embed_message(img, msg, cmap, coords)
        extracted = extract_message_length(stego, cmap, coords)
        assert extracted == len(msg) * 8

    def test_length_matches_short_message_rough(self):
        img = solid_rgb(512, 512)
        cmap = all_rough(512, 512)
        coords = sequential_coords(512, 512)
        msg = "Hi"
        stego = embed_message(img, msg, cmap, coords)
        extracted = extract_message_length(stego, cmap, coords)
        assert extracted == len(msg) * 8

    def test_length_matches_longer_message(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        msg = "Hello, World!"
        stego = embed_message(img, msg, cmap, coords)
        extracted = extract_message_length(stego, cmap, coords)
        assert extracted == len(msg) * 8

    def test_length_matches_single_char(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        stego = embed_message(img, "A", cmap, coords)
        assert extract_message_length(stego, cmap, coords) == 8

    def test_length_is_non_negative(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        stego = embed_message(img, "test", cmap, coords)
        assert extract_message_length(stego, cmap, coords) >= 0

    def test_length_deterministic(self):
        img = solid_rgb(512, 512)
        cmap = all_smooth(512, 512)
        coords = sequential_coords(512, 512)
        stego = embed_message(img, "repeat", cmap, coords)
        r1 = extract_message_length(stego, cmap, coords)
        r2 = extract_message_length(stego, cmap, coords)
        assert r1 == r2

    # --- TypeError: stego_image ---

    def test_raises_type_error_list_image(self):
        with pytest.raises(TypeError):
            extract_message_length([[[0, 0, 0]]], all_smooth(1, 1), [(0, 0)])

    def test_raises_type_error_none_image(self):
        with pytest.raises(TypeError):
            extract_message_length(
                None, all_smooth(512, 512), sequential_coords(512, 512)
            )

    # --- TypeError: classification_map ---

    def test_raises_type_error_list_cmap(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            extract_message_length(img, [[0, 1]], sequential_coords(512, 512))

    def test_raises_type_error_none_cmap(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            extract_message_length(img, None, sequential_coords(512, 512))

    # --- TypeError: pixel_coords ---

    def test_raises_type_error_coords_not_list(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            extract_message_length(img, all_smooth(512, 512), {(0, 0)})

    def test_raises_type_error_coords_list_of_lists(self):
        img = solid_rgb(512, 512)
        with pytest.raises(TypeError):
            extract_message_length(img, all_smooth(512, 512), [[0, 0]])

    # --- ValueError: stego_image ---

    def test_raises_value_error_wrong_shape_2d(self):
        with pytest.raises(ValueError):
            extract_message_length(
                np.zeros((512, 512), dtype=np.uint8),
                all_smooth(512, 512),
                sequential_coords(512, 512),
            )

    def test_raises_value_error_wrong_dtype(self):
        with pytest.raises(ValueError):
            extract_message_length(
                np.zeros((512, 512, 3), dtype=np.float32),
                all_smooth(512, 512),
                sequential_coords(512, 512),
            )

    # --- ValueError: classification_map ---

    def test_raises_value_error_cmap_shape_mismatch(self):
        img = solid_rgb(512, 512)
        with pytest.raises(ValueError):
            extract_message_length(
                img, all_smooth(256, 256), sequential_coords(512, 512)
            )

    def test_raises_value_error_cmap_invalid_values(self):
        img = solid_rgb(512, 512)
        bad_cmap = np.full((512, 512), 2, dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_message_length(img, bad_cmap, sequential_coords(512, 512))

    # --- ValueError: pixel_coords ---

    def test_raises_value_error_coord_out_of_bounds(self):
        img = solid_rgb(5, 5)
        with pytest.raises(ValueError):
            extract_message_length(img, all_smooth(5, 5), [(0, 0), (99, 99)])

    def test_raises_value_error_negative_coord(self):
        img = solid_rgb(5, 5)
        with pytest.raises(ValueError):
            extract_message_length(img, all_smooth(5, 5), [(-1, 0)])

    # --- ValueError: insufficient bits for 32-bit header ---

    def test_raises_value_error_too_few_coords_for_header(self):
        # 1x1 image, smooth → 2 bits per pixel; need 32 bits → need 16 pixels minimum
        img = solid_rgb(1, 1)
        with pytest.raises(ValueError):
            extract_message_length(img, all_smooth(1, 1), [(0, 0)])


# ---------------------------------------------------------------------------
# Tests for binary_to_text
# ---------------------------------------------------------------------------
class TestBinaryToText:
    # --- Return type ---

    def test_returns_string(self):
        assert isinstance(binary_to_text("01000001"), str)

    # --- Correctness ---

    def test_single_char_a(self):
        # 'A' = 01000001
        assert binary_to_text("01000001") == "A"

    def test_single_char_b(self):
        # 'B' = 01000010
        assert binary_to_text("01000010") == "B"

    def test_single_char_lowercase_a(self):
        # 'a' = 01100001
        assert binary_to_text("01100001") == "a"

    def test_space_character(self):
        # ' ' = 00100000
        assert binary_to_text("00100000") == " "

    def test_multi_char_string(self):
        assert binary_to_text("0100000101000010") == "AB"

    def test_full_word(self):
        assert binary_to_text("0100100001101001") == "Hi"

    def test_empty_string_returns_empty(self):
        assert binary_to_text("") == ""

    # --- Roundtrip with text_to_binary ---

    def test_roundtrip_single_char(self):
        assert binary_to_text(text_to_binary("A")) == "A"

    def test_roundtrip_sentence(self):
        msg = "Hello, World!"
        assert binary_to_text(text_to_binary(msg)) == msg

    def test_roundtrip_special_characters(self):
        msg = "!@#$%^&*()"
        assert binary_to_text(text_to_binary(msg)) == msg

    def test_roundtrip_numeric_string(self):
        msg = "1234567890"
        assert binary_to_text(text_to_binary(msg)) == msg

    def test_roundtrip_long_message(self):
        msg = "a" * 200
        assert binary_to_text(text_to_binary(msg)) == msg

    # --- TypeError ---

    def test_raises_type_error_for_int(self):
        with pytest.raises(TypeError):
            binary_to_text(123)

    def test_raises_type_error_for_none(self):
        with pytest.raises(TypeError):
            binary_to_text(None)

    def test_raises_type_error_for_list(self):
        with pytest.raises(TypeError):
            binary_to_text(["01000001"])

    def test_raises_type_error_for_bytes(self):
        with pytest.raises(TypeError):
            binary_to_text(b"01000001")

    # --- ValueError: non-binary characters ---

    def test_raises_value_error_for_digit_2(self):
        with pytest.raises(ValueError):
            binary_to_text("01000012")

    def test_raises_value_error_for_letter(self):
        with pytest.raises(ValueError):
            binary_to_text("0100001a")

    def test_raises_value_error_for_space_in_string(self):
        with pytest.raises(ValueError):
            binary_to_text("0100 001")

    # --- ValueError: length not multiple of 8 ---

    def test_raises_value_error_length_1(self):
        with pytest.raises(ValueError):
            binary_to_text("0")

    def test_raises_value_error_length_7(self):
        with pytest.raises(ValueError):
            binary_to_text("0100000")

    def test_raises_value_error_length_9(self):
        with pytest.raises(ValueError):
            binary_to_text("010000010")

    def test_raises_value_error_length_not_multiple_of_8(self):
        with pytest.raises(ValueError):
            binary_to_text("010000010100")


# -----------------------------
# Tests for extract_message
# -----------------------------
def test_extract_message_simple_roundtrip():
    """Test basic embed and extract roundtrip with independent LBP computation."""
    # Cover image setup
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)  # Pass RGB directly
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    # Embed
    secret_message = "Hello, World!"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    # Extract with stego's own LBP (realistic scenario)
    stego_classification = compute_lbp_classification(stego_img)  # Pass RGB directly

    # LBP should be identical after embedding (critical test!)
    assert np.array_equal(cover_classification, stego_classification), (
        "LBP classification changed after embedding - MSBs were modified!"
    )

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_empty_string():
    """Test extracting empty message."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = ""
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == ""


def test_extract_message_single_character():
    """Test extracting single character."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "A"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_long_text():
    """Test extracting longer message."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "The quick brown fox jumps over the lazy dog. " * 10
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_smooth_texture():
    """Test extraction with all smooth pixels."""
    # Create a uniform image (will likely be classified as smooth)
    cover_img = np.full((512, 512, 3), 128, dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Smooth texture test"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_rough_texture():
    """Test extraction with rough/mixed pixels."""
    # Create a random image (will likely have rough texture)
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Rough texture test"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_mixed_texture():
    """Test extraction with mixed smooth/rough texture."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Mixed texture message"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_special_characters():
    """Test extracting message with special characters."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Hello! @#$%^&*() 123 test?"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_multiline_text():
    """Test extracting multiline message."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Line 1\nLine 2\nLine 3\nLine 4"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_whitespace_preserved():
    """Test that whitespace is preserved in extraction."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "  leading and trailing  "
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_shuffled_pixels():
    """Test extraction with pseudorandom pixel order."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)

    # Shuffle pixel coordinates (simulating pseudorandom selection)
    np.random.seed(42)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]
    np.random.shuffle(pixel_coords)

    secret_message = "Pseudorandom pixel order"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_lbp_consistency():
    """Test that LBP classification remains consistent after embedding."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "LBP consistency test"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    # Compute stego LBP independently
    stego_classification = compute_lbp_classification(stego_img)

    # Classifications MUST be identical
    np.testing.assert_array_equal(
        cover_classification,
        stego_classification,
        err_msg="LBP classification changed after embedding - this breaks extraction!",
    )


def test_extract_message_green_channel_preserved():
    """Test that green channel is preserved after embedding (R&B only mode)."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Green channel preservation test"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    # Green channel (index 1) should be identical
    np.testing.assert_array_equal(
        cover_img[:, :, 1],
        stego_img[:, :, 1],
        err_msg="Green channel was modified during R&B only embedding!",
    )


def test_extract_message_returns_string():
    """Test that extraction always returns a string."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Type check"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert isinstance(extracted, str)


def test_extract_message_json_like_content():
    """Test extracting JSON-like structured content."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = '{"name": "test", "value": 123}'
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_repeated_characters():
    """Test extracting message with repeated characters."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "aaaaaabbbbbbcccccc"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_numbers_only():
    """Test extracting numeric-only message."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_classification = compute_lbp_classification(cover_img)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "1234567890"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_classification = compute_lbp_classification(stego_img)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message
