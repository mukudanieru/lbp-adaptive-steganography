import numpy as np
import pytest

from src.core.embedding import embed_message, text_to_binary
from src.core.preprocessing import img_to_grayscale
from src.core.lbp import compute_lbp_classification
from src.core.extraction import (
    extract_bits_from_pixel,
    binary_to_text,
    extract_message_length,
    extract_message,
)


# -----------------------------
# Tests for extract_bits_from_pixel
# -----------------------------
def test_extract_bits_from_pixel_1bit():
    pixel = [227, 136, 125]  # 227=11100011, 136=10001000, 125=01111101
    result = extract_bits_from_pixel(pixel, 1)

    # LSBs: 1, 0, 1
    assert result == "101"


def test_extract_bits_from_pixel_2bit():
    pixel = [227, 136, 125]
    result = extract_bits_from_pixel(pixel, 2)

    # 227 → 11
    # 136 → 00
    # 125 → 01
    assert result == "110001"


def test_extract_bits_invalid_bit_count():
    pixel = [100, 150, 200]
    with pytest.raises(ValueError):
        extract_bits_from_pixel(pixel, 3)


# -----------------------------
# Tests for extract_message_length
# -----------------------------
def test_extract_message_length_empty_message():
    """Test extracting length of empty message."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    # Embed empty message
    secret_message = ""
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    # Extract length
    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    # Empty message should have 0 bits
    assert extracted_length == 0


def test_extract_message_length_single_character():
    """Test extracting length of single character message."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "A"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    # "A" in UTF-8 = 8 bits
    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_short_message():
    """Test extracting length of short text message."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Hello"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_long_message():
    """Test extracting length of longer message."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = (
        "This is a longer test message with multiple words and punctuation!"
    )
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_with_smooth_pixels():
    """Test extraction with all smooth pixels (1 bit per channel)."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)  # All smooth
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Test message"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_with_rough_pixels():
    """Test extraction with all rough pixels (2 bits per channel)."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.ones((512, 512), dtype=np.uint8)  # All rough
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Test message"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_with_mixed_texture():
    """Test extraction with mixed smooth/rough texture map."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Mixed texture test"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_special_characters():
    """Test extraction with special characters in message."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "!@#$%^&*()"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_returns_integer():
    """Test that extracted length is always an integer."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Test"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    assert isinstance(extracted_length, int)


def test_extract_message_length_with_different_pixel_order():
    """Test that same pixel order gives consistent length extraction."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)

    # Use specific pixel order
    np.random.seed(42)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]
    np.random.shuffle(pixel_coords)

    secret_message = "Consistent extraction"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


def test_extract_message_length_large_value():
    """Test extraction with large message length (near 32-bit limit)."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.ones((512, 512), dtype=np.uint8)  # All rough for capacity
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    # Create a very long message
    secret_message = "X" * 10000  # Large message
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    extracted_length = extract_message_length(
        stego_img, classification_map, pixel_coords
    )

    expected_length = len(text_to_binary(secret_message))
    assert extracted_length == expected_length


# -----------------------------
# Tests for binary_to_text
# -----------------------------
def test_binary_to_text_basic():
    binary = "0100100001101001"  # "Hi"
    result = binary_to_text(binary)
    assert result == "Hi"


def test_binary_to_text_empty():
    assert binary_to_text("") == ""


def test_binary_to_text_invalid_length():
    # Not multiple of 8
    with pytest.raises(ValueError):
        binary_to_text("0101")


def test_extract_message_length():
    # Create fake 2x6 image (enough pixels)
    stego = np.zeros((2, 6, 3), dtype=np.uint8)

    # We'll manually embed 32 header bits using 1-bit per channel
    header_bits = format(8, "032b")

    bit_idx = 0
    coords = [(y, x) for y in range(2) for x in range(6)]
    classification_map = np.zeros((2, 6))  # all smooth (1 bit per channel)

    for y, x in coords:
        for c in range(3):
            if bit_idx >= 32:
                break
            stego[y, x, c] |= int(header_bits[bit_idx])
            bit_idx += 1
        if bit_idx >= 32:
            break

    length = extract_message_length(stego, classification_map, coords)

    assert length == 8


# -----------------------------
# Tests for extract_message
# -----------------------------
def test_extract_message_simple_roundtrip():
    """Test basic embed and extract roundtrip with independent LBP computation."""
    # Cover image setup
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    # Embed
    secret_message = "Hello, World!"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    # Extract with stego's own LBP (realistic scenario)
    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    # LBP should be identical after embedding (critical test!)
    assert np.array_equal(cover_classification, stego_classification), (
        "LBP classification changed after embedding - MSBs were modified!"
    )

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_empty_string():
    """Test extracting empty message."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = ""
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == ""


def test_extract_message_single_character():
    """Test extracting single character."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "A"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_long_text():
    """Test extracting longer message."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "The quick brown fox jumps over the lazy dog. " * 10
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_smooth_texture():
    """Test extraction with all smooth pixels."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)

    # Force all smooth for testing
    cover_classification[:] = 0
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Smooth texture test"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_rough_texture():
    """Test extraction with all rough pixels."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)

    # Force all rough for testing
    cover_classification[:] = 1
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Rough texture test"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_mixed_texture():
    """Test extraction with mixed smooth/rough texture."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Mixed texture message"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_special_characters():
    """Test extracting message with special characters."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Hello! @#$%^&*() 123 test?"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_multiline_text():
    """Test extracting multiline message."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Line 1\nLine 2\nLine 3\nLine 4"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_whitespace_preserved():
    """Test that whitespace is preserved in extraction."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "  leading and trailing  "
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_with_shuffled_pixels():
    """Test extraction with pseudorandom pixel order."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)

    # Shuffle pixel coordinates (simulating pseudorandom selection)
    np.random.seed(42)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]
    np.random.shuffle(pixel_coords)

    secret_message = "Pseudorandom pixel order"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message


def test_extract_message_lbp_consistency():
    """Test that LBP classification remains consistent after embedding."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "LBP consistency test"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    # Compute stego LBP independently
    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    # Classifications MUST be identical
    np.testing.assert_array_equal(
        cover_classification,
        stego_classification,
        err_msg="LBP classification changed after embedding - this breaks extraction!",
    )


def test_extract_message_returns_string():
    """Test that extraction always returns a string."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Type check"
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert isinstance(extracted, str)


def test_extract_message_json_like_content():
    """Test extracting JSON-like structured content."""
    cover_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cover_gray = img_to_grayscale(cover_img)
    cover_classification = compute_lbp_classification(cover_gray)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = '{"name": "test", "value": 123}'
    stego_img = embed_message(
        cover_img, secret_message, cover_classification, pixel_coords
    )

    stego_gray = img_to_grayscale(stego_img)
    stego_classification = compute_lbp_classification(stego_gray)

    extracted = extract_message(stego_img, stego_classification, pixel_coords)
    assert extracted == secret_message
