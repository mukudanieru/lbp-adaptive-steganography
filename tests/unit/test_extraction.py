import numpy as np

from src.core.embedding import embed_message, text_to_binary
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
def test_extract_bits_from_pixel_one_bit():
    """Test extracting 1 bit per channel from R&B only."""
    pixel = [227, 136, 124]  # R=227 (LSB=1), G=136 (skipped), B=124 (LSB=0)
    bits = extract_bits_from_pixel(pixel, num_bits=1)
    assert bits == "10"  # R LSB + B LSB


def test_extract_bits_from_pixel_two_bits():
    """Test extracting 2 bits per channel from R&B only."""
    pixel = [
        226,
        137,
        125,
    ]  # R=226 (last 2 bits=10), G=137 (skipped), B=125 (last 2 bits=01)
    bits = extract_bits_from_pixel(pixel, num_bits=2)
    assert bits == "1001"  # R 2 LSBs + B 2 LSBs


def test_extract_bits_from_pixel_all_zeros():
    """Test extracting bits when all LSBs are zero."""
    pixel = [224, 100, 128]  # R=224 (LSB=0), B=128 (LSB=0)
    bits = extract_bits_from_pixel(pixel, num_bits=1)
    assert bits == "00"


def test_extract_bits_from_pixel_all_ones():
    """Test extracting bits when all LSBs are one."""
    pixel = [225, 100, 129]  # R=225 (LSB=1), B=129 (LSB=1)
    bits = extract_bits_from_pixel(pixel, num_bits=1)
    assert bits == "11"


def test_extract_bits_from_pixel_green_ignored():
    """Test that green channel is completely ignored."""
    pixel1 = [227, 0, 124]
    pixel2 = [227, 255, 124]

    bits1 = extract_bits_from_pixel(pixel1, num_bits=1)
    bits2 = extract_bits_from_pixel(pixel2, num_bits=1)

    # Should be identical despite different green values
    assert bits1 == bits2 == "10"


def test_extract_bits_from_pixel_boundary_values():
    """Test extracting from boundary pixel values (0 and 255)."""
    pixel = [0, 128, 255]  # R=0 (LSB=0), B=255 (LSB=1)
    bits = extract_bits_from_pixel(pixel, num_bits=1)
    assert bits == "01"


def test_extract_bits_from_pixel_two_bits_pattern():
    """Test extracting 2-bit pattern."""
    pixel = [252, 100, 253]  # R=252 (00), B=253 (01)
    bits = extract_bits_from_pixel(pixel, num_bits=2)
    assert bits == "0001"


def test_extract_bits_from_pixel_returns_string():
    """Test that extraction returns a string."""
    pixel = [100, 150, 200]
    bits = extract_bits_from_pixel(pixel, num_bits=1)
    assert isinstance(bits, str)


def test_extract_bits_from_pixel_correct_length():
    """Test that extracted bits have correct length."""
    pixel = [123, 45, 67]

    bits_1 = extract_bits_from_pixel(pixel, num_bits=1)
    assert len(bits_1) == 2  # 1 bit × 2 channels

    bits_2 = extract_bits_from_pixel(pixel, num_bits=2)
    assert len(bits_2) == 4  # 2 bits × 2 channels


def test_extract_bits_from_pixel_binary_format():
    """Test that extracted bits are valid binary string."""
    pixel = [100, 150, 200]
    bits = extract_bits_from_pixel(pixel, num_bits=2)

    # Should only contain '0' and '1'
    assert all(bit in "01" for bit in bits)
    # Should be able to convert to int
    assert isinstance(int(bits, 2), int)


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
def test_binary_to_text_simple():
    """Test converting simple binary string to text."""
    binary = "01001000"  # 'H'
    text = binary_to_text(binary)
    assert text == "H"


def test_binary_to_text_multiple_chars():
    """Test converting multiple characters."""
    binary = "0100100001101001"  # "Hi"
    text = binary_to_text(binary)
    assert text == "Hi"


def test_binary_to_text_empty_string():
    """Test converting empty binary string."""
    binary = ""
    text = binary_to_text(binary)
    assert text == ""


def test_binary_to_text_full_word():
    """Test converting full word."""
    # "Hello" = 01001000 01100101 01101100 01101100 01101111
    binary = "0100100001100101011011000110110001101111"
    text = binary_to_text(binary)
    assert text == "Hello"


def test_binary_to_text_with_space():
    """Test converting text with space."""
    # "A B" = 01000001 00100000 01000010
    binary = "010000010010000001000010"
    text = binary_to_text(binary)
    assert text == "A B"


def test_binary_to_text_special_chars():
    """Test converting special characters."""
    # "!" = 00100001, "?" = 00111111
    binary = "0010000100111111"
    text = binary_to_text(binary)
    assert text == "!?"


def test_binary_to_text_numbers():
    """Test converting numeric characters."""
    # "123" = 00110001 00110010 00110011
    binary = "001100010011001000110011"
    text = binary_to_text(binary)
    assert text == "123"


def test_binary_to_text_returns_string():
    """Test that result is always a string."""
    binary = "01001000"
    text = binary_to_text(binary)
    assert isinstance(text, str)


def test_binary_to_text_newline():
    """Test converting newline character."""
    # "\n" = 00001010
    binary = "00001010"
    text = binary_to_text(binary)
    assert text == "\n"


def test_binary_to_text_mixed_content():
    """Test converting mixed alphanumeric and symbols."""
    # "A1!" = 01000001 00110001 00100001
    binary = "010000010011000100100001"
    text = binary_to_text(binary)
    assert text == "A1!"


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
