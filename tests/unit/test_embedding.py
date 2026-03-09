from src.core.embedding import (
    text_to_binary,
    get_binary_header,
    calculate_capacity,
    embed_bits_in_pixel,
    embed_message,
)
import numpy as np


# -----------------------------
# Tests for text_to_binary
# -----------------------------
def test_text_to_binary():
    # Test basic ASCII characters
    assert text_to_binary("A") == "01000001"
    assert text_to_binary("Hi") == "0100100001101001"

    # Test empty string
    assert text_to_binary("") == ""

    # Test space and special characters
    assert text_to_binary(" ") == "00100000"
    assert text_to_binary("!") == "00100001"

    # Test longer message
    result = text_to_binary("Hello")
    assert len(result) == 40  # 5 characters × 8 bits
    assert result == "0100100001100101011011000110110001101111"


# -----------------------------
# Tests for get_binary_header
# -----------------------------
def test_get_binary_header_empty_message():
    """Test header generation for empty binary message."""
    binary_message = ""
    header = get_binary_header(binary_message)
    assert len(header) == 32
    assert header == "00000000000000000000000000000000"
    assert int(header, 2) == 0


def test_get_binary_header_single_bit():
    """Test header generation for single bit message."""
    binary_message = "1"
    header = get_binary_header(binary_message)
    assert len(header) == 32
    assert int(header, 2) == 1


def test_get_binary_header_small_message():
    """Test header generation for small binary message."""
    binary_message = "10101010"  # 8 bits
    header = get_binary_header(binary_message)
    assert len(header) == 32
    assert int(header, 2) == 8


def test_get_binary_header_medium_message():
    """Test header generation for medium-sized message."""
    binary_message = "1" * 256  # 256 bits
    header = get_binary_header(binary_message)
    assert len(header) == 32
    assert int(header, 2) == 256


def test_get_binary_header_large_message():
    """Test header generation for large binary message."""
    binary_message = "0" * 10000  # 10,000 bits
    header = get_binary_header(binary_message)
    assert len(header) == 32
    assert int(header, 2) == 10000


def test_get_binary_header_max_32bit_value():
    """Test header generation at maximum 32-bit unsigned integer."""
    binary_message = "1" * 4294967295  # Max value for 32-bit unsigned int
    header = get_binary_header(binary_message)
    assert len(header) == 32
    assert int(header, 2) == 4294967295


def test_get_binary_header_consistency():
    """Test that same length produces same header."""
    binary_message1 = "1" * 100
    binary_message2 = "0" * 100
    header1 = get_binary_header(binary_message1)
    header2 = get_binary_header(binary_message2)
    assert header1 == header2


def test_get_binary_header_mixed_content():
    """Test header with mixed binary content."""
    binary_message = "110010101001" * 10  # 120 bits
    header = get_binary_header(binary_message)
    assert len(header) == 32
    assert int(header, 2) == 120


# -----------------------------
# Tests for calculate_capacity
# -----------------------------
FIXED_IMAGE_SIZE = 512 * 512


def test_calculate_capacity_all_smooth():
    """Test capacity calculation when all pixels are smooth."""
    classification_map = np.zeros(FIXED_IMAGE_SIZE, dtype=np.uint8)  # All smooth (0)
    capacity = calculate_capacity(classification_map, num_channels=3)
    expected = 3 * (262144 * 1 + 0 * 2)  # 3 * 262144 = 786,432 bits
    assert capacity == expected
    assert capacity == 786432


def test_calculate_capacity_all_rough():
    """Test capacity calculation when all pixels are rough."""
    classification_map = np.ones(FIXED_IMAGE_SIZE, dtype=np.uint8)  # All rough (1)
    capacity = calculate_capacity(classification_map, num_channels=3)
    expected = 3 * (0 * 1 + 262144 * 2)  # 3 * 524288 = 1,572,864 bits
    assert capacity == expected
    assert capacity == 1572864


def test_calculate_capacity_half_half():
    """Test capacity calculation with equal smooth and rough pixels."""
    classification_map = np.zeros(FIXED_IMAGE_SIZE, dtype=np.uint8)
    classification_map[: (FIXED_IMAGE_SIZE) // 2] = 1  # Half rough, half smooth
    capacity = calculate_capacity(classification_map, num_channels=3)
    expected = 3 * (131072 * 1 + 131072 * 2)  # 3 * 393216 = 1,179,648 bits
    assert capacity == expected
    assert capacity == 1179648


def test_calculate_capacity_single_channel():
    """Test capacity calculation with single channel (grayscale)."""
    classification_map = np.zeros(FIXED_IMAGE_SIZE, dtype=np.uint8)
    classification_map[::2] = 1  # Alternating pattern
    num_smooth = np.sum(classification_map == 0)
    num_rough = np.sum(classification_map == 1)
    capacity = calculate_capacity(classification_map, num_channels=1)
    expected = 1 * (num_smooth * 1 + num_rough * 2)
    assert capacity == expected


def test_calculate_capacity_four_channels():
    """Test capacity calculation with four channels (RGBA)."""
    classification_map = np.ones(FIXED_IMAGE_SIZE, dtype=np.uint8)
    capacity = calculate_capacity(classification_map, num_channels=4)
    expected = 4 * (0 * 1 + 262144 * 2)  # 4 * 524288 = 2,097,152 bits
    assert capacity == expected
    assert capacity == 2097152


def test_calculate_capacity_mostly_smooth():
    """Test capacity with 90% smooth, 10% rough."""
    classification_map = np.zeros(FIXED_IMAGE_SIZE, dtype=np.uint8)
    num_rough = int(0.1 * FIXED_IMAGE_SIZE)
    classification_map[:num_rough] = 1
    capacity = calculate_capacity(classification_map, num_channels=3)
    num_smooth = 262144 - num_rough
    expected = 3 * (num_smooth * 1 + num_rough * 2)
    assert capacity == expected


def test_calculate_capacity_mostly_rough():
    """Test capacity with 90% rough, 10% smooth."""
    classification_map = np.ones(FIXED_IMAGE_SIZE, dtype=np.uint8)
    num_smooth = int(0.1 * FIXED_IMAGE_SIZE)
    classification_map[:num_smooth] = 0
    capacity = calculate_capacity(classification_map, num_channels=3)
    num_rough = 262144 - num_smooth
    expected = 3 * (num_smooth * 1 + num_rough * 2)
    assert capacity == expected


def test_calculate_capacity_small_image():
    """Test capacity calculation with smaller image."""
    classification_map = np.zeros(100 * 100, dtype=np.uint8)  # 10,000 pixels
    classification_map[:5000] = 1  # Half rough
    capacity = calculate_capacity(classification_map, num_channels=3)
    expected = 3 * (5000 * 1 + 5000 * 2)  # 3 * 15000 = 45,000 bits
    assert capacity == expected
    assert capacity == 45000


def test_calculate_capacity_returns_integer():
    """Test that capacity is always an integer."""
    classification_map = np.random.randint(0, 2, size=FIXED_IMAGE_SIZE, dtype=np.uint8)
    capacity = calculate_capacity(classification_map, num_channels=3)
    assert isinstance(capacity, (int, np.integer))


def test_calculate_capacity_checkerboard_pattern():
    """Test capacity with checkerboard pattern."""
    classification_map = np.zeros(FIXED_IMAGE_SIZE, dtype=np.uint8)
    classification_map[::2] = 1  # Alternating 0, 1, 0, 1...
    num_smooth = np.sum(classification_map == 0)
    num_rough = np.sum(classification_map == 1)
    capacity = calculate_capacity(classification_map, num_channels=3)
    expected = 3 * (num_smooth * 1 + num_rough * 2)
    assert capacity == expected


# -----------------------------
# Tests for embed_bits_in_pixel
# -----------------------------
def test_embed_bits_single_bit_per_channel():
    """Test embedding 1 bit per channel (3 bits total)."""
    rgb_img = np.array([226, 137, 125], dtype=np.uint8)
    bits = "101"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=1)

    # Check LSB of each channel matches embedded bits
    assert result[0] & 1 == 1  # R channel LSB = 1
    assert result[1] & 1 == 0  # G channel LSB = 0
    assert result[2] & 1 == 1  # B channel LSB = 1

    # Check only LSB changed (difference must be ≤1 per channel)
    assert abs(int(result[0]) - int(rgb_img[0])) <= 1
    assert abs(int(result[1]) - int(rgb_img[1])) <= 1
    assert abs(int(result[2]) - int(rgb_img[2])) <= 1


def test_embed_bits_two_bits_per_channel():
    """Test embedding 2 bits per channel (6 bits total)."""
    rgb_img = np.array([228, 140, 127], dtype=np.uint8)
    bits = "101101"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=2)

    # Check last 2 bits of each channel
    assert result[0] & 0b11 == 0b10  # R channel last 2 bits = 10
    assert result[1] & 0b11 == 0b11  # G channel last 2 bits = 11
    assert result[2] & 0b11 == 0b01  # B channel last 2 bits = 01

    # Check only 2 LSBs changed (difference must be ≤3 per channel)
    assert abs(int(result[0]) - int(rgb_img[0])) <= 3
    assert abs(int(result[1]) - int(rgb_img[1])) <= 3
    assert abs(int(result[2]) - int(rgb_img[2])) <= 3


def test_embed_bits_preserves_msb():
    """Test that Most Significant Bits are preserved."""
    rgb_img = np.array([255, 128, 64], dtype=np.uint8)
    bits = "111"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=1)

    # MSBs (top 7 bits) should remain unchanged
    assert (result[0] >> 1) == (rgb_img[0] >> 1)
    assert (result[1] >> 1) == (rgb_img[1] >> 1)
    assert (result[2] >> 1) == (rgb_img[2] >> 1)


def test_embed_bits_all_zeros():
    """Test embedding all zero bits."""
    rgb_img = np.array([255, 255, 255], dtype=np.uint8)
    bits = "000"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=1)

    # All LSBs should be 0
    assert result[0] & 1 == 0
    assert result[1] & 1 == 0
    assert result[2] & 1 == 0


def test_embed_bits_all_ones():
    """Test embedding all one bits."""
    rgb_img = np.array([0, 0, 0], dtype=np.uint8)
    bits = "111"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=1)

    # All LSBs should be 1
    assert result[0] & 1 == 1
    assert result[1] & 1 == 1
    assert result[2] & 1 == 1


def test_embed_bits_no_change_needed():
    """Test when LSBs already match target bits."""
    rgb_img = np.array([227, 136, 125], dtype=np.uint8)  # LSBs: 1, 0, 1
    bits = "101"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=1)

    # Should remain unchanged
    np.testing.assert_array_equal(result, rgb_img)


def test_embed_bits_returns_uint8():
    """Test that result maintains uint8 dtype."""
    rgb_img = np.array([100, 150, 200], dtype=np.uint8)
    bits = "111"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=1)

    assert result.dtype == np.uint8


def test_embed_bits_boundary_values():
    """Test embedding at boundary pixel values (0 and 255)."""
    rgb_img = np.array([0, 255, 128], dtype=np.uint8)
    bits = "101"
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=1)

    # Check values stay within valid range
    assert 0 <= result[0] <= 255
    assert 0 <= result[1] <= 255
    assert 0 <= result[2] <= 255

    # Check LSBs are correct
    assert result[0] & 1 == 1
    assert result[1] & 1 == 0
    assert result[2] & 1 == 1


def test_embed_bits_two_bits_pattern():
    """Test specific 2-bit patterns."""
    rgb_img = np.array([252, 252, 252], dtype=np.uint8)  # Binary: 11111100 each
    bits = "001110"  # 00, 11, 10
    result = embed_bits_in_pixel(rgb_img, bits, num_bits=2)

    assert result[0] & 0b11 == 0b00
    assert result[1] & 0b11 == 0b11
    assert result[2] & 0b11 == 0b10


# -----------------------------
# Tests for embed_message
# -----------------------------
def test_embed_message_basic_integration():
    """Test basic message embedding completes without error."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

    # Generate pixel coordinates
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Hello, World!"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    # Check output shape and dtype are preserved
    assert stego_img.shape == rgb_img.shape
    assert stego_img.dtype == np.uint8


def test_embed_message_minimal_change():
    """Test that embedding causes minimal visual change."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)  # All smooth
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Test"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    # With 1-bit embedding, max change per pixel is ±1 per channel
    diff = np.abs(stego_img.astype(int) - rgb_img.astype(int))
    assert np.all(diff <= 1)


def test_embed_message_empty_string():
    """Test embedding empty message."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = ""
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    # Should still work (only header embedded)
    assert stego_img.shape == rgb_img.shape
    assert stego_img.dtype == np.uint8


def test_embed_message_single_character():
    """Test embedding single character."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "A"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    assert stego_img.shape == rgb_img.shape
    assert stego_img.dtype == np.uint8


def test_embed_message_only_affects_needed_pixels():
    """Test that only necessary pixels are modified."""
    rgb_img = np.full((512, 512, 3), 128, dtype=np.uint8)  # Uniform image
    classification_map = np.zeros(
        (512, 512), dtype=np.uint8
    )  # All smooth (1 bit/channel)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Hi"  # Short message
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    # Count how many pixels changed
    changed_pixels = np.any(stego_img != rgb_img, axis=2)
    num_changed = np.sum(changed_pixels)

    # Should change a small number of pixels (header + message)
    # "Hi" = 16 bits + 32-bit header = 48 bits total
    # At 3 bits per pixel (smooth), need ~16 pixels
    assert num_changed > 0  # Something changed
    assert num_changed < 100  # But not too many


def test_embed_message_preserves_unmodified_pixels():
    """Test that pixels beyond embedding range are unchanged."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.zeros((512, 512), dtype=np.uint8)

    # Only use first 100 pixels
    pixel_coords = [(y, x) for y in range(10) for x in range(10)]

    secret_message = "X"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    # Pixels outside the first 100 should be unchanged
    # Check bottom-right corner
    np.testing.assert_array_equal(stego_img[400:, 400:, :], rgb_img[400:, 400:, :])


def test_embed_message_different_classification_maps():
    """Test embedding with all-smooth vs all-rough classification."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]
    secret_message = "Test message"

    # All smooth (1 bit per channel)
    classification_smooth = np.zeros((512, 512), dtype=np.uint8)
    stego_smooth = embed_message(
        rgb_img.copy(), secret_message, classification_smooth, pixel_coords
    )

    # All rough (2 bits per channel)
    classification_rough = np.ones((512, 512), dtype=np.uint8)
    stego_rough = embed_message(
        rgb_img.copy(), secret_message, classification_rough, pixel_coords
    )

    # Both should succeed
    assert stego_smooth.shape == rgb_img.shape
    assert stego_rough.shape == rgb_img.shape

    # Rough should modify fewer pixels (higher capacity per pixel)
    changed_smooth = np.sum(np.any(stego_smooth != rgb_img, axis=2))
    changed_rough = np.sum(np.any(stego_rough != rgb_img, axis=2))
    assert changed_rough < changed_smooth


def test_embed_message_special_characters():
    """Test embedding message with special characters."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Hello! @#$%^&*() 123"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    assert stego_img.shape == rgb_img.shape
    assert stego_img.dtype == np.uint8


def test_embed_message_unicode_characters():
    """Test embedding message with unicode characters."""
    rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    classification_map = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    pixel_coords = [(y, x) for y in range(512) for x in range(512)]

    secret_message = "Hello 世界 🌍"
    stego_img = embed_message(rgb_img, secret_message, classification_map, pixel_coords)

    assert stego_img.shape == rgb_img.shape
    assert stego_img.dtype == np.uint8
