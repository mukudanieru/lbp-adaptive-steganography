import numpy as np
import pytest

from src.core.extraction import (
    extract_bits_from_pixel,
    binary_to_text,
    extract_message_length,
    extract_message,
)


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

    for (y, x) in coords:
        for c in range(3):
            if bit_idx >= 32:
                break
            stego[y, x, c] |= int(header_bits[bit_idx])
            bit_idx += 1
        if bit_idx >= 32:
            break

    length = extract_message_length(stego, classification_map, coords)

    assert length == 8


def test_extract_message_full_cycle():
    stego = np.zeros((4, 4, 3), dtype=np.uint8)
    classification_map = np.zeros((4, 4))  # smooth
    coords = [(y, x) for y in range(4) for x in range(4)]

    header = format(8, "032b")
    message_bits = "01000001"  # "A"
    payload = header + message_bits

    bit_idx = 0

    for (y, x) in coords:
        for c in range(3):
            if bit_idx >= len(payload):
                break
            stego[y, x, c] |= int(payload[bit_idx])
            bit_idx += 1
        if bit_idx >= len(payload):
            break

    result = extract_message(
        stego,
        password="unused_here",
        classification_map=classification_map,
        pixel_coords=coords,
    )

    assert result == "A"


def test_extract_message_full_cycle():
    stego = np.zeros((4, 4, 3), dtype=np.uint8)
    classification_map = np.zeros((4, 4))  # smooth
    coords = [(y, x) for y in range(4) for x in range(4)]

    header = format(8, "032b")
    message_bits = "01000001"  # "A"
    payload = header + message_bits

    bit_idx = 0

    for (y, x) in coords:
        for c in range(3):
            if bit_idx >= len(payload):
                break
            stego[y, x, c] |= int(payload[bit_idx])
            bit_idx += 1
        if bit_idx >= len(payload):
            break

    result = extract_message(
        stego,
        password="unused_here",
        classification_map=classification_map,
        pixel_coords=coords,
    )

    assert result == "A"
