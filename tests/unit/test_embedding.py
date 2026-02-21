from src.core.embedding import text_to_binary, calculate_capacity


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


def test_calculate_capacity():
    # Test all smooth pixels (1 bit per channel)
    # 4 smooth pixels × 3 channels × 1 bit = 12 bits
    assert calculate_capacity([0, 0, 0, 0]) == 12
    
    # Test all rough pixels (2 bits per channel)
    # 4 rough pixels × 3 channels × 2 bits = 24 bits
    assert calculate_capacity([1, 1, 1, 1]) == 24
    
    # Test mixed pixels
    # 2 smooth + 2 rough: 3 × (2×1 + 2×2) = 3 × 6 = 18 bits
    assert calculate_capacity([0, 1, 1, 0]) == 18
    
    # Test empty array
    assert calculate_capacity([]) == 0
    
    # Test single pixel
    assert calculate_capacity([0]) == 3   # smooth: 3 × 1 = 3
    assert calculate_capacity([1]) == 6   # rough: 3 × 2 = 6
    
    # Test custom number of channels (e.g., grayscale)
    assert calculate_capacity([0, 0, 1, 1], num_channels=1) == 6  # 1 × (2×1 + 2×2) = 6