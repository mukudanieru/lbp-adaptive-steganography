from src.core.pseudorandom import password_to_seed, generate_pixel_coordinates
import pytest


# --- Tests for password_to_seed ---
def test_password_to_seed():
    """
    Tests against your specific example: 'mypassword' → 9934964102539539065
    This ensures your SHA-256 and byte-to-int conversion logic is correct.
    """

    assert password_to_seed("mypassword") == 9934964102539539065


def test_password_to_seed_determinism():
    """Tests that the same password always produces the same seed."""

    assert password_to_seed("secret_password") == password_to_seed("secret_password")


def test_password_to_seed_sensitivity():
    """Tests that a small change in password results in a different seed."""

    assert password_to_seed("secret_password") != password_to_seed("Secret_password")


def test_password_to_seed_output_type():
    """Ensures the output is a 64-bit integer."""

    seed = password_to_seed("my_password")
    assert isinstance(seed, int)

    # Check if it fits in 64 bits (unsigned)
    assert 0 <= seed < 2**64


def test_password_to_seed_non_string():
    """
    Ensures a TypeError is raised when the input is not a string.
    This validates proper type checking for the password parameter.
    """

    with pytest.raises(TypeError):
        password_to_seed(12345)


def test_password_to_seed_empty_string():
    """
    Ensures a ValueError is raised when an empty string is provided.
    This validates that empty passwords are explicitly rejected.
    """

    with pytest.raises(ValueError):
        password_to_seed("")


# --- Tests for generate_pixel_coordinates ---
def test_pixel_gen_completeness():
    """Ensures every pixel is present in the list."""

    h, w = 512, 512
    coords = generate_pixel_coordinates(h, w, 9934964102539539065)
    assert len(coords) == h * w

    # Check if all coordinates are within bounds
    for y, x in coords:
        assert 0 <= y < h
        assert 0 <= x < w


def test_pixel_gen_uniqueness():
    """Ensures no pixel is selected twice."""

    h, w = 512, 512
    coords = generate_pixel_coordinates(h, w, 9934964102539539065)
    assert len(set(coords)) == len(coords)


def test_pixel_gen_determinism():
    """Ensures same seed produces the same shuffle sequence."""

    h, w, seed = 512, 512, 9934964102539539065
    sequence_1 = generate_pixel_coordinates(h, w, seed)
    sequence_2 = generate_pixel_coordinates(h, w, seed)
    assert sequence_1 == sequence_2


def test_pixel_gen_different_seeds():
    """Ensures different seeds produce different shuffle sequences."""

    h, w = 512, 512
    seq_a = generate_pixel_coordinates(h, w, 9934964102539539065)
    seq_b = generate_pixel_coordinates(h, w, 9934964102539539064)
    assert seq_a != seq_b


@pytest.mark.parametrize("h, w", [(1, 1), (1, 10), (10, 1)])
def test_pixel_gen_edge_cases(h, w):
    """Checks behavior for thin or single-pixel images."""

    coords = generate_pixel_coordinates(h, w, 9934964102539539065)
    assert len(coords) == h * w
