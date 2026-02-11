from src.core.pseudorandom import password_to_seed
import pytest


# --- Tests for password_to_seed ---
def test_password_to_seed():
    assert password_to_seed("mypassword") == 9934964102539539065


def test_password_to_seed_determinism():
    assert password_to_seed("secret_password") == password_to_seed("secret_password")


def test_password_to_seed_sensitivity():
    assert password_to_seed("secret_password") != password_to_seed("Secret_password")


def test_password_to_seed_output_type():
    """Ensures the output is a 64-bit integer."""
    seed = password_to_seed("my_password")
    assert isinstance(seed, int)

    # Check if it fits in 64 bits (unsigned)
    assert 0 <= seed < 2**64


def test_password_to_seed_non_string():
    with pytest.raises(TypeError):
        password_to_seed(12345)


def test_password_to_seed_empty_string():
    with pytest.raises(ValueError):
        password_to_seed("")
