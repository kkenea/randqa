import numpy as np
from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom
from main import bits_from_source


def test_bits_from_source_shape_and_values():
    """Test that bits_from_source produces correct shape and values."""
    n = 10_000
    bits = bits_from_source(LCG(seed=1), n)
    assert bits.shape == (n,)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)).issubset({0, 1})


def test_bits_from_source_multiple_generators():
    """Test bits_from_source with different generators."""
    n = 5_000

    # Test LCG
    lcg_bits = bits_from_source(LCG(seed=42), n)
    assert lcg_bits.shape == (n,)
    assert lcg_bits.dtype == np.uint8

    # Test XorShift
    xorshift_bits = bits_from_source(XorShift32(seed=42), n)
    assert xorshift_bits.shape == (n,)
    assert xorshift_bits.dtype == np.uint8

    # Test OS Random
    os_bits = bits_from_source(OSRandom(), n)
    assert os_bits.shape == (n,)
    assert os_bits.dtype == np.uint8


def test_bits_from_source_edge_cases():
    """Test bits_from_source with edge cases."""
    # Zero bits
    bits = bits_from_source(LCG(seed=1), 0)
    assert bits.shape == (0,)
    assert bits.dtype == np.uint8

    # Single bit
    bits = bits_from_source(LCG(seed=1), 1)
    assert bits.shape == (1,)
    assert bits.dtype == np.uint8
    assert bits[0] in [0, 1]

    # Large number of bits
    bits = bits_from_source(LCG(seed=1), 100_000)
    assert bits.shape == (100_000,)
    assert bits.dtype == np.uint8


def test_bits_from_source_consistency():
    """Test that bits_from_source produces consistent results for same seed."""
    n = 1_000
    gen1 = LCG(seed=123)
    gen2 = LCG(seed=123)

    bits1 = bits_from_source(gen1, n)
    bits2 = bits_from_source(gen2, n)

    np.testing.assert_array_equal(bits1, bits2, "Same seed should produce same bits")
