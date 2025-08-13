import numpy as np
from metrics.block_frequency import block_frequency_pvalue
from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom


def test_lcg_block_frequency_not_extreme():
    """Test that LCG block frequency is not extremely biased."""
    # LCG should not be extremely biased
    bits = bits_from_source(LCG(seed=42), 50_000)
    p = block_frequency_pvalue(bits, M=128)
    assert p > 0.001, f"LCG block frequency p={p} is too extreme"


def test_xorshift_block_frequency_reasonable():
    """Test that XorShift block frequency is reasonable."""
    # XorShift should be much better than LCG
    bits = bits_from_source(XorShift32(seed=42), 50_000)
    p = block_frequency_pvalue(bits, M=128)
    assert p > 0.001, f"XorShift block frequency p={p} is too extreme"


def test_os_random_block_frequency_reasonable():
    """Test that OS random block frequency is reasonable."""
    # OS random should be cryptographically secure
    bits = bits_from_source(OSRandom(), 50_000)
    p = block_frequency_pvalue(bits, M=128)
    assert p > 0.001, f"OS random block frequency p={p} is too extreme"


def test_block_frequency_different_block_sizes():
    """Test block frequency with different block sizes."""
    # Test different block sizes
    bits = bits_from_source(LCG(seed=42), 100_000)
    for M in [64, 128, 256]:
        p = block_frequency_pvalue(bits, M=M)
        assert 0 <= p <= 1, f"Invalid p-value {p} for M={M}"


def test_block_frequency_edge_cases():
    """Test block frequency with edge cases."""
    # All zeros
    bits = np.zeros(10_000, dtype=np.uint8)
    p = block_frequency_pvalue(bits, M=128)
    assert p == 0.0, f"All zeros should give p=0, got {p}"

    # All ones
    bits = np.ones(10_000, dtype=np.uint8)
    p = block_frequency_pvalue(bits, M=128)
    assert p == 0.0, f"All ones should give p=0, got {p}"

    # Perfect alternating pattern - balanced and should pass
    bits = np.tile([0, 1], 5_000).astype(np.uint8)
    p = block_frequency_pvalue(bits, M=128)
    assert p > 0.1, f"Perfect alternating should pass (p>0.1), got {p}"


def bits_from_source(src, n_bits):
    """Helper function to get bits from source."""
    if hasattr(src, "next_bytes"):
        n_bytes = (n_bits + 7) // 8
        raw = src.next_bytes(n_bytes)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bits = np.unpackbits(arr)[:n_bits]
        return bits.astype(np.uint8)
    return np.fromiter(
        (src.next_bit() for _ in range(n_bits)), count=n_bits, dtype=np.uint8
    )
