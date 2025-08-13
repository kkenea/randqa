import numpy as np
from metrics.monobit import monobit_pvalue
from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom


def test_lcg_monobit():
    """Test that LCG monobit test behavior is reasonable."""
    bits = bits_from_source(LCG(seed=42), 100_000)
    p = monobit_pvalue(bits)

    # Fail only if p is very small (highly unlikely imbalance)
    assert p > 0.0001, f"LCG monobit p={p} is too extreme"


def test_xorshift_monobit():
    """Test that XorShift monobit test is reasonable."""
    # XorShift should be better than LCG but may still have some bias
    bits = bits_from_source(XorShift32(seed=42), 100_000)
    p = monobit_pvalue(bits)
    assert p > 0.001, f"XorShift monobit p={p} is too extreme"


def test_os_random_monobit():
    """Test that OS random monobit test is reasonable."""
    # OS random should be cryptographically secure
    bits = bits_from_source(OSRandom(), 100_000)
    p = monobit_pvalue(bits)
    assert p > 0.001, f"OS random monobit p={p} is too extreme"


def test_monobit_edge_cases():
    """Test monobit test with edge cases."""
    # All zeros
    bits = np.zeros(10_000, dtype=np.uint8)
    p = monobit_pvalue(bits)
    assert p == 0.0, f"All zeros should give p=0, got {p}"

    # All ones
    bits = np.ones(10_000, dtype=np.uint8)
    p = monobit_pvalue(bits)
    assert p == 0.0, f"All ones should give p=0, got {p}"

    # Perfect alternating pattern - this is actually balanced and should pass
    bits = np.tile([0, 1], 5_000).astype(np.uint8)
    p = monobit_pvalue(bits)
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
