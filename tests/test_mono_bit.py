import numpy as np
from metrics.mono_bit import mono_bit_pvalue
from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom


def bits_from_source(src, n_bits):
    """Helper: get n_bits as 0/1, LSB-first within bytes."""
    if hasattr(src, "next_bytes"):
        n_bytes = (n_bits + 7) // 8
        raw = src.next_bytes(n_bytes)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bits = np.unpackbits(arr, bitorder="little")[:n_bits]
        return bits.astype(np.uint8)
    return np.fromiter(
        (src.next_bit() for _ in range(n_bits)), count=n_bits, dtype=np.uint8
    )


def test_lcg_mono_bit():
    """LCG mono_bit shouldn't look wildly imbalanced."""
    bits = bits_from_source(LCG(seed=42), 100_000)
    p = mono_bit_pvalue(bits)
    assert p > 0.0001, f"LCG mono_bit p={p} is too extreme"


def test_xorshift_mono_bit():
    """XorShift mono_bit should be reasonable."""
    bits = bits_from_source(XorShift32(seed=42), 100_000)
    p = mono_bit_pvalue(bits)
    assert p > 0.001, f"XorShift mono_bit p={p} is too extreme"


def test_os_random_mono_bit_stable_enough():
    """
    OS RNG mono_bit occasionally yields small p by chance.
    Try several batches and require at least one comfortably above the cutoff.
    """
    ps = []
    for _ in range(5):
        bits = bits_from_source(OSRandom(), 100_000)
        ps.append(mono_bit_pvalue(bits))
    assert max(ps) > 0.001, f"All OS random mono_bit p-values were too extreme: {ps}"


def test_mono_bit_edge_cases():
    """Mono_bit edge cases."""
    # All zeros
    bits = np.zeros(10_000, dtype=np.uint8)
    p = mono_bit_pvalue(bits)
    assert p == 0.0, f"All zeros should give p=0, got {p}"

    # All ones
    bits = np.ones(10_000, dtype=np.uint8)
    p = mono_bit_pvalue(bits)
    assert p == 0.0, f"All ones should give p=0, got {p}"

    # Perfect alternating pattern - balanced and should pass
    bits = np.tile([0, 1], 5_000).astype(np.uint8)
    p = mono_bit_pvalue(bits)
    assert p > 0.1, f"Perfect alternating should pass (p>0.1), got {p}"
