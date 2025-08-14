import numpy as np
from metrics.entropy import shannon_entropy_bits_per_bit
from metrics.compression import compression_ratio
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


def test_entropy_degenerate_and_balanced():
    """Entropy sanity checks."""
    # All zeros - very low entropy
    bits = np.zeros(10_000, dtype=np.uint8)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert entropy == 0.0, f"All zeros should give entropy=0, got {entropy}"

    # Perfect alternating - high entropy (1.0)
    alt = np.tile([0, 1], 50_000).astype(np.uint8)
    entropy = shannon_entropy_bits_per_bit(alt)
    assert entropy == 1.0, f"Perfect alternating should give entropy=1, got {entropy}"

    # Random data - should be close to 1.0
    rng = np.random.default_rng(42)
    random_bits = rng.integers(0, 2, size=100_000, dtype=np.uint8)
    entropy = shannon_entropy_bits_per_bit(random_bits)
    assert entropy > 0.98, f"Random data should have entropy>0.98, got {entropy}"


def test_entropy_different_generators():
    """Entropy stays within [0,1] across sources."""
    # LCG
    bits = bits_from_source(LCG(seed=42), 100_000)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert 0.0 <= entropy <= 1.0, f"Invalid entropy {entropy} for LCG"

    # XorShift
    bits = bits_from_source(XorShift32(seed=42), 100_000)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert 0.0 <= entropy <= 1.0, f"Invalid entropy {entropy} for XorShift"

    # OS Random
    bits = bits_from_source(OSRandom(), 100_000)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert 0.0 <= entropy <= 1.0, f"Invalid entropy {entropy} for OS Random"


def test_compression_random_vs_structured():
    """Compression ratio sanity checks."""
    # Random data - should be incompressible
    rng = np.random.default_rng(42)
    random_bytes = rng.integers(0, 256, size=10_000, dtype=np.uint8).tobytes()
    cr = compression_ratio(random_bytes)
    assert cr > 0.9, f"Random data should have cr>0.9, got {cr}"

    # All zeros - highly compressible
    zeros_bytes = np.zeros(10_000, dtype=np.uint8).tobytes()
    cr = compression_ratio(zeros_bytes)
    assert cr < 0.1, f"All zeros should have cr<0.1, got {cr}"


def test_compression_different_generators():
    """Compression signal across generators; use LSB-first packing."""
    # LCG - tends to be more compressible
    bits = bits_from_source(LCG(seed=42), 100_000)
    bytes_data = np.packbits(bits, bitorder="little").tobytes()
    cr = compression_ratio(bytes_data)
    assert cr < 0.5, f"LCG should be compressible (cr<0.5), got {cr}"

    # XorShift - closer to incompressible
    bits = bits_from_source(XorShift32(seed=42), 100_000)
    bytes_data = np.packbits(bits, bitorder="little").tobytes()
    cr = compression_ratio(bytes_data)
    assert cr > 0.8, f"XorShift should be less compressible (cr>0.8), got {cr}"

    # OS Random - incompressible
    bits = bits_from_source(OSRandom(), 100_000)
    bytes_data = np.packbits(bits, bitorder="little").tobytes()
    cr = compression_ratio(bytes_data)
    assert cr > 0.8, f"OS Random should be incompressible (cr>0.8), got {cr}"


def test_compression_edge_cases():
    """Compression edge cases."""
    # Empty bytes - compression ratio is 0.0 for empty data
    cr = compression_ratio(b"")
    assert cr == 0.0, f"Empty bytes should give cr=0, got {cr}"

    # Single byte - ratio can vary; should be non-negative
    cr = compression_ratio(b"\x00")
    assert cr >= 0.0, f"Single byte should give cr>=0, got {cr}"
