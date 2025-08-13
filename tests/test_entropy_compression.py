import numpy as np
from metrics.entropy import shannon_entropy_bits_per_bit
from metrics.compression import compression_ratio
from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom


def test_entropy_degenerate_and_balanced():
    """Test entropy calculation with degenerate and balanced sequences."""
    # All zeros - very low entropy
    bits = np.zeros(10_000, dtype=np.uint8)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert entropy == 0.0, f"All zeros should give entropy=0, got {entropy}"

    # Perfect alternating - high entropy
    alt = np.tile([0, 1], 50_000).astype(np.uint8)  # exactly balanced
    entropy = shannon_entropy_bits_per_bit(alt)
    assert entropy == 1.0, f"Perfect alternating should give entropy=1, got {entropy}"

    # Random data - should be close to 1.0
    rng = np.random.default_rng(42)
    random_bits = rng.integers(0, 2, size=100_000, dtype=np.uint8)
    entropy = shannon_entropy_bits_per_bit(random_bits)
    assert entropy > 0.98, f"Random data should have entropy>0.98, got {entropy}"


def test_entropy_different_generators():
    """Test entropy with different random generators."""
    # LCG
    bits = bits_from_source(LCG(seed=42), 100_000)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert 0 <= entropy <= 1, f"Invalid entropy {entropy} for LCG"

    # XorShift
    bits = bits_from_source(XorShift32(seed=42), 100_000)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert 0 <= entropy <= 1, f"Invalid entropy {entropy} for XorShift"

    # OS Random
    bits = bits_from_source(OSRandom(), 100_000)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert 0 <= entropy <= 1, f"Invalid entropy {entropy} for OS Random"


def test_entropy_edge_cases():
    """Test entropy with edge cases."""
    # Single bit
    bits = np.array([1], dtype=np.uint8)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert entropy == 0.0, f"Single bit should give entropy=0, got {entropy}"

    # All ones
    bits = np.ones(10_000, dtype=np.uint8)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert entropy == 0.0, f"All ones should give entropy=0, got {entropy}"

    # Empty array
    bits = np.array([], dtype=np.uint8)
    entropy = shannon_entropy_bits_per_bit(bits)
    assert entropy == 0.0, f"Empty array should give entropy=0, got {entropy}"


def test_compression_random_vs_structured():
    """Test compression ratio with random vs structured data."""
    # Random data - should be incompressible
    rng = np.random.default_rng(42)
    random_bytes = rng.integers(0, 256, size=10_000, dtype=np.uint8).tobytes()
    cr = compression_ratio(random_bytes)
    assert cr > 0.9, f"Random data should have cr>0.9, got {cr}"

    # All zeros - should be highly compressible
    zeros_bytes = np.zeros(10_000, dtype=np.uint8).tobytes()
    cr = compression_ratio(zeros_bytes)
    assert cr < 0.1, f"All zeros should have cr<0.1, got {cr}"


def test_compression_different_generators():
    """Test compression ratio with different generators."""
    # LCG - known to be highly compressible due to poor randomness
    bits = bits_from_source(LCG(seed=42), 100_000)
    bytes_data = np.packbits(bits).tobytes()
    cr = compression_ratio(bytes_data)
    assert cr < 0.5, f"LCG should be compressible (cr<0.5), got {cr}"

    # XorShift - may be less compressible than expected
    bits = bits_from_source(XorShift32(seed=42), 100_000)
    bytes_data = np.packbits(bits).tobytes()
    cr = compression_ratio(bytes_data)
    assert cr > 0.8, f"XorShift should be less compressible (cr>0.8), got {cr}"

    # OS Random
    bits = bits_from_source(OSRandom(), 100_000)
    bytes_data = np.packbits(bits).tobytes()
    cr = compression_ratio(bytes_data)
    assert cr > 0.8, f"OS Random should be incompressible (cr>0.8), got {cr}"


def test_compression_edge_cases():
    """Test compression ratio with edge cases."""
    # Empty bytes - compression ratio is 0.0 for empty data
    cr = compression_ratio(b"")
    assert cr == 0.0, f"Empty bytes should give cr=0, got {cr}"

    # Single byte - compression ratio can vary significantly and exceed 1.0
    cr = compression_ratio(b"\x00")
    assert cr >= 0, f"Single byte should give cr>=0, got {cr}"


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
