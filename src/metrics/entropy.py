import numpy as np


def shannon_entropy_bits_per_bit(bits: np.ndarray) -> float:
    """
    Empirical Shannon entropy H(X) in bits/bit for binary stream.
    H = -sum_x p(x) log2 p(x), x ∈ {0,1}.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    if n == 0:
        return 0.0
    p1 = float(bits.mean())
    p0 = 1.0 - p1
    h = 0.0
    if p0 > 0.0:
        h -= p0 * np.log2(p0)
    if p1 > 0.0:
        h -= p1 * np.log2(p1)
    return float(h)
