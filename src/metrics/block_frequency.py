import numpy as np
from scipy.special import gammaincc


def block_frequency_pvalue(bits: np.ndarray, M: int = 128) -> float:
    """
    NIST SP 800-22 Block Frequency Test.
    Partition into N blocks of size M (ignore remainder).
    For each block i: pi_i = mean of ones.
    X^2 = 4*M * sum_i (pi_i - 0.5)^2
    p = gammaincc(N/2, X^2/2)
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    if M <= 0:
        return 0.0
    N = n // M
    if N == 0:
        return 0.0

    trimmed = bits[: N * M].reshape(N, M)
    pis = trimmed.mean(axis=1)
    X2 = float(4.0 * M * np.sum((pis - 0.5) ** 2))
    return float(gammaincc(N / 2.0, X2 / 2.0))
