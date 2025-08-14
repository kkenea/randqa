# src/metrics/mono_bit.py
from __future__ import annotations
import numpy as np


def mono_bit_pvalue(bits: np.ndarray) -> float:
    """
    NIST SP 800-22 Mono_bit (Frequency) Test.

    Tests the overall balance of 0s and 1s in a binary sequence.
    Under the null hypothesis of randomness, the number of 1s should
    be approximately n/2, where n is the sequence length.

    Args:
        bits: Binary sequence as numpy array of 0s and 1s

    Returns:
        p-value: Probability of observing the given imbalance by chance.
                p > α indicates the sequence appears balanced (PASS).
                p ≤ α indicates significant imbalance (FAIL).

    Reference:
        NIST SP 800-22 Rev. 1a, Section 2.1
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = int(bits.size)

    if n == 0:
        return 1.0

    # Count 1s in the sequence
    ones = int(np.sum(bits))
    zeros = n - ones

    # Calculate the test statistic
    # S = (ones - zeros) / sqrt(n)
    # Under H0, S ~ N(0,1)
    S = (ones - zeros) / np.sqrt(n)

    # Convert to p-value using complementary error function
    # p = erfc(|S| / sqrt(2))
    from scipy.special import erfc

    p_value = erfc(abs(S) / np.sqrt(2))

    return float(p_value)
