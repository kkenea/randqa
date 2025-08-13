import numpy as np
from math import sqrt
from scipy.special import erfc


def runs_pvalue(bits: np.ndarray) -> float:
    """
    NIST SP 800-22 Runs Test.
    Preconditions: |pi - 0.5| < 2 / sqrt(n), where pi = mean(bits).
    Let V_n = number of runs (contiguous equal-bit segments).
    p = erfc(|V_n - 2n*pi*(1-pi)| / (2*sqrt(2n)*pi*(1-pi))).
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    if n < 2:
        return 0.0

    pi = float(bits.mean())
    tau = 2.0 / sqrt(n)
    if pi in (0.0, 1.0) or abs(pi - 0.5) >= tau:
        return 0.0  # precondition fail -> treat as strong evidence against H0

    transitions = int(np.sum(bits[1:] != bits[:-1]))
    V_n = transitions + 1
    num = abs(V_n - 2.0 * n * pi * (1.0 - pi))
    denom = 2.0 * sqrt(2.0 * n) * pi * (1.0 - pi)
    if denom == 0.0:
        return 0.0
    return float(erfc(num / denom))
