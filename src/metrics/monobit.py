import numpy as np
from math import sqrt
from scipy.special import erfc


def monobit_pvalue(bits: np.ndarray) -> float:
    """
    NIST SP 800-22 Monobit (Frequency) Test.
    H0: bits ~ i.i.d. Bernoulli(0.5).
    p = erfc(|S_obs| / sqrt(2n)), where S_obs = sum(2*xi - 1).
    """
    b = np.asarray(bits, dtype=np.int8)  # signed to avoid wraparound
    n = b.size
    if n == 0:
        return 0.0
    # avoid unsigned wrap
    s_obs = int(np.sum(np.where(b == 1, 1, -1), dtype=np.int64))
    return float(erfc(abs(s_obs) / sqrt(2.0 * n)))
