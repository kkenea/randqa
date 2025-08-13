# Adaptive Proportion Test (APT)
from __future__ import annotations
import numpy as np
from scipy.stats import binom


def adaptive_proportion_test(bits: np.ndarray, window: int = 512, alpha: float = 0.01):
    """
    SP 800-90B Adaptive Proportion Test (APT) for binary streams.
    Splits into non-overlapping windows of size window and checks the number of ones
    in each window lies within [L, U] derived from Binomial(n=window, p=0.5) with
    two-sided significance alpha.

    Returns a dict with:
      - pass: bool | None (None if not enough bits to run)
      - window, alpha
      - lower, upper (int bounds)
      - violations: list of {window_index, ones}
      - n: total bits seen
      - n_windows: number of windows evaluated
      - reason: present only when pass is None
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = int(bits.size)
    if window <= 0:
        raise ValueError("APT window must be > 0")

    # Not enough data to run APT at all
    if n < window:
        return {
            "pass": None,
            "window": int(window),
            "alpha": float(alpha),
            "lower": None,
            "upper": None,
            "violations": [],
            "n": n,
            "n_windows": 0,
            "reason": f"insufficient bits: n={n} < window={window}",
        }

    # Compute two-sided bounds using exact binomial quantiles
    lo = int(binom.ppf(alpha / 2.0, window, 0.5))
    # isf(q) = inverse survival function, upper tail; cast to int for bound
    hi = int(binom.isf(alpha / 2.0, window, 0.5))

    n_windows = n // window
    violations = []
    if n_windows > 0:
        reshaped = bits[: n_windows * window].reshape(n_windows, window)
        ones_per = reshaped.sum(axis=1).astype(int)
        for i, ones in enumerate(ones_per):
            if ones < lo or ones > hi:
                violations.append({"window_index": i, "ones": int(ones)})

    return {
        "pass": len(violations) == 0,
        "window": int(window),
        "alpha": float(alpha),
        "lower": lo,
        "upper": hi,
        "violations": violations,
        "n": n,
        "n_windows": int(n_windows),
    }
