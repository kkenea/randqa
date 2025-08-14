# Adaptive Proportion Test (APT)
from __future__ import annotations
import numpy as np
from scipy.stats import binom


def adaptive_proportion_test(
    bits: np.ndarray, window: int = 512, alpha: float = 0.01
) -> dict:
    """
    SP 800-90B Adaptive Proportion Test (APT) for binary streams.

    Splits the bitstream into non-overlapping windows of size window and
    checks that the number of ones in each window lies within inclusive bounds
    [lower, upper] derived from Binomial(n=window, p=0.5) with two-sided
    significance alpha.

    Returns a dict:
      {
        "pass": bool | None,      # None when not enough bits to run
        "window": int,
        "alpha": float,
        "lower": int | None,      # lower inclusive bound (None if not run)
        "upper": int | None,      # upper inclusive bound (None if not run)
        "violations": list[{"window_index": int, "ones": int}],
        "n": int,                 # total bits examined
        "n_windows": int,         # number of windows evaluated
        "reason": str             # present only when pass is None
      }
    """
    w = int(window)
    if w <= 0:
        raise ValueError("APT window must be > 0")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    b = np.asarray(bits, dtype=np.uint8)
    n = int(b.size)

    # Not enough data to run any APT window
    if n < w:
        return {
            "pass": None,
            "window": w,
            "alpha": a,
            "lower": None,
            "upper": None,
            "violations": [],
            "n": n,
            "n_windows": 0,
            "reason": f"insufficient bits: n={n} < window={w}",
        }

    # Compute exact two-sided binomial bounds (inclusive):
    # lower = PPF(alpha/2), upper = ISF(alpha/2)
    lower = int(binom.ppf(a / 2.0, w, 0.5))
    upper = int(
        binom.isf(a / 2.0, w, 0.5)
    )  # yields inclusive upper bound for our check

    n_windows = n // w
    violations: list[dict] = []

    if n_windows > 0:
        reshaped = b[: n_windows * w].reshape(n_windows, w)
        ones_per = reshaped.sum(axis=1).astype(int)
        for i, ones in enumerate(ones_per):
            if ones < lower or ones > upper:
                violations.append({"window_index": int(i), "ones": int(ones)})

    return {
        "pass": (len(violations) == 0),
        "window": w,
        "alpha": a,
        "lower": lower,
        "upper": upper,
        "violations": violations,
        "n": n,
        "n_windows": int(n_windows),
    }
