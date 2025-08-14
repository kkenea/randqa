# Repetition Count Test (RCT)
from __future__ import annotations
import numpy as np


def repetition_count_test(bits: np.ndarray, cutoff: int = 34) -> dict:
    """
    SP 800-90B Repetition Count Test (RCT) for a binary stream.

    Fails if any run of identical bits reaches/exceeds cutoff.

    Returns a dict:
      {
        "pass": bool | None,   # None when no data (test not run)
        "max_run": int,        # longest observed run length
        "cutoff": int,         # threshold used
        "approx_p": float|None,# very loose upper bound on false alarm prob ~ n * 2^(1 - cutoff)
        "n": int,              # number of bits inspected
        "reason": str          # present only when pass is None
      }

    Notes:
    - For empty input (n == 0), returns pass=None (NOT RUN) to avoid
      misclassifying a missing stream as a failure.
    - approx_p is an order-of-magnitude heuristic (not exact).
    """
    if cutoff is None or int(cutoff) < 2:
        raise ValueError("RCT cutoff must be an integer >= 2")

    b = np.asarray(bits, dtype=np.uint8)
    n = int(b.size)

    if n == 0:
        return {
            "pass": None,
            "max_run": 0,
            "cutoff": int(cutoff),
            "approx_p": None,
            "n": 0,
            "reason": "insufficient bits: n=0",
        }

    # Compute maximum run length
    runs = 1
    max_run = 1
    # Loop is simple and robust for arbitrary n (vectorization offers little gain here)
    for i in range(1, n):
        if b[i] == b[i - 1]:
            runs += 1
            if runs > max_run:
                max_run = runs
        else:
            runs = 1

    # Very loose false alarm upper bound:
    # expected count of runs-of-length-≥cutoff is approximately n * 2^(1 - cutoff)
    approx_p = min(1.0, float(n) * (2.0 ** (1 - int(cutoff))))

    return {
        "pass": bool(max_run < int(cutoff)),
        "max_run": int(max_run),
        "cutoff": int(cutoff),
        "approx_p": float(approx_p),
        "n": n,
    }
