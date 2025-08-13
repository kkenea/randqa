# Repetition Count Test (RCT)
from __future__ import annotations
import numpy as np


def repetition_count_test(bits: np.ndarray, cutoff: int = 34):
    """
    Fail if any run of identical bits reaches/exceeds `cutoff`.
    Returns dict: {pass, max_run, cutoff, approx_p}
    approx_p ~ n * 2^(1 - cutoff)  (loose upper bound for a false alarm)
    """
    b = np.asarray(bits, dtype=np.uint8)
    if b.size == 0:
        return {"pass": False, "max_run": 0, "cutoff": cutoff, "approx_p": 1.0}

    # Count max run length
    runs = 1
    max_run = 1
    for i in range(1, b.size):
        if b[i] == b[i - 1]:
            runs += 1
            if runs > max_run:
                max_run = runs
        else:
            runs = 1

    approx_p = min(1.0, float(b.size) * (2.0 ** (1 - cutoff)))
    return {
        "pass": max_run < cutoff,
        "max_run": int(max_run),
        "cutoff": int(cutoff),
        "approx_p": approx_p,
    }
