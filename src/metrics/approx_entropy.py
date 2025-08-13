from __future__ import annotations
import numpy as np
from scipy.special import gammaincc


def approximate_entropy_pvalue(bits: np.ndarray, m: int = 2) -> float:
    """
    NIST SP 800-22 Approximate Entropy test (default m=2).
    Implementation follows the spec: compute Φ_m and Φ_{m+1} using overlapping
    patterns with wrap-around; statistic:
        ApEn = Φ_m - Φ_{m+1}
        chi2 = 2 * n * (ln 2 - ApEn)
        p = gammaincc(2^{m-1}, chi2 / 2)
    """
    b = np.asarray(bits, dtype=np.uint8)
    n = b.size
    if n <= (m + 1):
        return 0.0

    def phi(mm: int) -> float:
        x = np.concatenate([b, b[: mm - 1]])  # wrap
        # windows -> integer codes
        weights = (1 << np.arange(mm)[::-1]).astype(np.uint32)
        windows = np.lib.stride_tricks.sliding_window_view(x, mm)
        idx = (windows @ weights).astype(np.int64)
        counts = np.bincount(idx, minlength=(1 << mm))[: (1 << mm)]
        probs = counts / float(n)
        nz = probs[probs > 0.0]
        return float(np.sum(nz * np.log(nz)))

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    ap_en = phi_m - phi_m1

    # chi2 small when ApEn ≈ ln 2
    ln2 = np.log(2.0)
    chi2 = 2.0 * n * max(0.0, (ln2 - ap_en))
    v = 1 << (m - 1)  # degrees of freedom
    return float(gammaincc(v, chi2 / 2.0))
