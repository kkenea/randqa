# src/util/fdr.py
from __future__ import annotations


def benjamini_hochberg(pdict: dict[str, float]) -> dict[str, float]:
    """
    Return BH-FDR q-values for named p-values.
    Correct step-up procedure: sort ascending, compute p*m/i,
    then take cumulative min from the end.
    """
    # Keep only present p-values
    items = [(k, float(v)) for k, v in pdict.items() if v is not None]
    m = len(items)
    if m == 0:
        return {}

    # Sort ascending by p
    items.sort(key=lambda kv: kv[1])  # [(name, p), ...]

    # Raw q_i = p_i * m / i
    raw = []
    for i, (name, p) in enumerate(items, start=1):
        q = (p * m) / i
        raw.append((name, min(q, 1.0)))

    # Cumulative min from the end to enforce monotonicity
    qvals_sorted = {}
    running = 1.0
    for name, q in reversed(raw):
        running = min(running, q)
        qvals_sorted[name] = running

    # Map back to original keys
    return {k: qvals_sorted.get(k, None) for k in pdict.keys()}
