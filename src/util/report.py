# src/util/report.py
from __future__ import annotations


def glossary_md(alpha: float) -> str:
    return f"""\
### Test glossary (α={alpha})
- **Mono_bit (Frequency):** Checks overall balance of 0/1 across the whole stream.
- **Runs:** Checks the count of runs (contiguous 0s/1s). Too many or too few implies non-random oscillation.
- **Block Frequency:** Splits data into fixed-size blocks and looks for local bias per block.
- **Approximate Entropy (m=2):** Detects repeating/local regularity using overlapping patterns; lower ApEn ⇒ more structure.
- **Shannon entropy (bits/bit):** 1.0 is ideal for Bernoulli(0.5); lower means bias or structure.
- **Compression ratio (zlib):** Random data should not compress (ratio ≈ 1.0).
- **ML next-bit accuracy:** Logistic regression predicts the next bit from the last *k* bits; accuracy > 0.55 suggests short-range predictability.

**SP 800-90B health tests**
- **Repetition Count (RCT):** Fails if any run of identical bits reaches the cutoff (e.g., 34). Large runs indicate a stuck/biased source.
- **Adaptive Proportion (APT):** Non-overlapping windows of size *W*; fails if any window's ones count falls outside [L, U] from Binomial(*W*, 0.5) at α/2.

**Multiple-testing (BH-FDR)**
- We report q-values controlling false discovery rate across p-value tests.  
  **Overall(FDR)** passes only if no test is rejected at level α *and* both health tests pass.
"""


def interpret(results: dict, alpha: float) -> list[str]:
    """Return bullet-point interpretations given results + alpha."""
    msgs: list[str] = []
    # p-values
    mono = results.get("mono_bit_p")
    runs = results.get("runs_p")
    blk = results.get("block_frequency_p")
    ent = results.get("shannon_entropy_bits_per_bit")
    cr = results.get("compression_ratio")
    acc = results.get("ml_accuracy")
    dec = results.get("decisions", {})

    if mono is not None and not dec.get("mono_bit_pass", False):
        msgs.append(f"Mono_bit p={mono:.4f} ≤ α: global 0/1 imbalance detected.")
    if runs is not None and not dec.get("runs_pass", False):
        msgs.append(f"Runs p={runs:.4f} ≤ α: abnormal alternation pattern.")
    if blk is not None and not dec.get("block_frequency_pass", False):
        msgs.append(f"Block Frequency p={blk:.4f} ≤ α: local bias within blocks.")
    if not dec.get("entropy_pass", True):
        msgs.append(f"Entropy {ent:.3f} < 0.98: per-bit unpredictability is low.")
    if not dec.get("compression_pass", True):
        msgs.append(
            f"Compression ratio {cr:.3f} < 0.95: stream is compressible (structure present)."
        )
    if acc is not None and not dec.get("ml_pass", True):
        msgs.append(
            f"ML accuracy {acc:.3f} > 0.55: next bit is predictable from short history."
        )
    if not msgs:
        msgs.append("All checks are consistent with randomness at the chosen α.")
    return msgs
