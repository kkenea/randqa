# src/util/config_help.py
from __future__ import annotations

CONFIG_HELP = {
    "source": (
        "Entropy source.\n"
        "- lcg: simple linear-congruential PRNG (weak; LSB alternates).\n"
        "- xorshift: fast PRNG (not cryptographic).\n"
        "- osrandom: OS CSPRNG via os.urandom (strong)."
    ),
    "bits": (
        "Number of bits to sample for analysis. More bits ⇒ more stable p-values "
        "but slower. NIST-style tests are usually run on ≥100k bits."
    ),
    "block_size": (
        "Block size M for the Block Frequency test. The bitstream is split into "
        "⌊n/M⌋ blocks; each block is checked for local bias. Typical M in [64, 512]."
    ),
    "ml_k": (
        "History window size k for the ML next-bit predictor (logistic regression). "
        "Larger k can reveal longer-range structure but needs more samples."
    ),
    "seed": (
        "Seed for PRNG sources (lcg/xorshift) to make results reproducible. "
        "Ignored for osrandom."
    ),
    "alpha": (
        "Significance level for statistical tests. We use α=0.01 by default: "
        "p>α ⇒ test passes; p≤α ⇒ rejects randomness for that test."
    ),
}


def config_warnings(n_bits: int, block_size: int, ml_k: int) -> list[str]:
    """Return human-readable warnings about current settings."""
    notes: list[str] = []
    if n_bits < 50_000:
        notes.append("Use ≥100k bits for more stable statistical tests.")
    # Block Frequency needs multiple blocks
    blocks = n_bits // block_size if block_size > 0 else 0
    if blocks < 20:
        notes.append(f"Block Frequency has only {blocks} blocks; aim for ≥20.")
    # Runs test precondition
    # Not computable here without data, but remind the user:
    notes.append("Runs test requires the fraction of ones to be near 0.5.")
    # ML sample size: number of (window→next) samples is n_bits-k
    samples = max(n_bits - ml_k, 0)
    if samples < 5_000:
        notes.append("ML predictor has few samples (<5k); accuracy may be noisy.")
    # Very large k
    if ml_k > 32 and n_bits < 500_000:
        notes.append("k>32 usually needs hundreds of thousands of bits.")
    return notes
