from __future__ import annotations
from typing import Protocol, runtime_checkable
import argparse
import json
from pathlib import Path

import numpy as np

from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom

from metrics.mono_bit import mono_bit_pvalue
from metrics.approx_entropy import approximate_entropy_pvalue
from health.rct import repetition_count_test
from health.apt import adaptive_proportion_test
from util.fdr import benjamini_hochberg
from metrics.runs import runs_pvalue
from metrics.block_frequency import block_frequency_pvalue
from metrics.entropy import shannon_entropy_bits_per_bit
from metrics.compression import compression_ratio
from ml.predictor import predictability_score

ALPHA = 0.01


@runtime_checkable
class BitSource(Protocol):
    def next_bit(self) -> int: ...
    def next_bytes(self, n: int) -> bytes: ...


def bits_from_source(src: BitSource, n_bits: int) -> np.ndarray:
    """Return n_bits as a NumPy uint8 array of 0/1."""
    if n_bits <= 0:
        return np.zeros(0, dtype=np.uint8)
    if hasattr(src, "next_bytes"):
        # Pull enough bytes and unpack LSB-first to match PRNG emission.
        n_bytes = (n_bits + 7) // 8
        raw = src.next_bytes(n_bytes)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bits = np.unpackbits(arr, bitorder="little")[:n_bits]
        return bits.astype(np.uint8)
    return np.fromiter(
        (src.next_bit() for _ in range(n_bits)), count=n_bits, dtype=np.uint8
    )


# Back-compat for web.py that imports _bits_from_source
_bits_from_source = bits_from_source


def _pack_bits_little(bits: np.ndarray) -> bytes:
    """
    Pack an arbitrary-length 0/1 bit array into bytes, using LSB-first within each byte.
    Preserves trailing bits by zero-padding the final (partial) byte instead of truncating.
    """
    b = np.asarray(bits, dtype=np.uint8)
    n = int(b.size)
    if n == 0:
        return b""
    rem = (-n) % 8  # zeros to add to complete the final byte
    if rem:
        b = np.concatenate([b, np.zeros(rem, dtype=np.uint8)])
    return np.packbits(b, bitorder="little").tobytes()


def run_tests(
    bits: np.ndarray,
    block_size: int,
    ml_k: int,
    *,
    alpha: float = ALPHA,
    rct_cutoff: int = 34,
    apt_window: int = 512,
) -> dict:
    """Compute p-values, health tests, metrics, and pass/fail decisions."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = int(bits.size)

    # Byte view for compression (preserve all bits by padding the last byte)
    raw_bytes = _pack_bits_little(bits)

    # p-value tests
    p_mono = mono_bit_pvalue(bits)
    p_runs = runs_pvalue(bits)
    p_blk = block_frequency_pvalue(bits, M=block_size)
    p_apen = approximate_entropy_pvalue(bits, m=2)

    # Supporting metrics
    entropy = shannon_entropy_bits_per_bit(bits)
    cr = compression_ratio(raw_bytes)
    acc = predictability_score(bits, k=ml_k)

    # Health tests (SP 800-90B)
    # Standardize empty-input behavior to "skipped" (pass=None)
    if n == 0:
        rct = {"pass": None, "max_run": 0, "cutoff": rct_cutoff, "approx_p": None}
        apt = {
            "pass": None,
            "window": apt_window,
            "alpha": alpha,
            "lower": None,
            "upper": None,
            "violations": [],
        }
    else:
        rct = repetition_count_test(bits, cutoff=rct_cutoff)
        apt = adaptive_proportion_test(bits, window=apt_window, alpha=alpha)

    # BH-FDR across p-value tests
    pvals = {
        "mono_bit": float(p_mono),
        "runs": float(p_runs),
        "block_frequency": float(p_blk),
        "approx_entropy": float(p_apen),
    }
    qvals = benjamini_hochberg(pvals)
    reject_fdr = {name: (q is not None and q <= alpha) for name, q in qvals.items()}

    # Helper: treat None (skipped) as "not a failure" for overall PASS
    def _not_failed(x: bool | None) -> bool:
        return False if x is False else True  # True for True/None

    decisions = {
        "mono_bit_pass": p_mono > alpha,
        "runs_pass": p_runs > alpha,
        "block_frequency_pass": p_blk > alpha,
        "approx_entropy_pass": p_apen > alpha,
        "ml_pass": (acc is None) or (acc <= 0.55),
        "compression_pass": cr >= 0.95,
        "entropy_pass": entropy >= 0.98,
        "rct_pass": rct["pass"],  # may be True/False/None
        "apt_pass": apt["pass"],  # may be True/False/None
        "overall_pass_fdr": (not any(reject_fdr.values()))
        and _not_failed(rct["pass"])
        and _not_failed(apt["pass"]),
    }

    return {
        "mono_bit_p": float(p_mono),
        "runs_p": float(p_runs),
        "block_frequency_p": float(p_blk),
        "approx_entropy_p": float(p_apen),
        "shannon_entropy_bits_per_bit": float(entropy),
        "compression_ratio": float(cr),
        "ml_accuracy": None if acc is None else float(acc),
        "health": {"rct": rct, "apt": apt},
        "pvals_raw": pvals,
        "pvals_fdr": qvals,
        "decisions": decisions,
        "alpha": alpha,
        "rct_cutoff": rct_cutoff,
        "apt_window": apt_window,
    }


def render_markdown(
    source_name: str, n_bits: int, block_size: int, ml_k: int, results: dict
) -> str:
    lines = []
    lines.append("# randqa Report")
    lines.append(f"- **Source:** {source_name}")
    lines.append(f"- **Sample size:** {n_bits} bits")
    lines.append(f"- **Block size (Block Frequency):** {block_size}")
    lines.append(f"- **ML window k:** {ml_k}")
    lines.append(f"- **Alpha:** {ALPHA}\n")

    lines.append("## Statistical Tests (p-values)")
    lines.append(
        f"- Mono_bit: `{results['mono_bit_p']:.6f}` — **{'PASS' if results['decisions']['mono_bit_pass'] else 'FAIL'}**"
    )
    lines.append(
        f"- Runs: `{results['runs_p']:.6f}` — **{'PASS' if results['decisions']['runs_pass'] else 'FAIL'}**"
    )
    lines.append(
        f"- Block Frequency: `{results['block_frequency_p']:.6f}` — **{'PASS' if results['decisions']['block_frequency_pass'] else 'FAIL'}**\n"
    )

    lines.append("## Supporting Metrics")
    lines.append(
        f"- Shannon entropy: `{results['shannon_entropy_bits_per_bit']:.5f}` bits/bit — **{'PASS' if results['decisions']['entropy_pass'] else 'WARN'}**"
    )
    lines.append(
        f"- Compression ratio (zlib): `{results['compression_ratio']:.5f}` — **{'PASS' if results['decisions']['compression_pass'] else 'WARN'}**"
    )
    lines.append("## ML Predictability")
    if results["ml_accuracy"] is not None:
        acc = results["ml_accuracy"]
        lines.append(
            f"- ML next-bit accuracy (k={ml_k}): `{acc:.5f}` — **{'PASS' if results['decisions']['ml_pass'] else 'FAIL'}**"
        )
    else:
        lines.append("- ML next-bit accuracy: `N/A` (insufficient data)")

    lines.append("\n## Interpretation")
    comments = []
    if not results["decisions"]["mono_bit_pass"]:
        comments.append("Imbalance in ones/zeros (mono_bit).")
    if not results["decisions"]["runs_pass"]:
        comments.append("Abnormal oscillation pattern (runs).")
    if not results["decisions"]["block_frequency_pass"]:
        comments.append("Local biases in fixed-size blocks.")
    if not results["decisions"]["ml_pass"]:
        comments.append("Short-history next-bit is predictably biased.")
    if not results["decisions"]["compression_pass"]:
        comments.append("Stream appears compressible (structure present).")
    if results["decisions"]["entropy_pass"] is False:
        comments.append("Entropy per bit is below ~1.0.")
    if not comments:
        comments.append("No significant deviations detected at α = 0.01.")
    lines.append("- " + " ".join(comments))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="randqa - Randomness quality assessment CLI"
    )
    parser.add_argument(
        "--source", choices=["lcg", "xorshift", "osrandom"], default="lcg"
    )
    parser.add_argument("--bits", type=int, default=100_000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--ml-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Seed for PRNG sources")
    parser.add_argument(
        "--out", type=str, default="results", help="Output directory for JSON/Markdown"
    )
    args = parser.parse_args()

    if args.source == "lcg":
        src = LCG(seed=args.seed)
    elif args.source == "xorshift":
        src = XorShift32(seed=args.seed)
    else:
        src = OSRandom()

    bits = bits_from_source(src, args.bits)
    results = run_tests(bits, block_size=args.block_size, ml_k=args.ml_k)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": args.source,
        "bits": args.bits,
        "block_size": args.block_size,
        "ml_k": args.ml_k,
        "alpha": ALPHA,
        "results": results,
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(payload, f, indent=2)

    md = render_markdown(args.source, args.bits, args.block_size, args.ml_k, results)
    with open(out_dir / "report.md", "w") as f:
        f.write(md)

    print(md)


if __name__ == "__main__":
    main()
