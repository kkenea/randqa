from __future__ import annotations


def rule_based_advice(results: dict) -> list[str]:
    """
    Produce actionable, human-readable guidance based on per-test decisions and health tests.
    Handles 'NOT RUN' (pass=None) cases for RCT/APT explicitly and avoids mislabeling them as failures.
    """
    d = results.get("decisions", {}) or {}
    health = results.get("health", {}) or {}
    rct = health.get("rct", {}) or {}
    apt = health.get("apt", {}) or {}

    msgs: list[str] = []

    # ---- Health tests (SP 800-90B) ----
    # RCT
    rct_pass = rct.get("pass")
    if rct_pass is False:
        cutoff = rct.get("cutoff", "?")
        max_run = rct.get("max_run", "?")
        msgs.append(
            f"Repetition Count Test failed: max run={max_run} ≥ cutoff {cutoff}. "
            f"Consider de-biasing/conditioning (e.g., Von Neumann extractor) or replacing the source."
        )
    elif rct_pass is None:
        reason = rct.get("reason", "test not run")
        n = rct.get("n")
        if n is not None:
            msgs.append(
                f"Repetition Count Test not run ({reason}); collected n={n}. "
                f"Provide at least one bit (n>0) to enable this health test."
            )
        else:
            msgs.append(
                f"Repetition Count Test not run ({reason}); provide data to enable this health test."
            )

    # APT
    apt_pass = apt.get("pass")
    if apt_pass is False:
        window = apt.get("window", "?")
        lower = apt.get("lower")
        upper = apt.get("upper")
        bounds = (
            f"[{lower},{upper}]"
            if (lower is not None and upper is not None)
            else "[L,U]"
        )
        violations = apt.get("violations", []) or []
        vcount = len(violations)
        example = ""
        if vcount > 0:
            # show up to 3 example window indices
            idxs = [str(v.get("window_index", "?")) for v in violations[:3]]
            example = f" (e.g., window index{'es' if len(idxs) > 1 else ''}: {', '.join(idxs)})"
        msgs.append(
            f"Adaptive Proportion Test failed: {vcount} window(s) of size {window} outside {bounds}{example}. "
            f"Review/condition the entropy source or reduce APT window to increase sensitivity with limited data."
        )
    elif apt_pass is None:
        reason = apt.get("reason", "test not run")
        n = apt.get("n")
        w = apt.get("window")
        if n is not None and w is not None:
            msgs.append(
                f"Adaptive Proportion Test not run ({reason}); collected n={n}, window={w}. "
                f"Provide at least {w} bits (prefer multiples of {w}) or reduce the window size."
            )
        else:
            msgs.append(
                f"Adaptive Proportion Test not run ({reason}); provide enough bits or reduce the window size."
            )

    # ---- NIST SP 800-22-style p-value tests ----
    if d.get("mono_bit_pass") is False:
        msgs.append(
            "Mono_bit failed: overall 0/1 imbalance; apply rejection sampling or XOR with a balanced source."
        )
    if d.get("runs_pass") is False:
        msgs.append(
            "Runs failed: oscillation pattern is abnormal; investigate LSB bias or linear structure; consider a cryptographic DRBG."
        )
    if d.get("block_frequency_pass") is False:
        msgs.append(
            "Block Frequency failed: local bias detected; strengthen entropy pool mixing or hash the pool before output."
        )
    if d.get("approx_entropy_pass") is False:
        msgs.append(
            "Approximate Entropy failed: repeating patterns present; consider larger state or non-linear mixing."
        )

    # ---- Supporting metrics ----
    if d.get("compression_pass") is False:
        msgs.append(
            "Stream is compressible (structure present). Avoid using this stream for keys/nonces without conditioning."
        )

    # ML predictability: only warn when it actually failed (>55% accuracy).
    # If the ML test was skipped upstream (accuracy None but treated as PASS), do not warn.
    if d.get("ml_pass") is False:
        msgs.append(
            "ML predictor exceeded 55% accuracy; next bit is predictable from short history. Switch sources or post-process with a secure PRF."
        )

    # If nothing triggered, provide a positive but practical recommendation.
    if not msgs:
        msgs.append(
            "No red flags at α=0.01. Proceed, but deploy periodic health testing and telemetry in production."
        )

    return msgs
