from __future__ import annotations


def rule_based_advice(results: dict) -> list[str]:
    d = results.get("decisions", {})
    msgs = []
    if not d.get("rct_pass", True):
        r = results["health"]["rct"]
        msgs.append(
            f"Repetition Count Test failed: max run={r['max_run']} ≥ cutoff {r['cutoff']}. Consider de-biasing or conditioning (e.g., Von Neumann extractor) or replacing the source."
        )
    if not d.get("apt_pass", True):
        a = results["health"]["apt"]
        msgs.append(
            f"Adaptive Proportion Test failed: some {a['window']}-bit windows outside [{a['lower']},{a['upper']}]. Review entropy source; increase conditioning or whiten the raw source."
        )
    if not d.get("monobit_pass", True):
        msgs.append(
            "Monobit failed: overall 0/1 imbalance; apply rejection sampling or XOR with a balanced source."
        )
    if not d.get("runs_pass", True):
        msgs.append(
            "Runs failed: oscillation pattern abnormal; investigate LSB bias or linear structure; use cryptographic DRBG."
        )
    if not d.get("block_frequency_pass", True):
        msgs.append(
            "Block Frequency failed: local bias detected; increase entropy pool mixing or hash the pool before output."
        )
    if not d.get("approx_entropy_pass", True):
        msgs.append(
            "Approximate Entropy failed: repeating patterns present; consider larger state or non-linear mixing."
        )
    if not d.get("compression_pass", True):
        msgs.append(
            "Stream is compressible; structure present. Avoid using this stream for keys/nonces without conditioning."
        )
    if not d.get("ml_pass", True):
        msgs.append(
            "ML predictor succeeded >55%; next bit is predictable from short history; switch sources or post-process with a secure PRF."
        )
    if not msgs:
        msgs.append(
            "No red flags detected at α=0.01. Continue with periodic health testing in production."
        )
    return msgs
