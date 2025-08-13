from __future__ import annotations

import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from advisor.llm import ai_advice
from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom
from main import bits_from_source, run_tests, render_markdown, ALPHA
from util.config_help import CONFIG_HELP
from util.report import interpret, glossary_md
from advisor.rules import rule_based_advice


def _make_source(name: str, seed: int):
    if name == "lcg":
        return LCG(seed=seed)
    if name == "xorshift":
        return XorShift32(seed=seed)
    if name == "osrandom":
        return OSRandom()
    raise ValueError("invalid source")


st.set_page_config(page_title="randqa", layout="wide")
st.title("randqa - Randomness Quality Assessment")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Input mode", ["PRNG Source", "Upload File", "Paste Hex"], index=0)
    file_bytes = None
    hex_bytes = None

    if mode == "PRNG Source":
        source = st.selectbox(
            "Source",
            ["lcg", "xorshift", "osrandom"],
            index=0,
            help=CONFIG_HELP["source"],
        )
        seed = st.number_input(
            "Seed (PRNG only)",
            min_value=0,
            max_value=2**32 - 1,
            value=42,
            step=1,
            disabled=(source == "osrandom"),
            help=CONFIG_HELP["seed"],
        )
    elif mode == "Upload File":
        up = st.file_uploader(
            "Upload bytes to analyze", type=None, accept_multiple_files=False
        )
        if up is not None:
            file_bytes = up.read()
            st.caption(f"Loaded {len(file_bytes)} bytes.")
    else:
        hex_str = st.text_area("Paste hex string (no 0x, spaces ok)", height=120)
        if hex_str.strip():
            try:
                hex_bytes = bytes.fromhex("".join(hex_str.split()))
                st.caption(f"Parsed {len(hex_bytes)} bytes from hex.")
            except Exception as e:
                st.error(f"Invalid hex: {e}")

    st.header("Parameters")
    bits_n = st.number_input(
        "Bits",
        min_value=10_000,
        max_value=2_000_000,
        value=100_000,
        step=10_000,
        help=CONFIG_HELP["bits"],
    )
    if mode == "Paste Hex":
        st.caption(
            f"For {int(bits_n)} bits, paste ≥ {int(bits_n) // 4} hex characters."
        )

    block_size = st.number_input(
        "Block size (Block Frequency)",
        min_value=8,
        max_value=4096,
        value=128,
        step=8,
        help=CONFIG_HELP["block_size"],
    )
    ml_k = st.number_input(
        "ML window k",
        min_value=2,
        max_value=64,
        value=8,
        step=1,
        help=CONFIG_HELP["ml_k"],
    )
    st.caption(CONFIG_HELP["alpha"])

    with st.expander("Advanced"):
        rct_cutoff = st.number_input(
            "RCT cutoff (max allowed run length)",
            min_value=4,
            max_value=256,
            value=34,
            step=1,
            help="SP 800-90B Repetition Count Test threshold.",
        )
        apt_window = st.number_input(
            "APT window size",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
            help="SP 800-90B Adaptive Proportion Test window size.",
        )

    st.header("Compare mode")
    compare = st.checkbox("Enable compare table (multi-sources)", value=False)
    sources_sel = []
    include_file = False
    include_hex = False
    if compare:
        sources_sel = st.multiselect(
            "Select sources",
            ["lcg", "xorshift", "osrandom"],
            default=["lcg", "xorshift", "osrandom"],
        )
        if mode == "Upload File" and file_bytes:
            include_file = st.checkbox("Include uploaded file", value=True)
        if mode == "Paste Hex" and hex_bytes:
            include_hex = st.checkbox("Include pasted hex", value=True)

    # AI (optional)
    st.header("AI (optional)")
    enable_ai = st.checkbox(
        "Enable AI recommendations",
        help="Uses API keys from environment variables (.env file) to provide AI-powered analysis recommendations.",
    )
    
    if enable_ai:
        # Check if any API keys are available in environment
        import os
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        
        available_providers = []
        if openai_key:
            available_providers.append("openai")
        if gemini_key:
            available_providers.append("gemini")
        if anthropic_key:
            available_providers.append("anthropic")
            
        if not available_providers:
            st.error("No API keys found in environment variables. Please set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in your .env file.")
            st.stop()
            
        provider = st.selectbox(
            "Provider",
            available_providers,
            index=0,
            help="Pick the LLM provider for recommendations.",
        )
        
        # Get API key from environment based on selection
        api_key_map = {
            "openai": openai_key,
            "gemini": gemini_key,
            "anthropic": anthropic_key
        }
        api_key = api_key_map[provider]
        
        # Get default model from environment or use defaults
        default_model = {
            "openai": os.environ.get("DEFAULT_OPENAI_MODEL", "gpt-4o-mini"),
            "gemini": os.environ.get("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash"),
            "anthropic": os.environ.get("DEFAULT_ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        }[provider]
        
        model_name = st.text_input("Model name", value=default_model)
        st.info(f"Using API key from environment variables for {provider}.")
        st.warning(
            "Privacy note: ONLY the summarized test results (no raw bitstream) are sent to the provider. "
            "Data leaves your machine when this is enabled. Disable if that is not acceptable."
        )

    run_btn = st.button("Analyze", use_container_width=True)

with st.expander("What do these settings do?"):
    st.markdown(glossary_md(ALPHA))


def _bits_from_bytes(raw: bytes, n_bits: int) -> np.ndarray:
    if not raw:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits[:n_bits].astype(np.uint8)


# Single analysis or compare mode
if run_btn:
    # Collect test items: list of (label, bits)
    items: list[tuple[str, np.ndarray]] = []
    raw_bits: dict[str, np.ndarray] = {}

    if compare:
        for s in sources_sel:
            src = _make_source(s, int(seed) if mode == "PRNG Source" else 42)
            b = bits_from_source(src, int(bits_n))
            items.append((f"source:{s}", b))
            raw_bits[f"source:{s}"] = b
        if include_file:
            b = _bits_from_bytes(file_bytes, int(bits_n))
            items.append(("file:upload", b))
            raw_bits["file:upload"] = b
        if include_hex:
            b = _bits_from_bytes(hex_bytes, int(bits_n))
            items.append(("hex:input", b))
            raw_bits["hex:input"] = b
    else:
        if mode == "PRNG Source":
            src = _make_source(source, int(seed))
            b = bits_from_source(src, int(bits_n))
            items.append((f"source:{source}", b))
            raw_bits[f"source:{source}"] = b
        elif mode == "Upload File" and file_bytes:
            b = _bits_from_bytes(file_bytes, int(bits_n))
            items.append(("file:upload", b))
            raw_bits["file:upload"] = b
        elif mode == "Paste Hex" and hex_bytes:
            b = _bits_from_bytes(hex_bytes, int(bits_n))
            items.append(("hex:input", b))
            raw_bits["hex:input"] = b
        else:
            st.error("No input provided.")
            st.stop()

    # Warn if bytes/hex payload shorter than requested
    for label, b in items:
        if (label.startswith("file:") or label.startswith("hex:")) and len(b) < int(
            bits_n
        ):
            requested = int(bits_n)
            available = int(len(b))
            need_hex_chars = requested // 4
            st.warning(
                f"'{label}' has only {available} bits available; analyzed {available} (requested {requested}). "
                f"For {requested} bits via hex, paste at least {need_hex_chars} hex characters."
            )

    # Run tests
    records = []
    detailed = {}
    with st.spinner("Running tests..."):
        for label, b in items:
            res = run_tests(
                b,
                block_size=int(block_size),
                ml_k=int(ml_k),
                alpha=ALPHA,
                rct_cutoff=int(rct_cutoff),
                apt_window=int(apt_window),
            )
            detailed[label] = res
            # Row for compare table
            # Handle APT "NOT RUN" based on health or based on insufficient windows
            apt_info = res["health"]["apt"]
            n_bits_here = len(b)
            n_windows_here = n_bits_here // int(apt_window)
            apt_pass_val = apt_info.get("pass")
            if n_windows_here == 0:
                apt_cell = "NOT RUN"
            else:
                apt_cell = (
                    "NOT RUN"
                    if apt_pass_val is None
                    else ("PASS" if res["decisions"]["apt_pass"] else "FAIL")
                )

            records.append(
                {
                    "item": label,
                    "monobit_p": round(res["monobit_p"], 6),
                    "runs_p": round(res["runs_p"], 6),
                    "blockfreq_p": round(res["block_frequency_p"], 6),
                    "approx_entropy_p": round(res["approx_entropy_p"], 6),
                    "entropy": round(res["shannon_entropy_bits_per_bit"], 5),
                    "compress_ratio": round(res["compression_ratio"], 5),
                    "ml_acc": None
                    if res["ml_accuracy"] is None
                    else round(res["ml_accuracy"], 5),
                    "RCT": "PASS" if res["decisions"]["rct_pass"] else "FAIL",
                    "APT": apt_cell,
                    "Overall(FDR)": "PASS"
                    if res["decisions"]["overall_pass_fdr"]
                    else "FAIL",
                }
            )

    # Compare table or single-item dashboard
    if compare and len(records) > 1:
        st.subheader("Compare results")
        st.dataframe(records, use_container_width=True)
        st.caption(
            "Overall(FDR): all p-value tests pass under BH-FDR and both health tests pass."
        )

        # Let user pick one item to view details
        chosen = st.selectbox("Inspect item", [r["item"] for r in records], index=0)
        res = detailed[chosen]
    else:
        # Single analysis
        chosen = items[0][0]
        res = detailed[chosen]

    # Bits actually analyzed for the chosen item
    selected_bits = raw_bits.get(chosen, np.array([], dtype=np.uint8))
    analyzed_bits = int(len(selected_bits))

    # Topline metrics with colors
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Monobit",
            f"{res['monobit_p']:.4f}",
            "PASS" if res["decisions"]["monobit_pass"] else "FAIL",
            delta_color="normal" if res["decisions"]["monobit_pass"] else "inverse",
        )
        st.metric(
            "Runs",
            f"{res['runs_p']:.4f}",
            "PASS" if res["decisions"]["runs_pass"] else "FAIL",
            delta_color="normal" if res["decisions"]["runs_pass"] else "inverse",
        )

    with col2:
        st.metric(
            "Block Frequency",
            f"{res['block_frequency_p']:.4f}",
            "PASS" if res["decisions"]["block_frequency_pass"] else "FAIL",
            delta_color="normal"
            if res["decisions"]["block_frequency_pass"]
            else "inverse",
        )
        st.metric(
            "Approx Entropy",
            f"{res['approx_entropy_p']:.4f}",
            "PASS" if res["decisions"]["approx_entropy_pass"] else "FAIL",
            delta_color="normal"
            if res["decisions"]["approx_entropy_pass"]
            else "inverse",
        )

    with col3:
        ml_acc = res["ml_accuracy"]
        if ml_acc is None:
            ml_display = "N/A"
            ml_status = "FAIL"
        else:
            ml_display = f"{ml_acc:.4f}"
            ml_status = "PASS" if res["decisions"]["ml_pass"] else "FAIL"
        st.metric(
            "ML Accuracy",
            ml_display,
            ml_status,
            delta_color="normal" if res["decisions"]["ml_pass"] else "inverse",
        )

    # Config notes
    st.caption(
        f"α={res['alpha']:.3f}, Block size={int(block_size)}, ML k={int(ml_k)}, "
        f"RCT cutoff={res['rct_cutoff']}, APT window={res['apt_window']}"
    )

    # Health tests
    st.subheader("Health Tests (SP 800-90B)")
    col1, col2 = st.columns(2)
    with col1:
        rct_info = res["health"]["rct"]
        st.metric(
            "Repetition Count",
            f"max run: {rct_info['max_run']}",
            "PASS" if rct_info["pass"] else "FAIL",
            delta_color="normal" if rct_info["pass"] else "inverse",
        )
        if not rct_info["pass"]:
            st.caption(f"Cutoff: {rct_info['cutoff']}")

    with col2:
        apt_info = res["health"]["apt"]
        if apt_info["pass"] is None:
            apt_display = "NOT RUN"
            apt_status = "N/A"
        else:
            apt_display = f"window: {apt_info['window']}"
            apt_status = "PASS" if apt_info["pass"] else "FAIL"
        st.metric(
            "Adaptive Proportion",
            apt_display,
            apt_status,
            delta_color="normal" if apt_info["pass"] else "inverse",
        )
        if apt_info["pass"] is False and apt_info["violations"]:
            st.caption(f"Violations: {len(apt_info['violations'])}")

    # FDR-adjusted p-values
    st.subheader("Multiple-testing (BH-FDR)")
    rows = []
    for name, p in res["pvals_raw"].items():
        q = res["pvals_fdr"].get(name)
        rows.append(
            {
                "test": name,
                "p": round(p, 6),
                "q (BH)": None if q is None else round(q, 6),
                "reject@α": bool(q is not None and q <= res["alpha"]),
            }
        )
    st.dataframe(rows, use_container_width=True)
    st.write(
        "**Overall verdict (FDR + health):**",
        "PASS" if res["decisions"]["overall_pass_fdr"] else "FAIL",
    )

    # Interpretation + advisor
    st.subheader("Interpretation")
    for msg in interpret(res, ALPHA):
        st.write(f"- {msg}")

    st.subheader("Recommendations")
    for msg in rule_based_advice(res):
        st.write(f"- {msg}")

    if enable_ai:
        st.markdown("**AI (LLM) Advice**")
        with st.spinner(f"Generating AI recommendations via {provider}..."):
            ai_text = ai_advice(
                res, provider=provider, api_key=api_key, model=model_name
            )
        st.write(ai_text if ai_text else "No AI advice returned.")

    # Reports side-by-side (use actual analyzed bit length)
    payload = {
        "item": chosen,
        "bits": analyzed_bits,
        "block_size": int(block_size),
        "ml_k": int(ml_k),
        "alpha": res["alpha"],
        "rct_cutoff": res["rct_cutoff"],
        "apt_window": res["apt_window"],
        "results": res,
    }
    md = render_markdown(chosen, analyzed_bits, int(block_size), int(ml_k), res)
    json_str = json.dumps(payload, indent=2)

    st.subheader("Reports")
    left, right = st.columns(2)
    with left:
        st.caption("Markdown Report")
        st.code(md, language="markdown")
        st.download_button(
            "Download Markdown",
            data=md.encode(),
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with right:
        st.caption("JSON Report")
        st.code(json_str, language="json")
        st.download_button(
            "Download JSON",
            data=json_str.encode(),
            file_name="report.json",
            mime="application/json",
            use_container_width=True,
        )

    # Visual: Block means with labeled axes (use selected item's bits)
    st.subheader("Quick Visual: Block means")
    if analyzed_bits > 0:
        M = int(block_size)
        N = analyzed_bits // M if M > 0 else 0
        if N > 0:
            trimmed = selected_bits[: N * M].reshape(N, M)
            block_means = trimmed.mean(axis=1)
            fig, ax = plt.subplots()
            ax.plot(range(N), block_means)
            ax.set_xlabel("Block index")
            ax.set_ylabel("Proportion of ones")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Increase bit count or reduce block size to see block means.")
    else:
        st.info("No bits available for the selected item.")
else:
    st.info("Choose input, set parameters, and click Analyze.")
