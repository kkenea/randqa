from __future__ import annotations
from flask import Flask, request, jsonify

from sources.lcg import LCG
from sources.xorshift import XorShift32
from sources.os_random import OSRandom
from main import bits_from_source, run_tests  # reuse CLI logic


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/analyze")
    def analyze():
        """
        Example:
          /analyze?source=lcg&bits=100000&block_size=128&ml_k=8&seed=42
          /analyze?source=osrandom&bits=200000
        """
        source = request.args.get("source", "lcg").lower()
        bits_n = int(request.args.get("bits", 100_000))
        block_size = int(request.args.get("block_size", 128))
        ml_k = int(request.args.get("ml_k", 8))
        seed = int(request.args.get("seed", 42))

        if source == "lcg":
            src = LCG(seed=seed)
        elif source == "xorshift":
            src = XorShift32(seed=seed)
        elif source == "osrandom":
            src = OSRandom()
        else:
            return jsonify({"error": "invalid source"}), 400

        bits = bits_from_source(src, bits_n)
        results = run_tests(bits, block_size=block_size, ml_k=ml_k)

        return jsonify(
            {
                "source": source,
                "bits": bits_n,
                "block_size": block_size,
                "ml_k": ml_k,
                "seed": None if source == "osrandom" else seed,
                "results": results,
            }
        )

    return app


def serve():
    app = create_app()
    import os

    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
