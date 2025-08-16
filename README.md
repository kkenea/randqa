# randqa - Randomness Quality Assessment

A toolkit for analyzing randomness quality in cryptographic applications. Implements NIST SP 800-22 statistical tests, SP 800-90B health tests, ML predictability analysis, and provides multiple interfaces (GUI, API, CLI) with optional AI recommendations.

## Features

- **Statistical Tests (SP 800-22):** Mono_bit, Runs, Block Frequency, Approximate Entropy
- **Health Tests (SP 800-90B):** Repetition Count Test (RCT), Adaptive Proportion Test (APT)
- **Machine Learning:** Next-bit predictability analysis using logistic regression
- **Multiple Interfaces:** Streamlit GUI, Flask API, command-line tools
- **AI Integration:** Optional LLM-powered analysis and recommendations
- **Comprehensive Reporting:** Markdown and JSON output formats

## Quick Start (Docker) - Recommended

Requirements: Docker & Docker Compose

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kkenea/randqa.git
   cd randqa
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see AI Recommendations Setup below)
   ```

3. **Start the services**:
   ```bash
   docker compose up --build
   ```

- GUI: http://localhost:8501  
- API: http://localhost:8000

> **Note:** It may take upto 2 minutes for build to complete.

## Quick Start (Local Python)

Requirements: Python 3.11+

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kkenea/randqa.git
   cd randqa
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see AI Recommendations Setup below)
   ```

### Using uv

```bash
uv sync --extra gui --extra web --extra ai
# GUI:
uv run python -m streamlit run src/gui.py
# API:
uv run randqa-web
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[gui,web,ai]"
# GUI:
streamlit run src/gui.py
# API:
randqa-web
```

Both methods default to:
- GUI at http://localhost:8501
- API at http://localhost:8000

## AI Recommendations Setup

To enable AI-powered analysis recommendations, you need to configure an API key for one LLM provider.

### Using Environment Variables (.env file)

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your API key** (only one provider needed):
   ```bash
   # AI Provider API Key (choose one)
   OPENAI_API_KEY=your-apikey
   # OR
   GEMINI_API_KEY=your-apikey
   # OR
   ANTHROPIC_API_KEY=your-apikey
   
   # Optional: Customize default model for your chosen provider
   DEFAULT_OPENAI_MODEL=gpt-4o-mini
   DEFAULT_GEMINI_MODEL=gemini-1.5-flash
   DEFAULT_ANTHROPIC_MODEL=claude-3-haiku-20240307
   ```

3. **Restart your containers** (if using Docker):
   ```bash
   docker compose down
   docker compose up --build
   ```

4. **For local Python**, set environment variables:
   ```bash
   # Choose one provider
   export OPENAI_API_KEY="your-apikey"
   # OR
   export GEMINI_API_KEY="your-apikey"
   # OR
   export ANTHROPIC_API_KEY="your-apikey"
   ```

### Supported Providers

- **OpenAI**: GPT models
- **Google Gemini**: Gemini Pro models
- **Anthropic**: Claude models

The GUI will automatically detect which providers have configured API keys and only show those options.

## Using the Streamlit GUI

### Input Modes
- PRNG Source: Built-ins (LCG, XorShift32, OS CSPRNG)
- Upload File: Analyze any binary blob
- Paste Hex: Parse and analyze a hex string

### Parameters
- Bits: Sample size (≥100k recommended for stable p-values)
- Block size: For Block Frequency (64-512)
- ML window k: History length for ML predictor (4-16)
- Advanced: RCT cutoff (default 34), APT window (default 512)

### Extras
- Compare mode: Side-by-side across multiple sources
- Reports: Markdown + JSON preview and downloads
- Charts: Block means with labeled axes
- AI Recommendations: Toggle in sidebar (select provider, uses API keys from environment)

## Web API

### Start the API

Local Python

```bash
uv run randqa-web
```

Docker Compose

```bash
docker compose up api
```

Server listens on http://localhost:8000 (override with PORT env).

### Endpoints

Health

```http
GET /health
```

Response:
```json
{"status":"ok"}
```

Analyze

```http
GET /analyze
```

Query Parameters:

| Name         | Type   | Default  | Notes                                                                 |
|--------------|--------|----------|-----------------------------------------------------------------------|
| `source`     | string | `lcg`    | `lcg` \| `xorshift` \| `osrandom`                                     |
| `bits`       | int    | `100000` | Requested sample size (typ. 1e5-2e5)                                  |
| `block_size` | int    | `128`    | Block size for Block Frequency (64-512)                               |
| `ml_k`       | int    | `8`      | History length for ML predictor (2-64)                                |
| `seed`       | int    | `42`     | Used for PRNGs; ignored by `osrandom`                                 |

Examples:

```bash
# Basic LCG analysis
curl "http://localhost:8000/analyze?source=lcg&bits=100000"

# XorShift with custom params
curl "http://localhost:8000/analyze?source=xorshift&bits=50000&block_size=256&ml_k=12&seed=123"

# OS CSPRNG
curl "http://localhost:8000/analyze?source=osrandom&bits=200000&block_size=128&ml_k=8"
```

Response:

```json
{
  "source": "lcg",
  "bits": 100000,
  "block_size": 128,
  "ml_k": 8,
  "seed": 42,
  "results": {
    "mono_bit_p": 1.0,
    "runs_p": 0.0,
    "block_frequency_p": 1.0,
    "approx_entropy_p": 0.0,
    "shannon_entropy_bits_per_bit": 1.0,
    "compression_ratio": 0.003,
    "ml_accuracy": 1.0,
    "health": {
      "rct": { "pass": true, "max_run": 1, "cutoff": 34, "approx_p": 0.0000116415 },
      "apt": { "pass": true, "window": 512, "alpha": 0.01, "lower": 227, "upper": 285, "violations": [] }
    },
    "pvals_raw": { "mono_bit": 1.0, "runs": 0.0, "block_frequency": 1.0, "approx_entropy": 0.0 },
    "pvals_fdr": { "mono_bit": 1.0, "runs": 0.0, "block_frequency": 1.0, "approx_entropy": 0.0 },
    "decisions": {
      "mono_bit_pass": true,
      "runs_pass": false,
      "block_frequency_pass": true,
      "approx_entropy_pass": false,
      "ml_pass": false,
      "compression_pass": false,
      "entropy_pass": true,
      "rct_pass": true,
      "apt_pass": true,
      "overall_pass_fdr": false
    },
    "alpha": 0.01,
    "rct_cutoff": 34,
    "apt_window": 512
  }
}
```

Errors:

```json
{"error": "invalid source"}
```

- 400 Bad Request for invalid source
- Other type/validation errors may return 500

## Interpreting Results

### Significance & FDR
- Per-test α: 0.01
- FDR: Benjamini-Hochberg across p-value tests
- Overall PASS: All p-value tests not rejected at α and both health tests pass

### Heuristics
- Entropy PASS if ≥ 0.98 bits/bit
- Compression ratio PASS if ≥ 0.95 (random data ≈ 1.0)
- ML predictability PASS if accuracy ≤ 0.55 (random ≈ 0.50)

## Development

```bash
# Install dev + all interfaces
uv sync --extra dev --extra web --extra gui --extra ai

# Tests
uv run pytest -q

# Lint (optional)
uv run ruff check .
```

## Tests

See [tests/README.md](tests/README.md) for detailed test coverage information.

## Troubleshooting

- GUI/Matplotlib missing: Install GUI dependencies with `uv sync --extra gui`
- API deps missing: Install API dependencies with `uv sync --extra web`
- Docker compose validation: Ensure environment: is a mapping (e.g., {}) if present
- Build issues: Rebuild with `docker compose build --no-cache` if needed

## Roadmap

- Add Cumulative Sums (SP 800-22) and Serial (m=2) tests
- Add Linear Complexity test
- Implement basic min-entropy estimators from SP 800-90B
- P-value diagnostics: histogram / Q-Q plot and numerical clamping
- ML: report accuracy vs. baseline, optional k auto-selection
- Provide known-good baselines (AES-CTR, ChaCha20) and non-crypto PRNG (MT19937)
- Stream/file handling: chunked processing for large inputs
- Features: export reports etc.
- Modular plugin system for tests and sources
- Additional SP 800-22 batteries (Random Excursions, Non-Overlapping/Overlapping Templates)
- Python module support: Install and import as import randqa
- Optional parallelization for large runs and CI integration

## References

- NIST SP 800-22 Rev. 1a - A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications (2010)
- NIST SP 800-90B - Recommendation for the Entropy Sources Used for Random Bit Generation (2018)
- Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical Software, 8(14)

## License

MIT License - see [LICENSE](LICENSE).
