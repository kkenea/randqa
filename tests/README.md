# Tests

This directory contains tests for the randqa package.

## Test Files

- `test_bits_utils.py` - Bit source utilities validation
- `test_mono_bit.py` - Mono_bit test validation
- `test_runs.py` - Runs test validation
- `test_block_frequency.py` - Block frequency test validation
- `test_approx_entropy.py` - Approximate entropy test validation
- `test_entropy_compression.py` - Entropy and compression validation
- `test_ml_predictor.py` - ML predictor validation
- `test_fdr.py` - False discovery rate validation

## Running Tests

Run all tests:
```bash
uv run pytest
```

Run specific test file:
```bash
uv run pytest tests/test_mono_bit.py -q
```

Run with coverage:
```bash
uv run pytest --cov=src
```