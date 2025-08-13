# Test Suite

This directory contains tests for the implemented components of the randqa tool.

## Test Coverage

### Randomness Sources
- `test_bits_utils.py` - Bit generation utilities and source integration

### Statistical Tests
- `test_monobit.py` - Monobit test validation
- `test_runs.py` - Runs test validation  
- `test_block_frequency.py` - Block frequency test validation
- `test_entropy_compression.py` - Entropy and compression metrics

### Machine Learning
- `test_ml_predictor.py` - ML predictability testing

### Utilities
- `test_fdr.py` - FDR (False Discovery Rate) testing

## Running Tests

```bash
# Run all tests
uv run pytest -q

# Run specific test file
uv run pytest tests/test_monobit.py -q

# Run with coverage
uv run pytest --cov=src --cov-report=html
```