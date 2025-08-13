# Statistical Tests & Metrics

This directory contains implementations of NIST SP 800-22 statistical tests and supporting metrics for randomness assessment.

## NIST SP 800-22 Tests

### `monobit.py` - Monobit (Frequency) Test
- Purpose: Tests overall balance of 0s and 1s
- Output: p-value indicating randomness quality

### `runs.py` - Runs Test
- Purpose: Tests count of consecutive 0s and 1s
- Output: p-value for run patterns

### `block_frequency.py` - Block Frequency Test
- Purpose: Tests local bias within fixed-size blocks
- Output: p-value for block-wise randomness

### `approx_entropy.py` - Approximate Entropy Test
- Purpose: Detects repeating patterns and local regularity
- Output: p-value for pattern detection

## Supporting Metrics

### `entropy.py` - Shannon Entropy
- Purpose: Measures information content per bit
- Output: Bits per bit (0.0 to 1.0)

### `compression.py` - Compression Ratio
- Purpose: Tests compressibility using zlib
- Output: Compression ratio (lower = more compressible) 