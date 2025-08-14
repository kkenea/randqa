# Metrics

This module contains statistical tests and metrics for randomness assessment.

## Statistical Tests

### `mono_bit.py` - Mono_bit (Frequency) Test

The mono_bit test checks the overall balance of 0s and 1s in a binary sequence. It's based on the NIST SP 800-22 specification.

**Function:** `monobit_pvalue(bits: np.ndarray) -> float`

**Returns:** p-value indicating the probability of observing the given imbalance by chance.

**Interpretation:** 
- p > α: Sequence appears balanced (PASS)
- p ≤ α: Sequence shows significant imbalance (FAIL)

### `runs.py` - Runs Test

The runs test counts the number of runs (contiguous sequences of identical bits) and compares against the expected distribution.

**Function:** `runs_pvalue(bits: np.ndarray) -> float`

**Returns:** p-value for the runs test.

### `block_frequency.py` - Block Frequency Test

Splits the sequence into blocks and tests the frequency of 1s within each block.

**Function:** `block_frequency_pvalue(bits: np.ndarray, M: int) -> float`

**Parameters:**
- `bits`: Binary sequence
- `M`: Block size

### `approx_entropy.py` - Approximate Entropy Test

Measures the regularity and predictability of patterns in the sequence.

**Function:** `approximate_entropy_pvalue(bits: np.ndarray, m: int = 2) -> float`

**Parameters:**
- `bits`: Binary sequence  
- `m`: Pattern length (default: 2)

## Entropy and Compression

### `entropy.py` - Shannon Entropy

Calculates the Shannon entropy per bit, measuring information content.

**Function:** `shannon_entropy_bits_per_bit(bits: np.ndarray) -> float`

**Returns:** Entropy in bits per bit (0.0 to 1.0).

### `compression.py` - Compression Ratio

Measures how compressible the data is using zlib compression.

**Function:** `compression_ratio(data: bytes) -> float`

**Returns:** Compression ratio (0.0 to 1.0, higher = less compressible). 