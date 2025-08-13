# Randomness Sources

This directory contains implementations of various random number generators for testing and comparison.

## Generators

### `lcg.py` - Linear Congruential Generator
- Purpose: Demonstrates poor randomness (known weaknesses)
- Parameters: Configurable seed, multiplier, increment, and modulus

### `xorshift.py` - XorShift32 Generator
- Purpose: Better than LCG but still not cryptographically secure
- Parameters: Configurable seed

### `os_random.py` - Operating System Random
- Purpose: Cryptographically secure random source
- Source: OS-provided entropy (e.g., /dev/urandom on Linux) 