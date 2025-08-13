# SP 800-90B Health Tests

This directory contains implementations of NIST SP 800-90B health tests for entropy sources.

## Tests

### `rct.py` - Repetition Count Test (RCT)
- Purpose: Detects stuck or biased entropy sources
- Method: Counts consecutive identical bits
- Threshold: Configurable cutoff (default: 34)
- Output: PASS/FAIL + maximum run length

### `apt.py` - Adaptive Proportion Test (APT)
- Purpose: Detects bias in proportion of 1s
- Method: Non-overlapping windows with binomial bounds
- Threshold: α = 0.01, configurable window size (default: 512)
- Output: PASS/FAIL + violation details 