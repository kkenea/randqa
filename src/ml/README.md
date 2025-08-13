# Machine Learning Predictability

This directory contains machine learning tools for assessing short-term predictability in random sequences.

## Components

### `predictor.py` - Next-Bit Predictor
- Purpose: Assesses predictability using logistic regression
- Method: Trains on k-bit history to predict next bit
- Parameters: 
  - k: History length (default: 8)
  - train_frac: Training split (default: 0.5)
- Output: Prediction accuracy (0.0 to 1.0)

## Interpretation

- Accuracy ≤ 0.55: PASS (random, not predictable)
- Accuracy > 0.55: FAIL (predictable patterns detected)
- Baseline: Random guessing ≈ 0.50 