import numpy as np
from ml.predictor import predictability_score


def test_predictability_on_alternating_sequence():
    """Test that alternating sequences are highly predictable."""
    # Perfectly predictable: 010101...
    bits = np.tile([0, 1], 10_000).astype(np.uint8)
    acc = predictability_score(bits, k=3, train_frac=0.5)
    assert acc is not None and acc > 0.8, (
        f"Alternating sequence should be predictable, got {acc}"
    )


def test_predictability_on_random_sequence():
    """Test that random sequences are not easily predictable."""
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=20_000, dtype=np.uint8)
    acc = predictability_score(bits, k=8, train_frac=0.5)
    assert acc is not None and acc <= 0.6, (
        f"Random sequence should not be predictable, got {acc}"
    )


def test_predictability_on_constant_sequence():
    """Test that constant sequences are highly predictable."""
    # All zeros - should return None since ML needs at least 2 classes
    bits = np.zeros(10_000, dtype=np.uint8)
    acc = predictability_score(bits, k=3, train_frac=0.5)
    assert acc is None, f"Constant sequence should return None, got {acc}"

    # All ones - should return None since ML needs at least 2 classes
    bits = np.ones(10_000, dtype=np.uint8)
    acc = predictability_score(bits, k=3, train_frac=0.5)
    assert acc is None, f"Constant sequence should return None, got {acc}"


def test_predictability_edge_cases():
    """Test predictability with edge cases."""
    # Very short sequence
    bits = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    acc = predictability_score(bits, k=2, train_frac=0.5)
    # Should return None for insufficient data
    assert acc is None, f"Very short sequence should return None, got {acc}"

    # Single bit
    bits = np.array([1], dtype=np.uint8)
    acc = predictability_score(bits, k=1, train_frac=0.5)
    assert acc is None, f"Single bit should return None, got {acc}"

    # Almost constant sequence (mostly zeros with one one)
    bits = np.zeros(10_000, dtype=np.uint8)
    bits[5000] = 1  # Add one bit of variety
    acc = predictability_score(bits, k=3, train_frac=0.5)
    # Should work since there are 2 classes, but be very predictable
    assert acc is not None, "Almost constant sequence should work"
    assert acc > 0.8, f"Almost constant sequence should be predictable, got {acc}"


def test_predictability_different_k_values():
    """Test predictability with different k values."""
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=15_000, dtype=np.uint8)

    # Test different k values
    for k in [2, 4, 8]:
        acc = predictability_score(bits, k=k, train_frac=0.5)
        assert acc is not None, f"Should get result for k={k}"
        assert 0 <= acc <= 1, f"Accuracy should be between 0 and 1, got {acc}"


def test_predictability_different_train_fractions():
    """Test predictability with different training fractions."""
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=20_000, dtype=np.uint8)

    # Test different training fractions
    for train_frac in [0.3, 0.5, 0.7]:
        acc = predictability_score(bits, k=4, train_frac=train_frac)
        assert acc is not None, f"Should get result for train_frac={train_frac}"
        assert 0 <= acc <= 1, f"Accuracy should be between 0 and 1, got {acc}"
