import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _to_supervised(bits: np.ndarray, k: int = 8):
    """
    Build supervised dataset: X = last k bits, y = next bit.
    Returns (X, y) as uint8 arrays or (None, None) if not enough data.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = int(bits.size)
    if n <= k:
        return None, None

    # windows shape: (n - k, k + 1)
    try:
        windows = np.lib.stride_tricks.sliding_window_view(bits, k + 1)
    except Exception:
        # Very small or odd inputs - just bail out cleanly
        return None, None

    if windows.size == 0:
        return None, None

    X = windows[:, :k]
    y = windows[:, -1]
    return X.astype(np.uint8, copy=False), y.astype(np.uint8, copy=False)


def predictability_score(
    bits: np.ndarray, k: int = 8, train_frac: float = 0.5
) -> float | None:
    """
    Train logistic regression to predict next bit from previous k bits.
    Returns accuracy in [0,1] on the hold-out split, or None if insufficient data.
    """
    try:
        X, y = _to_supervised(bits, k)
    except Exception:
        return None

    if X is None or y is None:
        return None

    m = int(X.shape[0])
    if m < 2:  # need at least 2 samples to split
        return None

    # Choose a conservative split to avoid empty train/test
    split = int(m * train_frac)
    if split <= 0 or split >= m:
        return None

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Training labels must have both classes; otherwise LR can't fit
    if np.unique(y_train).size < 2:
        return None

    # Binary LR; deterministic for tests via random_state
    try:
        model = LogisticRegression(max_iter=500, solver="liblinear", random_state=0)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
    except Exception:
        return None

    try:
        return float(accuracy_score(y_test, y_hat))
    except Exception:
        return None
