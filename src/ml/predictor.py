import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _to_supervised(bits: np.ndarray, k: int = 8):
    """
    Build supervised dataset: X = last k bits, y = next bit.
    Returns (X, y) as uint8 arrays or (None, None) if not enough data.
    """
    # Validate k
    if k is None or int(k) < 1:
        return None, None

    bits = np.asarray(bits, dtype=np.uint8)
    n = int(bits.size)
    if n <= k:
        return None, None

    # windows shape: (n - k, k + 1)
    try:
        windows = np.lib.stride_tricks.sliding_window_view(bits, k + 1)
    except Exception:
        # Very small or odd inputs — bail out cleanly
        return None, None

    if windows.size == 0 or windows.shape[0] < 1:
        return None, None

    X = windows[:, :k]
    y = windows[:, -1]
    # Ensure uint8 without copy when possible
    return X.astype(np.uint8, copy=False), y.astype(np.uint8, copy=False)


def predictability_score(
    bits: np.ndarray, k: int = 8, train_frac: float = 0.5
) -> float | None:
    """
    Train logistic regression to predict next bit from previous k bits.
    Returns accuracy in [0,1] on the hold-out split, or None if insufficient data.
    """
    # Validate params early
    try:
        k = int(k)
    except Exception:
        return None
    if k < 1:
        return None

    try:
        train_frac = float(train_frac)
    except Exception:
        return None
    if not (0.0 < train_frac < 1.0):
        return None

    try:
        X, y = _to_supervised(bits, k)
    except Exception:
        return None

    if X is None or y is None:
        return None

    m = int(X.shape[0])
    # Need at least 2 samples to split; be conservative with tiny m
    if m < 2:
        return None

    split = int(m * train_frac)
    if split <= 0 or split >= m:
        return None

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Training labels must contain both classes; otherwise LR cannot fit
    if np.unique(y_train).size < 2:
        return None

    # Binary LR; deterministic outcome for tests via random_state
    try:
        model = LogisticRegression(
            max_iter=500,
            solver="liblinear",
            random_state=0,
        )
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
    except Exception:
        return None

    try:
        return float(accuracy_score(y_test, y_hat))
    except Exception:
        return None
