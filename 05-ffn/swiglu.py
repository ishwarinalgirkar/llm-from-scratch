"""SiWGLU feed-forward variant scaffold."""

import numpy as np


def swiglu(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Simple SiWGLU-style forward (conceptual)."""
    a = x @ w1 + b1
    b = x @ w2 + b2
    return a * np.minimum(np.maximum(b, 0), 1)
