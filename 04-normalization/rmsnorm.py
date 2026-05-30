"""RMSNorm minimal numpy implementation."""

import numpy as np


def rms_norm(x: np.ndarray, eps: float = 1e-8):
    """Apply RMS normalization over the last dimension."""
    norm = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / norm
