"""LayerNorm minimal numpy implementation."""

import numpy as np


def layer_norm(x: np.ndarray, eps: float = 1e-5):
    """Apply layer normalization over the last dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)
