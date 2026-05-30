"""Scaled dot-product attention (numpy-based minimal version)."""

import numpy as np
from typing import Optional


def scaled_dot_product(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None):
    """Compute attention weights and output.

    q,k,v : (..., seq_len, d_k)
    """
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v
