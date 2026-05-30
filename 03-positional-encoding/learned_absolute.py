"""Learned absolute positional embeddings scaffold."""

import numpy as np


def learned_positional_embeddings(seq_len: int, d_model: int) -> np.ndarray:
    """Return learned positional embeddings initialized randomly."""
    return np.random.randn(seq_len, d_model) * 0.02
