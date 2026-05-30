"""LoRA (Low-Rank Adaptation) minimal scaffold."""

import numpy as np


def apply_lora(weight: np.ndarray, r: int, alpha: float = 1.0):
    """Return a low-rank update placeholder for a given weight matrix."""
    k, n = weight.shape
    a = np.zeros((k, r))
    b = np.zeros((r, n))
    return alpha * (a @ b)
