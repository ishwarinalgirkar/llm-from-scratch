"""Very small backprop-through-network utilities using numpy (educational)."""

import numpy as np


def simple_mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).mean())
