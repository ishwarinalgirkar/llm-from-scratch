"""Rotary positional embeddings (RoPE) minimal utilities."""

import numpy as np


def apply_rope(x: np.ndarray, seq_dim: int = -2) -> np.ndarray:
    """Apply a very small RoPE-like rotation to `x` (educational)."""
    # Placeholder: a real RoPE computes sin/cos and rotates half-dims.
    return x
