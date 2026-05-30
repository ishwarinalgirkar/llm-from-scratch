"""ALiBi positional bias utilities (sketch)."""

import numpy as np


def alibi_bias(seq_len: int, num_heads: int) -> np.ndarray:
    """Return an ALiBi-style bias tensor of shape (num_heads, seq_len, seq_len)."""
    slopes = np.linspace(1.0, 0.1, num_heads)
    arange = np.arange(seq_len)
    diff = arange[None, :] - arange[:, None]
    bias = -np.abs(diff)[None, :, :] * slopes[:, None, None]
    return bias
