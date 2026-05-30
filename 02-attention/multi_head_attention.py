"""Multi-head attention minimal implementation using numpy."""

import numpy as np
from typing import Optional

from .scaled_dot_product import scaled_dot_product


def multi_head_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, num_heads: int = 8, mask: Optional[np.ndarray] = None):
    """Compute multi-head attention by splitting last dimension."""
    batch, seq_len, d_model = q.shape
    assert d_model % num_heads == 0
    d_head = d_model // num_heads

    def split_heads(x):
        return x.reshape(batch, seq_len, num_heads, d_head).transpose(0,2,1,3)

    qh = split_heads(q)
    kh = split_heads(k)
    vh = split_heads(v)

    out = scaled_dot_product(qh, kh, vh, mask=None)
    out = out.transpose(0,2,1,3).reshape(batch, seq_len, d_model)
    return out
