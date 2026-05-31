import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import precompute_freqs, apply_rope

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        """
        Args:
            d_model:  total embedding dimension
            n_heads:  number of attention heads
            dropout:  attention dropout probability
            
        Things to initialise:
            - Q, K, V projections: Linear(d_model, d_model, bias=False)
            - output projection:   Linear(d_model, d_model, bias=False)
            - store n_heads, head_dim = d_model // n_heads
        """
        pass
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Args:
            x:   [B, T, d_model]
            cos: precomputed RoPE cos frequencies
            sin: precomputed RoPE sin frequencies
        
        Steps:
            1. Project x to Q, K, V          each [B, T, d_model]
            2. Split into heads              each [B, n_heads, T, head_dim]
            3. Apply RoPE to Q and K only    (not V)
            4. Scaled dot product attention  (use F.scaled_dot_product_attention)
            5. Merge heads                   [B, T, d_model]
            6. Output projection             [B, T, d_model]
        """
        pass