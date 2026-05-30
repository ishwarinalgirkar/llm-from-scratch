"""
LayerNorm vs RMSNorm — side by side
====================================
Paper (LayerNorm): Ba et al. 2016   - https://arxiv.org/abs/1607.06450
Paper (RMSNorm):   Zhang et al. 2019 - https://arxiv.org/abs/1910.07467

LayerNorm:
  mean = mean(x)
  var  = var(x)
  x̂   = (x - mean) / sqrt(var + eps)
  y    = x̂ * weight + bias           ← 2 learnable params: weight, bias

RMSNorm:
  rms  = sqrt(mean(x²) + eps)
  y    = (x / rms) * weight          ← 1 learnable param: weight only

Key insight:
  When mean(x) = 0, std(x) = RMS(x), so LayerNorm = RMSNorm.
  RMSNorm drops mean subtraction and bias — ~10% faster, same convergence.
  Used in: LLaMA, Mistral, Gemma, Falcon (all modern frontier models).
  LayerNorm used in: GPT-2, BERT, original Transformer.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: last dimension of input tensor (d_model)
            eps: numerical stability constant (prevents div by zero)

        Learnable parameters:
            weight (γ): scale, init to ones  — starts as identity
            bias   (β): shift, init to zeros — starts as no offset
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, dim]

        1. mean across last dim       → [B, T, 1]
        2. var  across last dim       → [B, T, 1]  (unbiased=False = divide by N)
        3. normalise: (x-mean)/√var   → [B, T, dim]
        4. scale + shift              → [B, T, dim]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: last dimension of input tensor (d_model)
            eps: numerical stability constant

        Learnable parameters:
            weight (γ): scale only, init to ones
            no bias — residual connections provide additive offset
        """
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, dim]

        1. rms = sqrt(mean(x²) + eps)   → [B, T, 1]
        2. normalise: x / rms            → [B, T, dim]
        3. scale: * weight               → [B, T, dim]
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("LayerNorm tests")
    print("=" * 55)

    ln = LayerNorm(64)
    x  = torch.randn(2, 10, 64)

    assert ln(x).shape == x.shape
    print(f"TEST 1 PASS — shape {x.shape} → {ln(x).shape}")

    assert len(list(ln.parameters())) == 2
    print(f"TEST 2 PASS — 2 params: {[p.shape for p in ln.parameters()]}")

    assert torch.allclose(ln.weight, torch.ones(64))
    assert torch.allclose(ln.bias,   torch.zeros(64))
    print("TEST 3 PASS — weight=ones, bias=zeros at init")

    x2  = torch.randn(32, 16, 64) * 10 + 5
    out = LayerNorm(64)(x2)
    assert out.mean(dim=-1).abs().max().item() < 1e-5
    assert (out.std(dim=-1) - 1.0).abs().max().item() < 1e-2
    print("TEST 4 PASS — output mean≈0, std≈1")

    torch.manual_seed(42)
    x3   = torch.randn(4, 8, 64)
    ref  = nn.LayerNorm(64, eps=1e-6)
    ours = LayerNorm(64)
    ours.weight.data = ref.weight.data.clone()
    ours.bias.data   = ref.bias.data.clone()
    diff = (ref(x3) - ours(x3)).abs().max().item()
    assert diff < 1e-5
    print(f"TEST 5 PASS — matches nn.LayerNorm, max diff={diff:.2e}")

    x4 = torch.randn(2, 8, 64, requires_grad=True)
    LayerNorm(64)(x4).sum().backward()
    assert x4.grad is not None
    print("TEST 6 PASS — gradients flow")

    print()
    print("=" * 55)
    print("RMSNorm tests")
    print("=" * 55)

    rn = RMSNorm(64)
    x  = torch.randn(2, 10, 64)

    assert rn(x).shape == x.shape
    print(f"TEST 1 PASS — shape {x.shape} → {rn(x).shape}")

    assert len(list(rn.parameters())) == 1
    print(f"TEST 2 PASS — 1 param: {[p.shape for p in rn.parameters()]}")

    x2  = torch.tensor([[[3.0, 4.0]]])
    rn2 = RMSNorm(2)
    y2  = rn2(x2)
    rms = torch.sqrt(torch.tensor(12.5) + 1e-6)
    assert torch.allclose(y2, x2 / rms, atol=1e-5)
    print(f"TEST 3 PASS — [3,4]/RMS({rms:.4f}) = {y2[0,0].tolist()}")

    x3 = torch.randn(2, 8, 64, requires_grad=True)
    RMSNorm(64)(x3).sum().backward()
    assert x3.grad is not None
    print("TEST 4 PASS — gradients flow")

    print()
    print("=" * 55)
    print("Side-by-side comparison")
    print("=" * 55)

    torch.manual_seed(7)
    dim = 8
    x5  = torch.randn(1, 1, dim)
    print(f"input:         {[round(v,4) for v in x5[0,0].tolist()]}")
    print(f"mean(x):       {x5.mean(dim=-1).item():.4f}")
    print(f"LayerNorm out: {[round(v,4) for v in LayerNorm(dim)(x5)[0,0].tolist()]}")
    print(f"RMSNorm  out:  {[round(v,4) for v in RMSNorm(dim)(x5)[0,0].tolist()]}")

    x6      = x5 - x5.mean(dim=-1, keepdim=True)
    diff_zm = (LayerNorm(dim)(x6) - RMSNorm(dim)(x6)).abs().max().item()
    print(f"\nWith mean(x)=0 forced:")
    print(f"  diff = {diff_zm:.2e}  ← they are identical when mean=0")

    print()
    print("=" * 55)
    print("Parameter count comparison")
    print("=" * 55)
    d = 512
    print(f"LayerNorm({d}): {sum(p.numel() for p in LayerNorm(d).parameters())} params "
          f"(weight + bias)")
    print(f"RMSNorm({d}):   {sum(p.numel() for p in RMSNorm(d).parameters())} params "
          f"(weight only)")
    print(f"Saving:         {d} params per norm layer — "
          f"small but multiplied by 2×N layers in the model")

    print()
    print("ALL TESTS PASSED")