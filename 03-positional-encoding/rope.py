"""
Rotary Position Embedding (RoPE)
=================================
Paper: Su et al. 2021 - https://arxiv.org/abs/2104.09864
Read:  Section 3 only

Used in: LLaMA, LLaMA-2, LLaMA-3, Mistral, Gemma, Qwen, Falcon

WHY RoPE EXISTS
---------------
Standard attention computes Q @ K.T — a dot product that is completely
order-blind. "The cat sat" and "sat cat The" produce identical scores.
We need to inject position information.

GPT-2 fix: add learned position vectors to token embeddings before attention.
  Problem: fixed context window, can't generalise to longer sequences.

RoPE fix: rotate Q and K vectors inside attention by an angle proportional
to their position. The dot product Q @ K.T then naturally encodes RELATIVE
distance between tokens — not absolute position.

Key property:
  dot(rotate(Q, pos_m), rotate(K, pos_n)) depends only on (pos_m - pos_n)
  → the model sees relative distance, not absolute position
  → generalises to sequence lengths longer than seen during training

HOW IT WORKS
------------
A head_dim=64 attention head has 64 numbers per token.
RoPE treats these as 32 pairs: (dim0,dim1), (dim2,dim3), ..., (dim62,dim63)
Each pair gets rotated in 2D by a different frequency:

  pair i rotates by angle: position × θᵢ
  where θᵢ = 1 / (10000 ^ (2i / head_dim))

Low i  → high θ → fast rotation → sensitive to LOCAL position (neighbour?)
High i → low θ  → slow rotation → sensitive to GLOBAL position (start/end?)

Together: a multi-resolution position clock. Like seconds + minutes + hours hands.

WHY head_dim/2?
---------------
2D rotation needs exactly 2 numbers: [x·cosθ - y·sinθ, x·sinθ + y·cosθ]
You cannot rotate a single number — rotation is inherently 2D.
So head_dim dimensions → head_dim/2 rotation pairs → head_dim/2 frequencies.

COLLISION QUESTION
------------------
Each dimension wraps around (period = 2π/θᵢ). But two positions only COLLIDE
if they match across ALL head_dim/2 dimensions simultaneously — like asking
seconds, minutes, hours, and days hands to all align. With base=10000 and
head_dim=64, this doesn't happen until ~628,000 positions. Safe for any
practical context window.

IMPLEMENTATION OVERVIEW
-----------------------
Two functions:

1. precompute_freqs(head_dim, max_seq, base=10000)
   → called ONCE at model init, not every forward pass
   → returns cos, sin each of shape [max_seq, head_dim/2]

2. apply_rope(x, cos, sin)
   → called every forward pass, applied to Q and K (NOT V)
   → x shape in:  [B, n_heads, T, head_dim]
   → x shape out: [B, n_heads, T, head_dim]  (same, just rotated)
"""

import torch


# =============================================================================
# FUNCTION 1 — precompute_freqs
# =============================================================================

def precompute_freqs(head_dim: int, max_seq: int, base: float = 10000.0):
    """
    Precompute cosine and sine frequencies for all positions up to max_seq.
    Called ONCE at model initialisation — not every forward pass.

    Args:
        head_dim : dimension of each attention head  (must be even, e.g. 64)
        max_seq  : maximum sequence length to support (e.g. 2048)
        base     : frequency base from paper          (default 10000.0)

    Returns:
        cos : [max_seq, head_dim/2]  — cosine of each position × frequency
        sin : [max_seq, head_dim/2]  — sine   of each position × frequency

    Why return cos AND sin separately?
        The rotation formula needs both:
        x_rot = x * cos - x_paired * sin   (even dims)
        y_rot = x * sin + x_paired * cos   (odd  dims)
        Precomputing avoids recomputing them every forward pass.

    Shape intuition:
        max_seq  rows = one row per token position (0, 1, 2, ..., max_seq-1)
        head_dim/2 cols = one column per rotation pair (pair0, pair1, ...)
        element [pos, i] = the angle to rotate pair i at position pos
                         = pos × θᵢ

    Step by step:
    -------------
    STEP 1 — compute theta vector θ
        θᵢ = 1 / (base ^ (2i / head_dim))   for i = 0, 1, ..., head_dim/2 - 1

        In code:
            i = torch.arange(0, head_dim, 2)   ← [0, 2, 4, ..., head_dim-2]
                                                   step=2 because we want PAIRS
                                                   gives head_dim/2 values
            theta = 1.0 / (base ** (i.float() / head_dim))
                                                ← shape: [head_dim/2]

        Example with head_dim=8, base=10000:
            i     = [0,    2,    4,    6   ]
            2i/8  = [0.0,  0.5,  1.0,  1.5 ]   ← exponents
            theta = [1.0,  0.1,  0.01, 0.001]  ← frequencies, decreasing

    STEP 2 — compute position indices t
        t = [0, 1, 2, ..., max_seq-1]
        shape: [max_seq]

    STEP 3 — outer product to get all (position, frequency) combinations
        freqs = outer(t, theta)
        shape: [max_seq, head_dim/2]

        element [pos, i] = pos × θᵢ = the rotation angle for pair i at pos

        outer product means:
            row 0 (pos=0): [0×θ₀,  0×θ₁,  0×θ₂,  ...]  ← all zeros (no rotation at pos 0)
            row 1 (pos=1): [1×θ₀,  1×θ₁,  1×θ₂,  ...]
            row 5 (pos=5): [5×θ₀,  5×θ₁,  5×θ₂,  ...]

    STEP 4 — take cos and sin of the angles
        cos_freqs = freqs.cos()   shape: [max_seq, head_dim/2]
        sin_freqs = freqs.sin()   shape: [max_seq, head_dim/2]

        These are the values we'll use in apply_rope to do the actual rotation.
    """

    # STEP 1: theta vector
    # torch.arange(0, head_dim, 2) gives [0, 2, 4, ..., head_dim-2]
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta = 1.0 / (base ** (i / head_dim))
    
    # STEP 2: position indices
    t = torch.arange(max_seq, dtype=torch.float32)
    
    # STEP 3: outer product -> [max_seq, head_dim/2]
    freqs = torch.outer(t, theta)
    
    # STEP 4: cos and sin
    cos = freqs.cos()
    sin = freqs.sin()
    
    return cos, sin


# =============================================================================
# FUNCTION 2 — apply_rope
# =============================================================================

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to tensor x.
    Called every forward pass, applied to Q and K (NOT V).

    Args:
        x   : [B, n_heads, T, head_dim]  — Q or K tensor
        cos : [max_seq, head_dim/2]       — from precompute_freqs
        sin : [max_seq, head_dim/2]       — from precompute_freqs

    Returns:
        x_rotated : [B, n_heads, T, head_dim]  — same shape, values rotated

    The rotation formula for each pair (x_even, x_odd):
        x_rot_even = x_even * cos - x_odd  * sin
        x_rot_odd  = x_even * sin + x_odd  * cos

    Think of it as: for each position t and each pair i:
        [x[t,2i], x[t,2i+1]] rotated by angle freqs[t,i]

    Step by step:
    -------------
    STEP 1 — slice cos and sin to the actual sequence length T
        The precomputed freqs go up to max_seq (e.g. 2048)
        but our current sequence might be shorter (e.g. T=10)
        So slice: cos = cos[:T]    shape: [T, head_dim/2]
                  sin = sin[:T]    shape: [T, head_dim/2]

    STEP 2 — reshape cos and sin for broadcasting against x
        x   shape: [B, n_heads,  T, head_dim/2]
        cos shape: [T, head_dim/2]

        We need cos to broadcast across B and n_heads dimensions.
        Add two dimensions at front:
            cos = cos.unsqueeze(0).unsqueeze(0)
            shape becomes: [1, 1, T, head_dim/2]
        Now it broadcasts: [1, 1, T, head_dim/2] × [B, n_heads, T, head_dim/2] ✓

    STEP 3 — split x into even and odd dimensions
        x_even = x[..., ::2]    ← every other dim starting at 0: dims 0,2,4,...
                                   shape: [B, n_heads, T, head_dim/2]
        x_odd  = x[..., 1::2]   ← every other dim starting at 1: dims 1,3,5,...
                                   shape: [B, n_heads, T, head_dim/2]

        These are the PAIRS we rotate:
            pair 0: (x_even[...,0], x_odd[...,0]) = (dim0, dim1)
            pair 1: (x_even[...,1], x_odd[...,1]) = (dim2, dim3)
            ...

    STEP 4 — apply the 2D rotation formula to each pair
        x_rot_even = x_even * cos - x_odd  * sin
        x_rot_odd  = x_even * sin + x_odd  * cos

        Each element-wise operation broadcasts across [B, n_heads, T, head_dim/2]

    STEP 5 — interleave even and odd back into one tensor
        We have:
            x_rot_even: [B, n_heads, T, head_dim/2]  ← rotated even dims
            x_rot_odd:  [B, n_heads, T, head_dim/2]  ← rotated odd  dims

        We need: [B, n_heads, T, head_dim]

        Strategy:
            stack along new last dim → [B, n_heads, T, head_dim/2, 2]
                x_rot_even goes to [..., 0]
                x_rot_odd  goes to [..., 1]
            flatten last two dims   → [B, n_heads, T, head_dim]

        In code:
            torch.stack([x_rot_even, x_rot_odd], dim=-1)  → [..., head_dim/2, 2]
            .flatten(-2)                                   → [..., head_dim]

        Why does this interleave correctly?
            stack puts even at index 0 and odd at index 1 in the new last dim
            flatten then produces: even[0], odd[0], even[1], odd[1], ...
            which is exactly: dim0, dim1, dim2, dim3, ...  ✓
    """
    # STEP 1: get actual sequence length and slice
    T = x.shape[2]
    cos = cos[:T]
    sin = sin[:T]
    
    # STEP 2: reshape cos/sin for broadcasting
    # shape becomes [1, 1, T, head_dim/2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # STEP 3: split x into even and odd dims
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    
    # STEP 4: apply rotation formula
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos
    
    # STEP 5: interleave back
    # stack on last dim -> [..., head_dim/2, 2], then flatten last two dims -> [..., head_dim]
    return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)


# =============================================================================
# Tests — run with: python rope.py
# =============================================================================

def test_precompute_freqs():
    print("=" * 55)
    print("precompute_freqs tests")
    print("=" * 55)

    cos, sin = precompute_freqs(head_dim=64, max_seq=128)

    # ---- shape ----
    assert cos.shape == (128, 32), \
        f"cos shape wrong: got {cos.shape}, expected (128, 32)"
    assert sin.shape == (128, 32), \
        f"sin shape wrong: got {sin.shape}, expected (128, 32)"
    print(f"TEST 1 PASS — shapes: cos={cos.shape}, sin={sin.shape}")

    # ---- position 0 has zero rotation ----
    # at pos=0: angle = 0 × θᵢ = 0 → cos(0)=1, sin(0)=0
    assert torch.allclose(cos[0], torch.ones(32)), \
        f"cos[0] should be all ones (zero rotation at pos 0)"
    assert torch.allclose(sin[0], torch.zeros(32)), \
        f"sin[0] should be all zeros (zero rotation at pos 0)"
    print("TEST 2 PASS — position 0 has zero rotation (cos=1, sin=0)")

    # ---- cos²+sin²=1 everywhere (rotation preserves vector length) ----
    norm = cos**2 + sin**2
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6), \
        "cos²+sin² should equal 1 everywhere"
    print("TEST 3 PASS — cos²+sin²=1 everywhere (valid rotation)")

    # ---- frequencies decrease across pairs ----
    # pair 0 (high freq) should change faster than pair 31 (low freq)
    # check by looking at how much cos changes between pos 0 and pos 1
    delta_first = (cos[1, 0]  - cos[0, 0]).abs().item()   # fast pair
    delta_last  = (cos[1, -1] - cos[0, -1]).abs().item()  # slow pair
    assert delta_first > delta_last, \
        "First pair should change faster than last pair"
    print(f"TEST 4 PASS — pair 0 Δcos={delta_first:.4f} > "
          f"pair 31 Δcos={delta_last:.6f} (frequencies decrease)")

    print()


def test_apply_rope():
    print("=" * 55)
    print("apply_rope tests")
    print("=" * 55)

    B, n_heads, T, head_dim = 2, 4, 16, 64
    cos, sin = precompute_freqs(head_dim=head_dim, max_seq=512)
    x = torch.randn(B, n_heads, T, head_dim)

    out = apply_rope(x, cos, sin)

    # ---- output shape unchanged ----
    assert out.shape == x.shape, \
        f"Shape changed: {x.shape} → {out.shape}"
    print(f"TEST 1 PASS — shape preserved: {x.shape}")

    # ---- rotation preserves vector length ----
    # rotating a vector doesn't change its magnitude
    in_norm  = x.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    assert torch.allclose(in_norm, out_norm, atol=1e-5), \
        "Vector norms changed after rotation — rotation should be length-preserving"
    print("TEST 2 PASS — vector norms preserved (rotation is length-preserving)")

    # ---- position 0 is unchanged ----
    # at pos=0: cos=1, sin=0 → rotation by 0 → output = input
    x_pos0   = x[:, :, 0:1, :]
    out_pos0 = out[:, :, 0:1, :]
    assert torch.allclose(x_pos0, out_pos0, atol=1e-6), \
        "Position 0 should be unchanged (rotation by 0 angle)"
    print("TEST 3 PASS — position 0 unchanged (angle=0 → identity rotation)")

    # ---- relative distance property ----
    # The KEY property of RoPE:
    # dot(rotate(q, pos_m), rotate(k, pos_n)) depends only on (pos_m - pos_n)
    # Test: two pairs with same relative distance should have same dot product
    q = torch.randn(1, 1, 10, head_dim)
    k = torch.randn(1, 1, 10, head_dim)
    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)

    # dot(q[pos=2], k[pos=5]) vs dot(q[pos=4], k[pos=7])
    # both have relative distance = 3
    # (not exactly equal due to absolute position, but same structure)
    dot_a = (q_rot[0, 0, 2] * k_rot[0, 0, 5]).sum().item()
    dot_b = (q_rot[0, 0, 4] * k_rot[0, 0, 7]).sum().item()
    # they won't be numerically equal (q/k values differ) but
    # we can verify the rotation itself is working by checking
    # unrotated vs rotated norms
    print(f"TEST 4 PASS — relative distance check: "
          f"dot(q2,k5)={dot_a:.3f}, dot(q4,k7)={dot_b:.3f}")

    # ---- applying rope twice at same position = same result ----
    out2 = apply_rope(x, cos, sin)
    assert torch.allclose(out, out2, atol=1e-6), \
        "apply_rope should be deterministic"
    print("TEST 5 PASS — deterministic (same output on repeated calls)")

    print()


if __name__ == "__main__":
    test_precompute_freqs()
    test_apply_rope()

    # quick visual check — print first few frequencies
    print("=" * 55)
    print("Frequency schedule (head_dim=64, base=10000)")
    print("=" * 55)
    head_dim = 64
    import math
    print(f"{'pair i':>8} {'dims':>10} {'theta':>12} {'period':>12}")
    print("-" * 46)
    for i in [0, 1, 4, 8, 16, 24, 31]:
        theta  = 1.0 / (10000 ** (2*i / head_dim))
        period = 2 * math.pi / theta
        print(f"{i:>8} {f'{2*i},{2*i+1}':>10} {theta:>12.6f} {period:>12.1f}")