"""M03: State-conditioned distribution write operator for Information Boltzmann.

Protocol from docs/INFORMATION_BOLTZMANN_MEMORY_HANDOFF.md Section 6.1:
- Interface: forward(x:[N,d], v:[N,d], token_embedding:[H], inverse=False) -> (x_new, v_new, diagnostics)
- Concatenates z = [x, v] in R^{N x 2d}.
- Two-layer alternating affine coupling.
- Conditioned on [z_a_i, token_embedding] with 4-head, width-64 Attention + MLP.
- Strictly invertible: inverse=True recovers exact inputs.
- Permutation-equivariant: all particles share weights with no particle-index embeddings.
- Zero-initialized output projections guarantee exact identity mapping at initialization.
- Bounded s and t activations:
    s = 0.05 * raw_s / sqrt(1 + raw_s^2)
    t = 0.05 * raw_t / sqrt(1 + raw_t^2)
"""
import os
import sys

# Prevent local scripts/ib_local from shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import math
import torch
from torch import nn
from torch.nn import functional as F


class AffineCouplingLayer(nn.Module):
    """Single token-conditioned, attention-based affine coupling layer."""
    def __init__(self, dim: int = 4, hidden_dim: int = 128, heads: int = 4, attn_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.attn_dim = attn_dim
        self.d_head = attn_dim // heads

        # Projections for K and V from invariant partition z_a
        self.key_proj = nn.Linear(dim, attn_dim)
        self.val_proj = nn.Linear(dim, attn_dim)

        # Projection for Q from [z_a, token_embedding]
        self.query_proj = nn.Linear(dim + hidden_dim, attn_dim)

        # MLP producing raw s and t
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim + dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, 2 * dim),
        )

        # Zero-initialize the final projection for exact identity mapping at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_active: torch.Tensor, z_invariant: torch.Tensor, token_embedding: torch.Tensor, inverse: bool = False, diagnostics: bool = False) -> tuple[torch.Tensor, dict | None]:
        """Transform z_active conditioned on z_invariant and token_embedding.

        z_active: [N, dim]
        z_invariant: [N, dim]
        token_embedding: [hidden_dim]
        """
        n = z_invariant.shape[0]
        device = z_invariant.device

        # Expand token embedding across particles: [N, hidden_dim]
        tok_exp = token_embedding.unsqueeze(0).expand(n, -1)
        q_in = torch.cat([z_invariant, tok_exp], dim=-1)  # [N, dim + hidden_dim]

        # Multi-head attention across particles
        Q = self.query_proj(q_in).view(n, self.heads, self.d_head).transpose(0, 1)  # [heads, N, d_head]
        K = self.key_proj(z_invariant).view(n, self.heads, self.d_head).transpose(0, 1)  # [heads, N, d_head]
        V = self.val_proj(z_invariant).view(n, self.heads, self.d_head).transpose(0, 1)  # [heads, N, d_head]

        att = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)  # [heads, N, d_head]
        att = att.transpose(0, 1).contiguous().view(n, self.attn_dim)  # [N, attn_dim]

        # Predict s and t
        mlp_in = torch.cat([att, z_invariant], dim=-1)
        raw_out = self.mlp(mlp_in)
        raw_s, raw_t = raw_out.chunk(2, dim=-1)

        # Bounded activations (SPEC §6.1)
        s = 0.05 * raw_s / torch.sqrt(1.0 + raw_s.square())
        t = 0.05 * raw_t / torch.sqrt(1.0 + raw_t.square())

        if inverse:
            z_new = (z_active - t) * torch.exp(-s)
        else:
            z_new = z_active * torch.exp(s) + t

        diag = None
        if diagnostics:
            diag = {
                'mean_abs_s': float(s.abs().mean().item()),
                'mean_abs_t': float(t.abs().mean().item()),
                'max_abs_s': float(s.abs().max().item()),
                'max_abs_t': float(t.abs().max().item()),
            }
        return z_new, diag


class WriteOperator(nn.Module):
    """M03 Two-layer alternating affine coupling distribution write operator."""
    def __init__(self, dim: int = 4, hidden_dim: int = 128, heads: int = 4, attn_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Layer 0: transforms v conditioned on x
        self.layer0 = AffineCouplingLayer(dim=dim, hidden_dim=hidden_dim, heads=heads, attn_dim=attn_dim)
        # Layer 1: transforms x conditioned on v
        self.layer1 = AffineCouplingLayer(dim=dim, hidden_dim=hidden_dim, heads=heads, attn_dim=attn_dim)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        token_embedding: torch.Tensor,
        inverse: bool = False,
        diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Apply state-conditioned distribution write mapping.

        x: [N, d]
        v: [N, d]
        token_embedding: [H]
        inverse: bool
        diagnostics: bool
        """
        if x.ndim != 2 or v.ndim != 2:
            raise ValueError(f"x and v must be 2D tensors [N, d], got {x.shape} and {v.shape}")
        if x.shape != v.shape:
            raise ValueError(f"x and v shapes must match, got {x.shape} vs {v.shape}")
        if token_embedding.ndim != 1 or token_embedding.shape[0] != self.hidden_dim:
            raise ValueError(f"token_embedding must have shape [{self.hidden_dim}], got {token_embedding.shape}")

        if not inverse:
            # Forward: Layer 0 updates v using x, then Layer 1 updates x using v_new
            v_new, diag0 = self.layer0(z_active=v, z_invariant=x, token_embedding=token_embedding, inverse=False, diagnostics=diagnostics)
            x_new, diag1 = self.layer1(z_active=x, z_invariant=v_new, token_embedding=token_embedding, inverse=False, diagnostics=diagnostics)
        else:
            # Inverse: Reverse layer order. Layer 1 inverts x using v_new, then Layer 0 inverts v using x
            x_orig, diag1 = self.layer1(z_active=x, z_invariant=v, token_embedding=token_embedding, inverse=True, diagnostics=diagnostics)
            v_orig, diag0 = self.layer0(z_active=v, z_invariant=x_orig, token_embedding=token_embedding, inverse=True, diagnostics=diagnostics)
            x_new, v_new = x_orig, v_orig

        diag = None
        if diagnostics:
            initial_ke = 0.5 * float(v.square().sum(-1).mean().item())
            initial_pe = 0.5 * float(x.square().sum(-1).mean().item())
            final_ke = 0.5 * float(v_new.square().sum(-1).mean().item())
            final_pe = 0.5 * float(x_new.square().sum(-1).mean().item())
            diag = {
                'delta_kinetic_energy': final_ke - initial_ke,
                'delta_potential_energy': final_pe - initial_pe,
                'layer0_diag': diag0,
                'layer1_diag': diag1,
            }
        return x_new, v_new, diag
