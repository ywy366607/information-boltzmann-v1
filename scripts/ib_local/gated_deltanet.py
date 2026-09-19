"""Capacity-matched and parameter-matched Gated DeltaNet baseline for BPE language modeling.

This model serves as the unconstrained (non-conservative) baseline to benchmark against
the Continuous Boltzmann Information Medium (CBIM). While CBIM enforces exact unitary
scattering, energy conservation, Cayley spatial transport, and passive quadratic radiation,
Gated DeltaNet (GDN) utilizes unconstrained associative memory matrices updated via
the delta rule:

    S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1}^T k_t) k_t^T

Parameters, vocabulary, embedding tying, and training conditions are matched 1:1 against
CBIM Torus3D (8x8x4, d=128) with ~332k non-embedding parameters.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GatedDeltaNetLayer(nn.Module):
    """Single layer of Gated DeltaNet with data-dependent decay and learning rate."""

    def __init__(self, d: int = 128, heads: int = 4):
        super().__init__()
        if d % heads != 0:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads})")
        self.d, self.heads = d, heads
        self.dk = self.dv = d // heads

        self.norm = nn.RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.g_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.alpha_proj = nn.Linear(d, heads, bias=True)
        self.beta_proj = nn.Linear(d, heads, bias=True)

        # Initializations:
        # Retention bias positive: default to holding memory (~0.92 retention on step 0)
        nn.init.constant_(self.alpha_proj.bias, 2.5)
        nn.init.constant_(self.beta_proj.bias, 0.0)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2 * 4))

    def forward(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass over sequence of length T with recurrent state S.

        Args:
            x: Input tensor of shape [B, T, d].
            state: Recurrent memory matrix of shape [B, H, dk, dv].

        Returns:
            x_next: Updated tensor [B, T, d].
            state_next: Updated memory matrix [B, H, dk, dv].
            diagnostics: Dict of scalar monitoring metrics.
        """
        B, T, d = x.shape
        x_norm = self.norm(x)

        q = self.q_proj(x_norm).view(B, T, self.heads, self.dk)
        k = self.k_proj(x_norm).view(B, T, self.heads, self.dk)
        v = self.v_proj(x_norm).view(B, T, self.heads, self.dv)
        g = self.g_proj(x_norm)

        alpha = torch.sigmoid(self.alpha_proj(x_norm))  # [B, T, H]
        beta = torch.sigmoid(self.beta_proj(x_norm))    # [B, T, H]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        S = state
        outputs = []
        errors = []

        for t in range(T):
            kt = k[:, t].unsqueeze(-1)  # [B, H, dk, 1]
            vt = v[:, t].unsqueeze(-1)  # [B, H, dv, 1]
            qt = q[:, t].unsqueeze(-1)  # [B, H, dk, 1]
            a_t = alpha[:, t, :, None, None]  # [B, H, 1, 1]
            b_t = beta[:, t, :, None, None]   # [B, H, 1, 1]

            # Retrieval & Prediction Error (Delta rule: v - S^T k)
            v_hat = torch.matmul(S.transpose(-1, -2), kt)  # [B, H, dv, 1]
            e_t = vt - v_hat  # [B, H, dv, 1]
            errors.append(e_t.detach().square().mean())

            # Associative Memory Update
            delta = torch.matmul(kt, e_t.transpose(-1, -2))  # [B, H, dk, dv]
            S = a_t * S + b_t * delta

            # Readout
            ot = torch.matmul(S.transpose(-1, -2), qt).squeeze(-1)  # [B, H, dv]
            outputs.append(ot)

        out = torch.stack(outputs, dim=1).reshape(B, T, d)
        out = self.o_proj(out * F.silu(g))

        diagnostics = {
            "alpha_mean": alpha.detach().mean(),
            "beta_mean": beta.detach().mean(),
            "error_norm": torch.stack(errors).mean().sqrt(),
            "state_norm": S.detach().square().mean().sqrt(),
        }
        return x + out, S, diagnostics


class GatedDeltaNetLM(nn.Module):
    """Language model using stacked Gated DeltaNet layers matching CBIM capacity."""

    architecture = "GatedDeltaNet-d128-L4-BPE"

    def __init__(
        self,
        vocab_size: int = 50257,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d = int(d)
        self.num_layers = int(layers)
        self.heads = int(heads)
        self.dk = self.dv = d // heads

        # We nest embedding inside source to match TruncatedInternalTimeGraphTrainer's
        # lexical parameter group convention ("source.embedding.weight").
        self.source = nn.Module()
        self.source.embedding = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.source.embedding.weight, std=0.02)

        self.layers = nn.ModuleList(
            [GatedDeltaNetLayer(d, heads) for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(d)
        self.decoder = nn.Linear(d, vocab_size, bias=False)
        self.decoder.weight = self.source.embedding.weight

    def initial_state(
        self, batch_size: int, device=None, dtype=None
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        dev = device or parameter.device
        dt = dtype or parameter.dtype
        return torch.zeros(
            self.num_layers, batch_size, self.heads, self.dk, self.dv,
            device=dev, dtype=dt,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Sequential or chunk forward pass.

        Args:
            input_ids: [B, T] token IDs.
            targets: [B, T] next token targets.
            state: [L, B, H, dk, dv] recurrent states across all layers.

        Returns:
            loss: Cross-entropy scalar loss (if targets provided).
            next_state: [L, B, H, dk, dv] updated states.
            diagnostics: Dict of metrics across layers.
        """
        B, T = input_ids.shape
        if state is None:
            state = self.initial_state(B, device=input_ids.device)

        h = self.source.embedding(input_ids)
        next_states = []
        diagnostics = {}

        for l, layer in enumerate(self.layers):
            h, next_s, layer_diag = layer(h, state[l])
            next_states.append(next_s)
            for k, v in layer_diag.items():
                diagnostics[f"l{l}_{k}"] = v

        h = self.final_norm(h)
        logits = self.decoder(h)
        next_state = torch.stack(next_states, dim=0)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )

        # Global summary diagnostics
        diagnostics["alpha_mean"] = torch.stack(
            [diagnostics[f"l{l}_alpha_mean"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["beta_mean"] = torch.stack(
            [diagnostics[f"l{l}_beta_mean"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["state_norm"] = torch.stack(
            [diagnostics[f"l{l}_state_norm"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["error_norm"] = torch.stack(
            [diagnostics[f"l{l}_error_norm"] for l in range(self.num_layers)]
        ).mean()

        return loss, next_state, diagnostics
