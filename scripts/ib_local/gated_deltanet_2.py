"""Official NVIDIA Gated DeltaNet-2 (GDN-2) architecture implementation.

Reference:
    Hatamizadeh, Choi, Kautz (NVIDIA Research)
    "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention" (arXiv:2605.22791)
    https://github.com/NVlabs/GatedDeltaNet-2

In GDN-2, the state update decouples key-side coordinate erasure and value-side writing:
    S_t = (I - k_t (b_t * k_t)^T) D_t S_{t-1} + k_t (w_t * v_t)^T

Where:
    - D_t in (0, 1)^{d_k}: channel-wise decay along key coordinates
    - b_t in (0, 1)^{d_k}: channel-wise erase gate
    - w_t in (0, 1)^{d_v}: channel-wise write gate
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GatedDeltaNet2Layer(nn.Module):
    """Single layer of Gated DeltaNet-2 with decoupled channel-wise erase, write, and decay."""

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

        # Decoupled channel-wise gates:
        # Channel-wise decay D_t along key/head dimension [heads, dk] = d
        self.decay_proj = nn.Linear(d, d, bias=True)
        # Channel-wise erase gate b_t along key/head dimension [heads, dk] = d
        self.erase_proj = nn.Linear(d, d, bias=True)
        # Channel-wise write gate w_t along value/head dimension [heads, dv] = d
        self.write_proj = nn.Linear(d, d, bias=True)

        # Initializations:
        # Decay bias initialized high to maintain initial long-term memory (~0.92 retention)
        nn.init.constant_(self.decay_proj.bias, 2.5)
        # Erase and write gates initialized neutral (sigmoid(0.0) = 0.5)
        nn.init.constant_(self.erase_proj.bias, 0.0)
        nn.init.constant_(self.write_proj.bias, 0.0)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2 * 3))

    def forward(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Sequential recurrence over sequence of length T.

        Args:
            x: Input tensor [B, T, d].
            state: Recurrent memory matrix [B, H, dk, dv].

        Returns:
            x_next: Updated tensor [B, T, d].
            state_next: Updated state [B, H, dk, dv].
            diagnostics: Dictionary of metrics.
        """
        B, T, d = x.shape
        x_norm = self.norm(x)

        q = self.q_proj(x_norm).view(B, T, self.heads, self.dk)
        k = self.k_proj(x_norm).view(B, T, self.heads, self.dk)
        v = self.v_proj(x_norm).view(B, T, self.heads, self.dv)
        g = self.g_proj(x_norm)

        D = torch.sigmoid(self.decay_proj(x_norm)).view(B, T, self.heads, self.dk)
        b = torch.sigmoid(self.erase_proj(x_norm)).view(B, T, self.heads, self.dk)
        w = torch.sigmoid(self.write_proj(x_norm)).view(B, T, self.heads, self.dv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        S = state
        outputs = []
        state_norms = []

        for t in range(T):
            kt = k[:, t]   # [B, H, dk]
            vt = v[:, t]   # [B, H, dv]
            qt = q[:, t]   # [B, H, dk]
            Dt = D[:, t]   # [B, H, dk]
            bt = b[:, t]   # [B, H, dk]
            wt = w[:, t]   # [B, H, dv]

            # 1. Channel-wise decay: S_decayed = D_t * S_{t-1}
            S_decayed = Dt.unsqueeze(-1) * S

            # 2. Decoupled channel-wise erase: k_t (b_t * k_t)^T S_decayed
            retrieved = torch.einsum("bhk,bhkd->bhd", bt * kt, S_decayed)
            erased = torch.einsum("bhk,bhd->bhkd", kt, retrieved)

            # 3. Decoupled channel-wise write: k_t (w_t * v_t)^T
            write = torch.einsum("bhk,bhd->bhkd", kt, wt * vt)

            # 4. State update: S_t = S_decayed - erased + write
            S = S_decayed - erased + write
            state_norms.append(S.detach().square().mean().sqrt())

            # 5. Readout: S_t^T q_t
            ot = torch.einsum("bhkd,bhk->bhd", S, qt)
            outputs.append(ot)

        out = torch.stack(outputs, dim=1).reshape(B, T, d)
        out = self.o_proj(out * F.silu(g))

        diagnostics = {
            "decay_mean": D.detach().mean(),
            "erase_mean": b.detach().mean(),
            "write_mean": w.detach().mean(),
            "state_norm": torch.stack(state_norms).mean(),
            "state_norm_final": state_norms[-1],
        }
        return x + out, S, diagnostics


class GatedDeltaNet2LM(nn.Module):
    """Language model using stacked Gated DeltaNet-2 layers."""

    architecture = "GatedDeltaNet2-d128-L3-BPE"

    def __init__(
        self,
        vocab_size: int = 50257,
        d: int = 128,
        layers: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d = int(d)
        self.num_layers = int(layers)
        self.heads = int(heads)
        self.dk = self.dv = d // heads

        self.source = nn.Module()
        self.source.embedding = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.source.embedding.weight, std=0.02)

        self.layers = nn.ModuleList(
            [GatedDeltaNet2Layer(d, heads) for _ in range(layers)]
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

        diagnostics["decay_mean"] = torch.stack(
            [diagnostics[f"l{l}_decay_mean"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["erase_mean"] = torch.stack(
            [diagnostics[f"l{l}_erase_mean"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["write_mean"] = torch.stack(
            [diagnostics[f"l{l}_write_mean"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["state_norm"] = torch.stack(
            [diagnostics[f"l{l}_state_norm"] for l in range(self.num_layers)]
        ).mean()
        diagnostics["state_norm_final"] = torch.stack(
            [diagnostics[f"l{l}_state_norm_final"] for l in range(self.num_layers)]
        ).mean()

        return loss, next_state, diagnostics
