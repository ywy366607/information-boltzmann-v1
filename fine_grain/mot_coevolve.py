"""MoT co-evolution: slices ‖ text tokens in ONE self-attention sequence.

True joint evolution (user requirement):
  seq = [slice_1..slice_G | text_1..text_L]
  seq' = SelfAttn(seq)   # same SDPA, both modalities update together
  X'   = deslice(slice' → points)   # vision keeps full-res field topology
  text' = text portion of seq'

This is NOT: continuous-H-only control, or vision-only slice SDPA then LLM.
Legacy continuous-(X,H) path remains in ``cross_modal_slice_loop.py`` for rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.frontends import FrontendOut
from fine_grain.mm_projector import MMProjector
from fine_grain.models import coords, sparse_deslice_weights


@dataclass
class MoTTrace:
    layer: int
    x_delta: float
    text_delta: float
    G: int
    L: int


class SoftSliceRead(nn.Module):
    """Point field → soft mass-normalized slices + assignment for deslice."""

    def __init__(self, dim: int, n_slices: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dh = dim // heads
        self.g = int(n_slices)
        self.in_x = nn.Linear(dim, dim)
        self.to_slice = nn.Linear(self.dh, self.g)
        nn.init.orthogonal_(self.to_slice.weight)
        self.temp = nn.Sequential(
            nn.Linear(self.dh, self.g), nn.GELU(),
            nn.Linear(self.g, 1), nn.GELU(),
        )
        self.bias = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

    def forward(self, x: torch.Tensor):
        """x: [B,N,C] → slices [B,G,C], w [B,N,G] (head-averaged assignment)."""
        B, N, C = x.shape
        xm = self.in_x(x).reshape(B, N, self.h, self.dh).permute(0, 2, 1, 3)
        temp = torch.clamp(self.temp(xm) + self.bias, min=0.01)
        logits = self.to_slice(xm)  # [B,H,N,G]
        w = F.softmax(logits / temp, dim=-1)
        mass = w.sum(2) + 1e-5
        tok = torch.einsum("bhnc,bhng->bhgc", xm, w) / mass.unsqueeze(-1)
        # merge heads → [B,G,C]
        slices = tok.permute(0, 2, 1, 3).reshape(B, self.g, C)
        w_pts = w.mean(dim=1)  # [B,N,G]
        return slices, w_pts


class MoTJointLayer(nn.Module):
    """One MoT step: joint self-attn on [slices | text], deslice slices → X."""

    def __init__(
        self,
        dim: int,
        n_slices: int,
        n_heads: int = 4,
        deslice_topk: int = 2,
        beta_deslice: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_slices = int(n_slices)
        self.deslice_topk = int(deslice_topk)
        self.beta_deslice = float(beta_deslice)

        self.read = SoftSliceRead(dim, n_slices, heads=n_heads)
        self.ln_s = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.slice_out = nn.Linear(dim, dim)
        self.text_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MoTTrace]:
        """
        x: [B,N,C] point field
        text: [B,L,C] text tokens (same channel width as slices)
        text_mask: [B,L] 1=keep, 0=pad
        Returns: x', text', slices_after, trace
        """
        B, N, C = x.shape
        L = text.shape[1]
        G = self.n_slices
        assert text.shape[0] == B and text.shape[2] == C

        x0, t0 = x, text
        slices, w_pts = self.read(x)  # [B,G,C], [B,N,G]

        # --- MoT sequence: slices first, then text ---
        seq = torch.cat([slices, text], dim=1)  # [B, G+L, C]
        # key padding: slices always valid; text uses mask
        if text_mask is None:
            text_mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
        else:
            text_mask = text_mask.bool()
        kpm = torch.cat([
            torch.zeros(B, G, device=x.device, dtype=torch.bool),
            ~text_mask,
        ], dim=1)  # True = ignore

        seq_n = self.ln_s(seq)
        # Non-causal joint attention: both modalities see each other (true co-evolve)
        attn_out, _ = self.attn(
            seq_n, seq_n, seq_n, key_padding_mask=kpm, need_weights=False,
        )
        seq = seq + attn_out
        seq = seq + self.ff(self.ln_ff(seq))

        slice_new = self.slice_out(seq[:, :G])
        text_new = self.text_out(seq[:, G:])
        # keep pad positions stable
        text_new = torch.where(text_mask.unsqueeze(-1), text_new, t0)

        # --- deslice updated slices back to full-res points ---
        w_write = sparse_deslice_weights(
            w_pts.unsqueeze(1), topk=self.deslice_topk,
        ).squeeze(1)  # [B,N,G]
        # residual write: scatter slice residual
        d_slice = slice_new - slices
        delta = torch.einsum("bng,bgc->bnc", w_write, d_slice)
        x = x + self.beta_deslice * delta

        tr = MoTTrace(
            layer=layer_idx,
            x_delta=float((x - x0).norm(dim=-1).mean().detach()),
            text_delta=float((text_new - t0).norm(dim=-1).mean().detach()),
            G=G,
            L=L,
        )
        return x, text_new, slice_new, tr


class MoTCoEvolveFrontend(nn.Module):
    """L MoT layers: joint slice‖text self-attn + deslice field; interface for LLM."""

    def __init__(
        self,
        d_llm: int,
        res: int = 32,
        T: int = 32,
        dim: int = 64,
        n_layers: int = 2,
        deslice_topk: int = 2,
        beta_deslice: float = 0.5,
        projector: str = "mlp",
        n_heads: int = 4,
    ):
        super().__init__()
        self.res = res
        self.T = int(T)
        self.n_layers = max(1, int(n_layers))
        self.d_llm = d_llm
        self.projector_kind = projector

        # channel width divisible by heads
        heads = n_heads
        dim_head = max(16, (self.T + heads - 1) // heads)
        dim_head = min(dim_head, 64)
        self.C = heads * dim_head
        mix_dim = self.C
        self.mix_dim = mix_dim
        self.heads = heads

        self.stem = nn.Linear(5, mix_dim)
        self.local = nn.Sequential(
            nn.Conv2d(mix_dim, mix_dim, 3, padding=1, groups=mix_dim),
            nn.Conv2d(mix_dim, mix_dim, 1),
        )
        # language embeddings (d_llm) → MoT channel
        self.text_in = nn.Linear(d_llm, mix_dim)
        self.text_out = nn.Linear(mix_dim, d_llm)

        self.layers = nn.ModuleList([
            MoTJointLayer(
                mix_dim, self.T, n_heads=heads,
                deslice_topk=deslice_topk, beta_deslice=beta_deslice,
            )
            for _ in range(self.n_layers)
        ])
        self.proj = MMProjector(mix_dim, d_llm, kind=projector)
        self._last_x = None
        self._last_text = None
        self._last_traces: List[MoTTrace] = []

    def _encode_points(self, img: torch.Tensor) -> torch.Tensor:
        B, _, R, _ = img.shape
        assert R == self.res, (R, self.res)
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = coords(R, img.device).expand(B, -1, -1)
        x = self.stem(torch.cat([pts, p], -1))
        g = x.transpose(1, 2).reshape(B, -1, R, R)
        x = x + self.local(g).flatten(2).transpose(1, 2)
        return x

    def forward_mot(
        self,
        img: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[MoTTrace]]:
        """
        text_emb: [B,L,d_llm] (e.g. LLM input embeddings)
        Returns: x, text_emb_out [B,L,d_llm], slices [B,G,mix], traces
        """
        x = self._encode_points(img)
        text = self.text_in(text_emb)
        traces = []
        slices = None
        for i, layer in enumerate(self.layers):
            x, text, slices, tr = layer(x, text, text_mask=text_mask, layer_idx=i)
            traces.append(tr)
        assert slices is not None
        text_llm = self.text_out(text)
        # residual with original emb for stability
        text_llm = text_llm + text_emb
        self._last_x = x.detach()
        self._last_text = text_llm.detach()
        self._last_traces = traces
        return x, text_llm, slices, traces

    def forward(
        self,
        img: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        external_h: Optional[torch.Tensor] = None,
    ) -> FrontendOut:
        """If text_emb is None, create a single learned null text token (vision-only)."""
        B = img.shape[0]
        if text_emb is None:
            # placeholder length-1 for API compat
            text_emb = torch.zeros(B, 1, self.d_llm, device=img.device, dtype=img.dtype)
            if external_h is not None:
                text_emb = text_emb + external_h.unsqueeze(1)
            text_mask = torch.ones(B, 1, device=img.device)
        x, text_llm, slices, traces = self.forward_mot(img, text_emb, text_mask)
        tok = self.proj(slices)
        return FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={
                "kind": "B_mot",
                "transolver": "3_mot",
                "mot": True,
                "joint_self_attn": True,
                "point_field": True,
                "N_points": int(x.shape[1]),
                "n_layers": self.n_layers,
                "T": self.T,
                "C": self.C,
                "tokens_are_interface_only": True,
                "replaces_legacy_slice_tokens": True,
                "layer_traces": [
                    {
                        "layer": t.layer,
                        "x_delta": t.x_delta,
                        "text_delta": t.text_delta,
                        "G": t.G,
                        "L": t.L,
                    }
                    for t in traces
                ],
            },
        )
