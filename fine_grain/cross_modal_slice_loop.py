"""Transolver3 multimodal core: full-res fields + shared workspace U.

**Transolver3 in multimodal = natural application of slice/deslice as
Read/Write into a temporary shared workspace — not “vision = slice tokens.”**

Legacy ``SliceFrontend`` (permanent G slice tokens → LLM) is **abandoned as
the product vision path**; slices remain only an ephemeral interact *view*.

**Thesis (locked):** pay off only if we *keep* full-res field ``X[B,N,C]``
(``N=H*W``) and **deslice-write** every step. Collapsing to static vision
tokens discards deslice and is *not* this product.

Coupled state evolution (not one-way FE → LLM tokens):

.. math::

    X_{t+1} = F_\\mathrm{vision}(X_t, H_t)

    H_{t+1} = F_\\mathrm{language}(H_t, \\mathrm{Read}(X_{t+1}))

Narrative (one discrete time step):

  full-res field X
    → text **reads** (soft mass pool → slices, query from H)
    → stage **judgment** in H
    → judgment **deslice-writes** back onto points
    → field **reorganizes** (Slice → global interact → deslice)
    → next step **re-reads** the live field

Slices are a temporary interact *view* (budget G). Projected LLM tokens after
the loop are an optional *interface*, never a replacement for X.

See ``results/published/FINAL_BIDIR_POINT_FIELD.md`` and
``results/published/NATIVE_MULTIMODAL_WORKSPACE.md`` (shared workspace U;
each modality keeps topology; Read_*/Write_* ports — not all→text-tokens).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.frontends import FrontendOut
from fine_grain.mm_projector import MMProjector
from fine_grain.models import (
    NS_STEPS_DEFAULT,
    AdaTempSlice,
    Block,
    apply_slice_flags,
    coords,
    deslice_support_size,
    sparse_deslice_weights,
)


@dataclass
class LayerTrace:
    """Diagnostics for one bidirectional step (X,H) → (X',H')."""

    layer: int
    x_delta_norm: float
    h_delta_norm: float
    read_entropy: float
    write_applied: bool
    support: float


class CrossModalSliceLayer(nn.Module):
    """One shared cell implementing F_vision + Read + F_language."""

    def __init__(
        self,
        mix_dim: int,
        h_dim: int,
        n_slices: int,
        heads: int = 4,
        dim_head: int = 16,
        deslice_topk: int = 2,
        write_h_into_x: bool = True,
        beta_write: float = 0.3,
        beta_cond: float = 1.0,
        stiefel: bool = True,
    ):
        super().__init__()
        self.mix_dim = mix_dim
        self.h_dim = h_dim
        self.n_slices = int(n_slices)
        self.write_h_into_x = bool(write_h_into_x)
        self.beta_write = float(beta_write)
        self.beta_cond = float(beta_cond)
        self.deslice_topk = int(deslice_topk)

        # --- F_vision pieces: H-condition + H-deslice-write + global reorganize ---
        self.h_to_cond = nn.Linear(h_dim, mix_dim)  # broadcast bias onto points
        self.h_to_slice_val = nn.Linear(h_dim, mix_dim)
        self.write_gate = nn.Linear(mix_dim + h_dim, self.n_slices)
        self.write_proj = nn.Linear(mix_dim, mix_dim)

        self.visual = Block(
            mix_dim,
            AdaTempSlice(
                mix_dim, heads=heads, dim_head=dim_head,
                slice_num=self.n_slices, norm="mass",
            ),
        )
        wrap = nn.Module()
        wrap.blocks = nn.ModuleList([self.visual])
        apply_slice_flags(
            wrap,
            dict(
                nog=True,
                stiefel_ns=bool(stiefel),
                deslice_topk=int(deslice_topk),
                ns_steps=NS_STEPS_DEFAULT,
            ),
        )

        # --- Read(X; H): query-conditioned soft pool over slices ---
        self.h_to_q = nn.Linear(h_dim, mix_dim)
        self.read_score = nn.Linear(mix_dim, 1)
        self.read_out = nn.Linear(mix_dim, h_dim)

        # --- F_language: residual text update from read ---
        self.h_ln = nn.LayerNorm(h_dim)
        self.h_mlp = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
        )

    # ------------------------------------------------------------------ F_vision

    def _write_h_into_field(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Judgment write-back: inject H into point field before reorganize.

        Soft slice gate from (mean X, H) → H-valued slices → deslice to points.
        Uses only **current** batch tensors (never stale mixer caches).
        """
        if not self.write_h_into_x:
            return x
        B, N, C = x.shape
        G = self.n_slices
        assert h.shape[0] == B, (h.shape, B)
        val = self.h_to_slice_val(h)  # [B,C]
        x_sum = x.mean(dim=1)
        gate = torch.softmax(self.write_gate(torch.cat([x_sum, h], dim=-1)), dim=-1)  # [B,G]
        slice_val = self.write_proj(gate.unsqueeze(-1) * val.unsqueeze(1))  # [B,G,C]

        # Prefer geometry from previous reorg if shapes match this batch; else gate.
        mix = self.visual.mix
        w = getattr(mix, "last_w", None)
        use_geo = (
            w is not None
            and torch.is_tensor(w)
            and w.dim() == 4
            and w.shape[0] == B
            and w.shape[2] == N
            and w.shape[3] == G
        )
        if use_geo:
            w_pts = w.mean(dim=1).to(dtype=x.dtype, device=x.device)  # [B,N,G]
            w_write = sparse_deslice_weights(
                w_pts.unsqueeze(1), topk=self.deslice_topk,
            ).squeeze(1)
        else:
            w_write = gate.unsqueeze(1).expand(B, N, G)

        delta = torch.einsum("bng,bgc->bnc", w_write, slice_val)
        return x + self.beta_write * delta

    def F_vision(self, x: torch.Tensor, h: torch.Tensor):
        """X_{t+1} = F_vision(X_t, H_t).

        Substeps:
          a) condition points by H (query bias)
          b) write H into field via slice/deslice (judgment → vision)
          c) global reorganize: Slice → interact → deslice → X'
        """
        assert x.shape[0] == h.shape[0], (x.shape, h.shape)
        # (a) H-conditioned residual bias on the point stream
        cond = self.h_to_cond(h).unsqueeze(1)  # [B,1,C]
        x_h = x + self.beta_cond * cond
        # (b) explicit judgment write-back into X
        x_h = self._write_h_into_field(x_h, h)
        # (c) visual global reorganize (field already carries H)
        x_new, slices = self.visual(x_h)
        return x_new, slices

    # ------------------------------------------------------------------ Read + F_language

    def Read(self, slices: torch.Tensor, h: torch.Tensor):
        """Dynamic read of slices of X_{t+1} with text query from H_t.

        Returns ``(read, attn, entropy)``; read is in H-space.
        """
        q = self.h_to_q(h).unsqueeze(1)  # [B,1,C]
        scored = self.read_score(slices + q).squeeze(-1)  # [B,G]
        attn = torch.softmax(scored, dim=-1)
        ent = -(attn * (attn.clamp_min(1e-8).log())).sum(-1).mean()
        pooled = torch.einsum("bg,bgc->bc", attn, slices)
        read = self.read_out(pooled)
        return read, attn, ent

    def F_language(self, h: torch.Tensor, read: torch.Tensor):
        """H_{t+1} = F_language(H_t, Read(X_{t+1})). Residual stage judgment."""
        delta = self.h_mlp(torch.cat([self.h_ln(h), read], dim=-1))
        return h + delta, delta

    # ------------------------------------------------------------------ one time step

    def forward(self, x: torch.Tensor, h: torch.Tensor, layer_idx: int = 0):
        """One bidirectional step: (X_t, H_t) → (X_{t+1}, H_{t+1})."""
        x0, h0 = x, h

        # X_{t+1} = F_vision(X_t, H_t)
        x, slices = self.F_vision(x, h)
        # Read(X_{t+1}) with query from H_t
        read, attn, ent = self.Read(slices, h0)
        # H_{t+1} = F_language(H_t, read)
        h, h_delta = self.F_language(h0, read)

        support = float("nan")
        mix = self.visual.mix
        if getattr(mix, "last_w_write", None) is not None:
            support = float(deslice_support_size(mix.last_w_write).mean())

        trace = LayerTrace(
            layer=layer_idx,
            x_delta_norm=float((x - x0).norm(dim=-1).mean().detach()),
            h_delta_norm=float(h_delta.norm(dim=-1).mean().detach()),
            read_entropy=float(ent.detach()),
            write_applied=self.write_h_into_x,
            support=support,
        )
        return x, h, slices, trace

    # Back-compat aliases (old test / docs names map onto the equation form)
    def step1_visual_global(self, x, h=None):
        """Deprecated name: use F_vision. If h is None, pure reorganize."""
        if h is None:
            return self.visual(x)
        return self.F_vision(x, h)

    def step2_query_read(self, slices, h):
        return self.Read(slices, h)

    def step3_text_update(self, h, read):
        return self.F_language(h, read)


class CrossModalSliceFrontend(nn.Module):
    """L steps of bidirectional (X,H) evolution, then project slices → LLM tokens.

    ``external_h`` seeds H_0 (e.g. last LLM hidden for outer two-look).
    """

    def __init__(
        self,
        d_llm: int,
        res: int = 32,
        T: int = 32,
        dim: int = 64,
        n_layers: int = 2,
        deslice_topk: int = 2,
        writeback: bool = True,
        beta_write: float = 0.3,
        shared_layer: bool = True,
        projector: str = "mlp",
        h_dim: Optional[int] = None,
    ):
        super().__init__()
        self.res = res
        self.T = int(T)
        self.n_layers = max(1, int(n_layers))
        self.d_llm = d_llm
        self.h_dim = int(h_dim or d_llm)
        self.projector_kind = projector
        self.shared_layer = bool(shared_layer)

        heads = 4
        dim_head = max(16, min(64, (self.T + heads - 1) // heads))
        self.heads, self.dim_head = heads, dim_head
        self.C = heads * dim_head
        mix_dim = self.C
        self.mix_dim = mix_dim

        self.stem = nn.Linear(5, mix_dim)
        self.local = nn.Sequential(
            nn.Conv2d(mix_dim, mix_dim, 3, padding=1, groups=mix_dim),
            nn.Conv2d(mix_dim, mix_dim, 1),
        )
        self.h0 = nn.Parameter(torch.zeros(1, self.h_dim))
        self.h_from_ext = nn.Linear(d_llm, self.h_dim)

        def _make_layer():
            return CrossModalSliceLayer(
                mix_dim=mix_dim,
                h_dim=self.h_dim,
                n_slices=self.T,
                heads=heads,
                dim_head=dim_head,
                deslice_topk=deslice_topk,
                write_h_into_x=writeback,
                beta_write=beta_write,
            )

        if self.shared_layer:
            self.layer = _make_layer()
            self.layers = None
        else:
            self.layer = None
            self.layers = nn.ModuleList([_make_layer() for _ in range(self.n_layers)])

        self.proj = MMProjector(mix_dim, d_llm, kind=projector)
        self._last_traces: List[LayerTrace] = []
        self._last_h: Optional[torch.Tensor] = None
        self._last_x: Optional[torch.Tensor] = None

    def _encode_points(self, img: torch.Tensor) -> torch.Tensor:
        B, _, R, _ = img.shape
        assert R == self.res, (R, self.res)
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = coords(R, img.device).expand(B, -1, -1)
        x = self.stem(torch.cat([pts, p], -1))
        g = x.transpose(1, 2).reshape(B, -1, R, R)
        x = x + self.local(g).flatten(2).transpose(1, 2)
        return x

    def forward(
        self,
        img: torch.Tensor,
        external_h: Optional[torch.Tensor] = None,
        n_layers: Optional[int] = None,
    ) -> FrontendOut:
        B = img.shape[0]
        L = max(1, int(n_layers if n_layers is not None else self.n_layers))

        x = self._encode_points(img)
        h = self.h0.expand(B, -1).to(dtype=x.dtype, device=x.device)
        if external_h is not None:
            h = h + self.h_from_ext(external_h.to(dtype=h.dtype))

        traces: List[LayerTrace] = []
        slices = None
        for i in range(L):
            cell = self.layer if self.shared_layer else self.layers[i]
            # (X_t, H_t) → (X_{t+1}, H_{t+1})
            x, h, slices, tr = cell(x, h, layer_idx=i)
            traces.append(tr)

        assert slices is not None
        # LLM tokens = interface only; working visual memory is still X (N=H*W).
        tok = self.proj(slices)
        self._last_traces = traces
        self._last_h = h.detach()
        self._last_x = x.detach()  # full-res point field after L steps
        wb = (
            self.layer.write_h_into_x if self.shared_layer
            else self.layers[0].write_h_into_x
        )
        B, N, C = x.shape
        return FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={
                "kind": "B_xmodal",
                "transolver": "3_multimodal",
                "thesis": "full_res_point_field_bidir",
                "replaces_legacy_slice_tokens": True,
                "contract": "X'=F_vision(X,H); H'=F_language(H,Read(X'))",
                "point_field": True,
                "N_points": int(N),
                "point_dim": int(C),
                "T": self.T,
                "C": self.C,
                "n_layers": L,
                "shared_layer": self.shared_layer,
                "writeback": bool(wb),
                "deslice": True,
                "tokens_are_interface_only": True,
                "res": self.res,
                "projector": self.projector_kind,
                "layer_traces": [
                    {
                        "layer": t.layer,
                        "x_delta_norm": t.x_delta_norm,
                        "h_delta_norm": t.h_delta_norm,
                        "read_entropy": t.read_entropy,
                        "write_applied": t.write_applied,
                        "support": t.support,
                    }
                    for t in traces
                ],
            },
        )


def run_layer_steps_dict(layer: CrossModalSliceLayer, x: torch.Tensor, h: torch.Tensor) -> dict:
    """Named tensors for one bidirectional step (tests / docs)."""
    x1, slices = layer.F_vision(x, h)
    read, attn, ent = layer.Read(slices, h)
    h2, delta = layer.F_language(h, read)
    return {
        "X_t": x,
        "H_t": h,
        "X_t1_F_vision": x1,
        "slices_from_X_t1": slices,
        "Read": read,
        "Read_attn": attn,
        "Read_entropy": ent,
        "H_t1_F_language": h2,
        "H_delta": delta,
        # aliases
        "1_x_after_visual": x1,
        "1_slices": slices,
        "2_read": read,
        "2_attn": attn,
        "2_entropy": ent,
        "3_h": h2,
        "3_h_delta": delta,
        "4_x_after_writeback": x1,  # write is inside F_vision now
        "4_w_write": None,
    }
