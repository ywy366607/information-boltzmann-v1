"""Visual Latent CoT over slice/deslice (no GDN-2, no LLM-BPTT).

Design (M0):
  - Point stream x is updated by sparse deslice write (Transolver-style).
  - Each latent step forms a continuous thought z and a query q = W_q z
    (same pattern as Q/K/V linear maps from a state vector).
  - z is contracted (tanh + scale |γ|<1) so the loop cannot explode
    (spectral-radius-style safety without full eigendecomp each step).
  - Training uses standard next-token CE on the *final* visual tokens + text;
    optional NextLat-style aux on consecutive z with stop-grad targets.
  - K is small and fixed (default 2); stop head is optional.

References conceptually: Coconut continuous thoughts; NextLat next-h prediction;
Transolver slice read / deslice write.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.frontends import FrontendOut, SliceFrontend
from fine_grain.mm_projector import MMProjector
from fine_grain.models import (
    NS_STEPS_DEFAULT,
    AdaTempSlice,
    Block,
    apply_slice_flags,
    coords,
    deslice_support_size,
)


@dataclass
class LatentCoTState:
    """Carries visual field + continuous thoughts across steps."""

    x: torch.Tensor              # [B, N, mix_dim] point features
    z: torch.Tensor              # [B, z_dim] continuous thought
    slices: Optional[torch.Tensor]  # [B, G, mix_dim]
    tokens: Optional[torch.Tensor]  # [B, G, d_llm] projected for LLM
    stop_logit: Optional[torch.Tensor] = None


class VisualLatentCoTSlice(nn.Module):
    """Iterative slice frontend: latent visual CoT then tokens for VLM.

    Args:
        latent_steps: max K continuous visual thoughts (fixed curriculum).
        beta: deslice residual damping (x ← x + β·Δ), |β|≤1 for stability.
        gamma: contraction on thought residual (|γ|<1).
    """

    def __init__(
        self,
        d_llm: int,
        res: int = 32,
        T: int = 32,
        dim: int = 64,
        depth: int = 2,
        deslice_topk: int = 2,
        stiefel: bool = True,
        projector: str = "mlp",
        latent_steps: int = 2,
        z_dim: Optional[int] = None,
        beta: float = 0.5,
        gamma: float = 0.9,
        use_stop_head: bool = False,
    ):
        super().__init__()
        self.res = res
        self.T = int(T)
        self.latent_steps = max(1, int(latent_steps))
        self.beta = float(beta)
        self.gamma = float(min(max(gamma, 0.0), 0.999))
        self.use_stop_head = use_stop_head
        self.projector_kind = projector

        heads = 4
        dim_head = max(16, (self.T + heads - 1) // heads)
        dim_head = min(dim_head, 64)
        self.heads, self.dim_head = heads, dim_head
        self.C = heads * dim_head
        mix_dim = self.C
        self.mix_dim = mix_dim
        self.z_dim = int(z_dim or min(128, d_llm))

        self.stem = nn.Linear(5, mix_dim)
        self.local = nn.Sequential(
            nn.Conv2d(mix_dim, mix_dim, 3, padding=1, groups=mix_dim),
            nn.Conv2d(mix_dim, mix_dim, 1),
        )
        # One refine block reused each latent step (weight sharing = UT style)
        self.refine = Block(
            mix_dim,
            AdaTempSlice(mix_dim, heads=heads, dim_head=dim_head,
                         slice_num=self.T, norm="mass"),
        )
        # Optional deeper stack for initial encode only
        self.init_blocks = nn.ModuleList([
            Block(mix_dim, AdaTempSlice(mix_dim, heads=heads, dim_head=dim_head,
                                        slice_num=self.T, norm="mass"))
            for _ in range(max(0, depth - 1))
        ])
        spec = dict(
            nog=True,
            stiefel_ns=bool(stiefel),
            deslice_topk=int(deslice_topk),
            ns_steps=NS_STEPS_DEFAULT,
        )

        class _Wrap(nn.Module):
            pass

        w = _Wrap()
        w.blocks = nn.ModuleList([self.refine] + list(self.init_blocks))
        apply_slice_flags(w, spec)

        # Continuous thought + query (like extra Q map from a state)
        self.z0 = nn.Parameter(torch.zeros(1, self.z_dim))
        self.pool_to_z = nn.Linear(mix_dim, self.z_dim)
        self.W_z = nn.Linear(self.z_dim, self.z_dim, bias=False)
        # Spectral-ish: scale W_z by gamma so map is contractive when ||W||~1
        nn.init.orthogonal_(self.W_z.weight)
        self.W_q = nn.Linear(self.z_dim, mix_dim)
        self.q_to_bias = nn.Linear(mix_dim, mix_dim)
        self.h_to_z = nn.Linear(d_llm, self.z_dim)  # last LLM hidden → thought
        self.stop_head = nn.Linear(self.z_dim, 1)
        self.proj = MMProjector(mix_dim, d_llm, kind=projector)

        # NextLat-style dynamics head (optional aux)
        self.next_z = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.GELU(),
            nn.Linear(self.z_dim, self.z_dim),
        )

        self._last_slot = {}
        self._last_z_traj: List[torch.Tensor] = []

    def _encode_points(self, img: torch.Tensor) -> torch.Tensor:
        B, _, R, _ = img.shape
        assert R == self.res
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = coords(R, img.device).expand(B, -1, -1)
        x = self.stem(torch.cat([pts, p], -1))
        g = x.transpose(1, 2).reshape(B, -1, R, R)
        x = x + self.local(g).flatten(2).transpose(1, 2)
        return x

    def _condition_x(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Broadcast query bias onto point stream (task/content-dependent look)."""
        # q already in mix_dim
        bias = self.q_to_bias(q).unsqueeze(1)  # [B,1,C]
        return x + bias

    def _step_thought(
        self, z: torch.Tensor, slices: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """z' = γ tanh(W_z z + pool(slices)); q = W_q z'."""
        if slices is not None:
            pooled = slices.mean(dim=1)
            z_in = z + self.pool_to_z(pooled)
        else:
            z_in = z
        # Contractive continuous thought (Coconut-like reuse of state, damped)
        z_new = self.gamma * torch.tanh(self.W_z(z_in))
        q = self.W_q(z_new)
        return z_new, q

    def _one_refine(
        self, x: torch.Tensor, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Condition → slice block → damped residual (deslice inside mixer)."""
        x_cond = self._condition_x(x, q)
        # Mixer returns (point_stream, slice_tokens)
        x_out, slices = self.refine(x_cond)
        # x_out already is residual-updated points inside Block; extra damp vs input
        x_next = x + self.beta * (x_out - x)
        return x_next, slices

    def forward(
        self,
        img: torch.Tensor,
        external_h: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> FrontendOut:
        """Run K latent visual steps; optional external LLM hidden for last query.

        external_h: [B, d_h] last-layer state — if provided, maps into z init via
        a learned affine (created lazily if dims match z_dim else linear).
        """
        B = img.shape[0]
        K = int(steps if steps is not None else self.latent_steps)
        K = max(1, K)

        x = self._encode_points(img)
        # warm-up blocks (optional depth)
        slices = None
        for b in self.init_blocks:
            x, slices = b(x)

        z = self.z0.expand(B, -1)
        if external_h is not None:
            # project LLM hidden into thought space (same spirit as Q map)
            z = z + self.h_to_z(external_h)

        z_traj = []
        stop_logit = None
        for k in range(K):
            z, q = self._step_thought(z, slices)
            z_traj.append(z)
            x, slices = self._one_refine(x, q)
            if self.use_stop_head:
                stop_logit = self.stop_head(z)

        assert slices is not None
        tok = self.proj(slices)
        self._last_z_traj = z_traj
        # slot metrics from refine mixer
        mix0 = self.refine.mix
        slot = {}
        w = getattr(mix0, "last_w", None)
        w_write = getattr(mix0, "last_w_write", None)
        if w is not None:
            mass = w.sum(2)
            pp = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
            slot["PR_mass"] = float((1.0 / pp.pow(2).sum(-1)).mean())
            slot["support"] = (
                float(deslice_support_size(w_write)) if w_write is not None else float("nan")
            )
        self._last_slot = slot

        return FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={
                "kind": "B_vlcot",
                "T": self.T,
                "C": self.C,
                "latent_steps": K,
                "beta": self.beta,
                "gamma": self.gamma,
                "deslice_topk": int(getattr(mix0, "deslice_topk", 0)),
                "stiefel_ns": bool(getattr(mix0, "stiefel_ns", False)),
                "slot": slot,
                "res": self.res,
                "projector": self.projector_kind,
            },
        )

    def nextlat_loss(self) -> torch.Tensor:
        """Auxiliary NextLat-style loss on consecutive thoughts (stop-grad target)."""
        traj = self._last_z_traj
        if len(traj) < 2:
            return torch.tensor(0.0, device=self.z0.device)
        loss = 0.0
        n = 0
        for i in range(len(traj) - 1):
            pred = self.next_z(traj[i])
            tgt = traj[i + 1].detach()
            loss = loss + F.smooth_l1_loss(pred, tgt)
            n += 1
        return loss / max(n, 1)


def build_vlcot_frontend(
    d_llm: int,
    res: int = 32,
    T: int = 32,
    **kwargs,
) -> VisualLatentCoTSlice:
    return VisualLatentCoTSlice(d_llm, res=res, T=T, **kwargs)


def encode_frontend(frontend: nn.Module, images: torch.Tensor, external_h=None) -> FrontendOut:
    """Call frontend; pass external_h only if supported (VL-CoT / cross-modal loop)."""
    from fine_grain.cross_modal_slice_loop import CrossModalSliceFrontend

    if isinstance(frontend, (VisualLatentCoTSlice, CrossModalSliceFrontend)):
        return frontend(images, external_h=external_h)
    return frontend(images)


class VLCoTBridge(nn.Module):
    """Frozen LLM + vision frontend with optional second look from last hidden.

    M1 training (no LLM BPTT over long rollouts):
      1) encode vis_0 = FE(img)
      2) teacher-forced LLM on [vis_0 | text] → last-layer h at end of prompt
      3) encode vis_1 = FE(img, external_h=h)   # query conditioned re-look
      4) CE loss on [vis_1 | text] (answer-only labels)
      5) optional: 0.5 * CE on pass-1 + NextLat aux on FE

    Inference helpers: refine_visual_from_h, digit-constrained generate.
    """

    def __init__(self, frontend: nn.Module, llm: nn.Module, pass1_ce_weight: float = 0.3):
        super().__init__()
        self.frontend = frontend
        self.llm = llm
        self.pass1_ce_weight = float(pass1_ce_weight)
        for p in self.llm.parameters():
            p.requires_grad_(False)

    def _run_llm(self, v, input_ids, attention_mask, text_labels=None, output_hidden=False):
        dtype = next(self.llm.parameters()).dtype
        if v.dtype != dtype:
            v = v.to(dtype=dtype)
        emb = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([v, emb], dim=1)
        T = v.shape[1]
        B, L = input_ids.shape
        vis_mask = torch.ones(B, T, device=input_ids.device, dtype=attention_mask.dtype)
        attn = torch.cat([vis_mask, attention_mask], dim=1)
        ignore = torch.full((B, T), -100, device=input_ids.device, dtype=input_ids.dtype)
        if text_labels is None:
            text_lab = input_ids.clone().masked_fill(attention_mask == 0, -100)
        else:
            text_lab = text_labels
        labels = torch.cat([ignore, text_lab], dim=1)
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels if text_labels is not None or not output_hidden else labels,
            output_hidden_states=output_hidden,
            use_cache=False,
        )
        return out, T

    def forward(self, images, input_ids, attention_mask, text_labels=None, two_look: bool = True):
        """Return (loss, meta). two_look enables LLM-h → second visual pass."""
        # Pass 1: initial visual look
        vis0 = encode_frontend(self.frontend, images, external_h=None)
        v0 = vis0.tokens
        meta = dict(vis0.meta)
        meta["T"] = vis0.T

        if not two_look or not isinstance(self.frontend, VisualLatentCoTSlice):
            out, _ = self._run_llm(v0, input_ids, attention_mask, text_labels=text_labels)
            meta["two_look"] = False
            aux = torch.tensor(0.0, device=images.device)
            if hasattr(self.frontend, "nextlat_loss"):
                aux = self.frontend.nextlat_loss()
            return out.loss + 0.0 * aux, meta

        # Hidden at last real prompt token (before answer span)
        # Use positions where text_labels == -100 as prompt; last such index.
        with torch.no_grad():
            out1, Tvis = self._run_llm(
                v0, input_ids, attention_mask, text_labels=text_labels, output_hidden=True,
            )
            hs = out1.hidden_states[-1]  # [B, Tvis+L, d]
            if text_labels is not None:
                # prompt positions in text = where label is -100 and mask=1
                prompt_mask = (text_labels < 0) & (attention_mask > 0)
            else:
                prompt_mask = attention_mask > 0
            # index of last prompt token in full sequence = Tvis + last_prompt_text_idx
            B = images.shape[0]
            h_list = []
            for i in range(B):
                idx = torch.where(prompt_mask[i])[0]
                if len(idx) == 0:
                    # fallback: last non-pad text
                    idx = torch.where(attention_mask[i] > 0)[0]
                ti = int(idx[-1].item()) if len(idx) else 0
                h_list.append(hs[i, Tvis + ti])
            h = torch.stack(h_list, dim=0)

        # Pass 2: re-look with control from LLM hidden (gradients flow into FE via pass2)
        vis1 = encode_frontend(self.frontend, images, external_h=h)
        v1 = vis1.tokens
        out2, _ = self._run_llm(v1, input_ids, attention_mask, text_labels=text_labels)
        loss = out2.loss
        if self.pass1_ce_weight > 0:
            # pass1 CE without retaining graph on frozen path extras
            out1b, _ = self._run_llm(v0, input_ids, attention_mask, text_labels=text_labels)
            loss = loss + self.pass1_ce_weight * out1b.loss
        if hasattr(self.frontend, "nextlat_loss"):
            loss = loss + 0.1 * self.frontend.nextlat_loss()
        meta.update(vis1.meta)
        meta["T"] = vis1.T
        meta["two_look"] = True
        return loss, meta


@torch.no_grad()
def greedy_digit_string(
    bridge: nn.Module,
    tokenizer,
    images: torch.Tensor,
    prompts,
    max_new: int = 6,
    refine_every: int = 1,
) -> list:
    """Greedy decode constrained to digit tokens; optional re-look each step.

    At each step, only allow tokens that decode to a single digit 0-9 (and
    common leading-space digit pieces). Optionally refresh visual tokens from
    last hidden (two-look) every ``refine_every`` steps.
    """
    device = images.device
    dtype = next(bridge.llm.parameters()).dtype
    # collect allowed first-piece digit ids
    allowed = set()
    for d in "0123456789":
        for variant in (d, " " + d, "\n" + d):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                allowed.add(int(ids[0]))
    allowed = sorted(allowed)
    if not allowed:
        allowed = None

    vis = encode_frontend(bridge.frontend, images, external_h=None)
    v = vis.tokens.to(dtype=dtype)
    Tvis = v.shape[1]
    B = images.shape[0]
    from fine_grain.vlm_data import tokenize_captions
    ids, mask = tokenize_captions(tokenizer, list(prompts), max_length=48)
    ids, mask = ids.to(device), mask.to(device)
    emb = bridge.llm.get_input_embeddings()(ids)
    inputs = torch.cat([v, emb], dim=1)
    attn = torch.cat([
        torch.ones(B, Tvis, device=device, dtype=mask.dtype), mask,
    ], dim=1)
    gen = ids
    for step in range(max_new):
        out = bridge.llm(
            inputs_embeds=inputs, attention_mask=attn,
            output_hidden_states=True, use_cache=False,
        )
        logits = out.logits[:, -1, :]
        if allowed is not None:
            mask_logits = torch.full_like(logits, -1e9)
            idx = torch.tensor(allowed, device=device, dtype=torch.long)
            mask_logits[:, idx] = logits[:, idx]
            logits = mask_logits
        next_id = logits.argmax(dim=-1, keepdim=True)
        gen = torch.cat([gen, next_id], dim=1)
        # optional re-look using last hidden of this step
        if (
            refine_every > 0
            and (step + 1) % refine_every == 0
            and isinstance(bridge.frontend, VisualLatentCoTSlice)
        ):
            h = out.hidden_states[-1][:, -1, :]
            vis = encode_frontend(bridge.frontend, images, external_h=h)
            v = vis.tokens.to(dtype=dtype)
            Tvis = v.shape[1]
            emb = bridge.llm.get_input_embeddings()(gen)
            inputs = torch.cat([v, emb], dim=1)
            attn = torch.ones(B, Tvis + gen.shape[1], device=device, dtype=mask.dtype)
        else:
            next_emb = bridge.llm.get_input_embeddings()(next_id)
            inputs = torch.cat([inputs, next_emb], dim=1)
            attn = torch.cat([
                attn, torch.ones(B, 1, device=device, dtype=attn.dtype),
            ], dim=1)
    texts = []
    prompt_lens = mask.sum(dim=1).tolist()
    for i in range(B):
        pl = int(prompt_lens[i])
        new_ids = gen[i, pl:]
        texts.append(tokenizer.decode(new_ids.tolist(), skip_special_tokens=True))
    return texts
