"""Transolver3 native multimodal VLM-OCR: vision–language co-evolution.

Product path (not legacy slice tokens):
  - Visual field X (full-res points) co-evolves with language state H
  - Temporary shared workspace U (slice interact view) via Read/Write
  - LLM sees **interface** tokens only (projected slices + optional H token);
    working visual memory remains X

Training:
  1) Seed H0 from prompt embeddings (Read_language → control)
  2) L steps: X'=F_vision(X,H), H'=F_language(H, Read(X'))
  3) Optional outer co-evolve: LLM hidden → re-seed H → second field pass
  4) Answer-only CE on [interface_tokens | text]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from fine_grain.cross_modal_slice_loop import CrossModalSliceFrontend
from fine_grain.frontends import FrontendOut, PatchFrontend
from fine_grain.mot_coevolve import MoTCoEvolveFrontend


class Transolver3OCRBridge(nn.Module):
    """Transolver3 field frontend + LLM; vision–text co-evolve.

    ``freeze_llm=True``: train vision/interface only (4GB-friendly baseline).
    ``freeze_llm=False``: **joint** training of field + language weights.
    """

    def __init__(
        self,
        frontend: CrossModalSliceFrontend,
        llm: nn.Module,
        coevolve_rounds: int = 2,
        pass1_ce_weight: float = 0.25,
        use_h_token: bool = True,
        freeze_llm: bool = True,
    ):
        super().__init__()
        assert isinstance(frontend, CrossModalSliceFrontend), type(frontend)
        self.frontend = frontend
        self.llm = llm
        self.coevolve_rounds = max(1, int(coevolve_rounds))
        self.pass1_ce_weight = float(pass1_ce_weight)
        self.use_h_token = bool(use_h_token)
        self.freeze_llm = bool(freeze_llm)
        d = int(frontend.d_llm)
        self.h_to_tok = nn.Linear(frontend.h_dim, d)
        for p in self.llm.parameters():
            p.requires_grad_(not self.freeze_llm)

    def _prompt_seed_h(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Read_language: pool prompt embeddings → seed for H0."""
        emb = self.llm.get_input_embeddings()(input_ids)  # [B,L,d]
        if text_labels is not None:
            # prompt = supervised-ignore positions that are real tokens
            pmask = (text_labels < 0) & (attention_mask > 0)
        else:
            pmask = attention_mask > 0
        # mean over prompt positions (avoid empty)
        w = pmask.float().unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        h = (emb * w).sum(dim=1) / denom
        return h

    def _interface_tokens(self, vis: FrontendOut) -> Tuple[torch.Tensor, dict]:
        """LLM interface only: slice projections + optional H token."""
        tok = vis.tokens
        meta = dict(vis.meta)
        if self.use_h_token and self.frontend._last_h is not None:
            # live H after co-evolve (use non-detached for train: recompute from fe)
            # frontend stores detach; re-get via forward path — caller passes h
            pass
        return tok, meta

    def _run_llm(
        self,
        v: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_labels=None,
        output_hidden: bool = False,
    ):
        dtype = next(self.llm.parameters()).dtype
        if v.dtype != dtype:
            v = v.to(dtype=dtype)
        emb = self.llm.get_input_embeddings()(input_ids)
        inputs = torch.cat([v, emb], dim=1)
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
            inputs_embeds=inputs,
            attention_mask=attn,
            labels=labels,
            output_hidden_states=output_hidden,
            use_cache=False,
        )
        return out, T

    def _pack_interface(self, tok: torch.Tensor, h_live: torch.Tensor) -> torch.Tensor:
        if not self.use_h_token:
            return tok
        ht = self.h_to_tok(h_live).unsqueeze(1)  # [B,1,d]
        return torch.cat([tok, ht], dim=1)

    def evolve_field(
        self,
        images: torch.Tensor,
        h_seed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, FrontendOut]:
        """Run frontend co-evolve; return (interface_v, h_live, FrontendOut).

        Note: frontend stores _last_h detached; we re-run a thin hook to keep h.
        """
        # Monkey: call encode path then recover h before detach by internal copy
        # CrossModalSliceFrontend detaches _last_h — add live h by re-implementing
        # one call and reading from a non-detach version via forward meta.
        fe = self.frontend
        B = images.shape[0]
        x = fe._encode_points(images)
        h = fe.h0.expand(B, -1).to(dtype=x.dtype, device=x.device)
        if h_seed is not None:
            h = h + fe.h_from_ext(h_seed.to(dtype=h.dtype))
        slices = None
        for i in range(fe.n_layers):
            cell = fe.layer if fe.shared_layer else fe.layers[i]
            x, h, slices, _ = cell(x, h, layer_idx=i)
        assert slices is not None
        tok = fe.proj(slices)
        fe._last_h = h.detach()
        fe._last_x = x.detach()
        out = FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={
                "kind": "B_xmodal",
                "transolver": "3_multimodal",
                "n_layers": fe.n_layers,
                "N_points": int(x.shape[1]),
                "tokens_are_interface_only": True,
                "point_field": True,
            },
        )
        return tok, h, out

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_labels=None,
    ):
        """Co-evolve then CE. Returns (loss, meta)."""
        # Seed H from language prompt (Read_language)
        h_seed = self._prompt_seed_h(input_ids, attention_mask, text_labels)
        tok, h, out0 = self.evolve_field(images, h_seed=h_seed)
        v = self._pack_interface(tok, h)
        meta = dict(out0.meta)
        meta["coevolve_rounds"] = self.coevolve_rounds
        meta["T_interface"] = int(v.shape[1])

        meta["freeze_llm"] = self.freeze_llm
        if self.coevolve_rounds <= 1:
            out, Tvis = self._run_llm(v, input_ids, attention_mask, text_labels=text_labels)
            meta["two_look"] = False
            meta["T"] = Tvis
            return out.loss, meta

        # Round 1: get LLM hidden as language write-back seed.
        # Joint train: keep graph on pass1 CE; detach only the h seed to limit VRAM.
        # Frozen: full no_grad on pass1 hidden extract.
        def _extract_h(out1, Tvis):
            hs = out1.hidden_states[-1]
            if text_labels is not None:
                pmask = (text_labels < 0) & (attention_mask > 0)
            else:
                pmask = attention_mask > 0
            B = images.shape[0]
            h_list = []
            for i in range(B):
                idx = torch.where(pmask[i])[0]
                if len(idx) == 0:
                    idx = torch.where(attention_mask[i] > 0)[0]
                ti = int(idx[-1].item()) if len(idx) else 0
                h_list.append(hs[i, Tvis + ti])
            return torch.stack(h_list, dim=0)

        if self.freeze_llm:
            with torch.no_grad():
                out1, Tvis = self._run_llm(
                    v.detach(), input_ids, attention_mask,
                    text_labels=text_labels, output_hidden=True,
                )
                h_llm = _extract_h(out1, Tvis)
            pass1_loss = None
        else:
            out1, Tvis = self._run_llm(
                v, input_ids, attention_mask,
                text_labels=text_labels, output_hidden=True,
            )
            h_llm = _extract_h(out1, Tvis).detach()
            pass1_loss = out1.loss

        # Re-seed field evolution with LLM judgment (Write_language → H → F_vision)
        tok2, h2, out2 = self.evolve_field(images, h_seed=h_llm)
        v2 = self._pack_interface(tok2, h2)
        out_final, T2 = self._run_llm(v2, input_ids, attention_mask, text_labels=text_labels)
        loss = out_final.loss
        if self.pass1_ce_weight > 0:
            if pass1_loss is not None:
                loss = loss + self.pass1_ce_weight * pass1_loss
            else:
                out_p1, _ = self._run_llm(v, input_ids, attention_mask, text_labels=text_labels)
                loss = loss + self.pass1_ce_weight * out_p1.loss
        meta.update(out2.meta)
        meta["two_look"] = True
        meta["T"] = T2
        meta["T_interface"] = int(v2.shape[1])
        meta["freeze_llm"] = self.freeze_llm
        return loss, meta


class PatchOCRBridge(nn.Module):
    """Baseline A: static patch tokens (optionally joint-train LLM)."""

    def __init__(self, frontend: PatchFrontend, llm: nn.Module, freeze_llm: bool = True):
        super().__init__()
        self.frontend = frontend
        self.llm = llm
        self.freeze_llm = bool(freeze_llm)
        for p in self.llm.parameters():
            p.requires_grad_(not self.freeze_llm)

    def forward(self, images, input_ids, attention_mask, text_labels=None):
        vis = self.frontend(images)
        v = vis.tokens
        dtype = next(self.llm.parameters()).dtype
        if v.dtype != dtype:
            v = v.to(dtype=dtype)
        emb = self.llm.get_input_embeddings()(input_ids)
        inputs = torch.cat([v, emb], dim=1)
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
            inputs_embeds=inputs, attention_mask=attn, labels=labels, use_cache=False,
        )
        meta = dict(vis.meta)
        meta["T"] = T
        meta["two_look"] = False
        meta["transolver"] = None
        meta["freeze_llm"] = self.freeze_llm
        return out.loss, meta


class MoTOCRBridge(nn.Module):
    """True MoT: slices ‖ text embeddings in one self-attn; deslice to X; LLM CE.

    Text side of MoT uses the same token sequence embeddings as the LLM path
    (refined, residual-added), so one joint attn updates vision field + text reps.

    ``freeze_llm=False``: joint training of MoT field + full LLM weights.
    """

    def __init__(
        self,
        frontend: MoTCoEvolveFrontend,
        llm: nn.Module,
        freeze_llm: bool = True,
    ):
        super().__init__()
        assert isinstance(frontend, MoTCoEvolveFrontend)
        self.frontend = frontend
        self.llm = llm
        self.freeze_llm = bool(freeze_llm)
        for p in self.llm.parameters():
            p.requires_grad_(not self.freeze_llm)

    def forward(self, images, input_ids, attention_mask, text_labels=None):
        emb = self.llm.get_input_embeddings()(input_ids)  # [B,L,d]
        dtype = next(self.llm.parameters()).dtype
        # MoT co-evolve in field + text emb space
        x, text_llm, slices, traces = self.frontend.forward_mot(
            images, emb.float(), text_mask=attention_mask,
        )
        v = self.frontend.proj(slices)
        if v.dtype != dtype:
            v = v.to(dtype=dtype)
        text_llm = text_llm.to(dtype=dtype)
        # LLM sees interface vision + MoT-refined text (not raw emb alone)
        inputs = torch.cat([v, text_llm], dim=1)
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
            inputs_embeds=inputs, attention_mask=attn, labels=labels, use_cache=False,
        )
        meta = {
            "kind": "B_mot",
            "transolver": "3_mot",
            "mot": True,
            "joint_self_attn": True,
            "T": T,
            "T_interface": T,
            "two_look": False,
            "n_layers": self.frontend.n_layers,
            "N_points": int(x.shape[1]),
            "point_field": True,
            "freeze_llm": self.freeze_llm,
            "layer_traces": [
                {"layer": t.layer, "x_delta": t.x_delta, "text_delta": t.text_delta}
                for t in traces
            ],
        }
        return out.loss, meta


def configure_llm_trainability(
    llm: nn.Module,
    *,
    freeze_llm: bool = True,
    last_n_layers: int = 0,
) -> dict:
    """Set which LLM params train.

    - freeze_llm=True: all frozen
    - freeze_llm=False, last_n_layers=0: full LLM trainable (heavy; OOM on 4GB)
    - freeze_llm=False, last_n_layers=K>0: only last K transformer blocks +
      embeddings + lm_head (default joint recipe for ~4GB)
    """
    for p in llm.parameters():
        p.requires_grad_(False)
    if freeze_llm:
        return {"mode": "frozen", "n_trainable": 0}

    n_layers = 0
    layers = None
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        layers = llm.model.layers
        n_layers = len(layers)
    elif hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
        layers = llm.transformer.h
        n_layers = len(layers)

    if last_n_layers and last_n_layers > 0 and layers is not None:
        k = min(int(last_n_layers), n_layers)
        start = n_layers - k
        for i in range(start, n_layers):
            for p in layers[i].parameters():
                p.requires_grad_(True)
        # embeddings + head
        if hasattr(llm, "get_input_embeddings"):
            emb = llm.get_input_embeddings()
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad_(True)
        for name in ("lm_head", "embed_out"):
            m = getattr(llm, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)
        n_tr = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        return {
            "mode": f"last_{k}_layers+embed+head",
            "n_layers_total": n_layers,
            "n_trainable": int(n_tr),
        }

    # full unfreeze
    for p in llm.parameters():
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    return {"mode": "full", "n_trainable": int(n_tr)}


def build_ocr_bridge(kind: str, d_llm: int, llm: nn.Module, args) -> nn.Module:
    k = kind.upper().replace("-", "_")
    freeze = bool(getattr(args, "freeze_llm", True))
    if k in ("A", "PATCH"):
        fe = PatchFrontend(
            d_llm, res=args.res, patch=args.patch, dim=args.dim,
            depth=args.depth, T=args.T, projector=args.projector,
        )
        return PatchOCRBridge(fe, llm, freeze_llm=freeze)
    if k in ("B_XMODAL", "XMODAL", "T3", "TRANSOLVER3", "B3"):
        # continuous-H path (rollback / ablation)
        fe = CrossModalSliceFrontend(
            d_llm, res=args.res, T=args.T, dim=args.dim,
            n_layers=args.n_layers, deslice_topk=args.topk,
            writeback=True, projector=args.projector,
            shared_layer=True,
        )
        return Transolver3OCRBridge(
            fe, llm,
            coevolve_rounds=args.coevolve_rounds,
            pass1_ce_weight=args.pass1_w,
            use_h_token=args.use_h_token,
            freeze_llm=freeze,
        )
    if k in ("B_MOT", "MOT", "T3_MOT", "MOT_COEVOLVE"):
        fe = MoTCoEvolveFrontend(
            d_llm, res=args.res, T=args.T, dim=args.dim,
            n_layers=args.n_layers, deslice_topk=args.topk,
            projector=args.projector,
        )
        return MoTOCRBridge(fe, llm, freeze_llm=freeze)
    raise ValueError(
        f"unknown kind {kind}; use A | B_xmodal (H-path) | B_mot (joint self-attn MoT). "
        f"Legacy slice-token B is not offered."
    )
