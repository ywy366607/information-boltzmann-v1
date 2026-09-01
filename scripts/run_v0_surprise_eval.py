#!/usr/bin/env python3
"""Run Phase 1 & 2 validation of Bayesian Surprise & JEPA error gates on Deep Dual-Stream Native MoT.

Features:
  - Deep 4-layer MoT stack (n_layers=4)
  - 9 experimental arms:
      1. baseline        - Standard dual-stream co-evolution (g=1.0)
      2. random          - Random gate control (g ~ Uniform(0,1))
      3. constant        - Constant gate control (g = 0.5)
      4. v0_jepa         - V0 Deterministic JEPA prediction error gate
      5. v0_shuffled     - Negative control: surprise values permuted across slices
      6. v0_reverse      - Negative control: reversed surprise gate (suppress surprise)
      7. v0_global_only  - Global layer-average surprise g(U_bar) (no spatial targeting)
      8. v0_spatial_only - Centered spatial contrast surprise g(max(0, U_j - U_bar))
      9. v1_bayes        - V1 Gaussian Bayesian surprise gate with learned uncertainty
  - Multi-seed paired analysis: Delta_s = Acc(arm, s) - Acc(baseline, s)
  - Per-task accuracy breakdown: OCR (1px digit), Kinks (corners count), Needle (color)
  - Step-by-step training trajectory logging (Loss(t), Acc(t))
  - Publication-ready diagnostic plots:
      - present/figs/eval_loss_acc_trajectories.png
      - present/figs/eval_task_breakdown.png
      - present/figs/eval_paired_deltas.png

Usage:
  python scripts/run_v0_surprise_eval.py --steps 300 --layers 4 --seeds 4 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.native_mot import NativeMoTStack
from fine_grain.vlm_data import COLORS, KINK_KS, OCR_DIGITS, make_vqa_batch


class DualStreamVQAModel(nn.Module):
    """NativeMoTStack + unified multi-task classification head for synthetic VQA."""

    def __init__(
        self,
        d_model: int = 128,
        d_llm: int | None = None,
        n_slices: int = 32,
        n_layers: int = 4,
        res: int = 32,
        surprise_mode: str = "baseline",
        surprise_beta: float = 1.5,
        prior_loss_coef: float = 0.1,
        detach_pred_target: bool = True,
        sigreg_coef: float = 0.0,
        use_stiefel: bool = True,
        deslice_topk: int = 2,
        deslice_write_sharpening: bool = False,
        prior_step_condition: bool = False,
        s_update: str = "raw",
        interact_prenorm: bool = False,
        trust_rho: float = 0.1,
        evidence_decay: float = 0.0,
        n_heads: int = 4,
        sigma_r: float = 1.0,
        gate_on: str = "u",
        deslice_write: str = "absolute",
        gate_h_local: bool = False,
        vfe_coef: float = 0.0,
        share_layers: bool = False,
        n_loops: int | None = None,
        deep_supervise: bool | None = None,
        lti_inject: bool = True,
        saccade: bool = False,
        saccade_gain: float = 1.0,
        saccade_inner: int = 2,
        saccade_halt: bool = False,
        saccade_halt_eps: float = 0.05,
        saccade_halt_train: bool = False,
        s2a: bool = False,
        s2a_ig_eps: float = 0.02,
        s2a_g_coef: float = 1.0,
        s2a_ent_coef: float = 0.05,
        s2a_halt_eps: float = 0.5,
        s_kalman_update: bool = False,
        s_lang_topk: int = 0,
        prior_write: float = 0.0,
        prior_write_by_t: bool = True,
        use_null_slice: bool = False,
        pack_by_surprise: bool = False,
        hard_admit: bool = False,
        use_yield_read: bool = False,
        use_ticket_read: bool = False,
        use_write_yield: bool = False,
        write_alpha: float = 1.0,
        use_residual_read: bool = False,
        use_modal_precision: bool = False,
        use_target_time: bool = False,
        use_target_time_adaln: bool = False,
        use_horizon_tokens: bool = False,
        gate_action_by_horizon: bool = False,
        history_size: int = 0,
        action_dim: int = 0,
        use_action_adaln: bool = False,
        use_action_tokens: bool = False,
        use_action_rel_bias: bool = False,
        use_action_transport: bool = False,
        use_action_slice_transition: bool = False,
        use_goal_adaln: bool = False,
        use_active_gdn2: bool = False,
        use_active_gdn2_history_transport: bool = False,
        active_gdn2_initial_trust: float = 0.0,
        terminal_token_atlas: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_llm = int(d_llm if d_llm is not None else d_model)
        self.lm = None
        self.lm_tok = None
        self.res = res
        self.n_heads = int(n_heads)
        self.surprise_mode = surprise_mode
        self.n_layers = n_layers
        self.share_layers = bool(share_layers)
        self.n_loops = int(n_layers if n_loops is None else n_loops)
        self.s2a = bool(s2a)
        self.s2a_ig_eps = float(s2a_ig_eps)
        self.s2a_g_coef = float(s2a_g_coef)
        self.s2a_ent_coef = float(s2a_ent_coef)
        if deep_supervise is None:
            self.deep_supervise = bool(self.share_layers or self.s2a)
        else:
            self.deep_supervise = bool(deep_supervise)
        self.lti_inject = bool(lti_inject)
        self.saccade = bool(saccade)
        self.prior_loss_coef = float(prior_loss_coef)
        self.detach_pred_target = bool(detach_pred_target)
        self.sigreg_coef = float(sigreg_coef)
        self.use_stiefel = bool(use_stiefel)
        self.deslice_topk = int(deslice_topk)
        self.deslice_write_sharpening = bool(deslice_write_sharpening)
        self.prior_step_condition = bool(prior_step_condition)
        self.s_update = str(s_update)
        self.interact_prenorm = bool(interact_prenorm)
        self.trust_rho = float(trust_rho)
        self.evidence_decay = float(evidence_decay)
        self.sigma_r = float(sigma_r)
        self.gate_on = str(gate_on)
        self.vfe_coef = float(vfe_coef)
        self.s_kalman_update = bool(s_kalman_update)
        self.s_lang_topk = int(s_lang_topk)

        # Text vocabulary embedding for synthetic prompts
        self.vocab = {
            "<pad>": 0, "What": 1, "color": 2, "is": 3, "the": 4, "small": 5, "square": 6,
            "How": 7, "many": 8, "corners": 9, "does": 10, "red": 11, "polyline": 12,
            "have": 13, "digit": 14, "drawn": 15, "with": 16, "thin": 17, "stroke": 18,
            "?": 19, "Answer:": 20,
        }
        for c in COLORS:
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        for k in KINK_KS:
            if str(k) not in self.vocab:
                self.vocab[str(k)] = len(self.vocab)
        for d in OCR_DIGITS:
            if d not in self.vocab:
                self.vocab[d] = len(self.vocab)

        self.embed = nn.Embedding(len(self.vocab) + 10, d_model)

        self.mot_stack = NativeMoTStack(
            d_llm=self.d_llm,
            res=res,
            d_x=d_model,
            d=d_model,
            n_slices=n_slices,
            n_layers=n_layers,
            n_heads=self.n_heads,
            deslice_topk=self.deslice_topk,
            deslice_write_sharpening=self.deslice_write_sharpening,
            prior_step_condition=self.prior_step_condition,
            use_stiefel=self.use_stiefel,
            local_kind="dw3",
            surprise_mode=surprise_mode,
            surprise_beta=surprise_beta,
            detach_pred_target=self.detach_pred_target,
            s_update=self.s_update,
            interact_prenorm=self.interact_prenorm,
            trust_rho=self.trust_rho,
            evidence_decay=self.evidence_decay,
            sigma_r=self.sigma_r,
            gate_on=self.gate_on,
            deslice_write=deslice_write,
            gate_h_local=gate_h_local,
            share_layers=self.share_layers,
            n_loops=self.n_loops,
            lti_inject=self.lti_inject,
            saccade=self.saccade,
            saccade_gain=saccade_gain,
            saccade_inner=saccade_inner,
            saccade_halt=saccade_halt,
            saccade_halt_eps=saccade_halt_eps,
            saccade_halt_train=saccade_halt_train,
            s_kalman_update=s_kalman_update,
            s_lang_topk=s_lang_topk,
            prior_write=prior_write,
            prior_write_by_t=prior_write_by_t,
            use_null_slice=use_null_slice,
            pack_by_surprise=pack_by_surprise,
            hard_admit=hard_admit,
            use_yield_read=use_yield_read,
            use_ticket_read=use_ticket_read,
            use_write_yield=use_write_yield,
            write_alpha=write_alpha,
            use_residual_read=use_residual_read,
            use_modal_precision=use_modal_precision,
            use_target_time=use_target_time,
            use_target_time_adaln=use_target_time_adaln,
            use_horizon_tokens=use_horizon_tokens,
            gate_action_by_horizon=gate_action_by_horizon,
            history_size=history_size,
            action_dim=action_dim,
            use_action_adaln=use_action_adaln,
            use_action_tokens=use_action_tokens,
            use_action_rel_bias=use_action_rel_bias,
            use_action_transport=use_action_transport,
            use_action_slice_transition=use_action_slice_transition,
            use_goal_adaln=use_goal_adaln,
            use_active_gdn2=use_active_gdn2,
            use_active_gdn2_history_transport=use_active_gdn2_history_transport,
            active_gdn2_initial_trust=active_gdn2_initial_trust,
            terminal_token_atlas=terminal_token_atlas,
        )

        # Multi-task answer classification heads
        self.answers = list(COLORS) + [str(k) for k in KINK_KS] + list(OCR_DIGITS)
        self.ans_to_idx = {a: i for i, a in enumerate(self.answers)}
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, len(self.answers)),
        )
        from fine_grain.s2a import AnswerGazeHead
        self.gaze_head = AnswerGazeHead(d_model) if self.s2a else None
        if self.gaze_head is not None:
            self.mot_stack.s2a_head = self.gaze_head
            self.mot_stack.s2a_eval_halt = True
            self.mot_stack.s2a_halt_eps = float(s2a_halt_eps)

    def encode_text(self, prompts: List[str], device: torch.device):
        """Toy word-embed or frozen LM *input* embeddings. MoT evolves H."""
        if self.lm is not None and self.lm_tok is not None:
            enc = self.lm_tok(
                list(prompts), padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            if next(self.lm.parameters()).device != ids.device:
                self.lm.to(device)
            emb = self.lm.get_input_embeddings()(ids)
            return emb.float(), am.bool()
        if self.lm is not None and self.lm_tok is None:
            raise RuntimeError(
                "encode_text(prompts) needs a tokenizer; use forward_tokens "
                "with input_ids or construct DualStreamOmni with lm_tok"
            )
        ids, mask = self.tokenize(prompts, device)
        return self.embed(ids), mask

    def tokenize(self, prompts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_list = []
        for p in prompts:
            words = p.replace("?", " ?").split()
            toks = [self.vocab.get(w, 0) for w in words] or [0]
            tokens_list.append(toks)
        max_len = max(len(t) for t in tokens_list)
        B = len(prompts)
        pad_t = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i, t in enumerate(tokens_list):
            pad_t[i, :len(t)] = torch.tensor(t, device=device)
            mask[i, :len(t)] = True
        return pad_t, mask

    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str],
        pi_x=None,
        n_loops=None,
        t=None,
        image_precision=None,
        text_precision=None,
        target_time=None,
        history_images=None,
        history_precision=None,
        action=None,
        action_precision=None,
        causal_state=None,
        x_init=None,
    ):
        B = images.shape[0]
        text_emb, mask = self.encode_text(prompts, images.device)

        X, H_out, tok, traces = self.mot_stack.forward_native(
            img=images, text_emb=text_emb, text_mask=mask, prompt_mask=mask,
            pi_x=1.0 if pi_x is None else pi_x,
            n_loops=n_loops,
            t=t,
            image_precision=image_precision,
            text_precision=text_precision,
            target_time=target_time,
            history_images=history_images,
            history_precision=history_precision,
            action=action,
            action_precision=action_precision,
            causal_state=causal_state,
            x_init=x_init,
        )

        mask_f = mask.unsqueeze(-1).float()

        def _pool(h_llm: torch.Tensor) -> torch.Tensor:
            return (h_llm * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        logits = self.head(_pool(H_out))
        logits_k = []
        if self.deep_supervise:
            text_residual = getattr(self.mot_stack, "_last_text_evidence", text_emb)
            for h_step in getattr(self.mot_stack, "_last_step_H", None) or []:
                h_llm = self.mot_stack.text_out(h_step) + text_residual
                logits_k.append(self.head(_pool(h_llm)))

        s2a_g_logits = []
        s2a_ig = []
        if self.s2a and self.gaze_head is not None and len(logits_k) >= 2:
            from fine_grain.s2a import kl_cat
            inner = max(1, int(getattr(self.mot_stack, "saccade_inner", 2)))
            for i in range(0, len(logits_k) - 1, inner):
                if i + 1 >= len(logits_k):
                    break
                p1 = F.softmax(logits_k[i].detach(), dim=-1)
                p2 = F.softmax(logits_k[i + 1].detach(), dim=-1)
                s2a_ig.append(kl_cat(p2, p1))
                s2a_g_logits.append(
                    self.gaze_head(self.mot_stack._pool_h(self.mot_stack._last_step_H[i], mask))
                )

        pred_loss = getattr(self.mot_stack, "_last_pred_loss", None)
        if pred_loss is None:
            pred_loss = logits.new_zeros(())
        vfe_loss = getattr(self.mot_stack, "_last_vfe_train_loss", None)
        if vfe_loss is None:
            vfe_loss = logits.new_zeros(())
        sigreg_loss = getattr(self.mot_stack, "_last_sigreg_loss", None)
        if sigreg_loss is None:
            sigreg_loss = logits.new_zeros(())
        return {
            "logits": logits,
            "logits_k": logits_k,
            "traces": traces,
            "X": X,
            "tokens": tok,
            "pred_loss": pred_loss,
            "vfe_loss": vfe_loss,
            "sigreg_loss": sigreg_loss,
            "rho": getattr(self.mot_stack, "_last_rho", {"rho_x": 0.0, "rho_h": 0.0}),
            "looks": getattr(self.mot_stack, "_last_looks", None),
            "s2a_g_logits": s2a_g_logits,
            "s2a_ig": s2a_ig,
            "causal_state": getattr(self.mot_stack, "_last_causal_state", None),
            "causal_prior_mu": getattr(self.mot_stack, "_last_causal_prior_mu", None),
            "causal_prior_logvar": getattr(
                self.mot_stack, "_last_causal_prior_logvar", None,
            ),
            "causal_posterior_mu": getattr(
                self.mot_stack, "_last_causal_posterior_mu", None,
            ),
            "causal_posterior_logvar": getattr(
                self.mot_stack, "_last_causal_posterior_logvar", None,
            ),
            "causal_diagnostics": getattr(
                self.mot_stack, "_last_causal_diagnostics", {},
            ),
        }

    def task_loss(self, out: Dict, targets: torch.Tensor) -> torch.Tensor:
        """CE + λ_pred · ||S_tgt − prior||² + λ_sig · SIGReg(S).

        S_tgt is S.detach() when detach_pred_target=True. SIGReg always uses live S.
        Val scripts should keep raw CE. Shared-Φ trains mean CE over every loop.
        """
        logits_k = out.get("logits_k") or []
        if self.deep_supervise and len(logits_k) > 0:
            ce = torch.stack([F.cross_entropy(lg, targets) for lg in logits_k]).mean()
        else:
            ce = F.cross_entropy(out["logits"], targets)
        loss = ce
        if self.s2a and out.get("s2a_g_logits") and out.get("s2a_ig"):
            from fine_grain.s2a import bernoulli_entropy
            bces = []
            ents = []
            for g_logit, ig in zip(out["s2a_g_logits"], out["s2a_ig"]):
                y = (ig.detach() > self.s2a_ig_eps).to(dtype=g_logit.dtype)
                bces.append(F.binary_cross_entropy_with_logits(g_logit, y))
                ents.append(bernoulli_entropy(torch.sigmoid(g_logit)).mean())
            loss = loss + self.s2a_g_coef * torch.stack(bces).mean()
            loss = loss - self.s2a_ent_coef * torch.stack(ents).mean()
        pred = out.get("pred_loss")
        if pred is not None and self.prior_loss_coef != 0.0:
            loss = loss + self.prior_loss_coef * pred
        vfe = out.get("vfe_loss")
        if vfe is not None and self.vfe_coef != 0.0:
            loss = loss + self.vfe_coef * vfe
        sig = out.get("sigreg_loss")
        if sig is not None and self.sigreg_coef != 0.0:
            loss = loss + self.sigreg_coef * sig
        return loss


def eval_model(
    model: DualStreamVQAModel,
    rng: np.random.Generator,
    val_batches: int,
    batch_size: int,
    res: int,
    device: torch.device,
) -> Dict:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    u_per_layer = []
    gap_per_layer = []
    F_per_layer = []
    rms_s_per_layer = []
    rms_d_per_layer = []
    rms_r_per_layer = []
    K_per_layer = []
    ce_per_layer = []
    rho_x_sum = 0.0
    rho_h_sum = 0.0
    looks_sum = 0.0
    looks_n = 0
    looks_task = {
        "color": {"sum": 0.0, "n": 0},
        "kinks": {"sum": 0.0, "n": 0},
        "ocr": {"sum": 0.0, "n": 0},
    }

    task_stats = {
        "color": {"correct": 0, "total": 0},
        "kinks": {"correct": 0, "total": 0},
        "ocr": {"correct": 0, "total": 0},
    }

    with torch.no_grad():
        for _ in range(val_batches):
            batch = make_vqa_batch(rng, batch=batch_size, res=res, mix=["ocr", "kinks", "color"])
            images = batch["image"].to(device)
            prompts = batch["prompt"]
            probes = batch["probe"]
            targets = torch.tensor([model.ans_to_idx[a] for a in batch["answer"]], device=device)

            out = model(images, prompts)
            logits = out["logits"]
            loss = F.cross_entropy(logits, targets)

            preds = logits.argmax(dim=-1)
            is_corr = (preds == targets).cpu().numpy()

            correct += int(is_corr.sum())
            total += len(targets)
            total_loss += float(loss.item()) * len(targets)

            looks = out.get("looks")
            if looks is not None:
                lk = looks.detach().float().cpu()
                looks_sum += float(lk.sum())
                looks_n += int(lk.numel())
            else:
                lk = None
            for i, p_kind in enumerate(probes):
                if p_kind in task_stats:
                    task_stats[p_kind]["total"] += 1
                    if is_corr[i]:
                        task_stats[p_kind]["correct"] += 1
                if lk is not None and p_kind in looks_task:
                    looks_task[p_kind]["sum"] += float(lk[i])
                    looks_task[p_kind]["n"] += 1

            traces = out.get("traces") or []
            if traces:
                layer_u = [tr.surprise_u for tr in traces]
                layer_gap = [getattr(tr, "vfe_gap", 0.0) for tr in traces]
                layer_F = [getattr(tr, "vfe_F", 0.0) for tr in traces]
                layer_rs = [tr.rms_s for tr in traces]
                layer_rd = [tr.rms_delta for tr in traces]
                layer_rr = [tr.rms_ratio for tr in traces]
                layer_K = [getattr(tr, "kalman_k", 0.0) for tr in traces]
            else:
                layer_u = layer_gap = layer_F = layer_rs = layer_rd = layer_rr = layer_K = []
            logits_k = out.get("logits_k") or []
            layer_ce = [float(F.cross_entropy(lg, targets).item()) for lg in logits_k]
            rho = out.get("rho") or {}
            rho_x_sum += float(rho.get("rho_x", 0.0))
            rho_h_sum += float(rho.get("rho_h", 0.0))
            if not u_per_layer:
                u_per_layer = layer_u
                gap_per_layer = layer_gap
                F_per_layer = layer_F
                rms_s_per_layer = layer_rs
                rms_d_per_layer = layer_rd
                rms_r_per_layer = layer_rr
                K_per_layer = layer_K
                ce_per_layer = layer_ce
            else:
                u_per_layer = [u1 + u2 for u1, u2 in zip(u_per_layer, layer_u)]
                gap_per_layer = [a + b for a, b in zip(gap_per_layer, layer_gap)]
                F_per_layer = [a + b for a, b in zip(F_per_layer, layer_F)]
                rms_s_per_layer = [a + b for a, b in zip(rms_s_per_layer, layer_rs)]
                rms_d_per_layer = [a + b for a, b in zip(rms_d_per_layer, layer_rd)]
                rms_r_per_layer = [a + b for a, b in zip(rms_r_per_layer, layer_rr)]
                K_per_layer = [a + b for a, b in zip(K_per_layer, layer_K)]
                if layer_ce and ce_per_layer:
                    ce_per_layer = [a + b for a, b in zip(ce_per_layer, layer_ce)]
                elif layer_ce and not ce_per_layer:
                    ce_per_layer = layer_ce

    u_avg = [u / val_batches for u in u_per_layer] if val_batches > 0 else []
    nval = max(1, val_batches)
    gap_avg = [x / nval for x in gap_per_layer]
    F_avg = [x / nval for x in F_per_layer]
    rms_s_avg = [x / nval for x in rms_s_per_layer]
    rms_d_avg = [x / nval for x in rms_d_per_layer]
    rms_r_avg = [x / nval for x in rms_r_per_layer]
    K_avg = [x / nval for x in K_per_layer]
    ce_avg = [x / nval for x in ce_per_layer]

    task_accs = {}
    for k, v in task_stats.items():
        task_accs[k] = v["correct"] / max(1, v["total"])

    return {
        "acc": correct / max(1, total),
        "loss": total_loss / max(1, total),
        "layer_surprise": u_avg,
        "layer_gap": gap_avg,
        "layer_F": F_avg,
        "layer_rms_s": rms_s_avg,
        "layer_rms_delta": rms_d_avg,
        "layer_rms_ratio": rms_r_avg,
        "layer_K": K_avg,
        "layer_ce": ce_avg,
        "rho_x": rho_x_sum / nval,
        "rho_h": rho_h_sum / nval,
        "task_accs": task_accs,
        "mean_looks": looks_sum / max(1, looks_n),
        "looks_task": {
            k: (v["sum"] / max(1, v["n"])) for k, v in looks_task.items()
        },
    }


def train_single_arm(
    arm: str,
    steps: int = 300,
    batch_size: int = 32,
    res: int = 32,
    d_model: int = 128,
    n_layers: int = 4,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
    eval_interval: int = 25,
) -> Dict:
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 999)

    model = DualStreamVQAModel(
        d_model=d_model,
        n_slices=32,
        n_layers=n_layers,
        res=res,
        surprise_mode=arm,
        surprise_beta=1.5,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    trajectory = []
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        batch = make_vqa_batch(rng_train, batch=batch_size, res=res, mix=["ocr", "kinks", "color"])
        images = batch["image"].to(dev)
        prompts = batch["prompt"]
        targets = torch.tensor([model.ans_to_idx[a] for a in batch["answer"]], device=dev)

        optimizer.zero_grad()
        out = model(images, prompts)
        loss = model.task_loss(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log trajectory periodically
        if step % eval_interval == 0 or step == steps:
            preds = out["logits"].argmax(dim=-1)
            train_acc = float((preds == targets).float().mean().item())
            trajectory.append({
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
            })

    elapsed = time.time() - t0

    # Final comprehensive evaluation on held-out seed data (20 batches = 640 samples)
    val_res = eval_model(model, rng_val, val_batches=20, batch_size=batch_size, res=res, device=dev)

    return {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "n_layers": n_layers,
        "val_acc": val_res["acc"],
        "val_loss": val_res["loss"],
        "task_accs": val_res["task_accs"],
        "layer_surprise": val_res["layer_surprise"],
        "trajectory": trajectory,
        "elapsed_sec": round(elapsed, 2),
    }


def render_diagnostic_plots(summary: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trajectory plot (Loss & Acc over steps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    arm_colors = {
        "baseline": "#94a3b8",
        "random": "#64748b",
        "constant": "#475569",
        "v0_jepa": "#38bdf8",
        "v0_shuffled": "#f59e0b",
        "v0_reverse": "#ef4444",
        "v0_global_only": "#a855f7",
        "v0_spatial_only": "#ec4899",
        "v1_bayes": "#2dd4bf",
    }

    for arm, s in summary.items():
        col = arm_colors.get(arm, "#ffffff")
        runs = s["runs"]
        # average trajectory over seeds
        all_trajs = [r["trajectory"] for r in runs if "trajectory" in r and r["trajectory"]]
        if not all_trajs:
            continue
        steps = [pt["step"] for pt in all_trajs[0]]
        losses = np.mean([[pt["train_loss"] for pt in t] for t in all_trajs], axis=0)
        accs = np.mean([[pt["train_acc"] for pt in t] for t in all_trajs], axis=0)

        lw = 2.5 if arm in ("v0_jepa", "v1_bayes", "baseline", "v0_reverse") else 1.5
        ls = "--" if "shuffled" in arm or "reverse" in arm else "-"
        ax1.plot(steps, losses, label=arm, color=col, linewidth=lw, linestyle=ls)
        ax2.plot(steps, accs * 100, label=arm, color=col, linewidth=lw, linestyle=ls)

    ax1.set_title("Training Loss Trajectory Loss(t)", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Steps", color="#cbd5e1")
    ax1.set_ylabel("Cross Entropy Loss", color="#cbd5e1")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    ax2.set_title("Training Accuracy Trajectory Acc(t) %", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Steps", color="#cbd5e1")
    ax2.set_ylabel("Batch Accuracy (%)", color="#cbd5e1")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    plt.tight_layout()
    traj_path = out_dir / "eval_loss_acc_trajectories.png"
    plt.savefig(traj_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved trajectory plot to {traj_path}")

    # 2. Per-Task Accuracy Breakdown (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    arms = list(summary.keys())
    tasks = ["ocr", "kinks", "color"]
    task_labels = ["1px OCR (Digit)", "Kinks (Polyline Corners)", "Needle (Color Square)"]
    x = np.arange(len(arms))
    width = 0.26

    task_colors = ["#38bdf8", "#fb7185", "#fbbf24"]

    for i, (t_key, t_label, t_col) in enumerate(zip(tasks, task_labels, task_colors)):
        means = []
        for arm in arms:
            t_accs = [r["task_accs"].get(t_key, 0.0) * 100 for r in summary[arm]["runs"]]
            means.append(np.mean(t_accs))
        offset = (i - 1) * width
        rects = ax.bar(x + offset, means, width, label=t_label, color=t_col, alpha=0.85, edgecolor="#0f172a")

    ax.set_title("Per-Task Validation Accuracy Breakdown Across Arms", color="white", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=25, ha="right", color="#e2e8f0", fontsize=10)
    ax.set_ylabel("Validation Accuracy (%)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b", axis="y")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=10)

    plt.tight_layout()
    task_path = out_dir / "eval_task_breakdown.png"
    plt.savefig(task_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved task breakdown plot to {task_path}")

    # 3. Paired Seed Deltas Distribution (relative to baseline)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    comp_arms = [a for a in arms if a != "baseline"]
    delta_means = [summary[a]["paired_gain_mean"] for a in comp_arms]
    delta_se = [summary[a]["paired_gain_se"] for a in comp_arms]

    bar_colors = ["#2dd4bf" if m > 0 else "#ef4444" for m in delta_means]
    bars = ax.bar(comp_arms, delta_means, yerr=delta_se, capsize=5, color=bar_colors, alpha=0.85, edgecolor="#0f172a")
    ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)

    for bar, m in zip(bars, delta_means):
        y_pos = bar.get_height() + (0.5 if m >= 0 else -1.2)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{m:+.2f}%",
                ha="center", va="bottom" if m >= 0 else "top", color="white", fontsize=9, fontweight="bold")

    ax.set_title("Paired Seed Accuracy Delta relative to Baseline (Delta_s = Acc_arm - Acc_baseline)",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("Paired Gain Delta (%)", color="#cbd5e1")
    ax.set_xticklabels(comp_arms, rotation=25, ha="right", color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b", axis="y")

    plt.tight_layout()
    delta_path = out_dir / "eval_paired_deltas.png"
    plt.savefig(delta_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved paired delta plot to {delta_path}")


def main():
    parser = argparse.ArgumentParser(description="Deep 4-Layer Bayesian Surprise & JEPA Validation.")
    parser.add_argument("--steps", type=int, default=300, help="Training steps per arm")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--res", type=int, default=32, help="Resolution")
    parser.add_argument("--dim", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of MoT layers")
    parser.add_argument("--seeds", type=int, default=4, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/published/v0_surprise_eval_table.json")
    args = parser.parse_args()

    arms = [
        "baseline",
        "random",
        "constant",
        "v0_jepa",
        "v0_shuffled",
        "v0_reverse",
        "v0_global_only",
        "v0_spatial_only",
        "v1_bayes",
    ]
    print(f"[Deep Eval] Starting validation across {len(arms)} arms with {args.seeds} seeds on {args.device.upper()} (L={args.layers})...")

    results = {}
    seed_list = [42 + s * 101 for s in range(args.seeds)]

    for arm in arms:
        results[arm] = []
        for seed in seed_list:
            res = train_single_arm(
                arm=arm,
                steps=args.steps,
                batch_size=args.batch,
                res=args.res,
                d_model=args.dim,
                n_layers=args.layers,
                seed=seed,
                device=args.device,
            )
            results[arm].append(res)
            print(f"  Arm: {arm:<16} Seed: {seed} -> Val Acc: {res['val_acc']*100:.2f}%, Loss: {res['val_loss']:.4f}, Tasks: { {k: f'{v*100:.1f}%' for k, v in res['task_accs'].items()} }")

    # Aggregate metrics & compute paired seed deltas
    summary = {}
    baseline_runs = {r["seed"]: r["val_acc"] for r in results["baseline"]}

    for arm, runs in results.items():
        accs = [r["val_acc"] * 100 for r in runs]
        losses = [r["val_loss"] for r in runs]

        # Paired deltas per seed
        paired_deltas = [(r["val_acc"] - baseline_runs[r["seed"]]) * 100 for r in runs]

        task_means = {}
        for t_k in ["ocr", "kinks", "color"]:
            t_vals = [r["task_accs"].get(t_k, 0.0) * 100 for r in runs]
            task_means[t_k] = float(np.mean(t_vals))

        summary[arm] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_loss": float(np.mean(losses)),
            "paired_gain_mean": float(np.mean(paired_deltas)),
            "paired_gain_se": float(np.std(paired_deltas) / np.sqrt(len(paired_deltas))),
            "task_means": task_means,
            "runs": runs,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'Arm':<18} {'Mean Acc (%)':<16} {'Paired Delta':<16} {'OCR Acc':<10} {'Kinks Acc':<10} {'Loss'}")
    print("=" * 80)
    for arm, s in summary.items():
        paired_str = f"{s['paired_gain_mean']:+.2f} ± {s['paired_gain_se']:.2f}%" if arm != "baseline" else "0.00% (Ref)"
        print(f"{arm:<18} {s['mean_acc']:.2f} ± {s['std_acc']:.2f}%     {paired_str:<16} {s['task_means']['ocr']:.1f}%      {s['task_means']['kinks']:.1f}%      {s['mean_loss']:.4f}")
    print("=" * 80)
    print(f"Results saved to {out_path}")

    # Generate diagnostic plots
    fig_dir = ROOT / "present" / "figs"
    render_diagnostic_plots(summary, fig_dir)


if __name__ == "__main__":
    main()
