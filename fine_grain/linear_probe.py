"""Non-generative linear probe on vision frontend tokens.

Same synthetic distribution as VQA (color 4-way + kinks 4-way, no angles).
LLM is never used: mean-pool vision tokens → linear heads.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.frontends import build_frontend
from fine_grain.vlm_data import KINK_KS, COLORS, make_vqa_batch

N_COLOR = len(COLORS)  # 4
N_KINKS = len(KINK_KS)  # 4 (labels 0..3 for k=5..8)


class VisionLinearProbe(nn.Module):
    """Frontend + mean-pool + dual linear heads (color / kinks)."""

    def __init__(self, frontend: nn.Module, d_feat: int, n_color: int = N_COLOR, n_kinks: int = N_KINKS):
        super().__init__()
        self.frontend = frontend
        self.color_head = nn.Linear(d_feat, n_color)
        self.kinks_head = nn.Linear(d_feat, n_kinks)

    def encode(self, images: torch.Tensor):
        out = self.frontend(images)
        # Fixed pool over vision tokens (no LLM, no attention over text).
        pooled = out.tokens.mean(dim=1)  # [B, d]
        return pooled, out

    def forward(self, images: torch.Tensor):
        pooled, out = self.encode(images)
        return {
            "color_logits": self.color_head(pooled),
            "kinks_logits": self.kinks_head(pooled),
            "pooled": pooled,
            "meta": out.meta,
            "T": out.T,
            "tokens": out.tokens,
        }


def labels_from_batch(data: Dict) -> Dict[str, torch.Tensor]:
    """Integer labels + boolean masks from make_vqa_batch tags."""
    tags = data["tags"]
    B = len(tags)
    color_y = torch.full((B,), -100, dtype=torch.long)
    kinks_y = torch.full((B,), -100, dtype=torch.long)
    is_color = torch.zeros(B, dtype=torch.bool)
    is_kinks = torch.zeros(B, dtype=torch.bool)
    for i, t in enumerate(tags):
        if t["probe"] == "color":
            color_y[i] = int(t["label"])
            is_color[i] = True
        elif t["probe"] == "kinks":
            kinks_y[i] = int(t["label"])
            is_kinks[i] = True
    return {
        "color_y": color_y,
        "kinks_y": kinks_y,
        "is_color": is_color,
        "is_kinks": is_kinks,
    }


def probe_loss(logits: Dict[str, torch.Tensor], lab: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean CE over present tasks in the batch (masked -100)."""
    losses = []
    if lab["is_color"].any():
        losses.append(F.cross_entropy(logits["color_logits"], lab["color_y"], ignore_index=-100))
    if lab["is_kinks"].any():
        losses.append(F.cross_entropy(logits["kinks_logits"], lab["kinks_y"], ignore_index=-100))
    if not losses:
        # empty mix edge case
        return logits["color_logits"].sum() * 0.0
    return sum(losses) / len(losses)


@torch.no_grad()
def eval_linear_probe(
    model: VisionLinearProbe,
    device: torch.device,
    res: int,
    n: int = 64,
    batch: int = 8,
    val_seed: int = 90_001,
    mix: Sequence[str] = ("color", "kinks"),
    collect_failures: int = 0,
) -> dict:
    """Held-out classification accuracy; optional failure sample dump."""
    model.eval()
    rng = np.random.default_rng(val_seed)
    by = {
        "color": {"hit": 0, "tot": 0},
        "kinks": {"hit": 0, "tot": 0},
    }
    failures: List[dict] = []
    left = n
    while left > 0:
        b = min(batch, left)
        data = make_vqa_batch(rng, b, res=res, mix=mix)
        img = data["image"].to(device)
        lab = labels_from_batch(data)
        out = model(img)
        color_pred = out["color_logits"].argmax(dim=-1).cpu()
        kinks_pred = out["kinks_logits"].argmax(dim=-1).cpu()
        for i, kind in enumerate(data["probe"]):
            if kind == "color":
                gold = int(lab["color_y"][i].item())
                pred = int(color_pred[i].item())
                ok = pred == gold
                by["color"]["tot"] += 1
                by["color"]["hit"] += int(ok)
                if not ok and len(failures) < collect_failures:
                    failures.append({
                        "probe": "color",
                        "gold": COLORS[gold] if 0 <= gold < len(COLORS) else gold,
                        "gold_id": gold,
                        "pred_id": pred,
                        "pred": COLORS[pred] if 0 <= pred < len(COLORS) else pred,
                        "answer": data["answer"][i],
                    })
            elif kind == "kinks":
                gold = int(lab["kinks_y"][i].item())
                pred = int(kinks_pred[i].item())
                ok = pred == gold
                by["kinks"]["tot"] += 1
                by["kinks"]["hit"] += int(ok)
                if not ok and len(failures) < collect_failures:
                    failures.append({
                        "probe": "kinks",
                        "gold_id": gold,
                        "gold_k": gold + 5,  # lab = k - 5
                        "pred_id": pred,
                        "pred_k": pred + 5,
                        "answer": data["answer"][i],
                    })
        left -= b
    model.train()
    result = {
        "acc_color": by["color"]["hit"] / max(by["color"]["tot"], 1),
        "n_color": by["color"]["tot"],
        "acc_kinks": by["kinks"]["hit"] / max(by["kinks"]["tot"], 1),
        "n_kinks": by["kinks"]["tot"],
        "acc_overall": (by["color"]["hit"] + by["kinks"]["hit"])
        / max(by["color"]["tot"] + by["kinks"]["tot"], 1),
        "n_overall": by["color"]["tot"] + by["kinks"]["tot"],
        "protocol": "linear_probe",
    }
    if collect_failures:
        result["failures"] = failures
    return result


def train_linear_probe(
    kind: str,
    T: int,
    *,
    d_feat: int = 640,
    res: int = 32,
    patch: int = 4,
    dim: int = 64,
    depth: int = 2,
    topk: int = 2,
    T_slice: Optional[int] = None,
    projector: str = "mlp",
    steps: int = 200,
    batch: int = 8,
    lr: float = 3e-4,
    seed: int = 0,
    val_seed: int = 90_001,
    probe_n: int = 64,
    device: Optional[torch.device] = None,
    log_every: int = 50,
    collect_failures: int = 8,
) -> dict:
    """Train frontend+heads only; return metrics and optional failure samples."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    fe = build_frontend(
        kind, d_feat, res=res, T=T, patch=patch,
        T_slice=T_slice, dim=dim, depth=depth, deslice_topk=topk,
        projector=projector,
    ).to(device)
    model = VisionLinearProbe(fe, d_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 7)
    hist = []
    t0 = __import__("time").time()
    model.train()
    meta_last = {}
    for step in range(1, steps + 1):
        data = make_vqa_batch(rng, batch, res=res, mix=("color", "kinks"))
        img = data["image"].to(device)
        lab = {k: v.to(device) for k, v in labels_from_batch(data).items()}
        out = model(img)
        loss = probe_loss(out, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        meta_last = dict(out["meta"])
        if "slot" in meta_last or hasattr(fe, "_last_slot"):
            slot = meta_last.get("slot") or getattr(fe, "_last_slot", {})
            if slot:
                meta_last["slot"] = slot
        if step % log_every == 0 or step == 1 or step == steps:
            hist.append({
                "step": step,
                "loss": float(loss.detach().float().item()),
                "T": int(out["T"]),
                "kind": kind,
            })
            print(
                f"  [probe {kind} T={out['T']}] step {step:4d} "
                f"loss={hist[-1]['loss']:.4f}",
                flush=True,
            )
    metrics = eval_linear_probe(
        model, device, res, n=probe_n, batch=min(8, batch),
        val_seed=val_seed, collect_failures=collect_failures,
    )
    # attach slot from last forward if B/C
    slot = getattr(fe, "_last_slot", {}) or meta_last.get("slot", {})
    return {
        "kind": kind,
        "T": int(hist[-1]["T"]) if hist else T,
        "protocol": "linear_probe",
        "final_loss": hist[-1]["loss"] if hist else float("nan"),
        "acc_overall": metrics["acc_overall"],
        "acc_color": metrics["acc_color"],
        "acc_kinks": metrics["acc_kinks"],
        "probe": metrics,
        "failures": metrics.get("failures", []),
        "slot_last": slot,
        "budget": meta_last.get("budget"),
        "seconds": __import__("time").time() - t0,
        "seed": seed,
        "status": "ok",
        "history": hist,
    }
