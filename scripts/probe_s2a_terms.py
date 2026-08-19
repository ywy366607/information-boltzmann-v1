#!/usr/bin/env python3
"""Decompose Active Inference terms on the S2a ckpt (not just acc)."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.s2a import kl_cat
from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel

CKPT = ROOT / "checkpoints" / "v1_bayes_2000step_s2a_l4i2_run1_best.pt"
OUT = ROOT / "results" / "published" / "s2a_ai_terms.json"


def _load(device):
    m = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms_dir", use_stiefel=True, deslice_topk=2, n_heads=4,
        gate_on="u", deslice_write="absolute", vfe_coef=0.1,
        saccade=True, saccade_inner=2, s2a=True,
    ).to(device)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    m.eval()
    m.mot_stack.s2a_eval_halt = False  # force both looks to measure IG
    return m


@torch.no_grad()
def main():
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = _load(device)
    rng = np.random.default_rng(123)
    buckets = {k: defaultdict(list) for k in ("color", "ocr", "kinks")}
    n_batches = 12
    for _ in range(n_batches):
        b = make_vqa_batch(rng, batch=32, res=32, mix=["ocr", "kinks", "color"])
        imgs = b["image"].to(device)
        out = m(imgs, b["prompt"])
        probes = b["probe"]
        traces = out["traces"]
        igs = out["s2a_ig"]  # 4 tensors [B]
        gs = [torch.sigmoid(x) for x in out["s2a_g_logits"]]
        logits_k = out["logits_k"]
        tgt = torch.tensor([m.ans_to_idx[a] for a in b["answer"]], device=device)
        for i, kind in enumerate(probes):
            rec = buckets[kind]
            rec["acc1"].append(int(logits_k[0][i].argmax() == tgt[i]))
            rec["acc_last"].append(int(out["logits"][i].argmax() == tgt[i]))
            p0 = F.softmax(logits_k[0][i], dim=-1)
            rec["ent1"].append(float((-(p0 * p0.clamp_min(1e-8).log())).sum()))
            rec["conf1"].append(float(p0.max()))
            for li in range(4):
                rec[f"U_l{li}a"].append(traces[2 * li].surprise_u)
                rec[f"U_l{li}b"].append(traces[2 * li + 1].surprise_u)
                rec[f"F_l{li}a"].append(traces[2 * li].vfe_F)
                rec[f"F_l{li}b"].append(traces[2 * li + 1].vfe_F)
                rec[f"gap_l{li}a"].append(traces[2 * li].vfe_gap)
                rec[f"gap_l{li}b"].append(traces[2 * li + 1].vfe_gap)
                rec[f"ig_l{li}"].append(float(igs[li][i]))
                rec[f"ghat_l{li}"].append(float(gs[li][i]))
                rec[f"open_l{li}"].append(int(float(gs[li][i]) > 0.5))
            # answer moved?
            rec["ans_changed"].append(int(logits_k[0][i].argmax() != out["logits"][i].argmax()))
            rec["ig_mean"].append(float(torch.stack([igs[li][i] for li in range(4)]).mean()))
            rec["ghat_mean"].append(float(torch.stack([gs[li][i] for li in range(4)]).mean()))

    summary = {}
    for kind, rec in buckets.items():
        n = len(rec["acc_last"])
        s = {"n": n}
        for k, vs in rec.items():
            s[k] = float(np.mean(vs))
        # correlation Ĝ vs IG per layer
        for li in range(4):
            ig = np.array(rec[f"ig_l{li}"])
            gh = np.array(rec[f"ghat_l{li}"])
            if ig.std() > 1e-8 and gh.std() > 1e-8:
                s[f"corr_g_ig_l{li}"] = float(np.corrcoef(ig, gh)[0, 1])
            else:
                s[f"corr_g_ig_l{li}"] = 0.0
        summary[kind] = s
        print(
            f"{kind:6s} n={n} acc1={s['acc1']:.3f} accL={s['acc_last']:.3f} "
            f"ent1={s['ent1']:.3f} IGbar={s['ig_mean']:.4f} Ĝbar={s['ghat_mean']:.3f} "
            f"open={np.mean([s[f'open_l{i}'] for i in range(4)]):.2f} "
            f"Δans={s['ans_changed']:.3f}",
            flush=True,
        )
        print(
            f"       U  {[round(s[f'U_l{i}a'],3) for i in range(4)]} -> {[round(s[f'U_l{i}b'],3) for i in range(4)]}",
            flush=True,
        )
        print(
            f"       F  {[round(s[f'F_l{i}a'],3) for i in range(4)]} -> {[round(s[f'F_l{i}b'],3) for i in range(4)]}",
            flush=True,
        )
        print(
            f"       gap{[round(s[f'gap_l{i}a'],3) for i in range(4)]} -> {[round(s[f'gap_l{i}b'],3) for i in range(4)]}",
            flush=True,
        )
        print(
            f"       IG {[round(s[f'ig_l{i}'],4) for i in range(4)]}  "
            f"Ĝ {[round(s[f'ghat_l{i}'],3) for i in range(4)]}  "
            f"corr {[round(s[f'corr_g_ig_l{i}'],3) for i in range(4)]}",
            flush=True,
        )

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
