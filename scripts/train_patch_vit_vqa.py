#!/usr/bin/env python3
"""2000-step traditional patch-ViT VQA on the same protocol as the triad.

python scripts/train_patch_vit_vqa.py --steps 2000 --runs 3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.patch_vit_vqa import PatchViTVQAModel
from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import eval_model


def train_one(run_idx: int, seed: int, steps: int, device: str, res: int = 32) -> dict:
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 9999)

    model = PatchViTVQAModel(d_model=128, n_layers=4, res=res, patch=4, n_heads=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    cached = []
    for _ in range(150):
        b = make_vqa_batch(rng_train, batch=32, res=res, mix=["ocr", "kinks", "color"])
        cached.append({
            "imgs": b["image"].to(dev),
            "prompts": b["prompt"],
            "targets": torch.tensor([model.ans_to_idx[a] for a in b["answer"]], device=dev),
        })

    best_acc = -1.0
    best_sd = None
    trajectory = []
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        b = cached[(step - 1) % len(cached)]
        opt.zero_grad()
        out = model(b["imgs"], b["prompts"])
        loss = model.task_loss(out, b["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 25 == 0 or step == steps:
            preds = out["logits"].argmax(dim=-1)
            train_acc = float((preds == b["targets"]).float().mean().item())
            quick = eval_model(model, rng_val, val_batches=5, batch_size=32, res=res, device=dev)
            if quick["acc"] > best_acc:
                best_acc = quick["acc"]
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            trajectory.append({
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_loss": quick["loss"],
                "val_acc": quick["acc"],
                "val_ocr": quick["task_accs"].get("ocr", 0.0),
                "val_kinks": quick["task_accs"].get("kinks", 0.0),
                "val_color": quick["task_accs"].get("color", 0.0),
            })
            if step % 200 == 0 or step == steps:
                print(
                    f"    [PATCH_VIT Run {run_idx+1}] Step {step:4d}/{steps}: "
                    f"Val Acc={quick['acc']*100:.1f}%, Loss={quick['loss']:.4f}, "
                    f"OCR={quick['task_accs'].get('ocr',0)*100:.1f}%, "
                    f"Color={quick['task_accs'].get('color',0)*100:.1f}%",
                    flush=True,
                )

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"patch_vit_2000step_run{run_idx+1}_best.pt"
    if best_sd is not None:
        torch.save(best_sd, ckpt_path)
        print(f"    Saved best checkpoint ({best_acc*100:.2f}%) to {ckpt_path}", flush=True)
        model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    final = eval_model(model, rng_val, val_batches=35, batch_size=32, res=res, device=dev)
    return {
        "arm": "patch_vit",
        "run_idx": run_idx + 1,
        "seed": seed,
        "steps": steps,
        "best_probe_acc": best_acc,
        "final_acc": final["acc"],
        "final_loss": final["loss"],
        "final_tasks": final["task_accs"],
        "trajectory": trajectory,
        "elapsed_sec": round(time.time() - t0, 2),
        "checkpoint": str(ckpt_path),
        "patch": 4,
        "n_layers": 4,
        "d_model": 128,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/published/patch_vit_2000step_table.json")
    args = parser.parse_args()
    seeds = [42, 1001, 7777][: args.runs]
    runs = []
    print(f"[PatchViT] {args.runs} runs × {args.steps} steps on {args.device}", flush=True)
    for i, seed in enumerate(seeds):
        print(f"\n--- Patch ViT Run {i+1}/{args.runs} (Seed: {seed}) ---", flush=True)
        rec = train_one(i, seed, args.steps, args.device)
        runs.append(rec)
        print(
            f"=== Done Run {i+1} in {rec['elapsed_sec']}s: "
            f"Acc={rec['final_acc']*100:.2f}% OCR={rec['final_tasks']['ocr']*100:.1f}% "
            f"Kinks={rec['final_tasks']['kinks']*100:.1f}% Color={rec['final_tasks']['color']*100:.1f}% ===",
            flush=True,
        )
    accs = [r["final_acc"] * 100 for r in runs]
    table = {
        "patch_vit": {
            "arm": "patch_vit",
            "steps": args.steps,
            "n_runs": args.runs,
            "patch": 4,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_loss": float(np.mean([r["final_loss"] for r in runs])),
            "mean_ocr": float(np.mean([r["final_tasks"]["ocr"] * 100 for r in runs])),
            "mean_kinks": float(np.mean([r["final_tasks"]["kinks"] * 100 for r in runs])),
            "mean_color": float(np.mean([r["final_tasks"]["color"] * 100 for r in runs])),
            "runs": runs,
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    p = table["patch_vit"]
    print(
        f"\nPATCH VIT  {p['mean_acc']:.2f} ± {p['std_acc']:.2f}  "
        f"ocr={p['mean_ocr']:.1f} kinks={p['mean_kinks']:.1f} color={p['mean_color']:.1f}",
        flush=True,
    )
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
