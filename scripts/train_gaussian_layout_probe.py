"""Matched scratch overfit: legacy vs correctly packed per-head Gaussians."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fine_grain.capability_tasks import capability_sample, collate_capability, CAPABILITY_CASES
from fine_grain.omni_model import DualStreamOmni
from fine_grain.gen_metrics import paired_digit_scores, paired_ink_iou
from scripts.audit_attention_write import forward_batch


@torch.no_grad()
def evaluate(model, samples):
    model.eval()
    out = forward_batch(model, samples, "cuda")
    rows = {}
    for case in CAPABILITY_CASES[:3]:
        idx = [i for i, s in enumerate(samples) if s["case"] == case]
        ds, iou, text, seg = [], [], [], []
        for i in idx:
            s = samples[i]
            ds.append(paired_digit_scores(out["rgb"][i], s["digit"], s["target_color"], s["target_place"])["paired_digit_top1"])
            iou.append(paired_ink_iou(out["rgb"][i:i+1], s["stroke"].cuda(), s["target_color"]))
            text.append(float(out["logits"][i].argmax() == model.ans_to_idx[s["answer"]]))
            mask = out["seg_logits"][i].argmax(0) > 0
            gold = s["target_seg"].cuda() > 0
            seg.append(float((mask & gold).sum() / (mask | gold).sum().clamp_min(1)))
        rows[case] = {"paired_digit_top1": float(np.mean(ds)), "paired_iou": float(np.mean(iou)),
                      "text_acc": float(np.mean(text)), "seg_iou": float(np.mean(seg))}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--layouts", nargs="+", default=["legacy", "per_head"])
    parser.add_argument("--out", default="results/published/gaussian_layout_probe.json")
    parser.add_argument("--init", default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--tag", default="probe")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.init is not None and len(args.layouts) != 1:
        raise ValueError("A continuation must specify one matching layout")
    base = torch.load("checkpoints/northstar_static32_unified_task_best.pt", map_location="cpu")["config"]
    rng = np.random.default_rng(91)
    samples = [capability_sample(rng, 32, case, digit=str(d), color="red", place="middle_center")
               for case in CAPABILITY_CASES[:3] for d in range(10)]
    initialization = "scratch" if args.init is None else "checkpoint continuation"
    results = {"scope": f"30 fixed 32px samples, ten digits at one color/address in three modes; {initialization}, one seed",
               "steps": args.steps, "seed": 42, "learning_rate": args.lr, "init": args.init, "arms": {}}
    for layout in args.layouts:
        torch.manual_seed(42)
        config = {**base, "control_prefix_attention": True, "use_attention_sink": False,
                  "gaussian_head_layout": layout, "deep_visual_likelihood_coef": 0.0}
        model = DualStreamOmni(**config).cuda()
        if args.init is not None:
            saved = torch.load(args.init, map_location="cpu")
            if saved["config"] != config:
                raise ValueError("Continuation config must exactly match the saved graph")
            model.load_state_dict(saved["state_dict"])
        if args.audit_only:
            if args.init is None:
                raise ValueError("audit-only needs --init")
            intact = evaluate(model, samples)
            changed = [dict(s) for s in samples]
            for offset in (0, 10, 20):
                for i in range(10):
                    key = "prompt" if offset == 0 else "image"
                    changed[offset + i][key] = samples[offset + (i + 1) % 10][key]
            interventions = evaluate(model, changed)
            audit = {"checkpoint": args.init, "scope": results["scope"],
                     "intervention": "rotate digit prompts for generation; rotate source images for current/edit; retain original targets",
                     "intact": intact, "changed": interventions}
            Path(args.out).write_text(json.dumps(audit, indent=2), encoding="utf-8")
            print(json.dumps(audit, indent=2), flush=True)
            return
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        warmup = args.steps // 3 if args.warmup_steps is None else args.warmup_steps
        results["warmup_steps"] = warmup
        checkpoint = f"checkpoints/gaussian_layout_{layout}_{args.tag}.pt"
        history = []
        best = -1.0
        for step in range(1, args.steps + 1):
            model.train()
            # Establish empty-canvas mapping, then train all three boundaries.
            part = samples[:10] if step <= warmup else samples
            batch = collate_capability(part)
            batch["t"] = torch.zeros(len(part), device="cuda")
            out = forward_batch(model, part, "cuda")
            loss, meta = model.omni_loss(out, batch, torch.device("cuda"))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            rate = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
            for g in optimizer.param_groups:
                g["lr"] = rate
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                ev = evaluate(model, samples)
                row = {"step": step, "loss": float(loss.detach()), "metrics": ev}
                history.append(row)
                print(json.dumps({"layout": layout, **row}), flush=True)
                score = np.mean([m["paired_digit_top1"] + m["paired_iou"] + m["text_acc"] + m["seg_iou"] for m in ev.values()])
                if score > best:
                    best = score
                    torch.save({"config": config, "state_dict": model.state_dict(), "step": step,
                                "scope": results["scope"]}, checkpoint)
        path = checkpoint
        model.load_state_dict(torch.load(path, map_location="cuda")["state_dict"])
        results["arms"][layout] = {"checkpoint": path, "config": config, "history": history,
                                    "reloaded_metrics": evaluate(model, samples)}
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        del model, optimizer, out, loss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
