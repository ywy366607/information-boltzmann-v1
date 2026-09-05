"""Train one corrected Slice/Pythia graph on a small, genuine multimodal bank."""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import html
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fine_grain.omni_model import DualStreamOmni
from fine_grain.real_capacity import (
    bank_metadata, build_real_bank, collate_real_capacity, forward_real_capacity,
)
from fine_grain.token_tasks import answer_token_accuracy
from scripts.train_sharegpt4o_t2i_overfit import edge_metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_graph_weights(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("lm.") for key in missing):
        raise RuntimeError("incomplete non-language checkpoint reload")


@torch.no_grad()
def decode_caption(model, sample, device, max_tokens=48):
    """Every emitted token reruns the same visual-language graph; no lm.generate."""
    tokenizer = model.lm_tok
    generated = []
    batch = collate_real_capacity(tokenizer, [sample], device)
    answer_positions = batch["labels"][0].ne(-100).nonzero().flatten()
    if not answer_positions.numel():
        raise ValueError("text evaluation requires a non-empty reference answer")
    # Reuse the collator's observed prefix, including its missing-BOS policy.
    # Gold answer IDs are never passed into a decoding forward call.
    prefix_end = int(answer_positions[0])
    prefix_ids = batch["input_ids"][:, :prefix_end].clone()
    prefix_precision = batch["text_precision"][:, :prefix_end].clone()
    prefix_mask = batch["visual_prompt_mask"][:, :prefix_end].clone()
    for _ in range(max_tokens):
        suffix = torch.tensor([generated], device=device, dtype=torch.long)
        ids = torch.cat([prefix_ids, suffix], dim=1)
        batch.update(input_ids=ids, attention_mask=torch.ones_like(ids), labels=None,
                     visual_prompt_mask=torch.cat([prefix_mask, torch.zeros_like(suffix, dtype=torch.bool)], dim=1),
                     text_precision=torch.cat([prefix_precision, torch.ones_like(suffix, dtype=torch.float32)], dim=1))
        out = forward_real_capacity(model, batch)
        next_id = int(out["token_logits"][0, out["n_vis_tokens"] + ids.shape[1] - 1].argmax())
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    expected = tokenizer.encode(" " + sample["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
    return {"text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "exact": generated == expected, "eos": generated[-1] == tokenizer.eos_token_id}


@torch.no_grad()
def evaluate(model, samples, device, decode=True):
    model.eval()
    rows, displays = [], []
    for sample in samples:
        batch = collate_real_capacity(model.lm_tok, [sample], device)
        out = forward_real_capacity(model, batch)  # Prediction never reads target_rgb.
        pred = out["rgb"].detach().cpu()
        target = sample["target_rgb"].unsqueeze(0)
        mse = float((pred - target).square().mean())
        edge_error, edge_corr = edge_metrics(pred, target)
        row = {"id": sample["id"], "task": sample["task"], "family": sample.get("family", "real"),
               "need_pix": sample["need_pix"], "horizon": sample["target_time"], "mse": mse,
               "psnr": -10 * math.log10(max(mse, 1e-12)),
               "edge_correlation": edge_corr, "edge_relative_mse": edge_error}
        display = {"sample": sample, "pred": pred[0]}
        if sample["need_seg"]:
            mask = out["seg_logits"].argmax(1)[0].cpu() > 0
            gold = sample["target_seg"] > 0
            row["seg_iou"] = float((mask & gold).sum() / (mask | gold).sum().clamp_min(1))
            display["mask"] = mask
        if sample.get("family") == "generation1px":
            rgb_mask = (pred[0, 0] > .5) & (pred[0, 1:].amax(0) < .25)
            gold = sample["target_seg"] > 0
            row["rgb_stroke_iou"] = float((rgb_mask & gold).sum() / (rgb_mask | gold).sum().clamp_min(1))
            background = ~gold
            near = torch.nn.functional.max_pool2d(gold[None, None].float(), 5, 1, 2)[0, 0].bool() & background
            row["rgb_background_flood"] = float((rgb_mask & background).sum() / background.sum().clamp_min(1))
            row["rgb_near_background_flood"] = float((rgb_mask & near).sum() / near.sum().clamp_min(1))
        if sample["need_text"]:
            row["token_nll"] = float(out["token_nll"].mean())
            row["token_accuracy"] = float(answer_token_accuracy(out, batch).mean())
            if decode:
                row["decoded"] = decode_caption(model, sample, device)
        del out
        if sample["task"] in ("t2t", "it2t"):
            controls = {}
            if sample["task"] == "t2t":
                other = next(s for s in samples if s["task"] == "t2t" and s["answer"] != sample["answer"])
                controls["text"] = {**sample, "prompt": other["prompt"]}
            else:
                by_id = {s["id"]: s for s in samples}
                controls["text"] = {**sample, "prompt": by_id[sample["qa_text_control_id"]]["prompt"]}
                controls["image"] = {**sample, "image": by_id[sample["qa_image_control_id"]]["image"]}
            for name, changed in controls.items():
                control = forward_real_capacity(model, collate_real_capacity(model.lm_tok, [changed], device))
                row[name + "_shuffle_nll_gap"] = float(control["token_nll"].mean()) - row["token_nll"]
                del control
        if sample["task"] in ("t2i", "it2i", "i2t", "future"):
            changed = dict(sample)
            if sample["task"] == "t2i":
                changed["prompt"] = next(s["prompt"] for s in samples
                                          if s["task"] == "t2i" and s["id"] != sample["id"]
                                          and s.get("family", "real") == sample.get("family", "real"))
            elif sample["task"] == "it2i":
                root = sample["id"].removesuffix("-identity")
                other = next(s for s in samples if s["task"] == "it2i"
                             and s["id"].removesuffix("-identity") == root and s["id"] != sample["id"])
                changed["prompt"] = other["prompt"]
            elif sample["task"] == "i2t":
                changed["image"] = next(s["image"] for s in samples
                                         if s["task"] == "i2t" and s["id"] != sample["id"]
                                         and s.get("family", "real") == sample.get("family", "real"))
            else:
                # Isolate temporal evidence, keeping the current image intact.
                changed["history_precision"] = torch.zeros_like(sample["history_precision"])
            control_batch = collate_real_capacity(model.lm_tok, [changed], device)
            control = forward_real_capacity(model, control_batch)
            control_rgb = control["rgb"].cpu()
            row["control_mse_gap"] = float((control_rgb - target).square().mean()) - mse
            row["control_output_rms"] = float((pred - control_rgb).square().mean().sqrt())
            if sample.get("family") == "generation1px":
                control_mask = (control_rgb[0, 0] > .5) & (control_rgb[0, 1:].amax(0) < .25)
                gold = sample["target_seg"] > 0
                control_iou = float((control_mask & gold).sum() / (control_mask | gold).sum().clamp_min(1))
                row["prompt_stroke_iou_gap"] = row["rgb_stroke_iou"] - control_iou
            if sample["need_text"]:
                row["image_shuffle_nll_gap"] = float(control["token_nll"].mean()) - row["token_nll"]
            if sample["task"] == "future":
                copy_error = (sample["image"].unsqueeze(0) - target).square()
                row["copy_mse"] = float(copy_error.mean())
                row["improvement_over_copy"] = row["copy_mse"] - mse
                change = (sample["image"] - target[0]).abs().mean(0) > 0.025
                row["changed_pixel_fraction"] = float(change.float().mean())
                row["changed_region_mse"] = float(
                    ((pred - target).square().mean(1)[0] * change).sum() / change.sum().clamp_min(1))
                row["changed_region_copy_mse"] = float(
                    (copy_error.mean(1)[0] * change).sum() / change.sum().clamp_min(1))
                if sample["target_time"] > 0:
                    zero_time = {**sample, "target_time": 0.0}
                    zero_out = forward_real_capacity(model, collate_real_capacity(model.lm_tok, [zero_time], device))
                    row["zero_horizon_mse_gap"] = float((zero_out["rgb"].cpu() - target).square().mean()) - mse
                    del zero_out
            del control
        rows.append(row)
        displays.append(display)
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
        if any(s.get("family", "real") != "real" for s in rows):
            groups[row["task"] + "_" + row.get("family", "real")].append(row)
        if row["task"] == "future":
            groups["future_positive_horizon" if row["horizon"] > 0 else "future_zero_horizon"].append(row)
    summary = {task: {"n": len(group), **{
        key: float(np.mean([r[key] for r in group if key in r]))
        for key in ("psnr", "edge_correlation", "seg_iou", "token_nll", "token_accuracy",
                    "image_shuffle_nll_gap", "improvement_over_copy") if any(key in r for r in group)
    }} for task, group in groups.items()}
    return {"summary": summary, "samples": rows}, displays


def image_uri(tensor):
    if tensor.ndim == 2:
        tensor = tensor.float().unsqueeze(0).expand(3, -1, -1)
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_gallery(path, displays, metrics, step):
    resolution = displays[0]["pred"].shape[-1]
    rows = {r["id"]: r for r in metrics["samples"]}
    cards = []
    for display in displays:
        sample = display["sample"]
        row = rows[sample["id"]]
        panels = [("Input / blank if missing", sample["image"]),
                  ("Real target", sample["target_rgb"]), (f"Native {resolution} output", display["pred"])]
        if "mask" in display:
            panels += [("Official foreground mask", sample["target_seg"]), ("Predicted foreground", display["mask"])]
        pictures = "".join(f'<figure><img src="{image_uri(x)}"><figcaption>{label}</figcaption></figure>'
                           for label, x in panels)
        caption = ""
        if sample["need_text"]:
            caption = (f'<p>Target answer: {html.escape(sample["answer"])}</p>'
                       f'<p>Model answer: {html.escape(row.get("decoded", {}).get("text", "not decoded"))}</p>')
        cards.append(f'<section><h2>{html.escape(sample["task"])} · {html.escape(sample["id"])}</h2>'
                     f'<p>{html.escape(sample.get("prompt", "")) or "No observed language"}</p>'
                     f'<div class="panels">{pictures}</div>{caption}'
                     f'<p>PSNR {row["psnr"]:.2f} dB · edge correlation {row["edge_correlation"]:.3f}</p></section>')
    page = (f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Real {resolution} · Slice capability preview</title>'
            '<style>body{background:#10151e;color:#e9eef6;font:16px system-ui;margin:30px}'
            'section{background:#192230;padding:20px;margin:20px 0;border-radius:14px}'
            '.panels{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}img{width:256px;height:256px;object-fit:contain}'
            'figcaption{color:#a8bed5;margin:6px 0}p{max-width:1000px}h2{font-size:19px}</style>'
            f'<h1>Real {resolution} · shared Slice–language graph · step {step}</h1>'
            '<p>Training candidate, not a completed champion. Real source/target images, frozen Pythia, '
            'no external generator. Foreground masks and consecutive frames are from DAVIS. '
            'Next-frame success must beat simply copying the current frame.</p>' + "".join(cards) + '</html>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    t2i = [d for d in displays if d["sample"]["task"] == "t2i"]
    if t2i:
        panel = torch.cat([torch.cat([d["sample"]["target_rgb"], d["pred"]], dim=2)
                           for d in t2i], dim=1)
        Image.open(io.BytesIO(base64.b64decode(image_uri(panel).split(",", 1)[1]))).save(path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt-manifest", required=True)
    parser.add_argument("--davis-manifest", required=True)
    parser.add_argument("--config-from", default="checkpoints/gaussian_layout_per_head_finish.pt")
    parser.add_argument("--tag", default="real256_joint")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--warmup-steps", type=int, default=160)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init", default=None)
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < args.steps or args.eval_every < 1:
        parser.error("warmup must leave joint steps, eval-every must be positive")
    if Path(args.tag).name != args.tag or args.tag in (".", ".."):
        parser.error("tag must be a simple filename")
    checkpoint = ROOT / "checkpoints" / f"{args.tag}.pt"
    report_path = ROOT / "results" / "published" / f"{args.tag}.json"
    gallery = ROOT / "present" / f"{args.tag}.html"
    if any(p.exists() for p in (checkpoint, report_path, gallery)):
        parser.error("existing outputs are protected; choose a new tag")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    source_hash = sha256(ROOT / args.config_from)
    config = torch.load(ROOT / args.config_from, map_location="cpu")["config"]
    config.update(res=256, language="pythia", lm_device=args.device,
                  pixel_loss_mode="gaussian_nll", use_active_gdn2=True,
                  active_gdn2_initial_trust=0.1, control_prefix_attention=True,
                  gaussian_head_layout="per_head", use_attention_sink=False,
                  deep_visual_likelihood_coef=0.0)
    model = DualStreamOmni(**config).to(args.device)
    config.update(lm_id=model.lm_id, lm_revision=model.lm_revision)
    if args.init:
        initial = torch.load(ROOT / args.init, map_location="cpu")
        if initial["config"] != config:
            parser.error("continuation must use exactly the same graph and LM")
        load_graph_weights(model, initial["state_dict"])
    samples = build_real_bank(args.sharegpt_manifest, args.davis_manifest)
    t2i = [s for s in samples if s["task"] == "t2i"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    report = {"status": "training", "config": config, "training": vars(args),
              "initialization": "same-graph weights; new optimizer/schedule" if args.init else "random multimodal weights; pretrained frozen Pythia",
              "scope": "14 fixed real/identity-derived samples; capacity only, not generalization",
              "counts": dict(Counter(s["task"] for s in samples)), "data": bank_metadata(samples),
              "seen": dict.fromkeys([s["id"] for s in samples], 0), "history": [],
              "checkpoint": str(checkpoint.relative_to(ROOT)), "gallery": str(gallery.relative_to(ROOT)),
              "config_source_sha256": source_hash, "language": model.language_meta()}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline, pictures = evaluate(model, samples, args.device, decode=False)
    report["baseline"] = baseline
    write_gallery(gallery, pictures, baseline, 0)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for step in range(1, args.steps + 1):
        sample = t2i[(step - 1) % len(t2i)] if step <= args.warmup_steps else samples[(step - args.warmup_steps - 1) % len(samples)]
        report["seen"][sample["id"]] += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = collate_real_capacity(model.lm_tok, [sample], args.device)
        out = forward_real_capacity(model, batch, with_posterior=True)
        loss, meta = model.omni_loss(out, batch, torch.device(args.device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: step {step}, sample {sample['id']}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                      error_if_nonfinite=True)
        rate = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if any(p.grad is not None or p.requires_grad for p in model.lm.parameters()):
            raise RuntimeError("frozen language-model boundary violated")
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "task": sample["task"], "loss": float(loss.detach()),
                              "lr": rate, "seconds": time.perf_counter() - started}), flush=True)
        del out, loss
        if step % args.eval_every == 0 or step == args.steps:
            metrics, pictures = evaluate(model, samples, args.device)
            report["history"].append({"step": step, "metrics": metrics})
            payload = {"config": config, "state_dict": model.non_lm_state_dict(), "step": step,
                       "optimizer": optimizer.state_dict(), "training": vars(args),
                       "language": model.language_meta(), "data": report["data"]}
            torch.save(payload, checkpoint)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            write_gallery(gallery, pictures, metrics, step)
            print(json.dumps({"evaluated_step": step, "summary": metrics["summary"]}), flush=True)
    # Independent construction reloads the single trainable graph plus its
    # explicitly identified immutable frozen language dependency.
    del model, optimizer, pictures, payload
    if args.device.startswith("cuda"):
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.empty_cache()
    saved = torch.load(checkpoint, map_location="cpu")
    model = DualStreamOmni(**saved["config"]).to(args.device)
    load_graph_weights(model, saved["state_dict"])
    metrics, pictures = evaluate(model, samples, args.device)
    report.update(status="completed_budget_candidate_only", reloaded=metrics,
                  seconds=time.perf_counter() - started, checkpoint_sha256=sha256(checkpoint))
    if sha256(ROOT / args.config_from) != source_hash:
        raise RuntimeError("reference checkpoint changed")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_gallery(gallery, pictures, metrics, args.steps)
    print(json.dumps({"status": report["status"], "summary": metrics["summary"],
                      "seconds": report["seconds"], "peak_allocated_mib": report.get("peak_allocated_mib")}), flush=True)


if __name__ == "__main__":
    main()
