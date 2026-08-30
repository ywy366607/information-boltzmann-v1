#!/usr/bin/env python3
"""Finite-bank natural-photo T2I content gate on the native Slice graph.

The source image is an all-zero field with ``image_precision=0``.  Frozen
Pythia provides prompt embeddings only; RGB is written by the existing
Slice--MoT--Deslice graph in one deterministic pass.  Prompt retrieval and a
rotated-prompt control prevent an unconditional average image from passing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import (
    capability_champion_kwargs,
    load_visual_champion,
    param_groups,
    set_optimization_phase,
)
from fine_grain.sharegpt4o_data import (
    collate_real_multimodal,
    counterfactual_real_samples,
    materialize_record,
)
from fine_grain.capability_tasks import collate_capability
from scripts.train_pythia_capabilities import (
    PROTECTED_CHECKPOINTS,
    make_cycle_eval_bank,
)
from scripts.train_sharegpt4o_pilot import static_audit


DEFAULT_IDS = (
    "freedom-t2i-6026",
    "freedom-t2i-43183",
    "freedom-t2i-34407",
    "freedom-t2i-3191",
)
DEFAULT_INIT = (
    ROOT / "checkpoints" /
    "omni_d64_pythia_sharegpt4o_i2t_ocr1px_joint_final.pt"
)
DEFAULT_CANDIDATE = (
    ROOT / "checkpoints" / "omni_d64_pythia_natural_t2i_overfit.pt"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def select_t2i_records(
    manifest: dict, ids: list[str] | tuple[str, ...], resolution: int,
) -> list[dict]:
    """Materialize an explicitly named, source-free T2I bank."""
    by_id = {
        str(row.get("id")): row
        for row in manifest.get("records", [])
        if row.get("task") == "t2i"
    }
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"T2I IDs missing from manifest: {missing}")
    rows = [materialize_record(by_id[sample_id], resolution) for sample_id in ids]
    if len(rows) < 2:
        raise ValueError("the causal T2I gate needs at least two prompts")
    for row in rows:
        if float(row["image_precision"]) != 0.0:
            raise RuntimeError("T2I input must have zero image precision")
        if bool(row.get("source_files")):
            raise RuntimeError("T2I capacity bank must not contain source images")
    return rows


def forward_t2i(model, batch: dict) -> dict:
    """Run the visual graph without invoking the frozen LM decoder."""
    return model.forward_tokens(
        batch["image"],
        batch["input_ids"],
        batch["attention_mask"],
        labels=None,
        visual_prompt_mask=batch["visual_prompt_mask"],
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
        score_tokens=False,
    )


def forward_static(model, batch: dict, device: torch.device) -> dict:
    """Run one established static boundary for capability distillation."""
    return model(
        batch["image"].to(device),
        list(batch["prompt"]),
        t=torch.zeros(len(batch["prompt"]), device=device),
        image_precision=batch["image_precision"].to(device),
        text_precision=batch["text_precision"].to(device),
        target_time=batch["target_time"].to(device),
    )


@torch.no_grad()
def cache_static_teacher(model, device: torch.device) -> dict:
    """Cache the input checkpoint's own RGB/seg terminal states."""
    model.eval()
    cached = {}
    for case in ("text_to_both", "image_to_current", "image_text_edit"):
        samples = make_cycle_eval_bank(model.res, case)
        rgb, seg = [], []
        for start in range(0, len(samples), 30):
            part = samples[start : start + 30]
            batch = collate_capability(part)
            if case == "image_to_current":
                batch["prompt"] = [""] * len(part)
                batch["text_precision"] = torch.zeros(len(part))
            out = forward_static(model, batch, device)
            rgb.append(out["rgb"].detach().cpu())
            seg.append(out["seg_logits"].detach().cpu())
        cached[case] = {
            "samples": samples,
            "rgb": torch.cat(rgb),
            "seg": torch.cat(seg),
        }
    return cached


def static_distillation_loss(
    model,
    teacher: dict,
    case: str,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Rehearse old capabilities without a second backbone or private head."""
    bank = teacher[case]
    part = [bank["samples"][int(index)] for index in indices]
    batch = collate_capability(part)
    if case == "image_to_current":
        batch["prompt"] = [""] * len(part)
        batch["text_precision"] = torch.zeros(len(part))
    out = forward_static(model, batch, device)
    target_rgb = bank["rgb"][indices].to(device)
    target_seg = bank["seg"][indices].to(device)
    return F.mse_loss(out["rgb"], target_rgb) + 0.1 * F.mse_loss(
        out["seg_logits"], target_seg,
    )


def pairwise_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred[:, None] - target[None, :]).pow(2).flatten(2).mean(dim=2)


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        image[:, :, :, 1:] - image[:, :, :, :-1],
        image[:, :, 1:] - image[:, :, :-1],
    )


def edge_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred_dx, pred_dy = image_gradients(pred)
    tgt_dx, tgt_dy = image_gradients(target)
    pred_edge = torch.cat([pred_dx.flatten(1), pred_dy.flatten(1)], dim=1)
    tgt_edge = torch.cat([tgt_dx.flatten(1), tgt_dy.flatten(1)], dim=1)
    relative_mse = (
        (pred_edge - tgt_edge).pow(2).mean(dim=1)
        / tgt_edge.pow(2).mean(dim=1).clamp_min(1e-8)
    )
    pred_centered = pred_edge - pred_edge.mean(dim=1, keepdim=True)
    tgt_centered = tgt_edge - tgt_edge.mean(dim=1, keepdim=True)
    correlation = (pred_centered * tgt_centered).sum(dim=1) / (
        pred_centered.norm(dim=1) * tgt_centered.norm(dim=1)
    ).clamp_min(1e-8)
    return float(relative_mse.mean()), float(correlation.mean())


def finite_t2i_metrics(
    pred: torch.Tensor, target: torch.Tensor, shuffled: torch.Tensor,
) -> dict:
    """Measure fidelity, prompt identity, and the rotated-prompt intervention."""
    energy = pairwise_mse(pred, target)
    diagonal = energy.diagonal()
    eye = torch.eye(len(pred), dtype=torch.bool, device=pred.device)
    nearest_wrong = energy.masked_fill(eye, float("inf")).min(dim=1).values
    shuffled_mse = (shuffled - target).pow(2).flatten(1).mean(dim=1)
    prompt_rms = (pred - shuffled).pow(2).mean().sqrt()
    mse = diagonal.mean()
    edge_relative_mse, edge_correlation = edge_metrics(pred, target)
    return {
        "n": len(pred),
        "rgb_mse": float(mse),
        "rgb_psnr": float(-10.0 * math.log10(max(float(mse), 1e-12))),
        "per_sample_psnr": [
            float(-10.0 * math.log10(max(float(value), 1e-12)))
            for value in diagonal
        ],
        "retrieval_top1": int((energy.argmin(dim=1) == torch.arange(
            len(pred), device=pred.device,
        )).sum()),
        "retrieval_margin": float((nearest_wrong - diagonal).mean()),
        "shuffled_rgb_mse": float(shuffled_mse.mean()),
        "shuffle_causal_gap": float((shuffled_mse - diagonal).mean()),
        "prompt_output_rms": float(prompt_rms),
        "edge_relative_mse": edge_relative_mse,
        "edge_correlation": edge_correlation,
    }


def natural_gate(metrics: dict) -> bool:
    """Strict finite-bank capacity gate; this is not a generalization claim."""
    return bool(
        metrics["retrieval_top1"] == metrics["n"]
        and metrics["rgb_psnr"] >= 20.0
        and metrics["shuffle_causal_gap"] >= 0.01
        and metrics["prompt_output_rms"] >= 0.05
        and metrics["edge_relative_mse"] <= 0.75
        and metrics["edge_correlation"] >= 0.50
    )


@torch.no_grad()
def evaluate(model, tokenizer, samples: list[dict], device) -> tuple[dict, torch.Tensor]:
    model.eval()
    batch = move_batch(collate_real_multimodal(tokenizer, samples), device)
    pred = forward_t2i(model, batch)["rgb"].clamp(0.0, 1.0)
    controls = counterfactual_real_samples(samples)
    control_batch = move_batch(collate_real_multimodal(tokenizer, controls), device)
    shuffled = forward_t2i(model, control_batch)["rgb"].clamp(0.0, 1.0)
    metrics = finite_t2i_metrics(pred, batch["target_rgb"], shuffled)
    metrics["passed"] = natural_gate(metrics)
    return metrics, pred.cpu()


def save_gallery(samples: list[dict], pred: torch.Tensor, path: Path) -> None:
    """Show original dataset targets above native-resolution predictions."""
    tile = 384
    header = 28
    canvas = Image.new(
        "RGB", (tile * len(samples), (tile + header) * 2), (16, 24, 39),
    )
    draw = ImageDraw.Draw(canvas)
    for column, (sample, output) in enumerate(zip(samples, pred)):
        x = column * tile
        draw.text(
            (x + 8, 7), "ShareGPT-4o target (original PNG)", fill=(226, 232, 240),
        )
        target_path = Path(str(sample["target_file"]))
        with Image.open(target_path) as source:
            target = ImageOps.contain(
                source.convert("RGB"), (tile, tile), Image.Resampling.LANCZOS,
            )
        target_panel = Image.new("RGB", (tile, tile), (0, 0, 0))
        target_panel.paste(
            target, ((tile - target.width) // 2, (tile - target.height) // 2),
        )
        canvas.paste(target_panel, (x, header))

        y = tile + header
        draw.text(
            (x + 8, y + 7),
            f"Slice generation ({pred.shape[-1]}x{pred.shape[-1]} native)",
            fill=(226, 232, 240),
        )
        array = (
            output.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255
        ).round().astype(np.uint8)
        generated = Image.fromarray(array).resize(
            (tile, tile), Image.Resampling.NEAREST,
        )
        canvas.paste(generated, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=r"D:\ml_cache\sharegpt4o\pilot_manifest.json",
    )
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS))
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument(
        "--out", default=str(
            ROOT / "results" / "published" /
            "sharegpt4o_natural_t2i_overfit.json"
        ),
    )
    parser.add_argument(
        "--gallery", default=str(
            ROOT / "present" / "figs" / "sharegpt4o_natural_t2i_overfit.png"
        ),
    )
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--n-slices", type=int, default=16)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--interface-lr", type=float, default=3e-4)
    parser.add_argument("--visual-lr", type=float, default=2e-4)
    parser.add_argument("--mse-coef", type=float, default=4.0)
    parser.add_argument("--nll-coef", type=float, default=1.0)
    parser.add_argument("--retrieval-coef", type=float, default=0.5)
    parser.add_argument("--retrieval-temperature", type=float, default=0.02)
    parser.add_argument("--shuffle-coef", type=float, default=1.0)
    parser.add_argument("--shuffle-margin", type=float, default=0.02)
    parser.add_argument("--edge-coef", type=float, default=4.0)
    parser.add_argument(
        "--phase", choices=("generation_write", "generation_capacity"),
        default="generation_write",
    )
    parser.add_argument("--rehearsal-coef", type=float, default=10.0)
    parser.add_argument("--rehearsal-batch", type=int, default=8)
    parser.add_argument(
        "--write-sharpening", action="store_true",
        help=(
            "Opt-in learned write-assignment temperature on Deslice "
            "(identity at init; the write counterpart of the read temp head)."
        ),
    )
    parser.add_argument(
        "--write-gamma", type=float, default=0.0,
        help=(
            "If >0, construct the sharpening parameter and FIX it at this "
            "gamma (frozen) for a prescribed-dose ablation; the scalar "
            "gradient is too weak for the optimizer to explore amplitude."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lm-device", default=None)
    parser.add_argument("--stop-on-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-static-audit", action="store_true")
    args = parser.parse_args()

    init_path = Path(args.init).resolve()
    candidate_path = Path(args.candidate).resolve()
    protected = {Path(path).resolve() for path in PROTECTED_CHECKPOINTS}
    if candidate_path == init_path or candidate_path in protected:
        raise SystemExit("refusing to overwrite a protected or input checkpoint")
    if not init_path.is_file():
        raise FileNotFoundError(init_path)
    init_hash = file_sha256(init_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    samples = select_t2i_records(manifest, ids, int(args.resolution))

    device = torch.device(args.device)
    lm_device = args.lm_device or args.device
    torch.manual_seed(20260825)
    model = DualStreamOmni(**capability_champion_kwargs(
        res=int(args.resolution), n_slices=int(args.n_slices),
        language="pythia", lm_device=lm_device,
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0,
        deslice_write_sharpening=bool(
            args.write_sharpening or float(args.write_gamma) > 0.0
        ),
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    load_report = load_visual_champion(
        model, init_path, skip_language_interface=False,
    )
    trainable = set_optimization_phase(model, args.phase)
    fixed_gamma = float(args.write_gamma)
    if fixed_gamma > 0.0:
        # Freeze after the phase pass so the prescribed dose is not re-opened.
        for layer in model.mot_stack.layers:
            raw = layer.deslice.write_gamma_raw
            if raw is not None:
                raw.data.fill_(float(torch.log(torch.tensor(fixed_gamma))))
                raw.requires_grad_(False)
        trainable = [
            name for name in trainable if "write_gamma_raw" not in name
        ]
    if any(parameter.requires_grad for parameter in model.lm.parameters()):
        raise RuntimeError("Pythia must remain frozen")
    optimizer = torch.optim.AdamW(
        param_groups(
            model,
            interface_lr=float(args.interface_lr),
            visual_lr=float(args.visual_lr),
        ),
        weight_decay=0.0,
    )
    tokenizer = model.lm_tok
    batch = move_batch(collate_real_multimodal(tokenizer, samples), device)
    teacher = cache_static_teacher(model, device)
    rehearsal_rng = np.random.default_rng(20260825)
    rehearsal_cases = tuple(teacher)

    before, _ = evaluate(model, tokenizer, samples, device)
    history = [{**before, "step": 0}]
    best_score = None
    best_state = None
    best_metrics = None
    best_step = 0
    started = time.time()
    completed_step = 0
    for step in range(1, int(args.steps) + 1):
        completed_step = step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = forward_t2i(model, batch)
        pred = out["rgb"]
        target = batch["target_rgb"]
        lv = out["rgb_lv"].clamp(-6.0, 3.0)
        error = (pred - target).pow(2)
        nll = 0.5 * (lv + error * torch.exp(-lv)).mean()
        mse = error.mean()
        pred_dx, pred_dy = image_gradients(pred)
        tgt_dx, tgt_dy = image_gradients(target)
        edge_loss = F.mse_loss(pred_dx, tgt_dx) + F.mse_loss(pred_dy, tgt_dy)
        energy = pairwise_mse(pred, target)
        labels = torch.arange(len(samples), device=device)
        retrieval = F.cross_entropy(
            -energy / float(args.retrieval_temperature), labels,
        )
        # Every source field is identical and missing; rotating the full-bank
        # outputs is exactly the rotated-prompt intervention, without a second
        # graph pass during optimization.  Evaluation still runs the explicit
        # counterfactual batch independently.
        shuffled = pred.roll(shifts=-1, dims=0)
        matched_mse = error.flatten(1).mean(dim=1)
        shuffled_mse = (shuffled - target).pow(2).flatten(1).mean(dim=1)
        causal = torch.relu(
            matched_mse - shuffled_mse + float(args.shuffle_margin),
        ).mean()
        rehearsal_case = rehearsal_cases[(step - 1) % len(rehearsal_cases)]
        if float(args.rehearsal_coef) != 0.0:
            rehearsal_bank = teacher[rehearsal_case]["samples"]
            rehearsal_indices = rehearsal_rng.choice(
                len(rehearsal_bank), size=int(args.rehearsal_batch), replace=False,
            )
            rehearsal = static_distillation_loss(
                model, teacher, rehearsal_case, rehearsal_indices, device,
            )
        else:
            rehearsal = pred.new_zeros(())
        loss = (
            float(args.nll_coef) * nll
            + float(args.mse_coef) * mse
            + float(args.retrieval_coef) * retrieval
            + float(args.shuffle_coef) * causal
            + float(args.edge_coef) * edge_loss
            + float(args.rehearsal_coef) * rehearsal
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            metrics, _ = evaluate(model, tokenizer, samples, device)
            metrics.update({
                "step": step,
                "train_loss": float(loss.detach()),
                "gaussian_nll": float(nll.detach()),
                "retrieval_loss": float(retrieval.detach()),
                "causal_hinge": float(causal.detach()),
                "edge_loss": float(edge_loss.detach()),
                "rehearsal_loss": float(rehearsal.detach()),
                "rehearsal_case": rehearsal_case,
            })
            history.append(metrics)
            score = (
                int(metrics["passed"]),
                int(metrics["retrieval_top1"]),
                float(metrics["rgb_psnr"]),
                float(metrics["shuffle_causal_gap"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_metrics = copy.deepcopy(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.non_lm_state_dict().items()
                }
            print(
                f"step={step:4d} psnr={metrics['rgb_psnr']:.2f} "
                f"retrieve={metrics['retrieval_top1']}/{len(samples)} "
                f"shuffle_gap={metrics['shuffle_causal_gap']:.4f} "
                f"prompt_rms={metrics['prompt_output_rms']:.4f} "
                f"pass={metrics['passed']}",
                flush=True,
            )
            if bool(args.stop_on_pass) and metrics["passed"]:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no evaluated candidate")
    model.load_state_dict(best_state, strict=False)
    final_metrics, final_pred = evaluate(model, tokenizer, samples, device)
    static = None if args.skip_static_audit else static_audit(
        model, device, batch_size=30,
    )
    natural_passed = bool(final_metrics["passed"])
    static_passed = bool(
        static is not None and all(static.get("gates", {}).values())
    )
    if file_sha256(init_path) != init_hash:
        raise RuntimeError("protected input checkpoint changed during training")
    record = {
        "schema": "sharegpt4o-natural-t2i-overfit-v2",
        "candidate_only": True,
        "admitted": bool(natural_passed and static_passed),
        "natural_capacity_gate_passed": natural_passed,
        "static_capability_gates_preserved": static_passed,
        "claim_scope": "finite-bank natural-image T2I content capacity only",
        "gate": {
            "rgb_psnr_min": 20.0,
            "retrieval": f"{len(samples)}/{len(samples)}",
            "shuffle_causal_gap_min": 0.01,
            "prompt_output_rms_min": 0.05,
            "edge_relative_mse_max": 0.75,
            "edge_correlation_min": 0.50,
        },
        "boundary": {
            "source_image": "all-zero field",
            "image_precision": 0.0,
            "text_precision": 1.0,
            "single_native_graph_pass": True,
            "lm_generate_called": False,
        },
        "dataset": "FreedomIntelligence/ShareGPT-4o-Image",
        "ids": ids,
        "prompts": [str(sample["prompt"]) for sample in samples],
        "resolution": int(args.resolution),
        "n_slices": int(args.n_slices),
        "phase": str(args.phase),
        "write_sharpening": bool(args.write_sharpening),
        "write_gamma_fixed": float(args.write_gamma),
        "pythia_frozen": True,
        "init": str(init_path),
        "init_sha256": init_hash,
        "candidate": str(candidate_path),
        "load_report": load_report,
        "trainable_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_names": trainable,
        "rehearsal": {
            "coefficient": float(args.rehearsal_coef),
            "batch": int(args.rehearsal_batch),
            "boundaries": list(rehearsal_cases),
            "teacher": "input checkpoint terminal RGB/seg states",
        },
        "steps": int(completed_step),
        "selected_step": int(best_step),
        "before": before,
        "after": final_metrics,
        "static_after": static,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.non_lm_state_dict(),
        "language": model.language_meta(),
        "candidate_only": True,
        "natural_t2i": {
            "schema": record["schema"],
            "ids": ids,
            "resolution": int(args.resolution),
            "selected_step": int(best_step),
            "metrics": final_metrics,
        },
    }, candidate_path)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    save_gallery(samples, final_pred, Path(args.gallery))
    print(
        f"wrote {output}; gallery={args.gallery}; "
        f"natural_passed={natural_passed}; admitted={record['admitted']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
