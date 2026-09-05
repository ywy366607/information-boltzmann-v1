#!/usr/bin/env python3
"""Train a candidate *single* 64px Slice champion under registered gates.

This is deliberately a gated continuation, not checkpoint averaging.  It
keeps the real-photo T2I visual graph as the base, imports only final,
read-only language likelihood adapters from the static token champion, and
alternates real T2I with the three ground-truth static boundaries.  A result
is a candidate unless every registered gate passes from the same saved state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import (
    capability_champion_kwargs,
    is_terminal_token_reader,
    is_token_interface,
    is_language_reader,
    is_generation_write,
    load_visual_champion,
    param_groups,
    set_optimization_phase,
)
from fine_grain.sharegpt4o_data import collate_real_multimodal
from scripts.train_pythia_capabilities import (
    current_gate,
    edit_gate,
    eval_current,
    eval_edit_static,
    eval_static_t2i,
    make_cycle_eval_bank,
    make_static_bank,
    sample_digit_group,
    static_one_step,
    t2i_gate,
    initialize_text_in_from_toy,
)
from scripts.train_pythia_tokens import (
    TOKEN_CASES,
    _forward_token_batch,
    answer_class_nll,
    collate_token_capabilities,
    counterfactual_token_samples,
    evaluate_graph_decode,
    evaluate_token_case,
    move_token_batch,
    sample_identified_group,
    token_gate,
)
from scripts.train_sharegpt4o_t2i_overfit import (
    edge_metrics,
    evaluate as evaluate_natural,
    file_sha256,
    finite_t2i_metrics,
    forward_t2i,
    image_gradients,
    pairwise_mse,
    save_gallery,
    select_t2i_records,
)


NATURAL_BASE = ROOT / "checkpoints" / "_natural_t2i_g8b_candidate.pt"
TOKEN_BASE = ROOT / "checkpoints" / "omni_d64_pythia_token_best.pt"
PROTECTED = {NATURAL_BASE.resolve(), TOKEN_BASE.resolve()}
DEFAULT_IDS = "freedom-t2i-34407,freedom-t2i-3191"


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        return raw["state_dict"]
    if isinstance(raw, dict) and all(torch.is_tensor(v) for v in raw.values()):
        return raw
    raise TypeError(f"unsupported checkpoint payload: {path}")


def copy_terminal_language_reader(model: DualStreamOmni, path: Path) -> dict:
    """Import only X-safe terminal token likelihood parameters.

    The source is a 16-Slice checkpoint; M-dependent routing is intentionally
    not copied.  In particular text_in and all non-terminal H blocks remain
    with the natural T2I graph because they can alter the visual evolution.
    """
    source, current = state_dict(path), model.state_dict()
    loaded, skipped = [], []
    for name, tensor in source.items():
        allow = (
            is_token_interface(name)
            or is_terminal_token_reader(model, name)
            or name.startswith("seg_head.")
        )
        if not allow:
            continue
        if name not in current or current[name].shape != tensor.shape:
            skipped.append(name)
            continue
        current[name] = tensor
        loaded.append(name)
    model.load_state_dict(current)
    return {"path": str(path), "loaded": loaded, "skipped": skipped}


def configure_trainables(
    model: DualStreamOmni, *, include_language_reader: bool = False,
) -> list[str]:
    """Open shared read/write and terminal likelihoods, never a private core."""
    # Establish normal generation-write semantics before adding only the two
    # shared terminal likelihoods needed by static segmentation/token ports.
    set_optimization_phase(model, "generation_write")
    names = []
    for name, parameter in model.named_parameters():
        allow = (
            is_generation_write(name)
            or (include_language_reader and is_language_reader(name))
            or is_token_interface(name)
            or is_terminal_token_reader(model, name)
            or name.startswith("seg_head.")
        )
        if name.startswith("lm.") or name.startswith(("embed.", "head.", "gaze_head.")):
            allow = False
        if "write_gamma_raw" in name:
            allow = False
        parameter.requires_grad_(allow)
        if allow:
            names.append(name)
    if any(parameter.requires_grad for parameter in model.lm.parameters()):
        raise RuntimeError("Pythia must stay frozen")
    return names


def fix_write_gamma(model: DualStreamOmni, gamma: float) -> None:
    if gamma <= 0:
        raise ValueError("the registered natural base requires fixed gamma > 0")
    for layer in model.mot_stack.layers:
        raw = layer.deslice.write_gamma_raw
        if raw is None:
            raise RuntimeError("construct with deslice_write_sharpening=True")
        raw.data.fill_(math.log(float(gamma)))
        raw.requires_grad_(False)


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def natural_loss(model, batch: dict, n_samples: int, args) -> tuple[torch.Tensor, dict]:
    out = forward_t2i(model, batch)
    prediction, target = out["rgb"], batch["target_rgb"]
    error = (prediction - target).pow(2)
    mse = error.mean()
    lv = out["rgb_lv"].clamp(-6.0, 3.0)
    nll = 0.5 * (lv + error * torch.exp(-lv)).mean()
    pdx, pdy = image_gradients(prediction)
    tdx, tdy = image_gradients(target)
    edge = F.mse_loss(pdx, tdx) + F.mse_loss(pdy, tdy)
    energy = pairwise_mse(prediction, target)
    retrieval = F.cross_entropy(
        -energy / float(args.retrieval_temperature),
        torch.arange(n_samples, device=prediction.device),
    )
    shuffled = prediction.roll(-1, dims=0)
    matched = error.flatten(1).mean(dim=1)
    shuffled_mse = (shuffled - target).pow(2).flatten(1).mean(dim=1)
    causal = torch.relu(matched - shuffled_mse + float(args.shuffle_margin)).mean()
    loss = (float(args.mse_coef) * mse + float(args.nll_coef) * nll
            + float(args.edge_coef) * edge + float(args.retrieval_coef) * retrieval
            + float(args.shuffle_coef) * causal)
    return loss, {"case": "natural_t2i", "mse": float(mse.detach()),
                  "nll": float(nll.detach()), "edge": float(edge.detach()),
                  "retrieval": float(retrieval.detach()), "causal": float(causal.detach())}


def token_loss(model, tokenizer, bank, case: str, rng, device, args):
    samples = sample_identified_group(bank, case, rng)
    if case == "text_to_both":
        batch = move_token_batch(collate_token_capabilities(tokenizer, samples), device)
        out = _forward_token_batch(model, batch, device)
        nll = out["token_nll"]
        loss = nll.mean() + float(args.token_class_coef) * answer_class_nll(
            out, batch, rows=range(len(samples)),
        )
        return loss, {"case": f"token_{case}", "nll": float(nll.mean().detach())}

    # Fused single forward pass for both samples and counterfactual control
    control = counterfactual_token_samples(samples, bank)
    n_s = len(samples)
    combined = samples + control
    batch = move_token_batch(collate_token_capabilities(tokenizer, combined), device)
    out = _forward_token_batch(model, batch, device)
    all_nll = out["token_nll"]
    nll = all_nll[:n_s]
    control_nll = all_nll[n_s:]

    loss = nll.mean() + float(args.token_class_coef) * answer_class_nll(
        out, batch, rows=range(n_s),
    )
    causal = torch.relu(nll - control_nll + float(args.token_shuffle_margin)).mean()
    loss = loss + float(args.token_shuffle_coef) * causal
    return loss, {"case": f"token_{case}", "nll": float(nll.mean().detach()), "causal": float(causal.detach())}


@torch.no_grad()
def assess(model, tokenizer, natural, device, batch: int, *, decode: bool) -> dict:
    natural_metrics, prediction = evaluate_natural(model, tokenizer, natural, device)
    static = {
        "t2i": eval_static_t2i(model, make_cycle_eval_bank(model.res, "text_to_both"), device, chunk=batch),
        "current": eval_current(model, make_cycle_eval_bank(model.res, "image_to_current"), device, chunk=batch),
        "edit": eval_edit_static(model, make_cycle_eval_bank(model.res, "image_text_edit"), device, chunk=batch),
    }
    static_gates = {"t2i": t2i_gate(static["t2i"]), "current": current_gate(static["current"]),
                    "edit": edit_gate(static["edit"])}
    token_train = {case: make_static_bank(model.res, case) for case in TOKEN_CASES}
    token_eval = {case: make_cycle_eval_bank(model.res, case) for case in TOKEN_CASES}
    token = {case: evaluate_token_case(model, tokenizer, token_eval[case], token_train[case], device, batch)
             for case in TOKEN_CASES}
    token_gates = {"t2t": token_gate(token["text_to_both"], require_image=False),
                   "i2t": token_gate(token["image_to_current"], require_image=True),
                   "it2t": token_gate(token["image_text_edit"], require_image=True)}
    graph = {}
    graph_gates = {}
    if decode:
        graph = {case: evaluate_graph_decode(model, tokenizer, token_eval[case], 30) for case in TOKEN_CASES}
        graph_gates = {case: row["exact"] >= .80 and row["all_steps_rerun_graph"]
                       for case, row in graph.items()}
    return {"natural": natural_metrics, "static": static, "static_gates": static_gates,
            "token": token, "token_gates": token_gates, "graph": graph,
            "graph_gates": graph_gates, "prediction": prediction}


def save_candidate(model, path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.non_lm_state_dict(), "language": model.language_meta(),
                "candidate_only": True, **record}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=r"D:\ml_cache\sharegpt4o\pilot_manifest.json")
    parser.add_argument("--ids", default=DEFAULT_IDS)
    parser.add_argument("--natural-base", default=str(NATURAL_BASE))
    parser.add_argument("--token-base", default=str(TOKEN_BASE))
    parser.add_argument("--candidate", default=str(ROOT / "checkpoints" / "_unified_u1_candidate.pt"))
    parser.add_argument("--out", default=str(ROOT / "results" / "published" / "unified_u1_candidate.json"))
    parser.add_argument("--gallery", default=str(ROOT / "present" / "figs" / "unified_u1_candidate.png"))
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n-slices", type=int, default=64)
    parser.add_argument("--write-gamma", type=float, default=8.0)
    parser.add_argument(
        "--lexical-ridge-init", action="store_true",
        help="Candidate-only: align Pythia text_in to the proven toy capability chart.",
    )
    parser.add_argument(
        "--lexical-source",
        default=str(ROOT / "checkpoints" / "northstar_slice_capability_best.pt"),
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--schedule", choices=("u1", "static_t2i", "static_t2i_language"), default="u1",
        help=(
            "u1 alternates every boundary; static_t2i is a candidate-only "
            "64px chart-closure control; static_t2i_language additionally "
            "opens the documented language-side MoT experts."
        ),
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--natural-batch", type=int, default=2)
    parser.add_argument("--interface-lr", type=float, default=1e-4)
    parser.add_argument("--visual-lr", type=float, default=1e-5)
    parser.add_argument("--mse-coef", type=float, default=4.0)
    parser.add_argument("--nll-coef", type=float, default=0.0)
    parser.add_argument("--edge-coef", type=float, default=8.0)
    parser.add_argument("--retrieval-coef", type=float, default=.5)
    parser.add_argument("--retrieval-temperature", type=float, default=.02)
    parser.add_argument("--shuffle-coef", type=float, default=1.0)
    parser.add_argument("--shuffle-margin", type=float, default=.02)
    parser.add_argument("--static-coef", type=float, default=1.0)
    parser.add_argument("--token-class-coef", type=float, default=1.0)
    parser.add_argument("--token-shuffle-coef", type=float, default=1.0)
    parser.add_argument("--token-shuffle-margin", type=float, default=.5)
    parser.add_argument("--use-lang-address", action="store_true", help="Enable prompt-conditioned spatial address prior in SliceRead")
    parser.add_argument("--lang-address-gain", type=float, default=0.5, help="Initial gain for language address prior")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lm-device", default=None)
    parser.add_argument("--smoke", action="store_true", help="one round-trip update; reports no admission metrics")
    parser.add_argument(
        "--smoke-task", default="natural_t2i",
        choices=("natural_t2i", "text_to_both", "image_to_current", "image_text_edit",
                 "token_text_to_both", "token_image_to_current", "token_image_text_edit"),
        help="which registered boundary to execute for --smoke",
    )
    args = parser.parse_args()

    natural_base, token_base, candidate = (Path(args.natural_base).resolve(), Path(args.token_base).resolve(),
                                           Path(args.candidate).resolve())
    if candidate in PROTECTED or candidate in {natural_base, token_base}:
        raise SystemExit("candidate must not overwrite either protected input")
    if int(args.resolution) != 64 or int(args.n_slices) != 64:
        raise SystemExit("U1 admission is registered for res=64, n_slices=64; make a new protocol for another scale")
    if not natural_base.is_file() or not token_base.is_file():
        raise FileNotFoundError("missing natural or token base checkpoint")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    natural = select_t2i_records(manifest, [x.strip() for x in args.ids.split(",") if x.strip()], args.resolution)
    if len(natural) != int(args.natural_batch):
        raise ValueError("U1 uses exactly its registered two-image natural bank")

    torch.manual_seed(20260904)
    rng = np.random.default_rng(20260904)
    device = torch.device(args.device)
    model = DualStreamOmni(**capability_champion_kwargs(
        res=args.resolution, n_slices=args.n_slices, language="pythia",
        lm_device=args.lm_device or args.device, pixel_loss_mode="gaussian_nll",
        s0_acc_coef=0.0, deslice_write_sharpening=True,
        use_lang_address=args.use_lang_address,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    natural_load = load_visual_champion(model, natural_base, skip_language_interface=False)
    token_overlay = copy_terminal_language_reader(model, token_base)
    lexical_init = (
        initialize_text_in_from_toy(model, args.lexical_source)
        if args.lexical_ridge_init else None
    )
    fix_write_gamma(model, args.write_gamma)
    trainable = configure_trainables(
        model, include_language_reader=args.schedule == "static_t2i_language",
    )
    optimizer = torch.optim.AdamW(param_groups(model, interface_lr=args.interface_lr,
                                                visual_lr=args.visual_lr), weight_decay=0.0)
    tokenizer = model.lm_tok
    natural_batch = move(collate_real_multimodal(tokenizer, natural), device)
    static_banks = {case: make_static_bank(model.res, case) for case in
                    ("text_to_both", "image_to_current", "image_text_edit")}
    token_banks = {case: make_static_bank(model.res, case) for case in TOKEN_CASES}
    sources = {"natural": {"path": str(natural_base), "sha256": file_sha256(natural_base)},
               "token": {"path": str(token_base), "sha256": file_sha256(token_base)}}
    history, started = [], time.time()
    cycle = ("natural", "text_to_both", "image_to_current", "image_text_edit",
             "token_text_to_both", "token_image_to_current", "token_image_text_edit")
    total = 1 if args.smoke else int(args.steps)
    for step in range(1, total + 1):
        task = (
            args.smoke_task.replace("natural_t2i", "natural")
            if args.smoke else (
                "text_to_both" if args.schedule in ("static_t2i", "static_t2i_language")
                else cycle[(step - 1) % len(cycle)]
            )
        )
        model.train(); optimizer.zero_grad(set_to_none=True)
        if task == "natural":
            loss, meta = natural_loss(model, natural_batch, len(natural), args)
        elif task.startswith("token_"):
            case = task[len("token_"):]
            loss, meta = token_loss(model, tokenizer, token_banks[case], case, rng, device, args)
        else:
            samples = (
                sample_digit_group(static_banks[task], rng)
                if task == "text_to_both"
                else [static_banks[task][int(rng.integers(len(static_banks[task])))]]
            )
            loss, meta = static_one_step(model, samples, static_banks[task], rng)
            loss = float(args.static_coef) * loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        meta.update({"step": step, "loss": float(loss.detach()), "elapsed_sec": time.time() - started})
        history.append(meta)
        print(f"step={step:04d} task={task} loss={meta['loss']:.4f}", flush=True)
        if not args.smoke and step % int(args.eval_every) == 0:
            # A cheap real-T2I check and recoverable candidate snapshot make a
            # long run diagnosable.  Static/token admission still happens only
            # in the final full evaluation below, from this same graph state.
            natural_progress, _ = evaluate_natural(model, tokenizer, natural, device)
            progress = {
                "schema": "unified-u1-progress-candidate",
                "admitted": False,
                "sources": sources,
                "step": step,
                "natural_progress": natural_progress,
                "history": history,
            }
            save_candidate(model, candidate, progress)
            print(
                f"progress step={step:04d} natural_gate={natural_progress['passed']} "
                f"edge_corr={natural_progress['edge_correlation']:.3f}",
                flush=True,
            )

    if args.smoke:
        record = {"schema": "unified-u1-smoke", "admitted": False, "sources": sources,
                  "natural_load": natural_load, "token_overlay": token_overlay,
                  "lexical_init": lexical_init,
                  "run": {"resolution": 64, "n_slices": 64, "write_gamma": args.write_gamma,
                          "pythia_frozen": True, "lm_generate_called": False,
                          "trainable_names": trainable}, "history": history}
    elif args.schedule in ("static_t2i", "static_t2i_language"):
        natural_metrics, prediction = evaluate_natural(model, tokenizer, natural, device)
        static = {
            "t2i": eval_static_t2i(model, make_cycle_eval_bank(model.res, "text_to_both"), device, chunk=args.batch),
            "current": eval_current(model, make_cycle_eval_bank(model.res, "image_to_current"), device, chunk=args.batch),
            "edit": eval_edit_static(model, make_cycle_eval_bank(model.res, "image_text_edit"), device, chunk=args.batch),
        }
        record = {
            "schema": "unified-u2a-static-chart-closure-control",
            "admitted": False,
            "candidate_only": True,
            "sources": sources,
            "natural_load": natural_load,
            "token_overlay": token_overlay,
            "lexical_init": lexical_init,
            "run": {"resolution": 64, "n_slices": 64, "write_gamma": args.write_gamma,
                    "schedule": args.schedule, "pythia_frozen": True,
                    "lm_generate_called": False, "trainable_names": trainable},
            "history": history,
            "natural": natural_metrics,
            "static": static,
            "static_gates": {"t2i": t2i_gate(static["t2i"]),
                             "current": current_gate(static["current"]),
                             "edit": edit_gate(static["edit"])},
        }
        save_gallery(natural, prediction, Path(args.gallery))
    else:
        report = assess(model, tokenizer, natural, device, args.batch, decode=True)
        all_gates = (bool(report["natural"]["passed"]) and all(report["static_gates"].values())
                     and all(report["token_gates"].values()) and all(report["graph_gates"].values()))
        report.pop("prediction")
        record = {"schema": "unified-u1-gated-single-graph", "admitted": bool(all_gates), "sources": sources,
                  "natural_load": natural_load, "token_overlay": token_overlay,
                  "lexical_init": lexical_init,
                  "run": {"resolution": 64, "n_slices": 64, "write_gamma": args.write_gamma,
                          "pythia_frozen": True, "lm_generate_called": False,
                          "phase": "generation_write + terminal likelihoods",
                          "trainable_names": trainable}, "history": history, "report": report}
        # The gallery is only an inspection aid; metrics above remain the gate.
        natural_metrics, prediction = evaluate_natural(model, tokenizer, natural, device)
        save_gallery(natural, prediction, Path(args.gallery))
        record["report"]["natural"] = natural_metrics
    save_candidate(model, candidate, record)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {args.out}; admitted={record['admitted']}", flush=True)


if __name__ == "__main__":
    main()
