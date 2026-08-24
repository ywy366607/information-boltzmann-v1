#!/usr/bin/env python3
"""Smoke-test five I/O ports on one Champion-B DualStream field loop.

  t2t / it2t / recon / t2i / i2i

python scripts/train_omni_probe.py --steps 800 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
import math

from fine_grain.flow_match import interpolate, ode_integrate, sample_t, velocity_target
from fine_grain.gen_metrics import (
    background_flood_rate,
    gen_free_scores,
    ink_centroid_error,
    paired_ink_iou,
)
from fine_grain.omni_tasks import (
    GRID_PLACES,
    TASKS,
    apply_t2i_gray_hint,
    make_omni_batch,
    one_sample,
)


def masked_psnr(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.unsqueeze(1).expand_as(pred)
    if float(mask.sum()) < 1:
        return 0.0
    mse = ((pred - tgt).pow(2) * m).sum() / m.sum().clamp_min(1.0)
    return float(-10.0 * torch.log10(mse.clamp_min(1e-8)))


def ink_color_match(pred: torch.Tensor, tgt: torch.Tensor, stroke: torch.Tensor) -> float:
    m = stroke.unsqueeze(1)
    w = m.sum(dim=(2, 3)).clamp_min(1.0)
    p = (pred * m).sum(dim=(2, 3)) / w
    t = (tgt * m).sum(dim=(2, 3)) / w
    return float((p.argmax(dim=-1) == t.argmax(dim=-1)).float().mean())


def flood_rate(pred: torch.Tensor, tgt: torch.Tensor, stroke: torch.Tensor) -> float:
    return background_flood_rate(pred, tgt, stroke)


def psnr(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    mse = (pred - tgt).pow(2).mean().clamp_min(1e-8)
    return float(-10.0 * torch.log10(mse))


def to_signed(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_unit(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5


@torch.no_grad()
def fm_generate(
    model: DualStreamOmni, x0: torch.Tensor, prompts, need_pix, n_steps: int,
    method: str = "heun",
    clamp: bool = True,
    cfg: float = 1.0,
) -> torch.Tensor:
    null = [""] * len(prompts)

    def step_fn(x, t):
        v_c = model(x, prompts, need_pix=need_pix, t=t)["v"]
        if cfg is None or abs(float(cfg) - 1.0) < 1e-6:
            return v_c
        v_u = model(x, null, need_pix=need_pix, t=t)["v"]
        return v_u + float(cfg) * (v_c - v_u)

    return ode_integrate(step_fn, x0, n_steps=n_steps, method=method, clamp=clamp)


def pixel_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).reshape(a.shape[0], -1).float().pow(2).mean(1).sqrt()


def f_iterate(step_fn, x: torch.Tensor, n_steps: int, halt_eps: float = 0.0):
    """Generic fixed-point probe retained for historical diagnostics.

    This helper does not establish that ``step_fn`` descends free energy.
    """
    n = max(1, int(n_steps))
    B = x.shape[0]
    alive = torch.ones(B, dtype=torch.bool, device=x.device)
    n_used = torch.zeros(B, device=x.device, dtype=torch.float32)
    eps = float(halt_eps)
    lead = (B,) + (1,) * (x.ndim - 1)
    for _ in range(n):
        if not bool(alive.any()):
            break
        xc = step_fn(x)
        rms = pixel_rms(xc, x)
        x = torch.where(alive.view(lead), xc, x)
        n_used = n_used + alive.float()
        if eps > 0.0:
            alive = alive & (rms > eps)
    return x, n_used


@torch.no_grad()
def f_generate(
    model: DualStreamOmni, x0: torch.Tensor, prompts, need_pix, n_steps: int,
    cfg: float = 1.0,
    halt_eps: float = 0.0,
    return_steps: bool = False,
):
    """Generate through one native full-resolution X-Slice-H field pass.

    The model's layers are the inference trajectory. The input is encoded once,
    X remains live across those layers, and RGB is decoded only at the end.
    ``n_steps`` and ``halt_eps`` remain for old CLI/checkpoint compatibility;
    they must not create an outer RGB→stem→RGB loop. A strong F-descent claim
    additionally requires a shared cross-step F evaluator (see NORTH_STAR.md).
    """
    B = x0.shape[0]
    null = [""] * len(prompts)
    # In the registered bridge, t=0 is the source/null boundary and t=1 data.
    zeros = torch.zeros(B, device=x0.device, dtype=x0.dtype)
    use_cfg = cfg is not None and abs(float(cfg) - 1.0) > 1e-6
    x = model(x0, prompts, pi_x=1.0, need_pix=need_pix, t=zeros)["x_pred"]
    if use_cfg:
        xu = model(x0, null, pi_x=1.0, need_pix=need_pix, t=zeros)["x_pred"]
        x = xu + float(cfg) * (x - xu)
    n_used = torch.ones(B, device=x0.device, dtype=torch.float32)
    if return_steps:
        return x, n_used
    return x


@torch.no_grad()
def s0_prompt_cosine(model: DualStreamOmni, device) -> float | None:
    """Kill criterion: S0 of two prompts must not be the same field."""
    if not getattr(model.mot_stack, "use_residual_read", False):
        return None
    layer = model.mot_stack.layers[0]
    if getattr(layer, "lang_s0", None) is None:
        return None
    model.eval()
    x = torch.full((2, 3, model.res, model.res), -1.0 if model.fm_signed else 0.0, device=device)
    prompts = [
        "Draw digit 7 with a thin green stroke blank image",
        "Draw digit 1 with a thin red stroke blank image",
    ]
    t = torch.ones(2, device=device)
    model(x, prompts, need_pix=[True, True], t=t)
    s0 = layer.last_s0
    if s0 is None or s0.shape[0] < 2:
        return None
    a, b = s0[0].reshape(1, -1), s0[1].reshape(1, -1)
    return float(F.cosine_similarity(a, b).item())


def eval_score(ev: dict, mix) -> float:
    """Select T2I by identity and requested address; PSNR only breaks ties."""
    scores = []
    for k in mix:
        rec = ev[k]
        if k in ("t2t", "i2t", "it2t"):
            scores.append(float(rec.get("acc", 0.0)))
        elif k == "t2i":
            scores.append(
                float(rec.get("digit_top1", 0.0))
                + float(rec.get("paired_iou", 0.0))
                + 0.001 * float(rec.get("psnr", 0.0))
            )
        else:
            scores.append(
                float(rec.get("digit_top1", 0.0))
                + 0.001 * float(rec.get("psnr", 0.0))
            )
    return float(sum(scores) / max(1, len(scores)))


PIX_KEYS = (
    "psnr", "stroke_psnr", "bg_psnr", "ink", "flood",
    "color_acc", "digit_iou", "digit_top1", "ink_frac",
    "paired_iou", "centroid_error", "f_steps",
)


@torch.no_grad()
def eval_ports(
    model: DualStreamOmni, rng, res, device, n=48, kinds=None,
    flow_steps: int = 0, flow_method: str = "heun",
    fm_x0: str = "pair",
    t2i_canvas: str = "paper",
    fm_signed: bool = False,
    cfg: float = 1.0,
    t2i_stroke_px: int = 1,
    t2i_place: str = "random",
    f_steps: int = 0,
    f_halt_eps: float = 0.0,
    t2i_digit=None,
    t2i_color: str | None = None,
) -> dict:
    model.eval()
    kinds = list(kinds or TASKS)
    stats = {
        k: {
            "n": 0, "acc": 0.0, "psnr": 0.0, "stroke_psnr": 0.0,
            "bg_psnr": 0.0, "ink": 0.0, "flood": 0.0,
            "color_acc": 0.0, "digit_iou": 0.0, "digit_top1": 0.0,
            "ink_frac": 0.0, "paired_iou": 0.0,
            "centroid_error": 0.0, "f_steps": 0.0,
        }
        for k in kinds
    }
    for kind in kinds:
        for _ in range(n):
            s = one_sample(
                rng, res, kind, t2i_canvas=t2i_canvas,
                t2i_stroke_px=t2i_stroke_px, t2i_place=t2i_place,
                t2i_digit=t2i_digit, t2i_color=t2i_color,
            )
            img = s["image"].to(device)
            if (flow_steps > 0 or f_steps > 0) and s["need_pix"]:
                if fm_x0 == "noise":
                    x0 = torch.randn_like(img)
                    clamp = False
                elif kind == "recon":
                    x0 = torch.rand_like(img)
                    clamp = not fm_signed
                    if fm_signed:
                        x0 = to_signed(x0)
                else:
                    x0 = to_signed(img) if fm_signed else img
                    clamp = not fm_signed
                if f_steps > 0:
                    z, n_used = f_generate(
                        model, x0, [s["prompt"]], [s["need_pix"]], f_steps, cfg=cfg,
                        halt_eps=f_halt_eps, return_steps=True,
                    )
                    stats[kind]["f_steps"] += float(n_used.mean())
                else:
                    z = fm_generate(
                        model, x0, [s["prompt"]], [s["need_pix"]], flow_steps,
                        method=flow_method, clamp=clamp, cfg=cfg,
                    )
                rgb = to_unit(z).clamp(0, 1) if fm_signed else z.clamp(0, 1)
            else:
                out = model(img, [s["prompt"]], need_pix=[s["need_pix"]])
                rgb = out["rgb"].clamp(0, 1)
            rec = stats[kind]
            rec["n"] += 1
            if s["need_text"]:
                if flow_steps > 0:
                    out = model(img, [s["prompt"]], need_pix=[s["need_pix"]])
                pred = out["logits"][0].argmax().item()
                gold = model.ans_to_idx.get(s["answer"], -1)
                rec["acc"] += float(pred == gold)
            if s["need_pix"]:
                tgt = s["target_rgb"].to(device)
                st = s["stroke"].to(device)
                rec["psnr"] += psnr(rgb, tgt)
                rec["stroke_psnr"] += masked_psnr(rgb, tgt, st)
                rec["bg_psnr"] += masked_psnr(rgb, tgt, 1.0 - st)
                rec["ink"] += ink_color_match(rgb, tgt, st)
                rec["flood"] += flood_rate(rgb, tgt, st)
                free = gen_free_scores(
                    rgb, s.get("digit", s["answer"]), s.get("color", "red"),
                )
                rec["color_acc"] += free["color_acc"]
                rec["digit_iou"] += free["digit_iou"]
                rec["digit_top1"] += free["digit_top1"]
                rec["ink_frac"] += free["ink_frac"]
                rec["paired_iou"] += paired_ink_iou(
                    rgb, st, s.get("color", "red"),
                )
                rec["centroid_error"] += ink_centroid_error(
                    rgb, st, s.get("color", "red"),
                )
    for k, rec in stats.items():
        n = max(1, rec["n"])
        for key in ("acc",) + PIX_KEYS:
            rec[key] /= n
    return stats


def render_gallery(
    model, rng, res, device, path: Path, kinds=None,
    flow_steps: int = 0, flow_method: str = "heun",
    fm_x0: str = "pair",
    t2i_canvas: str = "paper",
    fm_signed: bool = False,
    cfg: float = 1.0,
    t2i_stroke_px: int = 1,
    t2i_place: str = "random",
    f_steps: int = 0,
    f_halt_eps: float = 0.0,
    t2i_digit=None,
    t2i_color: str | None = None,
) -> None:
    model.eval()
    kinds = list(kinds or ["i2t", "it2t", "recon", "i2i", "t2i"])
    fig, axes = plt.subplots(len(kinds), 3, figsize=(8.6, 2.35 * len(kinds)), dpi=140)
    if len(kinds) == 1:
        axes = np.expand_dims(axes, 0)
    fig.patch.set_facecolor("#0b1120")
    trajectory = "field start" if f_steps > 0 else "ODE start"
    if fm_x0 == "noise":
        titles = [f"noise ({trajectory})", "target", "prediction"]
    elif t2i_canvas == "black":
        titles = [f"black ({trajectory})", "target", "prediction"]
    else:
        titles = [f"paper ({trajectory})", "target", "prediction"]
    for r, kind in enumerate(kinds):
        s = one_sample(
            rng, res, kind, t2i_canvas=t2i_canvas,
            t2i_stroke_px=t2i_stroke_px, t2i_place=t2i_place,
            t2i_digit=t2i_digit, t2i_color=t2i_color,
        )
        with torch.no_grad():
            img = s["image"].to(device)
            x0_vis = img
            if (flow_steps > 0 or f_steps > 0) and s["need_pix"]:
                if fm_x0 == "noise":
                    x0 = torch.randn_like(img)
                    clamp = False
                    x0_vis = (0.25 * x0 + 0.5).clamp(0, 1)
                elif kind == "recon":
                    x0 = torch.rand_like(img)
                    clamp = not fm_signed
                    x0_vis = x0
                    if fm_signed:
                        x0 = to_signed(x0)
                else:
                    x0 = to_signed(img) if fm_signed else img
                    clamp = not fm_signed
                    x0_vis = img
                if f_steps > 0:
                    z = f_generate(
                        model, x0, [s["prompt"]], [s["need_pix"]], f_steps, cfg=cfg,
                        halt_eps=f_halt_eps,
                    )
                else:
                    z = fm_generate(
                        model, x0, [s["prompt"]], [s["need_pix"]], flow_steps,
                        method=flow_method, clamp=clamp, cfg=cfg,
                    )
                rgb = to_unit(z).clamp(0, 1) if fm_signed else z.clamp(0, 1)
            else:
                rgb = model(img, [s["prompt"]], need_pix=[s["need_pix"]])["rgb"]
            pred = rgb[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        inp = x0_vis[0].detach().cpu().permute(1, 2, 0).numpy()
        tgt = s["target_rgb"][0].permute(1, 2, 0).numpy()
        for c, im in enumerate((inp, tgt, pred)):
            axes[r, c].imshow(np.clip(im, 0, 1))
            axes[r, c].axis("off")
            axes[r, c].set_facecolor("#0f172a")
            if r == 0:
                axes[r, c].set_title(titles[c], color="#e2e8f0", fontsize=10)
        axes[r, 0].set_ylabel(kind, color="#fbbf24", fontsize=10, fontweight="bold")
        axes[r, 0].text(
            0.0, -0.08, s["prompt"][:48], transform=axes[r, 0].transAxes,
            color="#94a3b8", fontsize=7.5, ha="left", va="top",
        )
    mode = f"{fm_x0}/{t2i_canvas}" if any(k == "t2i" for k in kinds) else "ports"
    solver = "native-field" if f_steps > 0 else f"{flow_method}{flow_steps or ''}"
    fig.suptitle(
        f"DualStream  d={model.d_model}  ·  {mode}  ·  {solver}",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"  saved {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="results/published/omni_scaled_2000step_table.json")
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--mix", type=str, default="t2t,i2t,it2t,recon,i2i,t2i")
    ap.add_argument(
        "--lr", type=float, default=None,
        help="Default: 1e-3 for core/active_f2 generation, otherwise 2e-4.",
    )
    ap.add_argument(
        "--lr-schedule", choices=["auto", "constant", "cosine"], default="auto",
        help="auto uses constant for generation core and cosine otherwise.",
    )
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--vfe-coef", type=float, default=0.1,
                    help="λ E[gap] on post_head. Canonical unified default=0.1; use 0 for history.")
    ap.add_argument("--gate-on", type=str, default="u", choices=["u", "gap", "f"])
    ap.add_argument("--deslice-write", type=str, default="increment",
                    choices=["absolute", "increment", "workspace"])
    ap.add_argument("--gate-h-local", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--s-lang-topk", type=int, default=0,
        help="Only top-k slices (by visual→text attn) take MoT Δ. 0 = all (default).",
    )
    ap.add_argument(
        "--flow-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="OT flow matching on pixel ports. Same graph; t conditions X.",
    )
    ap.add_argument("--flow-steps", type=int, default=8, help="ODE steps at eval.")
    ap.add_argument("--flow-method", type=str, default="heun", choices=["heun", "euler"])
    ap.add_argument(
        "--fm-x0", type=str, default="pair", choices=["pair", "noise"],
        help="pair=start from input image/paper; noise=JiT Gaussian start.",
    )
    ap.add_argument("--flow-t", type=str, default="uniform",
                    choices=["uniform", "logit_normal", "jit"])
    ap.add_argument(
        "--fm-signed", action=argparse.BooleanOptionalAction, default=False,
        help="JiT chart: x in [-1,1], linear x-pred, no sigmoid.",
    )
    ap.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale.")
    ap.add_argument("--cfg-drop", type=float, default=0.1, help="Train label-drop prob.")
    ap.add_argument(
        "--fm-pred", type=str, default="x", choices=["x", "v"],
        help="JiT x-prediction (clean image) vs v-prediction (velocity).",
    )
    ap.add_argument(
        "--t2i-hint-frac", type=float, default=0.0,
        help="Train-only: paint a gray digit on t2i paper with this probability.",
    )
    ap.add_argument(
        "--t2i-canvas", type=str, default="paper", choices=["paper", "black"],
        help="paper=product edit (ODE from this haystack). black=from-noise canvas.",
    )
    ap.add_argument(
        "--t2i-stroke-px", type=int, default=1,
        help="Dilate the target stroke (1=Bresenham 1px, 5=thick MNIST-like).",
    )
    ap.add_argument(
        "--t2i-place", type=str, default="random",
        choices=["random", "center", "grid", *GRID_PLACES],
        help="grid=prompt-controlled nine-grid address; center keeps the legacy fixed box.",
    )
    ap.add_argument("--t2i-digit", type=int, default=None, help="Lock t2i to one digit (capacity probe).")
    ap.add_argument("--t2i-color", type=str, default="", help="Lock t2i to one color.")
    ap.add_argument(
        "--prior-write", type=float, default=None,
        help="F-action Deslice(μp−S). Default: 1 for active_f2 generation, else 0.",
    )
    ap.add_argument(
        "--f-gen", action="store_true",
        help="Native persistent-field generation: one stack trajectory, one terminal decode.",
    )
    ap.add_argument(
        "--gen-recipe", choices=["core", "active_f2", "vfe"], default="active_f2",
        help=(
            "core=validated capacity recipe; active_f2=core plus F2 beliefs, "
            "clean language prior and prior-error action; vfe=historical bundle."
        ),
    )
    ap.add_argument(
        "--f-steps", type=int, default=4,
        help="Legacy compatibility only; native generation uses one X-Slice-H stack pass.",
    )
    ap.add_argument(
        "--f-halt-eps", type=float, default=0.0,
        help="Per-sample halt when RMS(Δx)≤eps. 0=always Tmax. Signed chart ~0.03.",
    )
    ap.add_argument(
        "--f-bptt", action=argparse.BooleanOptionalAction, default=False,
        help="Train action by BPTT through K writes. Off: teacher-forced data bridge.",
    )
    ap.add_argument(
        "--null-slice", action=argparse.BooleanOptionalAction, default=False,
        help="SliceRead softmax over M+1; last dim is ∅. Off = old row-softmax.",
    )
    ap.add_argument(
        "--sigreg-coef", type=float, default=0.1,
        help="LeJEPA SIGReg on S_y. 0 = off (single-class probes must be 0).",
    )
    ap.add_argument(
        "--pack-surprise", action=argparse.BooleanOptionalAction, default=False,
        help="BLT Read: low surprise packs into slice 0, high surprise keeps detail slices.",
    )
    ap.add_argument(
        "--hard-admit", action=argparse.BooleanOptionalAction, default=False,
        help="Write-side yield on Bayes U. Off = Read yield is the sparse mechanism.",
    )
    ap.add_argument(
        "--yield-read", action=argparse.BooleanOptionalAction, default=False,
        help="Per-head ReLU(w−τ_h) on SliceRead. Off: use ticket-read instead.",
    )
    ap.add_argument(
        "--ticket-read", action=argparse.BooleanOptionalAction, default=False,
        help="Read tickets = ||X − f(H,xy)||². Off: Read stays softmax. Yield is on write.",
    )
    ap.add_argument(
        "--write-yield", action=argparse.BooleanOptionalAction, default=False,
        help="Deslice Δ: u=sign(Δ)⊙ReLU(|Δ|−τ). Off: write follows residual-read w.",
    )
    ap.add_argument(
        "--write-alpha", type=float, default=1.0,
        help="Field leak X←αX+πu. 1=keep canvas (decoupled from τ).",
    )
    ap.add_argument(
        "--residual-read", action=argparse.BooleanOptionalAction, default=True,
        help="SliceRead(X−S0) with ℓ_∅=τ−γ e. Garbage (explained) is not read.",
    )
    args = ap.parse_args()

    if args.f_gen and args.f_bptt:
        raise ValueError(
            "--f-bptt trained the removed RGB round-trip rollout; native field "
            "generation keeps X inside one stack pass (docs/NORTH_STAR.md)."
        )

    dev = torch.device(args.device)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    rng_val = np.random.default_rng(9001)

    core_gen = bool(args.f_gen and args.gen_recipe == "core")
    active_f2_gen = bool(args.f_gen and args.gen_recipe == "active_f2")
    simple_gen = core_gen or active_f2_gen
    effective_prior_write = float(
        (1.0 if active_f2_gen else 0.0)
        if args.prior_write is None else args.prior_write
    )
    spatial_prompt_vocab = args.t2i_place == "grid" or args.t2i_place in GRID_PLACES
    model = DualStreamOmni(
        d_model=args.d_model, n_slices=args.n_slices, n_layers=4, res=args.res,
        n_heads=args.n_heads,
        surprise_mode="baseline" if core_gen else "v1_bayes",
        s_update="raw" if simple_gen else "rms_dir",
        prior_loss_coef=0.1 if active_f2_gen else (0.0 if core_gen else 0.1),
        sigreg_coef=0.0 if simple_gen else args.sigreg_coef,
        use_stiefel=not simple_gen,
        deslice_topk=0 if simple_gen else 2,
        use_null_slice=False if simple_gen else (args.null_slice or args.residual_read),
        pack_by_surprise=args.pack_surprise,
        hard_admit=args.hard_admit,
        use_yield_read=args.yield_read,
        use_ticket_read=args.ticket_read,
        use_write_yield=args.write_yield,
        write_alpha=args.write_alpha,
        use_residual_read=False if simple_gen else args.residual_read,
        gate_on=args.gate_on,
        deslice_write=args.deslice_write,
        gate_h_local=args.gate_h_local,
        vfe_coef=args.vfe_coef if active_f2_gen else (0.0 if core_gen else args.vfe_coef),
        s_lang_topk=args.s_lang_topk,
        fm_pred=args.fm_pred,
        fm_signed=args.fm_signed,
        prior_write=effective_prior_write,
        prior_write_by_t=not args.f_gen,
        pixel_loss_mode="balanced_bce" if simple_gen else "vfe",
        spatial_prompt_vocab=spatial_prompt_vocab,
    ).to(dev)
    if args.init:
        raw = torch.load(args.init, map_location="cpu")
        missing = model.load_state_dict(raw, strict=False)
        print(f"  loaded {args.init} missing={len(missing.missing_keys)}", flush=True)
    effective_lr = float(args.lr if args.lr is not None else (1e-3 if simple_gen else 2e-4))
    lr_schedule = (
        "constant" if simple_gen else "cosine"
    ) if args.lr_schedule == "auto" else args.lr_schedule
    opt = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    warmup = 0 if lr_schedule == "constant" else max(1, int(args.warmup_frac * args.steps))

    def lr_at(step_idx: int) -> float:
        if lr_schedule == "constant":
            return 1.0
        if step_idx < warmup:
            return (step_idx + 1) / warmup
        t = (step_idx - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)

    mix = [x.strip() for x in args.mix.split(",") if x.strip()]
    t0 = time.time()
    hist = []
    print(
        f"[omni] {args.steps} steps on {dev}  n_par={sum(p.numel() for p in model.parameters())}  "
        f"online data mix={mix}  lr={effective_lr} schedule={lr_schedule} "
        f"warmup={warmup}/{args.steps}  "
        f"flow_match={args.flow_match} fm_pred={args.fm_pred} write={args.deslice_write} "
        f"canvas={args.t2i_canvas} x0={args.fm_x0} signed={args.fm_signed} cfg={args.cfg} "
        f"f_gen={args.f_gen} gen_recipe={args.gen_recipe} "
        f"prior_write={effective_prior_write} "
        f"field_passes={1 if args.f_gen else 0} "
        f"deslice_topk={model.mot_stack.deslice_topk} "
        f"null_slice={model.mot_stack.use_null_slice} "
        f"pack_u={model.mot_stack.pack_by_surprise} "
        f"hard_admit={model.mot_stack.hard_admit} "
        f"yield_read={model.mot_stack.use_yield_read} "
        f"ticket_read={model.mot_stack.use_ticket_read} "
        f"write_yield={model.mot_stack.use_write_yield} "
        f"residual_read={model.mot_stack.use_residual_read} "
        f"write_alpha={model.mot_stack.write_alpha} sigreg={args.sigreg_coef}",
        flush=True,
    )
    flow_eval = 0 if args.f_gen else (int(args.flow_steps) if args.flow_match else 0)
    f_eval = int(args.f_steps) if args.f_gen else 0
    f_halt = float(args.f_halt_eps) if args.f_gen else 0.0
    eval_kw = dict(
        flow_steps=flow_eval, flow_method=args.flow_method,
        fm_x0=args.fm_x0, t2i_canvas=args.t2i_canvas,
        fm_signed=args.fm_signed, cfg=args.cfg,
        t2i_stroke_px=args.t2i_stroke_px, t2i_place=args.t2i_place,
        f_steps=f_eval, f_halt_eps=f_halt,
        t2i_digit=args.t2i_digit,
        t2i_color=(args.t2i_color or None),
    )
    tag = args.tag or "_".join(mix)
    ckpt_best = ROOT / "checkpoints" / f"omni_d{args.d_model}_{tag}_best.pt"
    ckpt_last = ROOT / "checkpoints" / f"omni_d{args.d_model}_{tag}_last.pt"
    ckpt_best.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_step = 0
    best_state = None
    best_eval = None

    for step in range(1, args.steps + 1):
        model.train()
        b = make_omni_batch(
            rng, args.batch, args.res, mix=mix, t2i_canvas=args.t2i_canvas,
            t2i_stroke_px=args.t2i_stroke_px, t2i_place=args.t2i_place,
            t2i_digit=args.t2i_digit,
            t2i_color=(args.t2i_color or None),
        )
        x0 = b["image"].to(dev)
        x1 = b["target_rgb"].to(dev)
        if args.t2i_hint_frac > 0.0:
            x0 = apply_t2i_gray_hint(
                x0, b["stroke"], b["kind"], args.t2i_hint_frac, rng,
            )
        t_fm = None
        prompts = list(b["prompt"])
        if args.f_gen and any(b["need_pix"]):
            if args.fm_signed:
                x0 = to_signed(x0)
                x1 = to_signed(x1)
            zeros = torch.zeros(x1.shape[0], device=dev)
            b["target_rgb"] = x1.detach().cpu()
            if args.cfg_drop > 0.0:
                drop = torch.rand(len(prompts)) < float(args.cfg_drop)
                prompts = ["" if d else p for d, p in zip(drop.tolist(), prompts)]
            # Same boundary at train and inference: source/null field at t=0,
            # one native stack trajectory, terminal observation loss.
            imgs = x0
            t_fm = zeros
            b["t"] = t_fm
            b["_action_bptt"] = 0
        elif args.f_gen:
            imgs = x0
            t_fm = None
            b["_action_bptt"] = 0
        elif args.flow_match and any(b["need_pix"]):
            if args.fm_signed:
                x0 = to_signed(x0)
                x1 = to_signed(x1)
            kinds = b["kind"]
            if args.fm_x0 == "noise":
                x0 = torch.randn_like(x1)
            elif any(k == "recon" for k in kinds):
                noise = torch.rand_like(x0) if not args.fm_signed else torch.randn_like(x0)
                for i, k in enumerate(kinds):
                    if k == "recon":
                        x0[i] = noise[i]
            t_fm = sample_t(x0.shape[0], dev, args.flow_t)
            imgs = interpolate(x0, x1, t_fm)
            b["v_tgt"] = velocity_target(x0, x1)
            b["t"] = t_fm
            b["target_rgb"] = x1.detach().cpu()
            if args.cfg_drop > 0.0:
                drop = torch.rand(len(prompts)) < float(args.cfg_drop)
                prompts = ["" if d else p for d, p in zip(drop.tolist(), prompts)]
        else:
            imgs = x0
        opt.zero_grad()
        n_act = int(b.get("_action_bptt") or 0)
        if n_act > 0:
            out = model.action_writes(
                imgs, prompts, need_pix=b["need_pix"], n_steps=n_act, t=t_fm,
            )
        else:
            out = model(imgs, prompts, need_pix=b["need_pix"], t=t_fm)
        loss, meta = model.omni_loss(out, b, dev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps:
            ev = eval_ports(
                model, rng_val, args.res, dev, n=24, kinds=mix, **eval_kw,
            )
            score = eval_score(ev, mix)
            hist.append({
                "step": step, "loss": float(loss.item()),
                "lr": opt.param_groups[0]["lr"], "eval": ev, "score": score,
            })
            bits = " ".join(
                f"{k}:" + (
                    f"acc={ev[k]['acc']*100:.0f}"
                    if k in ("t2t", "i2t", "it2t")
                    else (
                        f"psnr={ev[k]['psnr']:.1f}/str={ev[k]['stroke_psnr']:.1f}"
                        f"/top1={ev[k]['digit_top1']*100:.0f}"
                        f"/pair={ev[k]['paired_iou']*100:.0f}"
                        f"/pos={ev[k]['centroid_error']*100:.1f}"
                        f"/col={ev[k]['color_acc']*100:.0f}/fld={ev[k]['flood']*100:.0f}"
                        f"/K={ev[k].get('f_steps', 0):.1f}"
                    )
                )
                for k in mix
            )
            mark = ""
            if score > best_score:
                best_score = score
                best_step = step
                best_eval = ev
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt_best)
                mark = f"  *best={score:.4f}"
            s0c = s0_prompt_cosine(model, dev)
            s0bit = f"  s0cos={s0c:.4f}" if s0c is not None else ""
            model.train()
            print(
                f"  step {step:4d}/{args.steps} lr={opt.param_groups[0]['lr']:.2e} "
                f"loss={loss.item():.3f}  {bits}{s0bit}{mark}",
                flush=True,
            )

    torch.save(model.state_dict(), ckpt_last)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)
        print(f"  restored best @ step {best_step} score={best_score:.4f}", flush=True)
    ev = eval_ports(
        model, rng_val, args.res, dev, n=64, kinds=mix, **eval_kw,
    )
    ckpt = ckpt_best
    gallery = ROOT / "present" / "figs" / f"omni_{tag}_gallery.png"
    gal_kinds = mix if set(mix) <= {"t2i", "i2i", "recon"} else None
    if mix == ["t2i"]:
        gal_kinds = ["t2i"] * 5
    elif mix == ["i2i"]:
        gal_kinds = ["i2i"] * 5
    elif mix == ["recon"]:
        gal_kinds = ["recon"] * 5
    elif mix == ["i2t"]:
        gal_kinds = ["i2t"] * 4
    table = {
        "steps": args.steps,
        "res": args.res,
        "d_model": args.d_model,
        "n_slices": args.n_slices,
        "n_heads": args.n_heads,
        "elapsed_sec": round(time.time() - t0, 2),
        "n_par": int(sum(p.numel() for p in model.parameters())),
        "final": ev,
        "history": hist,
        "ckpt": str(ckpt),
        "ckpt_last": str(ckpt_last),
        "best_step": best_step,
        "best_score": best_score if best_score > float("-inf") else None,
        "best_eval_n24": best_eval,
        "gallery": str(gallery),
        "lr": effective_lr,
        "lr_schedule": lr_schedule,
        "warmup_frac": args.warmup_frac,
        "mix": mix,
        "vfe_coef": args.vfe_coef,
        "gate_on": args.gate_on,
        "deslice_write": args.deslice_write,
        "s_lang_topk": args.s_lang_topk,
        "flow_match": args.flow_match,
        "flow_steps": flow_eval,
        "f_gen": bool(args.f_gen),
        "gen_recipe": args.gen_recipe,
        "f_steps": f_eval,
        "f_halt_eps": f_halt,
        "t2i_hint_frac": args.t2i_hint_frac,
        "fm_pred": args.fm_pred,
        "fm_x0": args.fm_x0,
        "t2i_canvas": args.t2i_canvas,
        "t2i_stroke_px": args.t2i_stroke_px,
        "t2i_place": args.t2i_place,
        "spatial_prompt_vocab": spatial_prompt_vocab,
        "prior_write": effective_prior_write,
        "fm_signed": args.fm_signed,
        "cfg": args.cfg,
        "flow_t": args.flow_t,
        "note": (
            "Dedicated port run. The selected lr_schedule is recorded above. "
            "fm_signed: JiT [-1,1] chart + adaLN-Zero(t, pool(H)) on X/S. "
            "best.pt maximizes T2I digit_top1 + paired_iou + 0.001 PSNR; "
            "final metrics reload that ckpt. Native generation encodes once, "
            "evolves X-Slice-H through the stack, and decodes once."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print("\nFINAL", flush=True)
    for k, rec in ev.items():
        print(
            f"  {k:6s} acc={rec['acc']*100:5.1f}  psnr={rec['psnr']:5.2f}  "
            f"stroke={rec['stroke_psnr']:5.2f}  bg={rec['bg_psnr']:5.2f}  "
            f"ink={rec['ink']*100:5.1f}  flood={rec['flood']*100:5.1f}  "
            f"color={rec.get('color_acc',0)*100:5.1f}  "
            f"top1={rec.get('digit_top1',0)*100:5.1f}  "
            f"pair={rec.get('paired_iou',0)*100:5.1f}  "
            f"pos={rec.get('centroid_error',0)*100:5.1f}  "
            f"K={rec.get('f_steps',0):.1f}",
            flush=True,
        )
    print(f"saved {out}", flush=True)
    try:
        render_gallery(
            model, np.random.default_rng(7), args.res, dev, gallery,
            kinds=gal_kinds, flow_steps=flow_eval,
            flow_method=args.flow_method, fm_x0=args.fm_x0,
            t2i_canvas=args.t2i_canvas,
            fm_signed=args.fm_signed, cfg=args.cfg,
            t2i_stroke_px=args.t2i_stroke_px, t2i_place=args.t2i_place,
            f_steps=f_eval, f_halt_eps=f_halt,
            t2i_digit=args.t2i_digit,
            t2i_color=(args.t2i_color or None),
        )
    except Exception as exc:
        print(f"  gallery skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
