#!/usr/bin/env python3
"""Visualize the real hand-eye-brain path of native Slice generation.

The script loads the latest natural-image capacity candidate, starts from an
all-zero visual field, and records the actual language/Slice attention, read
assignments, sharpened Deslice write assignments, per-layer field writes, and
the generated RGB trajectory.  It also exports a compact unified-graph figure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import textwrap

os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import capability_champion_kwargs, load_visual_champion


CHECKPOINT = ROOT / "checkpoints" / "_natural_t2i_g8b_candidate.pt"
RESULT_JSON = ROOT / "results" / "published" / "natural_t2i_g8b_continue.json"
FIG_DIR = ROOT / "present" / "figs"
STATS_PATH = ROOT / "results" / "published" / "generation_interaction_stats.json"

BG = "#090d16"
PANEL = "#0f172a"
FG = "#f8fafc"
MUTED = "#94a3b8"
BLUE = "#38bdf8"
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
PURPLE = "#c084fc"
ROSE = "#fb7185"


def _clean_token(token: str) -> str:
    token = token.replace("Ġ", "▁").replace("Ċ", "↵")
    return token if len(token) <= 15 else token[:13] + "…"


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def _decode_steps(model, steps: list[torch.Tensor]) -> list[np.ndarray]:
    images = []
    with torch.no_grad():
        for state in steps:
            field = torch.sigmoid(model.decode_field(state))
            images.append(field[0].permute(1, 2, 0).float().cpu().numpy())
    return images


def _assignment_concentration(w: torch.Tensor) -> float:
    # Per-point share assigned to the dominant Slice; invariant to null mass.
    w = w.float().clamp_min(0)
    denom = w.sum(dim=-1).clamp_min(1e-8)
    return float((w.max(dim=-1).values / denom).mean().item())


def build_model(device: torch.device) -> DualStreamOmni:
    model = DualStreamOmni(**capability_champion_kwargs(
        res=64,
        n_slices=64,
        language="pythia",
        lm_device=str(device),
        pixel_loss_mode="gaussian_nll",
        s0_acc_coef=0.0,
        deslice_write_sharpening=True,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    report = load_visual_champion(
        model, CHECKPOINT, skip_language_interface=False,
    )
    if report.get("n_skipped", 0):
        raise RuntimeError(f"checkpoint shape mismatch: {report}")
    model.eval()
    model.mot_stack.set_record_field_trace(True)
    return model


def extract(model: DualStreamOmni, prompt: str, device: torch.device) -> dict:
    tok = model.lm_tok
    enc = tok(
        prompt, return_tensors="pt", add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    tokens = [_clean_token(t) for t in tok.convert_ids_to_tokens(ids[0].tolist())]
    blank = torch.zeros(1, 3, 64, 64, device=device)

    # Bypass the frozen decoder but retain a graph from prompt embeddings to RGB.
    emb = model.lm.get_input_embeddings()(ids).detach().float().requires_grad_(True)
    X, H, interface, traces = model.mot_stack.forward_native(
        img=blank,
        text_emb=emb,
        text_mask=mask.bool(),
        prompt_mask=mask.bool(),
        pi_x=1.0,
        image_precision=torch.zeros(1, device=device),
        text_precision=torch.ones(1, device=device),
    )
    out = model._fill_visual_outputs({"X": X}, blank)
    rgb = out["rgb"]
    dx = rgb[:, :, :, 1:] - rgb[:, :, :, :-1]
    dy = rgb[:, :, 1:, :] - rgb[:, :, :-1, :]
    # A single scalar that rewards both visible content and spatial structure.
    generation_energy = rgb.square().mean() + 4.0 * (
        dx.square().mean() + dy.square().mean()
    )
    token_grad = torch.autograd.grad(generation_energy, emb)[0]
    token_grad = token_grad.norm(dim=-1)[0].detach().cpu().numpy()

    layers = []
    eye_maps, hand_maps, prior_rows = [], [], []
    for index, layer in enumerate(model.mot_stack.layers):
        av = layer.mot.last_av[0].mean(dim=0).float()  # [M, M+T]
        at = layer.mot.last_at[0].mean(dim=0).float()  # [T, M+T]
        M = int(layer.M)
        s_to_h = float(av[:, M:].sum(dim=-1).mean().item())
        h_to_s = float(at[:, :M].sum(dim=-1).mean().item())

        prior = layer.surprise_gate.last_attn[0].mean(dim=(0, 1)).float()
        prior_rows.append(prior.detach().cpu().numpy())
        read_w = layer.last_w[0].float()       # [N,M]
        # last_w_write is the transported pre-Deslice assignment.  Apply the
        # actual γ sharpening used inside Deslice before visualizing the hand.
        write_w = layer.deslice._write_w(layer.last_w_write)[0].float()
        eye = read_w.max(dim=-1).values.reshape(64, 64)
        hand = layer.last_dW[0].float().norm(dim=-1).reshape(64, 64)
        eye_maps.append(eye.detach().cpu().numpy())
        hand_maps.append(hand.detach().cpu().numpy())
        gamma_raw = layer.deslice.write_gamma_raw
        gamma = 1.0 if gamma_raw is None else float(gamma_raw.detach().exp().item())
        layers.append({
            "layer": index,
            "s_to_h_mass": s_to_h,
            "h_to_s_mass": h_to_s,
            "read_concentration": _assignment_concentration(read_w),
            "write_concentration": _assignment_concentration(write_w),
            "write_rms": float(layer.last_dW.float().pow(2).mean().sqrt().item()),
            "write_gamma": gamma,
            "surprise_gate": float(layer.last_gate.float().mean().item()),
        })

    prior_array = np.stack(prior_rows)
    # Merge byte-pair pieces back into human-readable prompt words.
    spans = [(m.start(), m.end(), m.group(0).strip(".,;:!?")) for m in re.finditer(r"\S+", prompt)]
    groups: list[list[int]] = [[] for _ in spans]
    for token_index, (start, end) in enumerate(offsets):
        for word_index, (word_start, word_end, _word) in enumerate(spans):
            if max(start, word_start) < min(end, word_end):
                groups[word_index].append(token_index)
                break
    keep = [i for i, group in enumerate(groups) if group]
    word_labels = [spans[i][2] for i in keep]
    prior_words = np.stack([
        prior_array[:, groups[i]].sum(axis=1) for i in keep
    ], axis=1)
    grad_words = np.asarray([
        float(np.linalg.norm(token_grad[groups[i]])) for i in keep
    ])

    return {
        "tokens": tokens,
        "token_grad": token_grad,
        "prior": prior_array,
        "words": word_labels,
        "word_grad": grad_words,
        "prior_words": prior_words,
        "eye_maps": eye_maps,
        "hand_maps": hand_maps,
        "rgb_steps": _decode_steps(model, model.mot_stack._last_X_steps),
        "final_rgb": rgb[0].permute(1, 2, 0).detach().cpu().numpy(),
        "layers": layers,
        "trace": [trace.__dict__ for trace in traces],
        "interface_rms": float(interface.float().pow(2).mean().sqrt().item()),
        "prompt": prompt,
    }


def plot_generation(data: dict, out_path: Path) -> None:
    words = data["words"]
    prior = data["prior_words"]
    grad = data["word_grad"]
    # Keep prompt order while selecting the strongest twelve words.
    strength = prior.max(axis=0) + grad / max(float(grad.max()), 1e-12)
    selected = np.sort(np.argsort(strength)[-12:])
    labels = [words[i] for i in selected]

    fig = plt.figure(figsize=(18, 17), dpi=150, facecolor=BG)
    gs = GridSpec(
        5, 4, figure=fig,
        height_ratios=[1.15, 1, .23, 1, 1.05],
        hspace=.34, wspace=.18,
    )

    ax_prior = fig.add_subplot(gs[0, :2])
    _style(ax_prior)
    im = ax_prior.imshow(prior[:, selected], cmap="viridis", aspect="auto")
    ax_prior.set_yticks(range(4), [f"Layer {i}" for i in range(4)])
    ax_prior.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax_prior.set_title("BRAIN · Top-down language prior  Q_slice → H", color=FG, weight="bold")
    ax_prior.set_ylabel("Joint evolution depth", color=MUTED)
    cb = fig.colorbar(im, ax=ax_prior, fraction=.035, pad=.02)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    ax_grad = fig.add_subplot(gs[0, 2:])
    _style(ax_grad)
    g = grad[selected]
    order = np.argsort(g)
    ax_grad.barh(np.arange(len(order)), g[order], color=AMBER, alpha=.9)
    ax_grad.set_yticks(np.arange(len(order)), [labels[i] for i in order])
    ax_grad.yaxis.tick_right()
    ax_grad.tick_params(axis="y", labelleft=False, labelright=True, pad=3)
    ax_grad.set_title("BRAIN → HAND · ∂ generated field energy / ∂ token", color=FG, weight="bold")
    ax_grad.set_xlabel("Gradient norm", color=MUTED)
    ax_grad.grid(axis="x", color="#334155", alpha=.45)

    eye_max = max(float(np.max(x)) for x in data["eye_maps"])
    for layer in range(4):
        ax = fig.add_subplot(gs[1, layer])
        _style(ax)
        ax.imshow(data["eye_maps"][layer], cmap="magma", vmin=0, vmax=eye_max)
        metric = data["layers"][layer]
        ax.set_title(
            f"EYE L{layer} · raw SliceRead ownership\n"
            f"dominant share {metric['read_concentration']:.3f}",
            color=BLUE, weight="bold", fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[3, layer])
        _style(ax)
        # Normalize each layer locally so early writes remain visible.  Absolute
        # strength is retained numerically as RMS in the title.
        local_max = max(float(np.max(data["hand_maps"][layer])), 1e-12)
        ax.imshow(
            data["hand_maps"][layer], cmap="inferno", vmin=0,
            vmax=local_max,
        )
        ax.set_title(
            f"HAND L{layer} · write detail |ΔX| (local scale)\n"
            f"address {metric['write_concentration']:.3f} · RMS {metric['write_rms']:.3f}",
            color=ROSE, weight="bold", fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])

    ax_flow = fig.add_subplot(gs[2, :])
    ax_flow.set_facecolor(BG)
    ax_flow.axis("off")
    ax_flow.text(
        .02, .76,
        "PURPLE = diffuse Slice competition     YELLOW = strong local ownership",
        color=FG, fontsize=9.5, weight="bold", va="center",
    )
    ax_flow.text(
        .46, .25, "soft read address", color=BLUE, fontsize=9.5,
        weight="bold", ha="center", va="center",
    )
    ax_flow.annotate(
        "", xy=(.62, .25), xytext=(.55, .25),
        arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2),
    )
    ax_flow.text(
        .71, .25, "power sharpen  γ = 8", color=PURPLE, fontsize=9.5,
        weight="bold", ha="center", va="center",
    )
    ax_flow.annotate(
        "", xy=(.86, .25), xytext=(.81, .25),
        arrowprops=dict(arrowstyle="-|>", color=ROSE, lw=2),
    )
    ax_flow.text(
        .93, .25, "pixel write", color=ROSE, fontsize=9.5,
        weight="bold", ha="center", va="center",
    )

    raw_blank = np.zeros((64, 64, 3), dtype=np.float32)
    canvas_images = [raw_blank] + data["rgb_steps"][:5]
    canvas_titles = [
        "Actual RGB input\n(all-zero boundary)",
        "Encoded blank latent\n(decoded only for inspection)",
        "Canvas after Layer 0",
        "Canvas after Layer 1",
        "Canvas after Layer 2",
        "Canvas after Layer 3",
    ]
    step_grid = gs[4, :].subgridspec(1, 6, wspace=.08)
    for index, (image, title) in enumerate(zip(canvas_images, canvas_titles)):
        ax = fig.add_subplot(step_grid[0, index])
        _style(ax)
        ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
        ax.set_title(title, color=TEAL, weight="bold", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    wrapped = "\n".join(textwrap.wrap(data["prompt"], width=112))
    fig.suptitle(
        "Native Slice Generation · real cross-modal weights from a blank field\n" + wrapped,
        color=FG, fontsize=14, weight="bold", y=.995,
    )
    fig.text(
        .5, .01,
        "Eye = where Slices read · Brain = which words steer the field · Hand = where Deslice writes · hand maps use local color scales; RMS preserves absolute strength",
        ha="center", color=MUTED, fontsize=10,
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    def box(x, y, w, h, title, detail, color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=.04,rounding_size=.14",
            facecolor=PANEL, edgecolor=color, linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(x+w/2, y+h*.62, title, ha="center", va="center", color=FG, fontsize=13, weight="bold")
        ax.text(x+w/2, y+h*.28, detail, ha="center", va="center", color=MUTED, fontsize=9)

    def arrow(x1, y1, x2, y2, color, label="", rad=0.0):
        a = patches.FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
            linewidth=2.2, color=color, connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2+.2, label, color=color, ha="center", fontsize=9, weight="bold")

    ax.text(8, 8.55, "One Native Graph · Perception and Generation", ha="center", color=FG, fontsize=22, weight="bold")
    ax.text(8, 8.13, "Only boundary precision and likelihood ports change", ha="center", color=MUTED, fontsize=11)

    box(.5, 6.0, 3.0, 1.25, "BOUNDARY", "image · blank · history · action", BLUE)
    box(4.35, 5.65, 3.1, 1.95, "EYE · SliceRead", "full-resolution X → fixed M slices\nread assignment w_read", BLUE)
    box(8.0, 5.65, 3.1, 1.95, "BRAIN · S ↔ H", "shared MoT attention + VFE gate\nfrozen Pythia supplies H", AMBER)
    box(11.65, 5.65, 3.1, 1.95, "HAND · Deslice", "slice update → full-resolution ΔX\nsharpened w_write, γ=8", ROSE)
    arrow(3.5, 6.62, 4.35, 6.62, BLUE)
    arrow(7.45, 6.75, 8.0, 6.75, AMBER, "S→H")
    arrow(8.0, 6.25, 7.45, 6.25, TEAL, "H→S")
    arrow(11.1, 6.62, 11.65, 6.62, ROSE)
    feedback_path = MplPath(
        [(14.75, 6.05), (15.2, 5.0), (4.0, 5.0), (4.9, 5.65)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(patches.FancyArrowPatch(
        path=feedback_path, arrowstyle="-|>", mutation_scale=15,
        linewidth=2.2, color=PURPLE,
    ))
    ax.text(9.55, 4.82, "updated canvas returns to the eye", color=PURPLE,
            ha="center", fontsize=9, weight="bold")

    box(4.35, 3.55, 10.4, 1.0, "× 4 JOINT EVOLUTION LAYERS", "read current field → exchange language/vision messages → gate update → write back", PURPLE)
    arrow(9.55, 5.65, 9.55, 4.55, PURPLE)

    ports = [
        ("TOKEN", "T2T · I2T · IT2T", BLUE),
        ("RGB", "T2I · reconstruction · edit · future", TEAL),
        ("MASK", "segmentation", AMBER),
    ]
    for i, (name, detail, color) in enumerate(ports):
        x = 4.35 + i*3.55
        box(x, 1.65, 3.1, 1.15, name + " LIKELIHOOD", detail, color)
        arrow(9.55, 3.55, x+1.55, 2.8, color)

    tasks = "PERCEIVE   RECONSTRUCT   GENERATE   EDIT   SEGMENT   PREDICT"
    ax.text(8, .85, tasks, ha="center", color=FG, fontsize=13, weight="bold")
    ax.text(8, .42, "same parameters · same X–Slice–H chart · different evidence precision", ha="center", color=MUTED, fontsize=10)
    fig.savefig(out_path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    record = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    prompt = record["prompts"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(device)
    data = extract(model, prompt, device)
    plot_generation(data, FIG_DIR / "generation_cross_modal_interaction.png")
    plot_architecture(FIG_DIR / "unified_slice_architecture.png")

    serializable = {
        "checkpoint": str(CHECKPOINT),
        "prompt": data["prompt"],
        "tokens": data["tokens"],
        "token_generation_gradient": data["token_grad"].tolist(),
        "words": data["words"],
        "word_generation_gradient": data["word_grad"].tolist(),
        "layers": data["layers"],
        "interface_rms": data["interface_rms"],
        "source_image": "all-zero field",
        "image_precision": 0.0,
        "text_precision": 1.0,
        "lm_generate_called": False,
    }
    STATS_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps({
        "device": str(device),
        "figure": str(FIG_DIR / "generation_cross_modal_interaction.png"),
        "architecture": str(FIG_DIR / "unified_slice_architecture.png"),
        "stats": str(STATS_PATH),
        "layers": data["layers"],
    }, indent=2))


if __name__ == "__main__":
    main()
