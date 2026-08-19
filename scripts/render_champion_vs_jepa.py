#!/usr/bin/env python3
"""Render 4-arm showdown: Patch-ViT / ungated Baseline / JEPA / Champion B.

Figures under present/figs/:
  champion_vs_jepa_trajectories.png
  champion_vs_jepa_tasks.png
  champion_vs_jepa_summary_bars.png
  decision_saliency_layer_evolution.png
  saliency_snr_comparison.png
  pure_surprise_layer_comparison.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.ocr_1px import make_ocr_1px
from fine_grain.patch_vit_vqa import PatchViTVQAModel, compute_patch_layer_saliency
from scripts.run_v0_surprise_eval import DualStreamVQAModel
from scripts.visualize_decision_saliency_and_surprise import compute_layer_saliency
from scripts.visualize_surprise_heatmaps import run_heatmap_extraction

TRIAD = ROOT / "results" / "published" / "triad_2000step_table.json"
RMS_JSON = ROOT / "results" / "published" / "upd_rms_bayes_2000step_table.json"
JEPA_RMS_JSON = ROOT / "results" / "published" / "upd_rms_jepa_2000step_table.json"
PATCH_JSON = ROOT / "results" / "published" / "patch_vit_2000step_table.json"
OUT = ROOT / "present" / "figs"

CKPTS = {
    "patch_vit": ROOT / "checkpoints" / "patch_vit_2000step_run1_best.pt",
    "baseline": ROOT / "checkpoints" / "baseline_2000step_run2_best.pt",
    "v0_jepa": ROOT / "checkpoints" / "v0_jepa_2000step_upd_rms_run3_best.pt",
    "champ_b": ROOT / "checkpoints" / "v1_bayes_2000step_upd_rms_run3_best.pt",
}

STYLES = {
    "patch_vit": {
        "name": "Patch-ViT (p=4, shared TF)",
        "short": "Patch-ViT",
        "color": "#fb7185",
        "light": "#fda4af",
    },
    "baseline": {
        "name": "Baseline (ungated DualStream)",
        "short": "Baseline",
        "color": "#94a3b8",
        "light": "#cbd5e1",
    },
    "v0_jepa": {
        "name": "V0 JEPA + RMSNorm(Δ)",
        "short": "JEPA+RMS",
        "color": "#38bdf8",
        "light": "#7dd3fc",
    },
    "champ_b": {
        "name": "Champion B  (Bayes + RMSNorm Δ)",
        "short": "Champion B",
        "color": "#fbbf24",
        "light": "#fde68a",
    },
}

ORDER = ["patch_vit", "baseline", "v0_jepa", "champ_b"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_arms() -> Dict[str, dict]:
    triad = load_json(TRIAD)
    rms = load_json(RMS_JSON)["v1_bayes"]
    jepa_rms = load_json(JEPA_RMS_JSON)["v0_jepa"]
    arms = {
        "baseline": triad["baseline"],
        "v0_jepa": jepa_rms,
        "champ_b": rms,
    }
    if PATCH_JSON.exists():
        pj = load_json(PATCH_JSON)
        arms["patch_vit"] = pj.get("patch_vit", pj)
        # pick best seed ckpt
        best = max(arms["patch_vit"]["runs"], key=lambda r: r["final_acc"])
        CKPTS["patch_vit"] = ROOT / best.get(
            "checkpoint", f"checkpoints/patch_vit_2000step_run{best['run_idx']}_best.pt"
        )
    return arms


def shape_load(model: torch.nn.Module, ckpt_path: Path) -> Tuple[int, int]:
    raw = torch.load(ckpt_path, map_location="cpu")
    current = model.state_dict()
    matched = {k: v for k, v in raw.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    skipped = len(raw) - len(matched)
    missing = model.load_state_dict(matched, strict=False)
    print(
        f"  load {ckpt_path.name}: {len(matched)} ok, skip {skipped}, "
        f"missing {len(missing.missing_keys)}",
        flush=True,
    )
    return len(matched), skipped


def render_curves(arms: Dict[str, dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [k for k in ORDER if k in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.8), dpi=160)
    fig.patch.set_facecolor("#0b1120")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")

    for key in keys:
        data = arms[key]
        st = STYLES[key]
        runs = data["runs"]
        steps = [pt["step"] for pt in runs[0]["trajectory"]]
        losses = np.array([[pt["val_loss"] for pt in r["trajectory"]] for r in runs])
        accs = np.array([[pt["val_acc"] * 100 for pt in r["trajectory"]] for r in runs])
        lw = 3.1 if key == "champ_b" else 2.2
        for r in runs:
            ax1.plot(steps, [pt["val_loss"] for pt in r["trajectory"]], color=st["color"], alpha=0.16, lw=0.9)
            ax2.plot(steps, [pt["val_acc"] * 100 for pt in r["trajectory"]], color=st["color"], alpha=0.16, lw=0.9)
        m_l, s_l = losses.mean(0), losses.std(0)
        m_a, s_a = accs.mean(0), accs.std(0)
        ax1.plot(steps, m_l, color=st["color"], lw=lw, label=f"{st['name']}  {data['mean_loss']:.3f}")
        ax1.fill_between(steps, m_l - s_l, m_l + s_l, color=st["color"], alpha=0.10)
        ax2.plot(steps, m_a, color=st["color"], lw=lw, label=f"{st['name']}  {data['mean_acc']:.1f}±{data['std_acc']:.1f}%")
        ax2.fill_between(steps, m_a - s_a, m_a + s_a, color=st["color"], alpha=0.10)

    ax1.set_title("2000-step val loss", color="white", fontsize=12, fontweight="bold")
    ax2.set_title("2000-step val acc", color="white", fontsize=12, fontweight="bold")
    for ax, ylab in ((ax1, "Validation CE"), (ax2, "Validation accuracy (%)")):
        ax.set_xlabel("Training steps", color="#cbd5e1")
        ax.set_ylabel(ylab, color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, ls=":", alpha=0.3, color="#64748b")
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)
    plt.tight_layout()
    p = OUT / "champion_vs_jepa_trajectories.png"
    fig.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=160)
    fig.patch.set_facecolor("#0b1120")
    tasks = [("val_ocr", "1px OCR"), ("val_kinks", "Kinks (topology)"), ("val_color", "Color needle")]
    for ax, (tkey, title) in zip(axes, tasks):
        ax.set_facecolor("#0f172a")
        for key in keys:
            st = STYLES[key]
            runs = arms[key]["runs"]
            steps = [pt["step"] for pt in runs[0]["trajectory"]]
            mean = np.mean([[pt[tkey] * 100 for pt in r["trajectory"]] for r in runs], axis=0)
            ax.plot(steps, mean, color=st["color"], lw=3.0 if key == "champ_b" else 2.2, label=st["short"])
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training steps", color="#cbd5e1")
        ax.set_ylabel("Task accuracy (%)", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, ls=":", alpha=0.3, color="#64748b")
        ax.set_ylim(0, 105)
        if ax is axes[0]:
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)
    plt.tight_layout()
    p = OUT / "champion_vs_jepa_tasks.png"
    fig.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)

    fig, ax = plt.subplots(figsize=(12.2, 5.7), dpi=160)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    cats = ["Overall", "1px OCR", "Kinks", "Color"]
    x = np.arange(len(cats))
    n = len(keys)
    width = 0.78 / n
    for i, key in enumerate(keys):
        data = arms[key]
        st = STYLES[key]
        vals = [data["mean_acc"], data["mean_ocr"], data["mean_kinks"], data.get("mean_color", 100.0)]
        errs = [data["std_acc"], 0, 0, 0]
        rects = ax.bar(
            x + (i - (n - 1) / 2) * width, vals, width, yerr=errs, capsize=3,
            color=st["color"], alpha=0.92, label=st["name"], edgecolor="#0b1120", linewidth=0.3,
        )
        for r in rects:
            ax.annotate(
                f"{r.get_height():.1f}",
                xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color="#f8fafc", fontweight="bold",
            )
    ax.set_title("Asymptotic 2000-step showdown  (N=3 seeds)", color="white", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, color="#e2e8f0", fontsize=11)
    ax.set_ylabel("Accuracy (%)", color="#cbd5e1")
    ax.set_ylim(0, 116)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, ls=":", alpha=0.3, color="#64748b", axis="y")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.8)
    plt.tight_layout()
    p = OUT / "champion_vs_jepa_summary_bars.png"
    fig.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)


def build_models(device: torch.device, res: int, keys: List[str]) -> Dict[str, torch.nn.Module]:
    models = {}
    specs = {
        "baseline": dict(surprise_mode="baseline", s_update="raw", use_stiefel=True, deslice_topk=2),
        "v0_jepa": dict(surprise_mode="v0_jepa", s_update="rms", use_stiefel=True, deslice_topk=2),
        "champ_b": dict(surprise_mode="v1_bayes", s_update="rms", use_stiefel=True, deslice_topk=2),
    }
    for key in keys:
        if key == "patch_vit":
            m = PatchViTVQAModel(d_model=128, n_layers=4, res=res, patch=4, n_heads=4).to(device)
        else:
            m = DualStreamVQAModel(
                d_model=128, n_slices=32, n_layers=4, res=res, surprise_beta=1.5, **specs[key]
            ).to(device)
        if CKPTS[key].exists():
            shape_load(m, CKPTS[key])
        else:
            print(f"  missing ckpt {CKPTS[key]}", flush=True)
        m.eval()
        models[key] = m
    return models


def saliency_of(key: str, model, img, prompt, ans, res):
    if key == "patch_vit":
        return compute_patch_layer_saliency(model, img, prompt, ans, res)
    return compute_layer_saliency(model, img, prompt, ans, res)


def render_heatmaps(device: torch.device, keys: List[str], n_snr: int = 40) -> dict:
    res = 32
    models = build_models(device, res, keys)
    rng = np.random.default_rng(42)
    img, _, stroke = make_ocr_1px(rng, np.array([7]), res=res)
    img = img.to(device)
    stroke_np = stroke[0].view(res, res).cpu().numpy().astype(bool)
    rgb = img[0].permute(1, 2, 0).cpu().numpy()
    prompt = "Question: What digit is drawn with the thin stroke? Answer:"

    sal = {k: saliency_of(k, models[k], img, prompt, "7", res) for k in keys}

    n = len(keys)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.15 * n), dpi=150)
    if n == 1:
        axes = np.expand_dims(axes, 0)
    plt.subplots_adjust(wspace=0.22, hspace=0.38)
    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    single_snr = {}
    for row, key in enumerate(keys):
        st = STYLES[key]
        axes[row, 0].imshow(np.clip(rgb, 0, 1))
        axes[row, 0].set_title("Input  1px '7'", fontsize=9, color="white", fontweight="bold")
        axes[row, 0].axis("off")
        axes[row, 0].text(
            -0.16, 0.5, st["name"], transform=axes[row, 0].transAxes,
            fontsize=10, fontweight="bold", color=st["color"], rotation=90, va="center", ha="right",
        )
        snrs = []
        for l in range(4):
            g = sal[key][l]
            g_n = (g - g.min()) / (g.max() - g.min() + 1e-8)
            im = axes[row, l + 1].imshow(g_n, cmap="inferno")
            snr = float(g[stroke_np].mean() / (g[~stroke_np].mean() + 1e-8))
            snrs.append(snr)
            axes[row, l + 1].set_title(f"X_{l}  SNR={snr:.2f}×", fontsize=9, color=st["color"])
            axes[row, l + 1].axis("off")
            plt.colorbar(im, ax=axes[row, l + 1], fraction=0.046, pad=0.04)
        single_snr[key] = snrs

    fig.suptitle(
        "Decision saliency  ||∂ Logit_7 / ∂ X_l||    Patch-ViT · Baseline · JEPA · Champion B",
        fontsize=13, fontweight="bold", color="#f8fafc", y=0.995,
    )
    p = OUT / "decision_saliency_layer_evolution.png"
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)

    print(f"[viz] {n_snr}-sample saliency SNR …", flush=True)
    depth = {k: [[] for _ in range(4)] for k in keys}
    for _ in range(n_snr):
        d_val = int(rng.integers(0, 10))
        im_s, _, msk_s = make_ocr_1px(rng, np.array([d_val]), res=res)
        im_s = im_s.to(device)
        msk = msk_s[0].view(res, res).cpu().numpy().astype(bool)
        for key in keys:
            gl = saliency_of(key, models[key], im_s, prompt, str(d_val), res)
            for l in range(4):
                g = gl[l]
                depth[key][l].append(float(g[msk].mean() / (g[~msk].mean() + 1e-8)))

    fig, ax = plt.subplots(figsize=(11.4, 5.6), dpi=160)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    xs = np.arange(4)
    n = len(keys)
    width = 0.78 / n
    means = {k: [float(np.mean(depth[k][l])) for l in range(4)] for k in keys}
    stds = {k: [float(np.std(depth[k][l])) for l in range(4)] for k in keys}
    for i, key in enumerate(keys):
        off = (i - (n - 1) / 2) * width
        ax.bar(xs + off, means[key], width, yerr=stds[key], capsize=2.5,
               label=STYLES[key]["name"], color=STYLES[key]["color"], alpha=0.92)
        for l, v in enumerate(means[key]):
            ax.text(l + off, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2, color="#f8fafc")
    ax.set_title(f"Stroke saliency SNR across depth  (N={n_snr} held-out 1px digits)", color="white", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Field X_{l}" for l in range(4)], color="#e2e8f0")
    ax.set_ylabel("SNR  =  mean(grad_stroke) / mean(grad_bg)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, ls=":", alpha=0.3, color="#64748b", axis="y")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.8)
    plt.tight_layout()
    p = OUT / "saliency_snr_comparison.png"
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)
    for k in keys:
        print(f"    SNR {k}: {[round(v, 2) for v in means[k]]}", flush=True)

    # Surprise / energy: 4 rows. Patch = token energy; others = U (0 for baseline).
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.15 * n), dpi=150)
    if n == 1:
        axes = np.expand_dims(axes, 0)
    plt.subplots_adjust(wspace=0.22, hspace=0.38)
    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    for row, key in enumerate(keys):
        st = STYLES[key]
        axes[row, 0].imshow(np.clip(rgb, 0, 1))
        axes[row, 0].set_title("Input  1px '7'", fontsize=9, color="white", fontweight="bold")
        axes[row, 0].axis("off")
        axes[row, 0].text(
            -0.16, 0.5, st["short"], transform=axes[row, 0].transAxes,
            fontsize=10, fontweight="bold", color=st["color"], rotation=90, va="center", ha="right",
        )
        if key == "patch_vit":
            m = models[key]
            with torch.no_grad():
                _ = m(img, [prompt])
                states = m.last_patch_states[:4]
            for l in range(4):
                e = states[l][0].norm(dim=-1).reshape(m.grid, m.grid).cpu()
                up = e.repeat_interleave(m.patch, 0).repeat_interleave(m.patch, 1).numpy()
                im = axes[row, l + 1].imshow(up, cmap="magma")
                axes[row, l + 1].set_title(f"Patch energy L{l}", fontsize=9, color=st["color"])
                axes[row, l + 1].axis("off")
                plt.colorbar(im, ax=axes[row, l + 1], fraction=0.046, pad=0.04)
        else:
            ext = run_heatmap_extraction(models[key], img.cpu(), prompt, device)
            kind = "energy" if key == "baseline" else "U"
            src = ext["energies"][:4] if key == "baseline" else ext["surprises"]
            for l in range(4):
                im = axes[row, l + 1].imshow(src[l], cmap="magma")
                axes[row, l + 1].set_title(f"{st['short']}  {kind}_{l}", fontsize=9, color=st["color"])
                axes[row, l + 1].axis("off")
                plt.colorbar(im, ax=axes[row, l + 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Spatial maps across depth    Patch energy · Baseline field energy · JEPA/B surprise U",
        fontsize=13, fontweight="bold", color="#f8fafc", y=0.995,
    )
    p = OUT / "pure_surprise_layer_comparison.png"
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)

    return {"single_snr": single_snr, "mean_snr": means, "std_snr": stds}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-only", action="store_true")
    ap.add_argument("--heatmaps-only", action="store_true")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--snr-n", type=int, default=40)
    args = ap.parse_args()

    arms = collect_arms()
    if not args.heatmaps_only:
        print("[curves]", flush=True)
        for k, v in arms.items():
            print(f"  {k:12s} {v['mean_acc']:.2f}±{v['std_acc']:.2f}  ocr={v['mean_ocr']:.1f} kinks={v['mean_kinks']:.1f}", flush=True)
        render_curves(arms)
        if args.curves_only:
            return

    keys = [k for k in ORDER if k in arms]
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[heatmaps] device={device} arms={keys}", flush=True)
    stats = render_heatmaps(device, keys, n_snr=args.snr_n)
    meta = {
        k: {
            "mean_acc": arms[k]["mean_acc"],
            "std_acc": arms[k]["std_acc"],
            "ocr": arms[k]["mean_ocr"],
            "kinks": arms[k]["mean_kinks"],
            "color": arms[k].get("mean_color", 100.0),
        }
        for k in keys
    }
    meta["saliency"] = stats
    out_json = ROOT / "results" / "published" / "champion_vs_jepa_viz.json"
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
