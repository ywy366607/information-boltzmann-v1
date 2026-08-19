#!/usr/bin/env python3
"""One-seed 2000-step P0 residual-cell check vs published Champion B.

Does not overwrite published RMS ckpts / champion heatmap figs.
  1) VQA: v1_bayes + RMSNorm(Δ), seed 42, same recipe as upd_rms
  2) Compare numbers + surprise heatmaps to published Champion B (seed 42)
  3) Omni write ports recon/i2i/t2i at the last gen protocol (d=256, lr=2e-4)

  python scripts/run_p0_id_2000.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.ocr_1px import make_ocr_1px
from scripts.run_v0_surprise_eval import DualStreamVQAModel
from scripts.train_triad_2000steps import train_single_run
from scripts.visualize_surprise_heatmaps import run_heatmap_extraction

PUB_B = ROOT / "results" / "published" / "upd_rms_bayes_2000step_table.json"
FIG = ROOT / "present" / "figs"


def _seed42(table: dict) -> dict:
    arm = table.get("v1_bayes", table)
    for r in arm["runs"]:
        if int(r["seed"]) == 42:
            return r
    return arm["runs"][0]


def render_compare(p0: dict, pub: dict, out_json: Path) -> None:
    b = _seed42(pub)
    FIG.mkdir(parents=True, exist_ok=True)

    # bars
    labels = ["Overall", "1px OCR", "Kinks", "Color"]
    old = [
        b["final_acc"] * 100,
        b["final_tasks"]["ocr"] * 100,
        b["final_tasks"]["kinks"] * 100,
        b["final_tasks"]["color"] * 100,
    ]
    new = [
        p0["final_acc"] * 100,
        p0["final_tasks"]["ocr"] * 100,
        p0["final_tasks"]["kinks"] * 100,
        p0["final_tasks"]["color"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, old, w, label="Champion B  (pre-P0, seed 42)", color="#fbbf24", alpha=0.9)
    ax.bar(x + w / 2, new, w, label="P0 identity cell  (seed 42)", color="#2dd4bf", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#e2e8f0")
    ax.set_ylabel("Held-out acc (%)", color="#cbd5e1")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    ax.set_title("P0 residual cell vs Champion B   ·   2000 steps, seed 42", color="white", fontweight="bold")
    for xs, vals in ((x - w / 2, old), (x + w / 2, new)):
        for xi, v in zip(xs, vals):
            ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color="#f8fafc")
    fig.tight_layout()
    p_bar = FIG / "p0_id_vs_champb_bars.png"
    fig.savefig(p_bar, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p_bar}", flush=True)

    # trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3)
    sb = [pt["step"] for pt in b["trajectory"]]
    sp = [pt["step"] for pt in p0["trajectory"]]
    ax1.plot(sb, [pt["val_loss"] for pt in b["trajectory"]], color="#fbbf24", label="Champion B")
    ax1.plot(sp, [pt["val_loss"] for pt in p0["trajectory"]], color="#2dd4bf", label="P0 identity")
    ax1.set_title("Val loss", color="white")
    ax1.set_xlabel("step", color="#cbd5e1")
    ax2.plot(sb, [pt["val_acc"] * 100 for pt in b["trajectory"]], color="#fbbf24", label="Champion B")
    ax2.plot(sp, [pt["val_acc"] * 100 for pt in p0["trajectory"]], color="#2dd4bf", label="P0 identity")
    ax2.plot(sb, [pt["val_kinks"] * 100 for pt in b["trajectory"]], color="#fbbf24", ls="--", alpha=0.55, label="B kinks")
    ax2.plot(sp, [pt["val_kinks"] * 100 for pt in p0["trajectory"]], color="#2dd4bf", ls="--", alpha=0.55, label="P0 kinks")
    ax2.set_title("Val acc  (solid) / kinks (dash)", color="white")
    ax2.set_xlabel("step", color="#cbd5e1")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()
    p_tr = FIG / "p0_id_vs_champb_trajectories.png"
    fig.savefig(p_tr, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p_tr}", flush=True)

    summary = {
        "note": "P0 identity cell vs published Champion B. One seed (42). Old ckpts untouched.",
        "champion_b_seed42": {
            "final_acc": b["final_acc"],
            "final_tasks": b["final_tasks"],
            "final_loss": b["final_loss"],
            "ckpt": "checkpoints/v1_bayes_2000step_upd_rms_run1_best.pt",
        },
        "p0_identity_seed42": {
            "final_acc": p0["final_acc"],
            "final_tasks": p0["final_tasks"],
            "final_loss": p0["final_loss"],
            "checkpoint": p0.get("checkpoint"),
            "elapsed_sec": p0.get("elapsed_sec"),
        },
        "delta_pp": {
            "acc": (p0["final_acc"] - b["final_acc"]) * 100,
            "ocr": (p0["final_tasks"]["ocr"] - b["final_tasks"]["ocr"]) * 100,
            "kinks": (p0["final_tasks"]["kinks"] - b["final_tasks"]["kinks"]) * 100,
            "color": (p0["final_tasks"]["color"] - b["final_tasks"]["color"]) * 100,
        },
        "figs": [str(p_bar), str(p_tr)],
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved {out_json}", flush=True)
    print(
        f"  DELTA  acc={summary['delta_pp']['acc']:+.2f}pp  "
        f"ocr={summary['delta_pp']['ocr']:+.2f}  kinks={summary['delta_pp']['kinks']:+.2f}",
        flush=True,
    )


def render_p0_heatmaps(ckpt: Path, device: torch.device) -> None:
    model = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms", use_stiefel=True, deslice_topk=2,
    ).to(device)
    raw = torch.load(ckpt, map_location="cpu")
    missing = model.load_state_dict(raw, strict=False)
    print(f"  heatmap load missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}", flush=True)
    model.eval()

    rng = np.random.default_rng(0)
    img, _, _ = make_ocr_1px(rng, np.array([7]), res=32)
    prompt = "What digit is drawn with the thin stroke? Answer:"
    ext = run_heatmap_extraction(model, img, prompt, device)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    rgb = img[0].permute(1, 2, 0).numpy()
    axes[0, 0].imshow(np.clip(rgb, 0, 1))
    axes[0, 0].set_title("input 1px 7", color="white", fontsize=10)
    axes[1, 0].imshow(np.clip(rgb, 0, 1))
    axes[1, 0].set_title("input", color="white", fontsize=10)
    for l in range(4):
        im0 = axes[0, l + 1].imshow(ext["surprises"][l], cmap="magma")
        axes[0, l + 1].set_title(f"P0  U_{l}", color="#2dd4bf", fontsize=10)
        plt.colorbar(im0, ax=axes[0, l + 1], fraction=0.046, pad=0.03)
        im1 = axes[1, l + 1].imshow(ext["energies"][l + 1], cmap="viridis")
        axes[1, l + 1].set_title(f"P0  ||X_{l+1}||", color="#94a3b8", fontsize=10)
        plt.colorbar(im1, ax=axes[1, l + 1], fraction=0.046, pad=0.03)
    for ax in axes.flat:
        ax.axis("off")
        ax.set_facecolor("#0f172a")
    fig.suptitle("P0 identity cell  ·  surprise U and field energy  (seed 42 ckpt)", color="white", fontweight="bold")
    fig.tight_layout()
    p = FIG / "p0_id_heatmaps.png"
    fig.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)

    # KL split if present
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 12), dpi=130)
    fig.patch.set_facecolor("#0b1120")
    for l in range(4):
        axes[l, 0].imshow(ext["surprises"][l], cmap="magma")
        axes[l, 0].set_title(f"L{l}  U", color="#38bdf8", fontsize=9)
        axes[l, 1].imshow(ext["u_mu"][l], cmap="plasma")
        axes[l, 1].set_title(f"L{l}  U_mu", color="#fb7185", fontsize=9)
        axes[l, 2].imshow(ext["u_sigma"][l], cmap="cividis")
        axes[l, 2].set_title(f"L{l}  U_sigma", color="#a3e635", fontsize=9)
        for c in range(3):
            axes[l, c].axis("off")
            axes[l, c].set_facecolor("#0f172a")
    fig.suptitle("P0  V1 KL split on 1px 7", color="white", fontweight="bold")
    fig.tight_layout()
    p2 = FIG / "p0_id_kl_decomp.png"
    fig.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p2}", flush=True)


def run_omni(device: str) -> None:
    # Reuse the shipped trainer; unique tag so old omni ckpts stay put.
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_omni_probe.py"),
        "--steps", "2000",
        "--mix", "recon,i2i,t2i",
        "--lr", "0.0002",
        "--warmup-frac", "0.05",
        "--tag", "p0_id",
        "--out", str(ROOT / "results" / "published" / "omni_p0_id_2000step_table.json"),
        "--device", device,
    ]
    print("[omni]", " ".join(cmd), flush=True)
    import subprocess
    env = os.environ.copy()
    env.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
    env["PYTHONUNBUFFERED"] = "1"
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(f"omni trainer failed: {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-vqa", action="store_true")
    ap.add_argument("--skip-omni", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    vqa_out = ROOT / "results" / "published" / "p0_id_bayes_2000step_table.json"
    compare_out = ROOT / "results" / "published" / "p0_id_vs_champb.json"

    if not args.skip_vqa:
        print("=== P0 VQA  2000 steps  seed=42  v1_bayes + RMS  (Champion B recipe) ===", flush=True)
        res = train_single_run(
            arm="v1_bayes",
            run_idx=0,
            seed=42,
            steps=2000,
            device=args.device,
            prior_loss_coef=0.1,
            detach_pred_target=True,
            sigreg_coef=0.0,
            use_stiefel=True,
            deslice_topk=2,
            s_update="rms_dir",
            ckpt_tag="p0_id",
        )
        table = {
            "v1_bayes": {
                "arm": "v1_bayes",
                "steps": 2000,
                "n_runs": 1,
                "s_update": "rms_dir",
                "ckpt_tag": "p0_id",
                "note": "P0 residual cell. One seed. Does not replace upd_rms Champion B.",
                "mean_acc": res["final_acc"] * 100,
                "std_acc": 0.0,
                "mean_loss": res["final_loss"],
                "mean_ocr": res["final_tasks"].get("ocr", 0.0) * 100,
                "mean_kinks": res["final_tasks"].get("kinks", 0.0) * 100,
                "mean_color": res["final_tasks"].get("color", 0.0) * 100,
                "runs": [res],
            }
        }
        vqa_out.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"saved {vqa_out}", flush=True)
        print(
            f"P0 FINAL  acc={res['final_acc']*100:.2f}%  loss={res['final_loss']:.4f}  "
            f"ocr={res['final_tasks'].get('ocr',0)*100:.1f}  "
            f"kinks={res['final_tasks'].get('kinks',0)*100:.1f}",
            flush=True,
        )
        pub = json.loads(PUB_B.read_text(encoding="utf-8"))
        render_compare(res, pub, compare_out)
        render_p0_heatmaps(Path(res["checkpoint"]), torch.device(args.device))

    if not args.skip_omni:
        print("=== P0 omni write ports  recon,i2i,t2i  2000 steps  d=256 lr=2e-4 ===", flush=True)
        run_omni(args.device)


if __name__ == "__main__":
    main()
