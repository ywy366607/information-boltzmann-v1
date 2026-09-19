"""Generate publication-quality comparative learning curve and slope analysis.
Compares:
1. GDN-1 (Gated DeltaNet)
2. GDN-2 (NVIDIA Gated DeltaNet-2)
3. CBIM Matched Run 3 (Continuous Q8 + Quadratic Bath)
4. CBIM Run 4 (Continuous Q8 + Unified Dissipation)
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_metrics(p):
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    val = [r for r in rows if r.get('kind') == 'validation']
    steps = np.array([r['step'] for r in val])
    nlls = np.array([r['validation_nll'] for r in val])
    train = [r for r in rows if r.get('kind') == 'train']
    t_steps = np.array([r['step'] for r in train])
    t_energy = np.array([r.get('energy', 0.0) for r in train])
    return steps, nlls, t_steps, t_energy


def power_law(t, L_inf, A, alpha):
    return L_inf + A * (t ** (-alpha))


def main():
    gdn1_s, gdn1_l, _, _ = load_metrics('results/gated_deltanet_d128_3000/metrics.jsonl')
    gdn2_s, gdn2_l, _, _ = load_metrics('results/gated_deltanet_2_d128_3000/metrics.jsonl')
    r3_s, r3_l, _, _ = load_metrics('results/cbim_torus3d_w2_8x8x4_arm_c_continuous_q8_3000/metrics.jsonl')
    r4_s, r4_l, r4_ts, r4_te = load_metrics('results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/metrics.jsonl')

    # Fit extrapolation curves out to 6000 steps
    extrap_steps = np.linspace(250, 6000, 200)

    popt_gdn1, _ = curve_fit(power_law, gdn1_s[1:], gdn1_l[1:], p0=[5.0, 10.0, 0.3], bounds=([1.0, 0.1, 0.01], [10.0, 100.0, 2.0]), maxfev=10000)
    popt_gdn2, _ = curve_fit(power_law, gdn2_s[1:], gdn2_l[1:], p0=[5.0, 10.0, 0.3], bounds=([1.0, 0.1, 0.01], [10.0, 100.0, 2.0]), maxfev=10000)
    popt_r3, _ = curve_fit(power_law, r3_s[1:], r3_l[1:], p0=[5.0, 50.0, 0.5], bounds=([1.0, 0.1, 0.01], [10.0, 200.0, 2.0]), maxfev=10000)
    popt_r4, _ = curve_fit(power_law, r4_s[1:], r4_l[1:], p0=[5.0, 50.0, 0.5], bounds=([1.0, 0.1, 0.01], [10.0, 200.0, 2.0]), maxfev=10000)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b0d13")
    for ax in axes:
        ax.set_facecolor("#131620")
        ax.tick_params(colors="#8f9ba8")
        for spine in ax.spines.values():
            spine.set_color("#262b3a")
        ax.xaxis.label.set_color("#d0d7de")
        ax.yaxis.label.set_color("#d0d7de")
        ax.title.set_color("#f0f3f6")
        ax.grid(True, linestyle=":", alpha=0.3, color="#8f9ba8")

    # Panel 0: 3000-Step Validation Learning Curves
    axes[0].plot(gdn1_s[1:], gdn1_l[1:], 'o-', color="#58a6ff", linewidth=2.0, markersize=5, label=f"GDN-1 (Final: {gdn1_l[-1]:.4f})")
    axes[0].plot(gdn2_s[1:], gdn2_l[1:], 's-', color="#a371f7", linewidth=2.0, markersize=5, label=f"GDN-2 (Final: {gdn2_l[-1]:.4f})")
    axes[0].plot(r3_s[1:], r3_l[1:], '^-', color="#f85149", linewidth=2.0, markersize=5, label=f"CBIM Run 3 (Final: {r3_l[-1]:.4f})")
    axes[0].plot(r4_s[1:], r4_l[1:], 'D-', color="#3fb950", linewidth=2.5, markersize=6, label=f"CBIM Run 4 (Final: {r4_l[-1]:.4f})")
    axes[0].set_title("Validation NLL Learning Trajectory (0-3000 Steps)", fontsize=11, pad=8)
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Validation NLL (nats)")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#d0d7de", fontsize=9)

    # Panel 1: Power-Law Extrapolation out to 6000 Steps
    axes[1].plot(extrap_steps, power_law(extrap_steps, *popt_gdn1), '--', color="#58a6ff", linewidth=1.8, label=f"GDN-1 Extrap (6k: {power_law(6000, *popt_gdn1):.2f})")
    axes[1].plot(extrap_steps, power_law(extrap_steps, *popt_gdn2), '--', color="#a371f7", linewidth=1.8, label=f"GDN-2 Extrap (6k: {power_law(6000, *popt_gdn2):.2f})")
    axes[1].plot(extrap_steps, power_law(extrap_steps, *popt_r3), '--', color="#f85149", linewidth=1.8, label=f"Run 3 Extrap (6k: {power_law(6000, *popt_r3):.2f})")
    axes[1].plot(extrap_steps, power_law(extrap_steps, *popt_r4), '--', color="#3fb950", linewidth=2.2, label=f"Run 4 Extrap (6k: {power_law(6000, *popt_r4):.2f})")
    axes[1].axvline(3000, color="#8f9ba8", linestyle=":", alpha=0.6, label="T=3000 Budget Line")
    axes[1].set_title("Power-Law Asymptotic Extrapolation (Scaling to 6000 Steps)", fontsize=11, pad=8)
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Projected NLL (nats)")
    axes[1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#d0d7de", fontsize=9)

    # Panel 2: Initial Transient Inrush Overshoot (Why Step 10 Energy Peaks)
    early_mask = r4_ts <= 150
    axes[2].plot(r4_ts[early_mask], r4_te[early_mask], 'o-', color="#d29922", linewidth=2.5, markersize=5, label="Field Kinetic Energy E(t)")
    axes[2].axhline(r4_te[r4_ts >= 1000].mean(), color="#3fb950", linestyle="--", linewidth=1.8, label=f"NESS Steady-State Energy (E~{r4_te[r4_ts >= 1000].mean():.3f})")
    axes[2].annotate(f"Initial Acoustic Inrush Peak\nE={r4_te[r4_ts == 10][0]:.3f} (Step 10)\n[Vacuum Inflow >> Dissipation]",
                     xy=(10, r4_te[r4_ts == 10][0]), xytext=(35, 0.22),
                     arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.5),
                     color="#f85149", fontsize=8.5,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#f85149"))
    axes[2].set_title("Initial Transient Energy Surge: Vacuum Inrush vs NESS", fontsize=11, pad=8)
    axes[2].set_xlabel("Initial Steps (0 - 150)")
    axes[2].set_ylabel("Field Energy E")
    axes[2].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#d0d7de", fontsize=9)

    plt.tight_layout()
    out_path = Path("present/cbim_vs_gdn_convergence_slopes.png").resolve()
    plt.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    main()
