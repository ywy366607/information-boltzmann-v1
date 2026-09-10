"""Collect scaling data across particle counts N in [16, 64, 128, 256, 512, 1024] and fit thermodynamic scaling law."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy.optimize as opt


def fit_scaling_law(n_list: list[int], loss_list: list[float]) -> dict:
    """Fit power-law scaling: L(N) = L_inf + A * N^(-alpha)."""
    n_arr = np.array(n_list, dtype=float)
    l_arr = np.array(loss_list, dtype=float)

    if len(n_list) < 3:
        return {"alpha": 0.0, "l_inf": float(l_arr[-1]), "R2": 0.0}

    # Initial guess
    l_inf_init = min(l_arr) * 0.95
    p0 = [l_inf_init, 1.0, 0.5]

    def model(N, l_inf, a, alpha):
        return l_inf + a * (N ** (-alpha))

    try:
        popt, _ = opt.curve_fit(model, n_arr, l_arr, p0=p0, bounds=([0, 0, 0], [min(l_arr), 100, 2.0]), maxfev=5000)
        l_inf, a, alpha = popt
        pred = model(n_arr, *popt)
        ss_res = np.sum((l_arr - pred) ** 2)
        ss_tot = np.sum((l_arr - np.mean(l_arr)) ** 2)
        r2 = 1.0 - (ss_res / max(1e-12, ss_tot))
        return {
            "alpha": float(alpha),
            "A": float(a),
            "l_inf": float(l_inf),
            "R2": float(r2),
        }
    except Exception as e:
        # Fallback to log-log linear fit if 3-param fit fails
        log_n = np.log(n_arr)
        log_l = np.log(l_arr)
        slope, intercept = np.polyfit(log_n, log_l, 1)
        return {
            "alpha": float(-slope),
            "A": float(math.exp(intercept)),
            "l_inf": 0.0,
            "R2": 0.0,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/published/particle_scaling_laws.json"))
    args = parser.parse_args()

    # Collect available scaling results
    particle_runs = {}

    # Check N=16 baseline
    n16_summary = Path("results/information_boltzmann_live/summary.json")
    if n16_summary.exists():
        data16 = json.loads(n16_summary.read_text(encoding="utf-8"))
        particle_runs[16] = data16

    # Check scaling directories
    for n in [64, 128, 256, 512, 1024]:
        s_file = args.results_dir / f"scaling_n{n}" / "summary.json"
        if s_file.exists():
            particle_runs[n] = json.loads(s_file.read_text(encoding="utf-8"))

    print(f"Found scaling data for N = {sorted(particle_runs.keys())}")

    summary_table = []
    n_vals = sorted(particle_runs.keys())
    loss_vals = []

    for n in n_vals:
        run = particle_runs[n]
        nll = run["prequential_nll"]
        ppl = run["perplexity"]
        collisions = run["accepted_collisions"]
        sec = run["seconds"]
        loss_vals.append(nll)
        summary_table.append({
            "particles_N": n,
            "prequential_nll": nll,
            "perplexity": ppl,
            "accepted_collisions": collisions,
            "seconds": sec,
            "targets_per_second": run["targets_per_second"],
        })

    fit = fit_scaling_law(n_vals, loss_vals) if len(n_vals) >= 2 else {}

    output_data = {
        "particle_counts": n_vals,
        "runs": summary_table,
        "thermodynamic_scaling_fit": fit,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("PARTICLE COUNT SCALING STUDY & THERMODYNAMIC LIMIT FIT")
    print("=" * 80)
    print(f"{'Particles N':<14} | {'Prequential NLL':<18} | {'Perplexity':<14} | {'Collisions':<12} | {'Tokens/s':<10}")
    print("-" * 80)
    for row in summary_table:
        print(f"{row['particles_N']:<14} | {row['prequential_nll']:<18.4f} | {row['perplexity']:<14.2f} | {row['accepted_collisions']:<12} | {row['targets_per_second']:<10.1f}")
    print("=" * 80)
    if fit:
        print(f"Scaling Law Fit: L(N) = {fit.get('l_inf', 0):.4f} + {fit.get('A', 0):.4f} * N^(-{fit.get('alpha', 0):.4f}) (R^2 = {fit.get('R2', 0):.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
