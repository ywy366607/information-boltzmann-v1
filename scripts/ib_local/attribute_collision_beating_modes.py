"""Harmonic and Beating Attribution: Collision SO(124) Eigenfrequencies vs NLL FFT Peaks.

Mathematical Framework:
In the spatial DC subspace (k=0), Transport is identity (omega(0) = 0).
The field undergoes unitary rotations in the 124-dimensional nullspace driven by M_coll in SO(124).
Each 2D invariant eigenplane rotates as z_j(k) = A_j e^{i omega_j k} where omega_j = phi_j.

Because the Readout network is nonlinear and extracts second-moment variances e = (alpha * (v - r)^2):
Readout acts as a quadratic/nonlinear mixer, generating:
1. Fundamental frequencies: f_j = phi_j / (2*pi).
2. Harmonics: 2*f_j (from squaring), 3*f_j (from cubic nonlinearities).
3. Difference beatings: |f_i - f_j| (from cross-mode variance mixing).
4. Sum beatings: f_i + f_j (from cross-mode product mixing).

This script:
1. Loads the 61 exact positive eigenangles phi_j of M_coll.
2. Constructs the full candidate frequency dictionary (fundamentals, harmonics, sum/diff beatings).
3. Matches each of the top 8 NLL FFT peaks against the candidate dictionary.
4. Identifies the exact driving mode pairs (i, j) and computes relative matching errors.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import json
import numpy as np


def main():
    audit_path = Path("results/collision_eigenspectrum_and_1024fft_audit.json")
    if not audit_path.exists():
        raise FileNotFoundError(f"Audit file not found: {audit_path}")

    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eigen_angles = np.array(data["collision_audit"]["eigen_angles_rad"])
    fft_peaks = data["fft_1024_peaks"][:8]

    # Fundamental frequencies f_j = phi_j / (2*pi) in cycles/microstep
    f_fund = eigen_angles / (2.0 * np.pi)
    num_modes = len(f_fund)

    candidates = []

    # 1. Fundamentals
    for i in range(num_modes):
        candidates.append({
            "freq": float(f_fund[i]),
            "period": float(1.0 / (f_fund[i] + 1e-12)),
            "type": "Fundamental",
            "desc": f"f_{i} (T={1.0/f_fund[i]:.1f})",
            "modes": [i],
        })

    # 2. Harmonics: 2*f_j, 3*f_j
    for i in range(num_modes):
        f2 = 2.0 * f_fund[i]
        candidates.append({
            "freq": float(f2),
            "period": float(1.0 / (f2 + 1e-12)),
            "type": "2nd Harmonic",
            "desc": f"2*f_{i} (T_base={1.0/f_fund[i]:.1f})",
            "modes": [i],
        })
        f3 = 3.0 * f_fund[i]
        candidates.append({
            "freq": float(f3),
            "period": float(1.0 / (f3 + 1e-12)),
            "type": "3rd Harmonic",
            "desc": f"3*f_{i} (T_base={1.0/f_fund[i]:.1f})",
            "modes": [i],
        })

    # 3. Difference Beatings: |f_i - f_j|
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            f_diff = abs(f_fund[i] - f_fund[j])
            if f_diff > 1e-6:
                candidates.append({
                    "freq": float(f_diff),
                    "period": float(1.0 / f_diff),
                    "type": "Difference Beating",
                    "desc": f"|f_{i} - f_{j}| (T_i={1.0/f_fund[i]:.1f}, T_j={1.0/f_fund[j]:.1f})",
                    "modes": [i, j],
                })

    # 4. Sum Beatings: f_i + f_j
    for i in range(num_modes):
        for j in range(i, num_modes):
            f_sum = f_fund[i] + f_fund[j]
            candidates.append({
                "freq": float(f_sum),
                "period": float(1.0 / (f_sum + 1e-12)),
                "type": "Sum Beating",
                "desc": f"f_{i} + f_{j} (T_i={1.0/f_fund[i]:.1f}, T_j={1.0/f_fund[j]:.1f})",
                "modes": [i, j],
            })

    print("\n" + "=" * 110)
    print("      COLLISION SO(124) EIGENFREQUENCY -> NLL FFT PEAK ATTRIBUTION TABLE")
    print("=" * 110)
    print(f"{'FFT Peak':<18} | {'FFT Freq':<12} | {'Best Match Type':<20} | {'Matched Theory':<30} | {'Rel Err %'}")
    print("-" * 110)

    matched_results = []
    for p in fft_peaks:
        target_f = p["freq"]
        target_T = p["period_microsteps"]
        power = p["power"]

        # Find closest candidate
        best_cand = min(candidates, key=lambda c: abs(c["freq"] - target_f))
        rel_err = abs(best_cand["freq"] - target_f) / target_f * 100.0

        match_info = {
            "fft_period_steps": target_T,
            "fft_freq": target_f,
            "fft_power": power,
            "matched_type": best_cand["type"],
            "matched_desc": best_cand["desc"],
            "matched_freq": best_cand["freq"],
            "matched_period_steps": best_cand["period"],
            "rel_err_pct": rel_err,
        }
        matched_results.append(match_info)

        fft_label = f"T = {target_T:5.1f} steps"
        fft_f_str = f"{target_f:.6f}"
        matched_t_str = f"{best_cand['desc']} (T={best_cand['period']:.1f})"
        print(f"{fft_label:<18} | {fft_f_str:<12} | {best_cand['type']:<20} | {matched_t_str:<30} | {rel_err:5.2f}%")

    print("=" * 110)

    # Save output
    out_file = Path("results/collision_frequency_attribution_table.json")
    out_file.write_text(json.dumps(matched_results, indent=2), encoding="utf-8")
    print(f"\nAttribution table saved to {out_file}")


if __name__ == "__main__":
    main()
