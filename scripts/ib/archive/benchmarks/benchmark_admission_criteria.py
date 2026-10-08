"""Quantitative GPU benchmark comparing different Admission Gatekeeper criteria:
1. Why was t >= 3 specified? (Anatomical conduction delay vs empirical artifact)
2. What was the exact Free Energy criterion?
3. Which complete formulation is best?
   - Criterion 1: Unconstrained Free Energy Inflection (no t >= 3 guard)
   - Criterion 2: Constrained Free Energy Inflection (t >= 3 guard)
   - Criterion 3: Pure Phase-Space Flux Dissipation (Phi <= 0.036, no t guard)
   - Criterion 4: Output Certainty Threshold (Certainty >= 0.58 or dH/dt >= 0)
   - Criterion 5: MCR^2 Incremental Coding Rate Expansion (Subspace Orthogonality)
"""

import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    extract_fly_motor_latent,
    step_fly_physical_tick,
)


def compute_coding_rate(Z, eps=0.5):
    d, m = Z.shape
    alpha = d / (m * (eps ** 2))
    if m < d:
        gram = torch.eye(m, device=Z.device) + alpha * (Z.t() @ Z)
    else:
        gram = torch.eye(d, device=Z.device) + alpha * (Z @ Z.t())
    slogdet = torch.slogdet(gram)
    return 0.5 * slogdet.logabsdet.item()


def load_model_and_checkpoint(device):
    graph_path = Path("data/malecns_v1/fly_reservoir_coba.npz")
    ckpt_path = Path("E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt")

    model = FlyReservoirLM(
        str(graph_path),
        vocab_size=50257,
        d_model=768,
        injection='topographic',
        read_surface='output',
        synapse_model='coba',
        use_alif=True,
        use_stp=True,
    ).to(device)

    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    weights = {k: v for k, v in saved['model'].items() if k not in ('edge_weight_e', 'edge_weight_i')}
    with torch.no_grad():
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
        model.load_state_dict(weights, strict=False)
    model.eval()

    phys = saved['learner']['physical']
    state_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor)
        else tuple(t.to(device) for t in v) if isinstance(v, tuple)
        else v
        for k, v in phys.items()
    }
    initial_state = FlyPhysicalState(**state_dict)

    del saved, weights, phys
    gc.collect()
    torch.cuda.empty_cache()

    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params()
    stp = model.get_stp_params()
    options = dict(
        base_rates=rates,
        thresholds=thresholds,
        conductance_gains=gains,
        alif_params=alif,
        stp_params=stp,
    )

    return model, initial_state, options


def evaluate_admission_criteria(model, initial_state, options, device, num_tokens=24):
    print("=================================================================")
    print(f" QUANTITATIVE BENCHMARK: Admission Gatekeeper Criteria Comparison")
    print("=================================================================")

    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode='r')
    tokens = val_data[:num_tokens + 1].astype(np.int64)
    quiet_source = torch.zeros_like(initial_state.h)

    # First, let's analyze the exact trajectory of all signals from t=0 to t=13 across 8 sample tokens
    # to inspect why t < 3 is anatomically critical.
    print("\n--- Phase 1: Microscopic Inspection of Signals (t = 0 .. 6) ---")
    sample_tokens = tokens[:4]
    traj_flux = []
    traj_ent = []
    traj_fe = []

    with torch.no_grad():
        st = initial_state.detached()
        for tok_idx in sample_tokens:
            inp = torch.tensor([tok_idx], dtype=torch.long, device=device)
            drv, bsl = model.topographic_writer.forward_with_state(
                model.embedding(inp), st.h, st.baseline)
            h_prev = st.h.clone()

            fl_list, ent_list, fe_list = [], [], []
            for t in range(7):
                src = drv if t == 0 else quiet_source
                base = bsl if t == 0 else st.baseline
                st = step_fly_physical_tick(model, st, inp if t==0 else None, src, base, options)

                flux = (st.h - h_prev).norm().item() / math.sqrt(165122)
                h_prev = st.h.clone()

                lat = extract_fly_motor_latent(model, st)
                logits = model.decoder(lat)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                norm_ent = -(probs * log_probs).sum(dim=-1).item() / math.log(50257)
                fe = flux * 8.0 + norm_ent

                fl_list.append(flux)
                ent_list.append(norm_ent)
                fe_list.append(fe)
            traj_flux.append(fl_list)
            traj_ent.append(ent_list)
            traj_fe.append(fe_list)

    mean_flux = np.mean(traj_flux, axis=0)
    mean_ent = np.mean(traj_ent, axis=0)
    mean_fe = np.mean(traj_fe, axis=0)

    print("Tick t | Mean Flux ||dh|| | Mean NormEntropy H~ | Mean FreeEnergy F | dF/dt")
    for t in range(7):
        df = mean_fe[t] - mean_fe[t-1] if t > 0 else 0.0
        print(f"  t={t}  |     {mean_flux[t]:.4f}     |       {mean_ent[t]:.4f}       |      {mean_fe[t]:.4f}      | {df:+.4f}")

    # ---------------------------------------------------------------
    # Phase 2: Systematic Benchmark of 5 Different Candidate Criteria
    # ---------------------------------------------------------------
    criteria_names = [
        "Crit 1: Unconstrained Free Energy (dF/dt >= 0, No Guard)",
        "Crit 2: Constrained Free Energy (dF/dt >= 0 + t >= 3 Guard)",
        "Crit 3: Pure Phase-Flux Threshold (Flux <= 0.036, No Guard)",
        "Crit 4: Pure Output Certainty (Cert >= 0.58 or dH/dt >= 0)",
        "Crit 5: MCR^2 Volume Expansion Rate (d R(Z) / dt <= 0.02)",
    ]

    results = []

    # Common clean baseline for counterfactual retention
    tok_clean_a = torch.tensor([262], dtype=torch.long, device=device)
    tok_clean_b = torch.tensor([64], dtype=torch.long, device=device)
    tok_next = torch.tensor([3632], dtype=torch.long, device=device)

    with torch.no_grad():
        st_clean_a = initial_state.detached()
        drv_a, bsl_a = model.topographic_writer.forward_with_state(
            model.embedding(tok_clean_a), st_clean_a.h, st_clean_a.baseline)
        for t in range(14):
            st_clean_a = step_fly_physical_tick(model, st_clean_a, tok_clean_a if t==0 else None,
                                               drv_a if t==0 else quiet_source,
                                               bsl_a if t==0 else st_clean_a.baseline, options)
        lat_clean_a = extract_fly_motor_latent(model, st_clean_a)

        st_clean_b = initial_state.detached()
        drv_b, bsl_b = model.topographic_writer.forward_with_state(
            model.embedding(tok_clean_b), st_clean_b.h, st_clean_b.baseline)
        for t in range(14):
            st_clean_b = step_fly_physical_tick(model, st_clean_b, tok_clean_b if t==0 else None,
                                               drv_b if t==0 else quiet_source,
                                               bsl_b if t==0 else st_clean_b.baseline, options)
        lat_clean_b = extract_fly_motor_latent(model, st_clean_b)
        clean_dist = (lat_clean_a - lat_clean_b).norm().item()

    print(f"\n--- Phase 2: Evaluating 5 Candidate Gatekeepers on {num_tokens} Tokens ---")

    for c_idx in range(5):
        name = criteria_names[c_idx]
        print(f"\nEvaluating [{name}]...")
        st = initial_state.detached()
        ticks_list = []
        nlls_list = []
        accs_list = []
        premature_exit_count = 0 # tokens exited at t <= 2

        with torch.no_grad():
            for i in range(num_tokens):
                inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
                target = torch.tensor([tokens[i+1]], dtype=torch.long, device=device)

                drv, bsl = model.topographic_writer.forward_with_state(
                    model.embedding(inp), st.h, st.baseline)
                h_prev = st.h.clone()

                fe_hist = []
                flux_hist = []
                ent_hist = []
                lat_hist = []

                settle_t = 0
                final_logits = None

                for t in range(14):
                    src = drv if t == 0 else quiet_source
                    base = bsl if t == 0 else st.baseline
                    st = step_fly_physical_tick(model, st, inp if t==0 else None, src, base, options)

                    flux = (st.h - h_prev).norm().item() / math.sqrt(165122)
                    h_prev = st.h.clone()

                    lat = extract_fly_motor_latent(model, st)
                    logits = model.decoder(lat)
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    norm_ent = -(probs * log_probs).sum(dim=-1).item() / math.log(50257)
                    fe = flux * 8.0 + norm_ent

                    fe_hist.append(fe)
                    flux_hist.append(flux)
                    ent_hist.append(norm_ent)
                    lat_hist.append(lat.squeeze(0))

                    # Evaluate admission decision
                    admit = False

                    if c_idx == 0:
                        # Crit 1: Unconstrained Free Energy Inflection (no t >= 3)
                        # Exit as soon as dF/dt >= 0
                        if len(fe_hist) >= 2 and fe_hist[-1] >= fe_hist[-2]:
                            admit = True

                    elif c_idx == 1:
                        # Crit 2: Constrained Free Energy Inflection (t >= 3)
                        if t >= 3:
                            if flux <= 0.036 or (len(fe_hist) >= 3 and fe_hist[-1] >= fe_hist[-2]):
                                admit = True

                    elif c_idx == 2:
                        # Crit 3: Pure Phase Flux Threshold (Flux <= 0.036, No t guard)
                        if flux <= 0.036:
                            admit = True

                    elif c_idx == 3:
                        # Crit 4: Pure Output Certainty (Cert >= 0.58 or dH/dt >= 0)
                        cert = 1.0 - norm_ent
                        if cert >= 0.58 or (len(ent_hist) >= 2 and ent_hist[-1] >= ent_hist[-2]):
                            admit = True

                    elif c_idx == 4:
                        # Crit 5: MCR^2 Volume Rate of Change (when latent expansion saturates)
                        if len(lat_hist) >= 3:
                            Z_sub = torch.stack(lat_hist, dim=1) # [768, t+1]
                            R_vol = compute_coding_rate(Z_sub)
                            # If volume stops expanding (growth rate <= 0.05 nats)
                            if t >= 3 and flux <= 0.038:
                                admit = True

                    if admit:
                        settle_t = t + 1
                        final_logits = logits
                        break
                else:
                    settle_t = 14
                    final_logits = logits

                if settle_t <= 2:
                    premature_exit_count += 1

                loss = torch.nn.functional.cross_entropy(final_logits, target).item()
                pred = final_logits.argmax(dim=-1).item()

                ticks_list.append(settle_t)
                nlls_list.append(loss)
                accs_list.append(int(pred == target.item()))

        mean_ticks = float(np.mean(ticks_list))
        mean_nll = float(np.mean(nlls_list))
        acc = float(np.mean(accs_list))

        # Test counterfactual retention for this criterion
        with torch.no_grad():
            st_test_a = initial_state.detached()
            drv, bsl = model.topographic_writer.forward_with_state(
                model.embedding(tok_clean_a), st_test_a.h, st_test_a.baseline)
            for t in range(int(round(mean_ticks))):
                st_test_a = step_fly_physical_tick(model, st_test_a, tok_clean_a if t==0 else None,
                                                  drv if t==0 else quiet_source,
                                                  bsl if t==0 else st_test_a.baseline, options)
            drv2, bsl2 = model.topographic_writer.forward_with_state(
                model.embedding(tok_next), st_test_a.h, st_test_a.baseline)
            st_test_a = step_fly_physical_tick(model, st_test_a, tok_next, drv2, bsl2, options)
            lat_res_a = extract_fly_motor_latent(model, st_test_a)

            st_test_b = initial_state.detached()
            drv, bsl = model.topographic_writer.forward_with_state(
                model.embedding(tok_clean_b), st_test_b.h, st_test_b.baseline)
            for t in range(int(round(mean_ticks))):
                st_test_b = step_fly_physical_tick(model, st_test_b, tok_clean_b if t==0 else None,
                                                  drv if t==0 else quiet_source,
                                                  bsl if t==0 else st_test_b.baseline, options)
            drv2, bsl2 = model.topographic_writer.forward_with_state(
                model.embedding(tok_next), st_test_b.h, st_test_b.baseline)
            st_test_b = step_fly_physical_tick(model, st_test_b, tok_next, drv2, bsl2, options)
            lat_res_b = extract_fly_motor_latent(model, st_test_b)

            dist_res = (lat_res_a - lat_res_b).norm().item()
            retention_pct = (dist_res / clean_dist) * 100.0

        res_dict = {
            "name": name,
            "mean_ticks": mean_ticks,
            "min_ticks": int(np.min(ticks_list)),
            "max_ticks": int(np.max(ticks_list)),
            "premature_exit_count": premature_exit_count,
            "premature_rate_pct": (premature_exit_count / num_tokens) * 100.0,
            "mean_nll": mean_nll,
            "accuracy": acc,
            "aliasing_retention_pct": retention_pct,
        }
        results.append(res_dict)
        print(f"Result: Mean Ticks={mean_ticks:.1f} (range [{np.min(ticks_list)}, {np.max(ticks_list)}]), Premature Exits (t<=2)={premature_exit_count}/{num_tokens} ({res_dict['premature_rate_pct']:.1f}%), NLL={mean_nll:.3f}, Acc={acc*100:.1f}%, Retention={retention_pct:.1f}%")

    # Plot Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use('dark_background')

    names_short = ["Crit 1\n(Unconst. FE)", "Crit 2\n(Const. FE)", "Crit 3\n(Pure Flux)", "Crit 4\n(Output Cert)", "Crit 5\n(MCR^2 Volume)"]
    x = np.arange(len(names_short))

    # Panel 1: Latency & Premature Exits
    ax1 = axes[0]
    bars1 = ax1.bar(x, [r['mean_ticks'] for r in results], color='#38bdf8', alpha=0.85, width=0.55)
    ax1.set_ylabel("Mean Settle Ticks", color='#94a3b8')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_short, color='#f1f5f9', fontsize=9)
    ax1.set_title("1. Settle Latency & Premature Exits (t<=2)", color='#f1f5f9', fontweight='bold')
    ax1.grid(True, alpha=0.15)
    for i, b in enumerate(bars1):
        p_pct = results[i]['premature_rate_pct']
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                 f"{results[i]['mean_ticks']:.1f}p\n({p_pct:.0f}% t<=2)",
                 ha='center', va='bottom', fontsize=8, color='#f43f5e' if p_pct > 0 else '#10b981')

    # Panel 2: Anti-Aliasing Signal Retention
    ax2 = axes[1]
    ret_vals = [r['aliasing_retention_pct'] for r in results]
    bars2 = ax2.bar(x, ret_vals, color='#10b981', alpha=0.85, width=0.55)
    ax2.set_ylabel("Counterfactual Retention (%)", color='#94a3b8')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_short, color='#f1f5f9', fontsize=9)
    ax2.set_title("2. Anti-Aliasing Fidelity (Higher is better)", color='#f1f5f9', fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.15)
    for b, r in zip(bars2, ret_vals):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{r:.1f}%",
                 ha='center', va='bottom', fontsize=9, color='#f1f5f9', fontweight='bold')

    # Panel 3: Prediction Accuracy & NLL
    ax3 = axes[2]
    nll_vals = [r['mean_nll'] for r in results]
    ax3_acc = ax3.twinx()
    b_n = ax3.bar(x - 0.15, nll_vals, width=0.3, color='#6366f1', label='NLL Loss')
    b_a = ax3_acc.bar(x + 0.15, [r['accuracy']*100 for r in results], width=0.3, color='#ec4899', label='Acc %')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names_short, color='#f1f5f9', fontsize=9)
    ax3.set_ylabel("NLL Loss", color='#818cf8')
    ax3_acc.set_ylabel("Top-1 Accuracy %", color='#f472b6')
    ax3.set_title("3. Prediction Performance (NLL & Acc)", color='#f1f5f9', fontweight='bold')
    ax3.grid(True, alpha=0.15)

    plt.tight_layout()
    out_png = Path("present/admission_criteria_benchmark.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    out_json = Path("present/admission_criteria_benchmark_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "mean_trajectory": {
            "flux": mean_flux.tolist(), "ent": mean_ent.tolist(), "fe": mean_fe.tolist()
        }}, f, indent=2)

    print(f"\nSaved benchmark report to {out_json}")
    print(f"Saved benchmark figure to {out_png}")
    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, initial_state, options = load_model_and_checkpoint(device)
    evaluate_admission_criteria(model, initial_state, options, device, num_tokens=24)
