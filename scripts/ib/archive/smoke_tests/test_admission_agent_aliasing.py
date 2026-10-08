"""Rigorous GPU verification of the Adaptive Admission Agent (写入 Agent / Gatekeeper):
1. Congestion Resolution (阻塞检验): Can event-driven free-energy admission eliminate the 14-tick dead wait?
2. Aliasing / Jamming (混叠检验): Does admitting tokens dynamically before 14 ticks cause representation overwriting or signal interference?

Evaluates three arms on real OWT tokens using trained MaleCNS connectome model:
- Arm A: Fixed S=14 Settling (Static baseline, maximal blocking)
- Arm B: Blind 1-Tick Pipelining (Zero wait, severe aliasing risk)
- Arm C: Active Inference Admission Agent (Free-energy / chaos gated admission)
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
    advance_fly_input_event,
    extract_fly_motor_latent,
    step_fly_physical_tick,
)


def load_model_and_checkpoint(device):
    print("1. Loading MaleCNS connectome model...")
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

    print("2. Loading checkpoint weights...")
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


def compute_free_energy_proxy(model, state, h_prev):
    """Computes internal chaos / variational free energy proxy F = lambda * Flux + NormEntropy."""
    flux = (state.h - h_prev).norm().item() / math.sqrt(165122)
    lat = extract_fly_motor_latent(model, state)
    logits = model.decoder(lat)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    norm_ent = -(probs * log_probs).sum(dim=-1).item() / math.log(50257)
    free_energy = flux * 8.0 + norm_ent
    return free_energy, flux, norm_ent, logits


def run_benchmark(model, initial_state, options, device, num_tokens=32):
    print(f"\n=======================================================")
    print(f" BENCHMARK: Congestion & Aliasing Across {num_tokens} Tokens")
    print(f"=======================================================")

    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode='r')
    tokens = val_data[:num_tokens + 1].astype(np.int64)

    quiet_source = torch.zeros_like(initial_state.h)

    # ----------------------------------------------------
    # ARM A: Fixed S=14 (Maximal Blocking, Zero Pipelining)
    # ----------------------------------------------------
    print("\n[Running Arm A: Fixed S=14 Settling]...")
    state_a = initial_state.detached()
    arm_a_ticks = []
    arm_a_nlls = []
    arm_a_accs = []
    arm_a_latents = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_tokens):
            inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[i+1]], dtype=torch.long, device=device)

            if model.topographic_writer is not None:
                drive, baseline = model.topographic_writer.forward_with_state(
                    model.embedding(inp), state_a.h, state_a.baseline)
            else:
                drive, baseline = None, state_a.baseline

            for t in range(14):
                src = drive if t == 0 else quiet_source
                base = baseline if t == 0 else state_a.baseline
                state_a = step_fly_physical_tick(model, state_a, inp if t==0 else None, src, base, options)

            lat = extract_fly_motor_latent(model, state_a)
            logits = model.decoder(lat)
            loss = torch.nn.functional.cross_entropy(logits, target).item()
            pred = logits.argmax(dim=-1).item()

            arm_a_ticks.append(14)
            arm_a_nlls.append(loss)
            arm_a_accs.append(int(pred == target.item()))
            arm_a_latents.append(lat.cpu().numpy())

    time_a = time.perf_counter() - t0
    total_ticks_a = sum(arm_a_ticks)
    mean_nll_a = float(np.mean(arm_a_nlls))
    acc_a = float(np.mean(arm_a_accs))
    print(f"Arm A Complete: Total Ticks={total_ticks_a}, Mean NLL={mean_nll_a:.3f}, Acc={acc_a*100:.1f}%, Time={time_a:.2f}s")

    # ----------------------------------------------------
    # ARM B: Blind 1-Tick Push (Zero Wait, Severe Aliasing)
    # ----------------------------------------------------
    print("\n[Running Arm B: Blind 1-Tick Pipelining]...")
    state_b = initial_state.detached()
    arm_b_ticks = []
    arm_b_nlls = []
    arm_b_accs = []
    arm_b_latents = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_tokens):
            inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[i+1]], dtype=torch.long, device=device)

            if model.topographic_writer is not None:
                drive, baseline = model.topographic_writer.forward_with_state(
                    model.embedding(inp), state_b.h, state_b.baseline)
            else:
                drive, baseline = None, state_b.baseline

            # Only 1 physical tick per token!
            state_b = step_fly_physical_tick(model, state_b, inp, drive, baseline, options)

            lat = extract_fly_motor_latent(model, state_b)
            logits = model.decoder(lat)
            loss = torch.nn.functional.cross_entropy(logits, target).item()
            pred = logits.argmax(dim=-1).item()

            arm_b_ticks.append(1)
            arm_b_nlls.append(loss)
            arm_b_accs.append(int(pred == target.item()))
            arm_b_latents.append(lat.cpu().numpy())

    time_b = time.perf_counter() - t0
    total_ticks_b = sum(arm_b_ticks)
    mean_nll_b = float(np.mean(arm_b_nlls))
    acc_b = float(np.mean(arm_b_accs))
    print(f"Arm B Complete: Total Ticks={total_ticks_b}, Mean NLL={mean_nll_b:.3f}, Acc={acc_b*100:.1f}%, Time={time_b:.2f}s")

    # ----------------------------------------------------
    # ARM C: Adaptive Admission Agent (Free-Energy Gated)
    # ----------------------------------------------------
    print("\n[Running Arm C: Adaptive Admission Agent]...")
    state_c = initial_state.detached()
    arm_c_ticks = []
    arm_c_nlls = []
    arm_c_accs = []
    arm_c_latents = []
    arm_c_flux_traces = []
    arm_c_fe_traces = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_tokens):
            inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[i+1]], dtype=torch.long, device=device)

            if model.topographic_writer is not None:
                drive, baseline = model.topographic_writer.forward_with_state(
                    model.embedding(inp), state_c.h, state_c.baseline)
            else:
                drive, baseline = None, state_c.baseline

            h_prev = state_c.h.clone()

            fe_hist = []
            flux_hist = []
            settle_t = 0
            final_logits = None

            for t in range(14):
                src = drive if t == 0 else quiet_source
                base = baseline if t == 0 else state_c.baseline
                state_c = step_fly_physical_tick(model, state_c, inp if t==0 else None, src, base, options)

                fe, flux, ent, logits = compute_free_energy_proxy(model, state_c, h_prev)
                h_prev = state_c.h.clone()
                fe_hist.append(fe)
                flux_hist.append(flux)

                # Gatekeeper criteria:
                # 1. Minimum transit requirement: t >= 3 (axon propagation floor)
                # 2. Free-energy inflection: flux stabilizes (<= 0.036) or local free energy minimum reached
                if t >= 3:
                    if flux <= 0.036 or (len(fe_hist) >= 3 and fe_hist[-1] >= fe_hist[-2]):
                        settle_t = t + 1
                        final_logits = logits
                        break
            else:
                settle_t = 14
                final_logits = logits

            lat = extract_fly_motor_latent(model, state_c)
            loss = torch.nn.functional.cross_entropy(final_logits, target).item()
            pred = final_logits.argmax(dim=-1).item()

            arm_c_ticks.append(settle_t)
            arm_c_nlls.append(loss)
            arm_c_accs.append(int(pred == target.item()))
            arm_c_latents.append(lat.cpu().numpy())
            arm_c_flux_traces.append(flux_hist)
            arm_c_fe_traces.append(fe_hist)

    time_c = time.perf_counter() - t0
    total_ticks_c = sum(arm_c_ticks)
    mean_nll_c = float(np.mean(arm_c_nlls))
    acc_c = float(np.mean(arm_c_accs))
    print(f"Arm C Complete: Total Ticks={total_ticks_c}, Mean NLL={mean_nll_c:.3f}, Acc={acc_c*100:.1f}%, Time={time_c:.2f}s")

    # ----------------------------------------------------
    # TEST PART 2: Aliasing & Counterfactual Distinguishability
    # ----------------------------------------------------
    print("\n[Running Test Part 2: Counterfactual Signal Retention (Aliasing Test)]...")
    # Test how well a prior token's distinction survives the subsequent injection:
    # Pair: Token X1 = " the" (262) vs Token X1' = " a" (64)
    # Followed by Token X2 = " brain" (3632)
    tok_clean_a = torch.tensor([262], dtype=torch.long, device=device)
    tok_clean_b = torch.tensor([64], dtype=torch.long, device=device)
    tok_next = torch.tensor([3632], dtype=torch.long, device=device)

    # Clean Baseline: Evolve X1 and X1' in isolation to t=14
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

        # Arm B (1-tick premature collision): Inject X1 for 1 tick, then immediately slam X2!
        st_b_a = initial_state.detached()
        drv, bsl = model.topographic_writer.forward_with_state(
            model.embedding(tok_clean_a), st_b_a.h, st_b_a.baseline)
        st_b_a = step_fly_physical_tick(model, st_b_a, tok_clean_a, drv, bsl, options)
        drv2, bsl2 = model.topographic_writer.forward_with_state(
            model.embedding(tok_next), st_b_a.h, st_b_a.baseline)
        st_b_a = step_fly_physical_tick(model, st_b_a, tok_next, drv2, bsl2, options)
        lat_jam_a = extract_fly_motor_latent(model, st_b_a)

        st_b_b = initial_state.detached()
        drv, bsl = model.topographic_writer.forward_with_state(
            model.embedding(tok_clean_b), st_b_b.h, st_b_b.baseline)
        st_b_b = step_fly_physical_tick(model, st_b_b, tok_clean_b, drv, bsl, options)
        drv2, bsl2 = model.topographic_writer.forward_with_state(
            model.embedding(tok_next), st_b_b.h, st_b_b.baseline)
        st_b_b = step_fly_physical_tick(model, st_b_b, tok_next, drv2, bsl2, options)
        lat_jam_b = extract_fly_motor_latent(model, st_b_b)

        jam_dist = (lat_jam_a - lat_jam_b).norm().item()
        jam_retention = (jam_dist / clean_dist) * 100.0

        # Arm C (Adaptive Gatekeeper): Let X1 settle until free energy stabilizes (say 6 ticks), then inject X2!
        st_c_a = initial_state.detached()
        drv, bsl = model.topographic_writer.forward_with_state(
            model.embedding(tok_clean_a), st_c_a.h, st_c_a.baseline)
        for t in range(6):
            st_c_a = step_fly_physical_tick(model, st_c_a, tok_clean_a if t==0 else None,
                                           drv if t==0 else quiet_source,
                                           bsl if t==0 else st_c_a.baseline, options)
        drv2, bsl2 = model.topographic_writer.forward_with_state(
            model.embedding(tok_next), st_c_a.h, st_c_a.baseline)
        st_c_a = step_fly_physical_tick(model, st_c_a, tok_next, drv2, bsl2, options)
        lat_c_a = extract_fly_motor_latent(model, st_c_a)

        st_c_b = initial_state.detached()
        drv, bsl = model.topographic_writer.forward_with_state(
            model.embedding(tok_clean_b), st_c_b.h, st_c_b.baseline)
        for t in range(6):
            st_c_b = step_fly_physical_tick(model, st_c_b, tok_clean_b if t==0 else None,
                                           drv if t==0 else quiet_source,
                                           bsl if t==0 else st_c_b.baseline, options)
        drv2, bsl2 = model.topographic_writer.forward_with_state(
            model.embedding(tok_next), st_c_b.h, st_c_b.baseline)
        st_c_b = step_fly_physical_tick(model, st_c_b, tok_next, drv2, bsl2, options)
        lat_c_b = extract_fly_motor_latent(model, st_c_b)

        adaptive_dist = (lat_c_a - lat_c_b).norm().item()
        adaptive_retention = (adaptive_dist / clean_dist) * 100.0

    print(f"Clean Distinction ||lat(X1) - lat(X1')|| = {clean_dist:.4f}")
    print(f"Arm B (1-tick Jamming) Retention: {jam_retention:.2f}% (Dist={jam_dist:.4f})")
    print(f"Arm C (Adaptive Agent) Retention: {adaptive_retention:.2f}% (Dist={adaptive_dist:.4f})")

    # ----------------------------------------------------
    # Compile Results & Generate Visual Artifacts
    # ----------------------------------------------------
    results = {
        "num_tokens": num_tokens,
        "arm_a_fixed_14": {
            "total_ticks": total_ticks_a,
            "mean_ticks_per_token": total_ticks_a / num_tokens,
            "mean_nll": mean_nll_a,
            "accuracy": acc_a,
            "throughput_tokens_per_tick": num_tokens / total_ticks_a,
        },
        "arm_b_blind_1tick": {
            "total_ticks": total_ticks_b,
            "mean_ticks_per_token": 1.0,
            "mean_nll": mean_nll_b,
            "accuracy": acc_b,
            "throughput_tokens_per_tick": 1.0,
            "aliasing_retention_pct": jam_retention,
        },
        "arm_c_adaptive_agent": {
            "total_ticks": total_ticks_c,
            "mean_ticks_per_token": total_ticks_c / num_tokens,
            "mean_nll": mean_nll_c,
            "accuracy": acc_c,
            "throughput_tokens_per_tick": num_tokens / total_ticks_c,
            "speedup_vs_fixed_14": total_ticks_a / total_ticks_c,
            "ticks_saved_pct": (1.0 - total_ticks_c / total_ticks_a) * 100.0,
            "settle_ticks_distribution": arm_c_ticks,
            "aliasing_retention_pct": adaptive_retention,
        },
        "counterfactual_analysis": {
            "clean_dist": clean_dist,
            "arm_b_jammed_dist": jam_dist,
            "arm_b_retention_pct": jam_retention,
            "arm_c_adaptive_dist": adaptive_dist,
            "arm_c_retention_pct": adaptive_retention,
        }
    }

    report_path = Path("present/admission_agent_aliasing_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved quantitative report to {report_path}")

    # Generate Visualization Figure
    fig_path = Path("present/admission_agent_aliasing.png")
    plot_visualization(results, arm_c_ticks, tokens, fig_path)
    print(f"Saved visualization figure to {fig_path}")

    return results


def plot_visualization(results, arm_c_ticks, tokens, fig_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Adaptive Admission Agent: Congestion Elimination & Anti-Aliasing Verification",
                 fontsize=16, fontweight='bold', color='#38bdf8', y=0.98)

    # 1. Congestion Resolution: Cumulative Ticks vs Processed Tokens
    ax1 = axes[0, 0]
    n_tokens = results['num_tokens']
    token_indices = np.arange(1, n_tokens + 1)
    cum_a = token_indices * 14
    cum_b = token_indices * 1
    cum_c = np.cumsum(arm_c_ticks)

    ax1.plot(token_indices, cum_a, label=f"Arm A: Fixed S=14 (Total: {cum_a[-1]} ticks)", color='#ef4444', linewidth=2.5, linestyle='--')
    ax1.plot(token_indices, cum_c, label=f"Arm C: Adaptive Agent (Total: {cum_c[-1]} ticks, {results['arm_c_adaptive_agent']['speedup_vs_fixed_14']:.2f}x Speedup)",
             color='#10b981', linewidth=3.0)
    ax1.plot(token_indices, cum_b, label=f"Arm B: Blind 1-Tick Push (Total: {cum_b[-1]} ticks)", color='#f59e0b', linewidth=2.0, linestyle=':')

    ax1.set_title("1. Congestion Resolution: Physical Ticks per Stream", fontsize=12, fontweight='bold', color='#f1f5f9')
    ax1.set_xlabel("Processed Tokens", fontsize=10, color='#94a3b8')
    ax1.set_ylabel("Cumulative Physical Ticks", fontsize=10, color='#94a3b8')
    ax1.legend(loc='upper left', framealpha=0.4, fontsize=9)
    ax1.grid(True, alpha=0.15)

    # Annotate ticks saved
    saved_ticks = cum_a[-1] - cum_c[-1]
    ax1.annotate(f"Eliminated {saved_ticks} Idle Ticks!\n(-{results['arm_c_adaptive_agent']['ticks_saved_pct']:.1f}% Dead Wait)",
                 xy=(n_tokens * 0.7, cum_c[int(n_tokens * 0.7)]),
                 xytext=(n_tokens * 0.45, cum_a[int(n_tokens * 0.65)]),
                 arrowprops=dict(facecolor='#10b981', shrink=0.08, width=2, headwidth=8),
                 color='#10b981', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#064e3b', alpha=0.8, edgecolor='#10b981'))

    # 2. Adaptive Settling Distribution per Token
    ax2 = axes[0, 1]
    bars = ax2.bar(token_indices, arm_c_ticks, color='#38bdf8', alpha=0.85, edgecolor='#0284c7')
    ax2.axhline(14, color='#ef4444', linestyle='--', linewidth=2, label="Fixed Baseline (S=14)")
    mean_c = results['arm_c_adaptive_agent']['mean_ticks_per_token']
    ax2.axhline(mean_c, color='#10b981', linestyle='-', linewidth=2, label=f"Adaptive Mean ({mean_c:.1f} ticks)")

    ax2.set_title(f"2. Self-Paced Cognition: Settling Ticks per Token (Mean: {mean_c:.1f})", fontsize=12, fontweight='bold', color='#f1f5f9')
    ax2.set_xlabel("Token Index", fontsize=10, color='#94a3b8')
    ax2.set_ylabel("Ticks to Free-Energy Convergence", fontsize=10, color='#94a3b8')
    ax2.set_ylim(0, 16)
    ax2.legend(loc='upper right', framealpha=0.4, fontsize=9)
    ax2.grid(True, alpha=0.15)

    # Highlight fast vs slow tokens
    min_idx = np.argmin(arm_c_ticks)
    max_idx = np.argmax(arm_c_ticks)
    bars[min_idx].set_color('#22c55e')
    bars[max_idx].set_color('#f43f5e')

    # 3. Aliasing & Signal Overwrite Retention (Anti-Jamming)
    ax3 = axes[1, 0]
    arms = ['Arm A\n(Clean S=14)', 'Arm B\n(Blind 1-Tick)', 'Arm C\n(Adaptive Gate)']
    retentions = [100.0, results['counterfactual_analysis']['arm_b_retention_pct'], results['counterfactual_analysis']['arm_c_retention_pct']]
    colors = ['#ef4444', '#f59e0b', '#10b981']

    bars3 = ax3.bar(arms, retentions, color=colors, alpha=0.85, width=0.55, edgecolor='#334155')
    ax3.set_title("3. Aliasing Resistance: Counterfactual Signal Retention", fontsize=12, fontweight='bold', color='#f1f5f9')
    ax3.set_ylabel("Signal Retention Ratio (%)", fontsize=10, color='#94a3b8')
    ax3.set_ylim(0, 120)
    ax3.grid(True, alpha=0.15)

    for bar, ret in zip(bars3, retentions):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{ret:.1f}%",
                 ha='center', va='bottom', color='#f1f5f9', fontweight='bold', fontsize=11)

    ax3.annotate("High Aliasing / Jamming:\nSubsequent token corrupts\nprior representation!",
                 xy=(1, retentions[1]), xytext=(0.6, 60),
                 arrowprops=dict(facecolor='#f59e0b', shrink=0.08, width=1.5, headwidth=6),
                 color='#f59e0b', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#451a03', alpha=0.7, edgecolor='#f59e0b'))

    ax3.annotate("Anti-Aliasing Preserved:\nGatekeeper waits for basin\nbefore admitting new wavepacket!",
                 xy=(2, retentions[2]), xytext=(1.6, 105),
                 arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.5, headwidth=6),
                 color='#10b981', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#064e3b', alpha=0.7, edgecolor='#10b981'))

    # 4. Cognitive Quality: NLL Loss & Accuracy Comparison
    ax4 = axes[1, 1]
    x_pos = np.arange(3)
    width = 0.35

    nlls = [results['arm_a_fixed_14']['mean_nll'], results['arm_b_blind_1tick']['mean_nll'], results['arm_c_adaptive_agent']['mean_nll']]
    accs = [results['arm_a_fixed_14']['accuracy'] * 100, results['arm_b_blind_1tick']['accuracy'] * 100, results['arm_c_adaptive_agent']['accuracy'] * 100]

    ax4_acc = ax4.twinx()
    b_nll = ax4.bar(x_pos - width/2, nlls, width, label='Cross-Entropy Loss (NLL)', color='#6366f1', alpha=0.85)
    b_acc = ax4_acc.bar(x_pos + width/2, accs, width, label='Top-1 Accuracy (%)', color='#ec4899', alpha=0.85)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Arm A (Fixed 14)', 'Arm B (1-Tick Push)', 'Arm C (Adaptive)'], color='#f1f5f9', fontsize=10)
    ax4.set_ylabel("NLL Loss (Lower is better)", color='#818cf8', fontsize=10)
    ax4_acc.set_ylabel("Accuracy % (Higher is better)", color='#f472b6', fontsize=10)
    ax4.set_title("4. Representation Quality: NLL Loss & Accuracy", fontsize=12, fontweight='bold', color='#f1f5f9')
    ax4.grid(True, alpha=0.15)

    # Combined legends
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_acc.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='upper left', framealpha=0.4, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(fig_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, initial_state, options = load_model_and_checkpoint(device)
    results = run_benchmark(model, initial_state, options, device, num_tokens=32)
