"""Empirical GPU verification for:
1. Hierarchical Clocks (Adaptive Certainty-Triggered Exit vs Fixed 14-Tick Settling)
3. Differential Axonal Latency Compensation & Temporal Binding Window (Neck vs Foot).

Runs on GPU using trained MaleCNS connectome model (q8_fly_ctm_settle14_100k).
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


def load_model_and_channels(device):
    print("1. Loading MaleCNS connectome and model architecture...")
    graph_path = Path("data/malecns_v1/fly_reservoir_coba.npz")
    ckpt_path = Path("E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt")

    # Load architecture directly to CUDA
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

    print("2. Loading checkpoint weights with memory mapping...")
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

    # Anatomical sensory and central channels
    npz = np.load(graph_path, allow_pickle=True)
    sc_names = npz['superclass_names']
    sc_ids = npz['superclass_id']
    coords = npz['coords_um']

    neck_idx = np.where(sc_names[sc_ids] == 'cb_sensory')[0]
    foot_idx = np.where(sc_names[sc_ids] == 'vnc_sensory')[0]
    central_idx = np.where(sc_names[sc_ids] == 'cb_intrinsic')[0]

    print(f"Loaded Neck sensory (cb_sensory): {len(neck_idx):,d} neurons, mean Z={coords[neck_idx, 2].mean():.1f} μm")
    print(f"Loaded Foot sensory (vnc_sensory): {len(foot_idx):,d} neurons, mean Z={coords[foot_idx, 2].mean():.1f} μm")
    print(f"Loaded Central Brain (cb_intrinsic): {len(central_idx):,d} neurons, mean Z={coords[central_idx, 2].mean():.1f} μm")

    channels = {
        'neck_idx': neck_idx,
        'foot_idx': foot_idx,
        'central_idx': central_idx,
        'coords': coords,
    }
    return model, initial_state, channels


def run_experiment_1_hierarchical_clocks(model, initial_state, device, num_tokens=64):
    """Experiment 1: Fixed 14 Settle vs Zero Settle vs Adaptive Certainty-Triggered Exit."""
    print(f"\n=======================================================")
    print(f" EXPERIMENT 1: Hierarchical Clocks (Adaptive vs Fixed)")
    print(f"=======================================================")

    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode='r')
    tokens = val_data[:num_tokens + 1].astype(np.int64)

    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params()
    stp = model.get_stp_params()

    current_state = initial_state

    # Data logs across tokens
    fixed_0_nll, fixed_0_acc = [], []
    fixed_14_nll, fixed_14_acc = [], []
    adaptive_nll, adaptive_acc, adaptive_ticks = [], [], []

    # Threshold for adaptive exit: exit when certainty reaches 0.58 or peak
    certainty_threshold = 0.58

    print(f"Streaming {num_tokens} validation tokens through connectome...")
    with torch.no_grad():
        for i in range(num_tokens):
            inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[i+1]], dtype=torch.long, device=device)

            # Advance 1 input event + 14 settle ticks
            next_state, tick_states = advance_fly_input_event(
                model, current_state, inp,
                settle_ticks=14,
                writer_baseline_clock='input',
                base_rates=rates, thresholds=thresholds,
                conductance_gains=gains, alif_params=alif,
                stp_params=stp, return_ticks=True,
            )

            # Evaluate each tick
            tick_nlls = []
            tick_certs = []
            tick_preds = []

            for st in tick_states:
                lat = extract_fly_motor_latent(model, st)
                logits = model.decoder(lat)
                loss = torch.nn.functional.cross_entropy(logits, target).item()
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).item() / np.log(50257)
                cert = 1.0 - entropy
                pred = logits.argmax(dim=-1).item()

                tick_nlls.append(loss)
                tick_certs.append(cert)
                tick_preds.append(pred)

            # 1. Fixed 0 ticks
            fixed_0_nll.append(tick_nlls[0])
            fixed_0_acc.append(int(tick_preds[0] == target.item()))

            # 2. Fixed 14 ticks
            fixed_14_nll.append(tick_nlls[14])
            fixed_14_acc.append(int(tick_preds[14] == target.item()))

            # 3. Adaptive Exit Time: first tick where cert >= threshold, or peak certainty
            exit_tick = 14
            for t_idx, c in enumerate(tick_certs):
                if c >= certainty_threshold:
                    exit_tick = t_idx
                    break
            else:
                # If never reaches threshold, take local peak
                exit_tick = int(np.argmax(tick_certs))

            adaptive_ticks.append(exit_tick)
            adaptive_nll.append(tick_nlls[exit_tick])
            adaptive_acc.append(int(tick_preds[exit_tick] == target.item()))

            # Commit physical state
            current_state = next_state

    # Summary
    f0_mean_nll = float(np.mean(fixed_0_nll))
    f0_mean_acc = float(np.mean(fixed_0_acc)) * 100.0

    f14_mean_nll = float(np.mean(fixed_14_nll))
    f14_mean_acc = float(np.mean(fixed_14_acc)) * 100.0

    ad_mean_nll = float(np.mean(adaptive_nll))
    ad_mean_acc = float(np.mean(adaptive_acc)) * 100.0
    ad_mean_tick = float(np.mean(adaptive_ticks))

    tick_dist = np.bincount(adaptive_ticks, minlength=15)

    print(f"\n--- Experiment 1 Results (N = {num_tokens} tokens) ---")
    print(f"Strategy                  | Avg Settle Ticks | Mean NLL | Top-1 Accuracy (%)")
    print(f"----------------------------------------------------------------------------")
    print(f"Zero Settle (Fixed t=0)   |      0.0 ticks   |  {f0_mean_nll:7.3f} |     {f0_mean_acc:5.1f}%")
    print(f"Fixed Settling (Fixed t=14)|    14.0 ticks   |  {f14_mean_nll:7.3f} |     {f14_mean_acc:5.1f}%")
    print(f"Adaptive Certainty-Exit   |      {ad_mean_tick:4.1f} ticks   |  {ad_mean_nll:7.3f} |     {ad_mean_acc:5.1f}%")
    print(f"\nAdaptive exit tick distribution (0..14):")
    for t_val in range(15):
        if tick_dist[t_val] > 0:
            pct = tick_dist[t_val] / num_tokens * 100.0
            print(f"  t={t_val:2d}: {tick_dist[t_val]:2d} tokens ({pct:5.1f}%)")

    return {
        'fixed_0': {'latency': 0.0, 'nll': f0_mean_nll, 'acc': f0_mean_acc},
        'fixed_14': {'latency': 14.0, 'nll': f14_mean_nll, 'acc': f14_mean_acc},
        'adaptive': {'latency': ad_mean_tick, 'nll': ad_mean_nll, 'acc': ad_mean_acc, 'distribution': tick_dist.tolist()},
    }


def run_experiment_3_temporal_binding(model, initial_state, channels, device):
    """Experiment 3: Differential Conduction Delays (Neck vs Foot) & Temporal Binding Window."""
    print(f"\n=======================================================")
    print(f" EXPERIMENT 3: Differential Latency & Temporal Binding")
    print(f"=======================================================")

    neck_idx = channels['neck_idx']
    foot_idx = channels['foot_idx']
    central_idx = torch.as_tensor(channels['central_idx'], device=device)

    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params()
    stp = model.get_stp_params()
    options = dict(base_rates=rates, thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)

    quiet_source = torch.zeros_like(initial_state.h)

    # 1. Measure Isolated Impulse Response for Neck vs Foot
    sim_ticks = 16
    def measure_impulse_response(inject_indices, pulse_duration=2):
        state = initial_state.detached()
        central_activity = []
        central_spikes = []
        for t in range(sim_ticks):
            source = quiet_source.clone()
            if t < pulse_duration and len(inject_indices) > 0:
                source[0, inject_indices] = 8.0
            state = step_fly_physical_tick(model, state, None, source, state.baseline, options)
            mean_h = state.h[0, central_idx].mean().item()
            spk_cnt = (state.ring[0][0, central_idx] > 0).sum().item()
            central_activity.append(mean_h)
            central_spikes.append(spk_cnt)
        return np.array(central_activity), np.array(central_spikes)

    print("Measuring Neck (cb_sensory) impulse response to Central Brain...")
    resp_neck, spk_neck = measure_impulse_response(neck_idx)

    print("Measuring Foot (vnc_sensory) impulse response to Central Brain...")
    resp_foot, spk_foot = measure_impulse_response(foot_idx)

    # Baselines
    baseline_resp, baseline_spk = measure_impulse_response([], pulse_duration=0)
    delta_neck = spk_neck - baseline_spk
    delta_foot = spk_foot - baseline_spk

    t_neck_peak = int(np.argmax(delta_neck))
    t_foot_peak = int(np.argmax(delta_foot))
    latency_diff = t_foot_peak - t_neck_peak

    print(f"\n--- Central Brain Spike Impulse Response ---")
    print(f"  Neck -> Central Brain Peak Latency: t = {t_neck_peak} ticks (Delta Spikes = {delta_neck.max():+d})")
    print(f"  Foot -> Central Brain Peak Latency: t = {t_foot_peak} ticks (Delta Spikes = {delta_foot.max():+d})")
    print(f"  Differential Conduction Delay Δt   : {latency_diff} physical ticks")

    # 2. Test Coincidence Detection & Temporal Binding Window across varying physical interval ΔT
    intervals = [0, 1, 2, 4, 6, 8, 10, 12]
    binding_ratios = []
    actual_peaks = []
    linear_expected = []

    print("\nTesting temporal binding across inter-stimulus intervals ΔT ∈ [0, 12] ticks...")
    for dt in intervals:
        state = initial_state.detached()
        sim_spk = []
        for t in range(sim_ticks):
            source = quiet_source.clone()
            # Neck pulse at t = 0..1
            if t < 2:
                source[0, neck_idx] = 8.0
            # Foot pulse at t = dt..dt+1
            if dt <= t < dt + 2:
                source[0, foot_idx] = 8.0

            state = step_fly_physical_tick(model, state, None, source, state.baseline, options)
            spk_cnt = (state.ring[0][0, central_idx] > 0).sum().item()
            sim_spk.append(spk_cnt)

        sim_delta = np.array(sim_spk) - baseline_spk

        # Construct linear sum (Neck at 0 + Foot shifted by dt)
        shifted_foot = np.zeros_like(delta_foot)
        if dt < len(delta_foot):
            shifted_foot[dt:] = delta_foot[:len(delta_foot)-dt]
        lin_sum = delta_neck + shifted_foot

        actual_peak = float(sim_delta.max())
        lin_peak = float(lin_sum.max())
        supralinear_ratio = (actual_peak - lin_peak) / max(lin_peak, 1.0)

        binding_ratios.append(supralinear_ratio)
        actual_peaks.append(actual_peak)
        linear_expected.append(lin_peak)

        print(f"  Interval ΔT = {dt:2d} ticks: Actual Peak = {actual_peak:.0f} spikes, Linear Sum = {lin_peak:.0f} spikes, Non-linear Binding = {supralinear_ratio*100:+.2f}%")

    return {
        't_neck_peak': t_neck_peak,
        't_foot_peak': t_foot_peak,
        'latency_diff': latency_diff,
        'delta_neck': delta_neck.tolist(),
        'delta_foot': delta_foot.tolist(),
        'intervals': intervals,
        'binding_ratios': binding_ratios,
        'actual_peaks': actual_peaks,
        'linear_expected': linear_expected,
    }


def generate_verification_figure(exp1_results, exp3_results, out_path):
    print(f"\nGenerating visual figure: {out_path}...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#071018')
    fig.suptitle(
        "Empirical GPU Verification: Hierarchical Clocks and Differential Temporal Binding\n"
        "Adult Drosophila Connectome (MaleCNS v1.0, COBA ALIF+STP S=14 Checkpoint)",
        fontsize=14, fontweight='bold', color='#E0F2FE', y=0.98
    )

    # Panel 1: Experiment 1 Latency vs Accuracy Comparison
    ax1 = axes[0, 0]
    ax1.set_facecolor('#0B1924')
    ax1.grid(True, color='#1E293B', linestyle=':', alpha=0.7)

    methods = ['Zero Settle\n(Fixed t=0)', 'Adaptive Exit\n(Certainty-Triggered)', 'Fixed Settling\n(Fixed t=14)']
    latencies = [exp1_results['fixed_0']['latency'], exp1_results['adaptive']['latency'], exp1_results['fixed_14']['latency']]
    accs = [exp1_results['fixed_0']['acc'], exp1_results['adaptive']['acc'], exp1_results['fixed_14']['acc']]
    nlls = [exp1_results['fixed_0']['nll'], exp1_results['adaptive']['nll'], exp1_results['fixed_14']['nll']]

    bars = ax1.bar(methods, latencies, color=['#64748B', '#38BDF8', '#F59E0B'], width=0.5, alpha=0.9)
    for bar, lat, acc in zip(bars, latencies, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{lat:.1f} ticks\n({acc:.1f}% Acc)", color='#CBD5E1', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax1.set_ylabel("Mean Physical Settling Ticks Required", color='#38BDF8', fontsize=10)
    ax1.set_ylim(0, 16.5)
    ax1.tick_params(colors='#CBD5E1', labelsize=9)
    ax1.set_title("A. Hierarchical Clock: Latency Savings", color='#38BDF8', fontsize=11, fontweight='bold', pad=8)

    # Panel 2: Experiment 1 Adaptive Exit Tick Histogram
    ax2 = axes[0, 1]
    ax2.set_facecolor('#0B1924')
    ax2.grid(True, color='#1E293B', linestyle=':', alpha=0.7)

    dist = exp1_results['adaptive']['distribution']
    ticks_x = np.arange(len(dist))
    ax2.bar(ticks_x, dist, color='#4ADE80', width=0.7, alpha=0.85)
    ax2.axvline(x=exp1_results['adaptive']['latency'], color='#FACC15', linestyle='--', linewidth=2.0, label=f"Mean Exit = {exp1_results['adaptive']['latency']:.1f} ticks")

    ax2.set_xlabel("Physical Exit Tick (t)", color='#94A3B8', fontsize=10)
    ax2.set_ylabel("Number of Validation Tokens", color='#94A3B8', fontsize=10)
    ax2.set_xticks(range(0, 15, 2))
    ax2.tick_params(colors='#CBD5E1', labelsize=9)
    ax2.set_title("B. Dynamic Decision Firing Distribution", color='#4ADE80', fontsize=11, fontweight='bold', pad=8)
    leg2 = ax2.legend(facecolor='#071018', edgecolor='#1E293B')
    for text in leg2.get_texts():
        text.set_color('#CBD5E1')

    # Panel 3: Experiment 3 Differential Latency (Neck vs Foot Impulse Response)
    ax3 = axes[1, 0]
    ax3.set_facecolor('#0B1924')
    ax3.grid(True, color='#1E293B', linestyle=':', alpha=0.7)

    d_neck = exp3_results['delta_neck']
    d_foot = exp3_results['delta_foot']
    t_axis = np.arange(len(d_neck))

    ax3.plot(t_axis, d_neck, color='#00E5FF', linewidth=2.5, marker='o', label=f"Neck (cb_sensory, Peak t={exp3_results['t_neck_peak']})")
    ax3.plot(t_axis, d_foot, color='#FF9100', linewidth=2.5, marker='s', label=f"Foot (vnc_sensory, Peak t={exp3_results['t_foot_peak']})")

    ax3.axvspan(exp3_results['t_neck_peak'], exp3_results['t_foot_peak'], color='#E040FB', alpha=0.15, label=f"Δt Delay = {exp3_results['latency_diff']} ticks")

    ax3.set_xlabel("Physical Ticks After Impulse", color='#94A3B8', fontsize=10)
    ax3.set_ylabel("Central Brain Evoked Spikes (ΔSpikes)", color='#94A3B8', fontsize=10)
    ax3.set_xticks(range(0, 16, 2))
    ax3.tick_params(colors='#CBD5E1', labelsize=9)
    ax3.set_title("C. Differential Axonal Latency (Neck vs Foot)", color='#38BDF8', fontsize=11, fontweight='bold', pad=8)
    leg3 = ax3.legend(facecolor='#071018', edgecolor='#1E293B')
    for text in leg3.get_texts():
        text.set_color('#CBD5E1')

    # Panel 4: Experiment 3 Temporal Binding Window (TBW Curve)
    ax4 = axes[1, 1]
    ax4.set_facecolor('#0B1924')
    ax4.grid(True, color='#1E293B', linestyle=':', alpha=0.7)

    intervals = exp3_results['intervals']
    ratios = [r * 100 for r in exp3_results['binding_ratios']]

    ax4.plot(intervals, ratios, color='#E040FB', linewidth=2.8, marker='D', markersize=7, label="Non-linear Binding Gain (%)")
    ax4.axhline(y=0, color='#64748B', linestyle='--', linewidth=1.2)

    # Highlight TBW (where binding gain is strongest)
    ax4.fill_between(intervals, 0, ratios, where=[r > 0 for r in ratios], color='#E040FB', alpha=0.2, label="Temporal Binding Window (TBW)")

    ax4.set_xlabel("External Inter-Stimulus Interval ΔT (Ticks between Neck & Foot)", color='#94A3B8', fontsize=10)
    ax4.set_ylabel("Binding Non-linearity Gain over Linear Sum (%)", color='#E040FB', fontsize=10)
    ax4.set_xticks(intervals)
    ax4.tick_params(colors='#CBD5E1', labelsize=9)
    ax4.set_title("D. Temporal Binding Window (Coincidence Detection)", color='#E040FB', fontsize=11, fontweight='bold', pad=8)
    leg4 = ax4.legend(facecolor='#071018', edgecolor='#1E293B')
    for text in leg4.get_texts():
        text.set_color('#CBD5E1')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Figure saved to: {out_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running validation on device: {device}")

    model, initial_state, channels = load_model_and_channels(device)

    # Run Experiment 1
    exp1_results = run_experiment_1_hierarchical_clocks(model, initial_state, device, num_tokens=64)

    # Run Experiment 3
    exp3_results = run_experiment_3_temporal_binding(model, initial_state, channels, device)

    # Generate Figure
    fig_path = Path("present/verification_hierarchical_clock_and_binding.png")
    generate_verification_figure(exp1_results, exp3_results, fig_path)

    # Save summary report JSON
    report = {
        'experiment_1_hierarchical_clocks': exp1_results,
        'experiment_3_temporal_binding': exp3_results,
        'figure_path': str(fig_path),
    }
    report_path = Path("present/verification_report.json")
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nAll validations complete! Report saved to {report_path}")


if __name__ == '__main__':
    main()
