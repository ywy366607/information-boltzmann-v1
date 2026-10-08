"""Rigorous empirical evaluation of physical conduction delay compensation in HX-1.

Investigates three falsifiable scientific hypotheses:
1. Temporal Timestamp Alignment (14x14 Matrix): Does internal simulation step k
   maximally correlate with real physical state at physical tick tau = k?
2. Counterfactual Horizon Ablation vs Physical Settle Equivalence: Does reading out
   at k=14 match physical waiting (settle_ticks = 14) while drastically outperforming
   uncompensated immediate readout (k=0)?
3. Macro Wavefront Tracking: Does signal propagation through Node 0 -> 1 -> 2 -> 3
   in the internal simulator match the real MaleCNS physical connectome arrival times?
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState, collect_fly_quiet_trajectory,
    advance_fly_input_event, step_fly_physical_tick
)


def load_model_and_checkpoint(checkpoint_path, graph_path):
    print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = saved['config']
    
    model = FlyReservoirLM(
        Path(graph_path), vocab_size=50257, d_model=config.get('d_model', 768),
        injection='topographic', read_surface='output', synapse_model='coba',
        use_alif=True, use_stp=True, decoder_bias=config.get('decoder_bias', True),
        read_centering=config.get('read_centering', False),
        use_read_gamma_trace=config.get('use_read_gamma_trace', False),
        init_read_gamma=config.get('init_read_gamma', 'anatomical'),
        use_latent_predictor=False,
        use_graph_observer=True,
        lambda_obs=1.0, lambda_sigreg=0.2,
        max_horizon=14, dagger_beta=0.5
    ).cuda()
    
    weights = {k: v for k, v in saved['model'].items()
               if k not in ('edge_weight_e', 'edge_weight_i')}
    with torch.no_grad():
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
        model.load_state_dict(weights, strict=False)
    model.eval()
    return model, saved


def test_14x14_timestamp_alignment(model, val_tokens, num_windows=16, window_size=32):
    """Test 1: Compute 14x14 cross-horizon alignment matrix M(k, tau)."""
    print("\n" + "="*70, flush=True)
    print("TEST 1: 14x14 Temporal Timestamp Alignment Matrix M(k, tau)", flush=True)
    print("="*70, flush=True)
    
    obs = model.graph_observer
    sim_matrix = torch.zeros(14, 14, device='cuda')
    sample_count = 0
    
    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params() if model.use_alif else None
    stp = model.get_stp_params() if model.use_stp else None
    
    n = model.n_neurons
    state = FlyPhysicalState(
        torch.zeros(1, n, device='cuda'),
        tuple(torch.zeros(1, n, device='cuda') for _ in range(4)),
        torch.zeros(1, n, device='cuda'),
        torch.zeros(1, n, device='cuda'),
        torch.zeros(1, n, device='cuda'),
        torch.ones(1, n, device='cuda'),
        stp[0].clone().expand(1, n).contiguous(),
        torch.zeros(1, model.topographic_writer.n_total, device='cuda'),
        torch.zeros(1, n, device='cuda'),
        torch.empty(0, device='cuda'),
        torch.empty(0, device='cuda'),
        torch.empty(0, device='cuda')
    )
    
    with torch.no_grad():
        for win_idx in range(num_windows):
            chunk = val_tokens[win_idx * window_size : (win_idx + 1) * window_size]
            for tok in chunk:
                token_tensor = torch.tensor([tok], device='cuda', dtype=torch.long)
                state = advance_fly_input_event(
                    model, state, token_tensor, settle_ticks=0,
                    writer_baseline_clock='physical',
                    base_rates=rates, thresholds=thresholds,
                    conductance_gains=gains, alif_params=alif, stp_params=stp
                )
            
            # Now at terminal token: get internal simulation rollout k=1..14
            z0 = obs.encoders(state.h) # [1, 4, d]
            sim_hops = obs.simulate_hops(z0, num_hops=14) # list of 14 [1, 4, d]
            z_motor_k = torch.stack([hop[0, 3] for hop in sim_hops], dim=0) # [14, d]
            
            # Ground truth physical quiet trajectory tau=1..14
            quiet_h_list, _ = collect_fly_quiet_trajectory(
                model, state, num_ticks=14, writer_baseline_clock='physical',
                base_rates=rates, thresholds=thresholds,
                conductance_gains=gains, alif_params=alif, stp_params=stp,
                return_h=True
            )
            quiet_h = torch.cat(quiet_h_list, dim=0) # [14, n]
            z_phys_tau = obs.encoders(quiet_h)[:, 3, :] # [14, d]
            
            # Compute cosine similarity and normalized innovation error between all pairs (k, tau)
            k_norm = F.normalize(z_motor_k, dim=-1)
            tau_norm = F.normalize(z_phys_tau, dim=-1)
            M = torch.mm(k_norm, tau_norm.t()) # [14, 14]
            sim_matrix += M
            sample_count += 1
            
    sim_matrix = (sim_matrix / sample_count).cpu().numpy()
    
    # Analyze diagonal vs off-diagonal
    diag_vals = np.diag(sim_matrix)
    diag_mean = float(np.mean(diag_vals))
    off_diag_mask = ~np.eye(14, dtype=bool)
    off_diag_mean = float(np.mean(sim_matrix[off_diag_mask]))
    
    peak_tau_for_k = np.argmax(sim_matrix, axis=1) + 1 # 1-indexed
    
    print("\n14x14 Alignment Matrix (k: Simulation Step rows, tau: Physical Time cols):")
    header = "k \\ tau | " + " ".join(f"{t:4d}" for t in range(1, 15)) + " | argmax(tau)"
    print(header)
    print("-" * len(header))
    for k in range(14):
        row_str = f" k={k+1:2d}  | " + " ".join(f"{sim_matrix[k, t]:.2f}" for t in range(14))
        row_str += f" | tau={peak_tau_for_k[k]:2d}"
        print(row_str)
        
    print(f"\nDiagonal Mean M(k=tau):     {diag_mean:.4f}")
    print(f"Off-diagonal Mean M(k!=tau): {off_diag_mean:.4f}")
    print(f"Diagonal Contrast Ratio:     {diag_mean / max(off_diag_mean, 1e-6):.2f}x")
    
    return {
        'sim_matrix': sim_matrix.tolist(),
        'diag_mean': diag_mean,
        'off_diag_mean': off_diag_mean,
        'peak_tau_for_k': peak_tau_for_k.tolist()
    }


def test_counterfactual_horizon_ablation(model, val_tokens, num_eval_tokens=256):
    """Test 2: Counterfactual Readout Horizon Ablation & Physical Settle Equivalence."""
    print("\n" + "="*70, flush=True)
    print("TEST 2: Horizon Ablation & Physical Settle Equivalence (Language NLL)", flush=True)
    print("="*70, flush=True)
    
    tokens = val_tokens[:num_eval_tokens]
    obs = model.graph_observer
    
    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params() if model.use_alif else None
    stp = model.get_stp_params() if model.use_stp else None
    n = model.n_neurons
    
    def fresh_state():
        return FlyPhysicalState(
            torch.zeros(1, n, device='cuda'),
            tuple(torch.zeros(1, n, device='cuda') for _ in range(4)),
            torch.zeros(1, n, device='cuda'),
            torch.zeros(1, n, device='cuda'),
            torch.zeros(1, n, device='cuda'),
            torch.ones(1, n, device='cuda'),
            stp[0].clone().expand(1, n).contiguous(),
            torch.zeros(1, model.topographic_writer.n_total, device='cuda'),
            torch.zeros(1, n, device='cuda'),
            torch.empty(0, device='cuda'),
            torch.empty(0, device='cuda'),
            torch.empty(0, device='cuda')
        )
    
    horizons_to_test = [0, 1, 2, 4, 7, 10, 14]
    results_sim = {}
    
    # 2.1 Test different simulation horizons k
    for k in horizons_to_test:
        state = fresh_state()
        scores = []
        with torch.no_grad():
            for i in range(len(tokens) - 1):
                tok = int(tokens[i])
                target = int(tokens[i+1])
                token_tensor = torch.tensor([tok], device='cuda', dtype=torch.long)
                state = advance_fly_input_event(
                    model, state, token_tensor, settle_ticks=0,
                    writer_baseline_clock='physical',
                    base_rates=rates, thresholds=thresholds,
                    conductance_gains=gains, alif_params=alif, stp_params=stp
                )
                
                # Encode and rollout to k
                z0 = obs.encoders(state.h) # [1, 4, d]
                if k == 0:
                    z_readout = z0[0, 3] # [d]
                else:
                    sim_hops = obs.simulate_hops(z0, num_hops=k) # list of k [1, 4, d]
                    z_readout = sim_hops[-1][0, 3] # [d]
                
                normed = model.read_norm(z_readout.unsqueeze(0))
                logits = model.decoder(normed)
                nll = F.cross_entropy(logits, torch.tensor([target], device='cuda')).item()
                scores.append(nll)
        mean_nll = float(np.mean(scores))
        results_sim[k] = mean_nll
        print(f"Internal Simulation Horizon k={k:2d}: NLL = {mean_nll:.4f}", flush=True)

    # 2.2 Test physical waiting (settle_ticks = tau, direct physical readout at k=0)
    settle_ticks_to_test = [0, 1, 4, 7, 14]
    results_phys = {}
    print("\nPhysical Waiting (settle_ticks=tau, direct motor readout without simulation):", flush=True)
    for tau in settle_ticks_to_test:
        state = fresh_state()
        scores = []
        with torch.no_grad():
            for i in range(len(tokens) - 1):
                tok = int(tokens[i])
                target = int(tokens[i+1])
                token_tensor = torch.tensor([tok], device='cuda', dtype=torch.long)
                state = advance_fly_input_event(
                    model, state, token_tensor, settle_ticks=tau,
                    writer_baseline_clock='physical',
                    base_rates=rates, thresholds=thresholds,
                    conductance_gains=gains, alif_params=alif, stp_params=stp
                )
                
                # Direct motor readout from physical state
                z_phys = obs.encoders(state.h)[0, 3]
                normed = model.read_norm(z_phys.unsqueeze(0))
                logits = model.decoder(normed)
                nll = F.cross_entropy(logits, torch.tensor([target], device='cuda')).item()
                scores.append(nll)
        mean_nll = float(np.mean(scores))
        results_phys[tau] = mean_nll
        print(f"Physical Settle tau={tau:2d} ticks: NLL = {mean_nll:.4f}", flush=True)

    return {'results_sim': results_sim, 'results_phys': results_phys}


def test_wavefront_propagation(model):
    """Test 3: Macro Region Wavefront Tracking across Sensory -> Hub -> Premotor -> Motor."""
    print("\n" + "="*70, flush=True)
    print("TEST 3: Wavefront Penetration across Macro Connectome Regions", flush=True)
    print("="*70, flush=True)
    
    obs = model.graph_observer
    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params() if model.use_alif else None
    stp = model.get_stp_params() if model.use_stp else None
    n = model.n_neurons
    
    def fresh_state():
        return FlyPhysicalState(
            torch.zeros(1, n, device='cuda'),
            tuple(torch.zeros(1, n, device='cuda') for _ in range(4)),
            torch.zeros(1, n, device='cuda'),
            torch.zeros(1, n, device='cuda'),
            torch.zeros(1, n, device='cuda'),
            torch.ones(1, n, device='cuda'),
            stp[0].clone().expand(1, n).contiguous(),
            torch.zeros(1, model.topographic_writer.n_total, device='cuda'),
            torch.zeros(1, n, device='cuda'),
            torch.empty(0, device='cuda'),
            torch.empty(0, device='cuda'),
            torch.empty(0, device='cuda')
        )
    
    # Inject impulse into sensory region only
    state = fresh_state()
    token = 100
    token_tensor = torch.tensor([token], device='cuda', dtype=torch.long)
    state = advance_fly_input_event(
        model, state, token_tensor, settle_ticks=0,
        writer_baseline_clock='physical',
        base_rates=rates, thresholds=thresholds,
        conductance_gains=gains, alif_params=alif, stp_params=stp
    )
    
    # Reference unperturbed baseline
    state_ref = fresh_state()
    z_ref0 = obs.encoders(state_ref.h)[0] # [4, d]
    
    # 1. Track physical quiet propagation of differential response across 14 ticks
    phys_deltas = {r: [] for r in range(4)}
    current = state
    options = dict(base_rates=rates, thresholds=thresholds,
                   conductance_gains=gains, alif_params=alif, stp_params=stp)
    baseline = current.baseline
    with torch.no_grad():
        for tau in range(14):
            z = obs.encoders(current.h)[0] # [4, d]
            diff = (z - z_ref0).norm(dim=-1) # [4]
            for r in range(4):
                phys_deltas[r].append(float(diff[r].item()))
            current = step_fly_physical_tick(model, current, None, torch.zeros_like(current.h), baseline, options)
            
    # 2. Track internal simulator rollout across 14 steps
    sim_deltas = {r: [] for r in range(4)}
    with torch.no_grad():
        z0 = obs.encoders(state.h) # [1, 4, d]
        sim_hops = obs.simulate_hops(z0, num_hops=14) # 14 [1, 4, d]
        for k in range(14):
            diff = (sim_hops[k][0] - z_ref0).norm(dim=-1) # [4]
            for r in range(4):
                sim_deltas[r].append(float(diff[r].item()))
                
    region_names = ["Sensory (Node 0)", "Central Hub (Node 1)", "Premotor (Node 2)", "Motor (Node 3)"]
    print("\nWavefront Peak Response (in ticks / steps):")
    for r in range(4):
        phys_peak = int(np.argmax(phys_deltas[r])) + 1
        sim_peak = int(np.argmax(sim_deltas[r])) + 1
        print(f"Region {region_names[r]:22s} | Physical Peak tau={phys_peak:2d} | Simulator Peak k={sim_peak:2d}")
        
    return {'phys_deltas': phys_deltas, 'sim_deltas': sim_deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('E:/ib_checkpoints/q8_fly_bptt32_hx1_dagger_query_100k/last.pt'))
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    parser.add_argument('--val-data', type=Path, default=Path('data/ib_owt_gpt2/validation.npy'))
    parser.add_argument('--output', type=Path, default=Path('results/published/fly_delay_compensation_verification_20261007.json'))
    args = parser.parse_args()
    
    val_tokens = np.load(args.val_data, mmap_mode='r')
    model, _ = load_model_and_checkpoint(args.checkpoint, args.graph)
    
    res1 = test_14x14_timestamp_alignment(model, val_tokens, num_windows=32)
    res2 = test_counterfactual_horizon_ablation(model, val_tokens, num_eval_tokens=512)
    res3 = test_wavefront_propagation(model)
    
    final_report = {
        'timestamp': '2026-10-07',
        'checkpoint': str(args.checkpoint),
        'test_1_alignment_matrix': res1,
        'test_2_horizon_ablation': res2,
        'test_3_wavefront_propagation': res3
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)
    print(f"\nVerification report successfully written to {args.output}")


if __name__ == '__main__':
    main()
