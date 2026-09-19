"""Signal, noise and retention measurement on frozen Information Boltzmann checkpoint.

Protocol from docs/INFORMATION_BOLTZMANN_MEMORY_HANDOFF.md Section 3 (M01):
- Paired design across 8 held-out text sites from OWT validation set (avoiding first 4224 tokens).
- 512 tokens history, 256 tokens continuation, 2 independent noise streams.
- Branch A1, A2: True history A, different noise streams.
- Branch B1, B2: Non-overlapping history B, paired with noise streams 1 and 2.
- Continuation text S: Identical text, absolute clock aligned, paired proposals/noise.
- Permutation-invariant moment metrics (centroid, covariance, spectrum, mean feature).
- Prediction distribution JS divergence in stable log-domain.
- Token-by-token CE and historical gain G(k) = CE(B->S, k) - CE(A->S, k) binned by [1-4, 5-16, 17-64, 65-256].
- Temperature comparisons: T=0.1 (default), T=0.03, T=0.0.
"""
import os
import sys

# Prevent local scripts/ib_local from shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_bpe_window import clock_inputs
from scripts.ib_fused_ou import FusedOU
from scripts.ib_local.device_collision import collision_device
from scripts.ib_local.sampling import sample_window
from scripts.ib_local.window import LocalWindow


class RetentionWindow(LocalWindow):
    """LocalWindow with configurable temperature in kick and no recomputation."""
    def __init__(self, vocab=50257, hidden=128, particles=256, steps=4, collision_hidden=32, temperature=0.1):
        super().__init__(vocab=vocab, hidden=hidden, particles=particles, steps=steps, collision_hidden=collision_hidden)
        self.temperature = float(temperature)
        self.recompute = False

    def kick(self, x, v, shared, noise, h):
        force = self.core.force
        drive = 0.5 * force.net[2](force.net[1](F.linear(x, force.net[0].weight[:, :4]) + shared)).tanh()
        gamma = force.damping(x)
        if self.temperature > 0.0:
            if x.is_cuda:
                return FusedOU.apply(v, -x + drive, gamma, noise, h, self.temperature)
            decay = torch.exp(-gamma * h)
            integ = -torch.expm1(-gamma * h) / gamma
            std = torch.sqrt(torch.clamp(-self.temperature * torch.expm1(-2 * gamma * h), min=0.0))
            return decay * v + integ * (-x + drive) + std * noise
        else:
            # T=0: omit stochastic fluctuation term, preserve damping and drive
            decay = torch.exp(-gamma * h)
            integ = -torch.expm1(-gamma * h) / gamma
            return decay * v + integ * (-x + drive)

    def evolve(self, x, v, shared, clocks, noise, layers):
        dt = 1.0 / self.steps
        for half in range(2):
            drive_input = shared + clocks[half]
            v = self.kick(x, v, drive_input, noise[half * 2], dt / 4.0)
            x = x + (dt / 2.0) * v
            v = self.kick(x, v, drive_input, noise[half * 2 + 1], dt / 4.0)
            if half == 0:
                v, lp, accepted = collision_device(x, v, self.core.collision, layers, layer_fn=self.layer_fn)
        return x, v, lp, accepted


def compute_js_divergence(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """Stable Jensen-Shannon divergence between two logits vectors in float64.

    Guarantees: non-negative, finite, and JS(P, P) == 0.0.
    """
    l1 = F.log_softmax(logits1.double(), dim=-1)
    l2 = F.log_softmax(logits2.double(), dim=-1)
    # log M = log((exp(l1) + exp(l2)) / 2) = logaddexp(l1, l2) - ln 2
    log_m = torch.logaddexp(l1, l2) - math.log(2.0)
    kl1 = (l1.exp() * (l1 - log_m)).sum()
    kl2 = (l2.exp() * (l2 - log_m)).sum()
    js = 0.5 * (kl1 + kl2)
    return max(0.0, float(js.item()))


def extract_state_observables(x: torch.Tensor, v: torch.Tensor, model: nn.Module) -> dict:
    """Extract permutation-invariant moments and pre-readout features."""
    # x, v: [N, d]
    n, d = x.shape
    cx = x.mean(dim=0)  # [d]
    cv = v.mean(dim=0)  # [d]
    tilde_x = x - cx
    tilde_v = v - cv

    cov_x = (tilde_x.T @ tilde_x) / n  # [d, d]
    cov_v = (tilde_v.T @ tilde_v) / n  # [d, d]

    # Spectra (eigenvalues sorted descending)
    eig_x = torch.linalg.eigvalsh(cov_x).flip(0)  # [d]
    eig_v = torch.linalg.eigvalsh(cov_v).flip(0)  # [d]

    z = torch.cat((x, v), dim=-1)  # [N, 2d]
    mean_feat = model.core.features(z).mean(dim=0)  # [H]

    logits = F.linear(mean_feat, model.core.decoder.weight, model.core.decoder.bias)  # [V]

    return {
        'centroid_x': cx.cpu(),
        'centroid_v': cv.cpu(),
        'cov_diag_x': cov_x.diag().cpu(),
        'cov_diag_v': cov_v.diag().cpu(),
        'spec_x': eig_x.cpu(),
        'spec_v': eig_v.cpu(),
        'cov_x': cov_x.cpu(),
        'cov_v': cov_v.cpu(),
        'mean_feat': mean_feat.cpu(),
        'logits': logits.cpu(),
    }


def compute_bootstrap_ci(site_values: list[float], n_resamples: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Compute mean and 95% percentile bootstrap confidence interval across sites."""
    arr = np.array(site_values, dtype=np.float64)
    n = len(arr)
    mean_val = float(arr.mean())
    if n <= 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = arr[indices].mean(axis=1)
    ci_low = float(np.percentile(resampled_means, 2.5))
    ci_high = float(np.percentile(resampled_means, 97.5))
    return mean_val, ci_low, ci_high


@torch.no_grad()
def run_retention_experiment(
    checkpoint_path: Path,
    data_path: Path,
    temperature: float,
    device: str = 'cuda',
    n_sites: int = 8,
    history_len: int = 512,
    horizon_len: int = 256,
) -> dict:
    """Run paired retention measurement for a given temperature."""
    raw_bytes = checkpoint_path.read_bytes()
    ckpt_hash = hashlib.sha256(raw_bytes).hexdigest()
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = saved['config']
    n_particles = cfg['particles']
    hidden = cfg['hidden']

    model = RetentionWindow(
        vocab=50257,
        hidden=hidden,
        particles=n_particles,
        steps=4,
        collision_hidden=32,
        temperature=temperature,
    ).to(device).eval()
    model.load_state_dict(saved['model'])
    model.requires_grad_(False)

    # Reference check on base LocalWindow to verify exact implementation equivalence at T=0.1
    ref_model = LocalWindow(vocab=50257, hidden=hidden, particles=n_particles).to(device).eval()
    ref_model.load_state_dict(saved['model'])
    ref_model.requires_grad_(False)

    data = np.load(data_path / 'validation.npy', mmap_mode='r')

    # Establish 8 non-overlapping sites beyond the 4224 tokens checkpoint boundary
    site_configs = []
    for s in range(n_sites):
        start_a = 8192 + s * 11000
        start_b = start_a + 4096
        span_a = (start_a, start_a + history_len + horizon_len)
        span_b = (start_b, start_b + history_len)
        assert span_a[1] <= len(data) and span_b[1] <= len(data)

        eos_a = (np.where(data[span_a[0]:span_a[1]] == 50256)[0] + span_a[0]).tolist()
        eos_b = (np.where(data[span_b[0]:span_b[1]] == 50256)[0] + span_b[0]).tolist()
        site_configs.append({
            'site': s,
            'start_A': start_a,
            'end_A_hist': start_a + history_len,
            'end_S': start_a + history_len + horizon_len,
            'start_B': start_b,
            'end_B_hist': start_b + history_len,
            'eos_A_offsets': eos_a,
            'eos_B_offsets': eos_b,
        })

    def step_single_token(state, prev_token, target_token, clock_time, tables, noise):
        x, v = state
        force = model.core.force
        layer = force.net[0]
        in_tensor = torch.tensor([prev_token], device=device, dtype=torch.long)
        shared = F.linear(force.embedding(in_tensor), layer.weight[:, 4:-2], layer.bias)
        clocks = clock_inputs(clock_time, 1, model.steps, device)
        clock_proj = F.linear(clocks, layer.weight[:, -2:])
        for s in range(model.steps):
            noise_slice = noise[s * 4:s * 4 + 4]
            tab = tables[s]
            x, v, _, _ = model.evolve(x, v, shared[0], clock_proj[0, s], noise_slice, tab)
        return x, v

    # Record actual gamma distribution across state space
    sample_x = torch.randn(512, 4, device=device)
    gamma_samples = model.core.force.damping(sample_x).cpu().numpy().flatten()
    gamma_stats = {
        'mean': float(gamma_samples.mean()),
        'std': float(gamma_samples.std()),
        'min': float(gamma_samples.min()),
        'max': float(gamma_samples.max()),
        'percentiles': {str(p): float(np.percentile(gamma_samples, p)) for p in [10, 25, 50, 75, 90]},
    }

    site_results = []

    for s_cfg in site_configs:
        site = s_cfg['site']
        start_a = s_cfg['start_A']
        start_b = s_cfg['start_B']
        t_site_start = time.time()

        # Pre-sample all tables and noise for Stream 1 and Stream 2
        # Stream 1
        rng_h1 = np.random.default_rng(site * 1000 + 303)
        gen_h1 = torch.Generator(device=device).manual_seed(site * 1000 + 404)
        tables_h1 = [sample_window(rng_h1, 1, model.steps, n_particles, device)[0] for _ in range(history_len)]
        noise_h1 = [torch.randn(model.steps * 4, n_particles, 4, device=device, generator=gen_h1) for _ in range(history_len)]

        # Stream 2
        rng_h2 = np.random.default_rng(site * 1000 + 505)
        gen_h2 = torch.Generator(device=device).manual_seed(site * 1000 + 606)
        tables_h2 = [sample_window(rng_h2, 1, model.steps, n_particles, device)[0] for _ in range(history_len)]
        noise_h2 = [torch.randn(model.steps * 4, n_particles, 4, device=device, generator=gen_h2) for _ in range(history_len)]

        # Continuation streams
        rng_c1 = np.random.default_rng(site * 1000 + 707)
        gen_c1 = torch.Generator(device=device).manual_seed(site * 1000 + 808)
        tables_c1 = [sample_window(rng_c1, 1, model.steps, n_particles, device)[0] for _ in range(horizon_len)]
        noise_c1 = [torch.randn(model.steps * 4, n_particles, 4, device=device, generator=gen_c1) for _ in range(horizon_len)]

        rng_c2 = np.random.default_rng(site * 1000 + 909)
        gen_c2 = torch.Generator(device=device).manual_seed(site * 1000 + 1010)
        tables_c2 = [sample_window(rng_c2, 1, model.steps, n_particles, device)[0] for _ in range(horizon_len)]
        noise_c2 = [torch.randn(model.steps * 4, n_particles, 4, device=device, generator=gen_c2) for _ in range(horizon_len)]

        # Birth
        gen_b1 = torch.Generator(device=device).manual_seed(site * 1000 + 101)
        birth_1 = model.core.initialize(torch.tensor([50256], device=device), gen_b1)
        state_A1 = (birth_1.x.clone(), birth_1.v.clone())
        state_B1 = (birth_1.x.clone(), birth_1.v.clone())

        gen_b2 = torch.Generator(device=device).manual_seed(site * 1000 + 202)
        birth_2 = model.core.initialize(torch.tensor([50256], device=device), gen_b2)
        state_A2 = (birth_2.x.clone(), birth_2.v.clone())
        state_B2 = (birth_2.x.clone(), birth_2.v.clone())

        # Precompute sequence projections in single batched GEMMs
        force = model.core.force
        layer = force.net[0]

        # History A
        prev_a_list = [50256 if t == 0 else int(data[start_a + t - 1]) for t in range(history_len)]
        shared_a = F.linear(force.embedding(torch.tensor(prev_a_list, device=device, dtype=torch.long)), layer.weight[:, 4:-2], layer.bias)
        clocks_a = clock_inputs(0, history_len, model.steps, device)
        clock_proj_a = F.linear(clocks_a, layer.weight[:, -2:])

        # History B
        prev_b_list = [50256 if t == 0 else int(data[start_b + t - 1]) for t in range(history_len)]
        shared_b = F.linear(force.embedding(torch.tensor(prev_b_list, device=device, dtype=torch.long)), layer.weight[:, 4:-2], layer.bias)
        clocks_b = clock_inputs(0, history_len, model.steps, device)
        clock_proj_b = F.linear(clocks_b, layer.weight[:, -2:])

        # Continuation S
        prev_s_list = [int(data[start_a + history_len + k - 1]) for k in range(horizon_len)]
        shared_s = F.linear(force.embedding(torch.tensor(prev_s_list, device=device, dtype=torch.long)), layer.weight[:, 4:-2], layer.bias)
        clocks_s = clock_inputs(history_len, horizon_len, model.steps, device)
        clock_proj_s = F.linear(clocks_s, layer.weight[:, -2:])

        # Check equivalence with ref_model on first step if T == 0.1 and site == 0
        if site == 0 and abs(temperature - 0.1) < 1e-6:
            ref_x, ref_v = ref_model.evolve(
                birth_1.x.clone(), birth_1.v.clone(),
                shared_a[0],
                clock_proj_a[0, 0],
                noise_h1[0][:4],
                tables_h1[0][0],
            )[:2]
            cur_x, cur_v = model.evolve(
                birth_1.x.clone(), birth_1.v.clone(),
                shared_a[0],
                clock_proj_a[0, 0],
                noise_h1[0][:4],
                tables_h1[0][0],
            )[:2]
            torch.testing.assert_close(cur_x, ref_x, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(cur_v, ref_v, atol=1e-6, rtol=1e-5)

        # 1. History Phase: 512 tokens
        def advance_sequence(state, shared_seq, clock_proj_seq, tables_seq, noise_seq):
            x, v = state
            for t in range(len(shared_seq)):
                sh = shared_seq[t]
                cp = clock_proj_seq[t]
                tab = tables_seq[t]
                ns = noise_seq[t]
                for s in range(model.steps):
                    x, v, _, _ = model.evolve(x, v, sh, cp[s], ns[s * 4:s * 4 + 4], tab[s])
            return x, v

        state_A1 = advance_sequence(state_A1, shared_a, clock_proj_a, tables_h1, noise_h1)
        state_A2 = advance_sequence(state_A2, shared_a, clock_proj_a, tables_h2, noise_h2)
        state_B1 = advance_sequence(state_B1, shared_b, clock_proj_b, tables_h1, noise_h1)
        state_B2 = advance_sequence(state_B2, shared_b, clock_proj_b, tables_h2, noise_h2)

        # 2. Continuation Phase: 256 tokens on identical text S
        obs_timeline = []

        for k in range(horizon_len):
            tgt_s = int(data[start_a + history_len + k])
            sh = shared_s[k]
            cp = clock_proj_s[k]
            tab_1, ns_1 = tables_c1[k], noise_c1[k]
            tab_2, ns_2 = tables_c2[k], noise_c2[k]

            xA1, vA1 = state_A1
            xB1, vB1 = state_B1
            xA2, vA2 = state_A2
            xB2, vB2 = state_B2

            for s in range(model.steps):
                xA1, vA1, _, _ = model.evolve(xA1, vA1, sh, cp[s], ns_1[s * 4:s * 4 + 4], tab_1[s])
                xB1, vB1, _, _ = model.evolve(xB1, vB1, sh, cp[s], ns_1[s * 4:s * 4 + 4], tab_1[s])
                xA2, vA2, _, _ = model.evolve(xA2, vA2, sh, cp[s], ns_2[s * 4:s * 4 + 4], tab_2[s])
                xB2, vB2, _, _ = model.evolve(xB2, vB2, sh, cp[s], ns_2[s * 4:s * 4 + 4], tab_2[s])

            state_A1 = (xA1, vA1)
            state_B1 = (xB1, vB1)
            state_A2 = (xA2, vA2)
            state_B2 = (xB2, vB2)

            # Extract observables for all 4 branches
            obs_A1 = extract_state_observables(*state_A1, model)
            obs_A2 = extract_state_observables(*state_A2, model)
            obs_B1 = extract_state_observables(*state_B1, model)
            obs_B2 = extract_state_observables(*state_B2, model)

            # CE losses on target token
            tgt_t = torch.tensor([tgt_s], dtype=torch.long)
            ce_A1 = float(F.cross_entropy(obs_A1['logits'].unsqueeze(0), tgt_t).item())
            ce_A2 = float(F.cross_entropy(obs_A2['logits'].unsqueeze(0), tgt_t).item())
            ce_B1 = float(F.cross_entropy(obs_B1['logits'].unsqueeze(0), tgt_t).item())
            ce_B2 = float(F.cross_entropy(obs_B2['logits'].unsqueeze(0), tgt_t).item())

            ce_A = 0.5 * (ce_A1 + ce_A2)
            ce_B = 0.5 * (ce_B1 + ce_B2)
            gain_k = ce_B - ce_A  # Positive indicates true history A improved prediction over B

            # JS distances on predictive distributions
            js_A1_B1 = compute_js_divergence(obs_A1['logits'], obs_B1['logits'])
            js_A2_B2 = compute_js_divergence(obs_A2['logits'], obs_B2['logits'])
            js_hist = 0.5 * (js_A1_B1 + js_A2_B2)
            js_noise = compute_js_divergence(obs_A1['logits'], obs_A2['logits'])

            # D_history and D_noise for each observable
            d_hist = {}
            d_noise = {}
            for key in ['centroid_x', 'centroid_v', 'cov_diag_x', 'cov_diag_v', 'spec_x', 'spec_v', 'cov_x', 'cov_v', 'mean_feat']:
                v_A1 = obs_A1[key]
                v_A2 = obs_A2[key]
                v_B1 = obs_B1[key]
                v_B2 = obs_B2[key]

                diff_hist_1 = float((v_A1 - v_B1).square().sum().item())
                diff_hist_2 = float((v_A2 - v_B2).square().sum().item())
                d_hist[key] = 0.5 * (diff_hist_1 + diff_hist_2)

                diff_noise = float((v_A1 - v_A2).square().sum().item())
                d_noise[key] = diff_noise

            obs_timeline.append({
                'k': k,
                'ce_A': ce_A,
                'ce_B': ce_B,
                'gain': gain_k,
                'js_history': js_hist,
                'js_noise': js_noise,
                'D_history': d_hist,
                'D_noise': d_noise,
            })

        t_site_elapsed = time.time() - t_site_start
        gain_1_4 = float(np.mean([obs_timeline[k]['gain'] for k in range(min(4, len(obs_timeline)))]))
        print(f"  [T={temperature}] Site {site+1}/{len(site_configs)} (A={start_a}, B={start_b}) completed in {t_site_elapsed:.1f}s | G[1-4]={gain_1_4:+.5f} | k=0: JS_hist={obs_timeline[0]['js_history']:.5f}, JS_noise={obs_timeline[0]['js_noise']:.5f}", flush=True)

        site_results.append({
            'site': site,
            'config': s_cfg,
            'timeline': obs_timeline,
        })

    # Aggregate interval metrics for historical gain G(k)
    # Intervals: 1-4 (k: 0..3), 5-16 (k: 4..15), 17-64 (k: 16..63), 65-256 (k: 64..255)
    intervals = {
        '1-4': (0, 4),
        '5-16': (4, 16),
        '17-64': (16, 64),
        '65-256': (64, 256),
    }

    gain_summary = {}
    for inv_name, (start_k, end_k) in intervals.items():
        if start_k >= horizon_len:
            continue
        clamped_end = min(end_k, horizon_len)
        site_means = []
        for s_res in site_results:
            gains = [s_res['timeline'][k]['gain'] for k in range(start_k, clamped_end)]
            site_means.append(float(np.mean(gains)))
        mean_g, ci_low, ci_high = compute_bootstrap_ci(site_means)
        gain_summary[inv_name] = {
            'mean': mean_g,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'per_site': site_means,
        }

    # Aggregate timeline D_history and D_noise averages across sites
    mean_timeline = []
    for k in range(horizon_len):
        mean_js_h = float(np.mean([s['timeline'][k]['js_history'] for s in site_results]))
        mean_js_n = float(np.mean([s['timeline'][k]['js_noise'] for s in site_results]))
        mean_g = float(np.mean([s['timeline'][k]['gain'] for s in site_results]))
        mean_ce_a = float(np.mean([s['timeline'][k]['ce_A'] for s in site_results]))
        mean_ce_b = float(np.mean([s['timeline'][k]['ce_B'] for s in site_results]))

        d_hist_avg = {}
        d_noise_avg = {}
        for key in site_results[0]['timeline'][k]['D_history']:
            d_hist_avg[key] = float(np.mean([s['timeline'][k]['D_history'][key] for s in site_results]))
            d_noise_avg[key] = float(np.mean([s['timeline'][k]['D_noise'][key] for s in site_results]))

        mean_timeline.append({
            'k': k,
            'ce_A': mean_ce_a,
            'ce_B': mean_ce_b,
            'gain': mean_g,
            'js_history': mean_js_h,
            'js_noise': mean_js_n,
            'D_history': d_hist_avg,
            'D_noise': d_noise_avg,
        })

    # Verify model weights unchanged
    post_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert ckpt_hash == post_hash, "Model weights file modified during measurement!"

    return {
        'temperature': temperature,
        'checkpoint_sha256': ckpt_hash,
        'step': saved['step'],
        'gamma_stats': gamma_stats,
        'gain_summary': gain_summary,
        'mean_timeline': mean_timeline,
        'sites': site_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('results/ib_local_bpe_256_3000_v2/age_003000.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, default=Path('results/published/ib_signal_retention_3000.json'))
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.1, 0.03, 0.0])
    parser.add_argument('--sites', type=int, default=8)
    parser.add_argument('--history', type=int, default=512)
    parser.add_argument('--horizon', type=int, default=256)
    args = parser.parse_args()

    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.35)

    print(f"=== Starting M01 Signal Retention Measurement ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Sites: {args.sites}, History: {args.history}, Horizon: {args.horizon}")

    all_results = {}
    for temp in args.temperatures:
        print(f"\n--- Running evaluation for Temperature T = {temp} ---")
        res = run_retention_experiment(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            temperature=temp,
            n_sites=args.sites,
            history_len=args.history,
            horizon_len=args.horizon,
        )
        all_results[f"T_{temp}"] = res
        print(f"Gain Summary for T={temp}:")
        for inv, stats in res['gain_summary'].items():
            print(f"  [{inv}]: mean = {stats['mean']:+.5f} (95% CI: [{stats['ci_95_low']:+.5f}, {stats['ci_95_high']:+.5f}])")
        print(f"  Timeline samples:")
        for idx in [0, min(10, len(res['mean_timeline']) - 1), min(100, len(res['mean_timeline']) - 1)]:
            pt = res['mean_timeline'][idx]
            print(f"    k={pt['k']:3d}: JS_hist={pt['js_history']:.6f}, JS_noise={pt['js_noise']:.6f}, D_feat_hist={pt['D_history']['mean_feat']:.6f}, D_feat_noise={pt['D_noise']['mean_feat']:.6f}")

    # Compact JSON report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'protocol': 'M01 Signal, noise and retention measurement on frozen weights (docs/INFORMATION_BOLTZMANN_MEMORY_HANDOFF.md)',
        'checkpoint_path': str(args.checkpoint),
        'checkpoint_sha256': all_results[f"T_{args.temperatures[0]}"]['checkpoint_sha256'],
        'n_sites': args.sites,
        'history_len': args.history,
        'horizon_len': args.horizon,
        'temperatures': args.temperatures,
        'results': all_results,
    }

    # Write summary report
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\n[DONE] Successfully saved M01 report to {args.output}")


if __name__ == '__main__':
    main()
