"""Single-factor causal measurement of gamma scaling on Arm B (step 3250 checkpoint).

Protocol:
- Evaluates checkpoint results/ib_fork_arm_b_3250/age_003250.pt.
- Preserves relative spatial profile of gamma(x), scaling it multiplicatively by s_gamma in {1.0, 0.5, 0.25}.
- Retains all other rules, temperature T=0.1, paired text sites, proposals, and RNG seeds.
- Measures:
    1. History difference retention trajectory D_history(k) and JS divergence.
    2. Correct history prediction gain G(k) binned across [1-4], [5-16], [17-64], [65-256] with bootstrap 95% CIs.
    3. Effect of clearing state every token: Delta NLL(reset_each_token) = NLL(reset) - NLL(keep).
    4. Kinetic energy, potential energy, and phase-space variance stability.
    5. Physical OU noise standard deviation scaling under the damping-thermal bath coupling.
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


class GammaScaledRetentionWindow(LocalWindow):
    """LocalWindow with configurable gamma scaling and temperature."""
    def __init__(self, vocab=50257, hidden=128, particles=256, steps=4, collision_hidden=32, temperature=0.1, gamma_scale=1.0):
        super().__init__(vocab=vocab, hidden=hidden, particles=particles, steps=steps, collision_hidden=collision_hidden)
        self.temperature = float(temperature)
        self.gamma_scale = float(gamma_scale)
        self.recompute = False

    def kick(self, x, v, shared, noise, h):
        force = self.core.force
        drive = 0.5 * force.net[2](force.net[1](F.linear(x, force.net[0].weight[:, :4]) + shared)).tanh()
        # Scale gamma while preserving its learned spatial profile
        gamma = force.damping(x) * self.gamma_scale
        if self.temperature > 0.0:
            if x.is_cuda:
                return FusedOU.apply(v, -x + drive, gamma, noise, h, self.temperature)
            decay = torch.exp(-gamma * h)
            integ = -torch.expm1(-gamma * h) / gamma
            std = torch.sqrt(torch.clamp(-self.temperature * torch.expm1(-2 * gamma * h), min=0.0))
            return decay * v + integ * (-x + drive) + std * noise
        else:
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
    """Stable Jensen-Shannon divergence between two logits vectors in float64."""
    l1 = F.log_softmax(logits1.double(), dim=-1)
    l2 = F.log_softmax(logits2.double(), dim=-1)
    log_m = torch.logaddexp(l1, l2) - math.log(2.0)
    kl1 = (l1.exp() * (l1 - log_m)).sum()
    kl2 = (l2.exp() * (l2 - log_m)).sum()
    js = 0.5 * (kl1 + kl2)
    return max(0.0, float(js.item()))


def extract_state_observables(x: torch.Tensor, v: torch.Tensor, model: nn.Module) -> dict:
    """Extract permutation-invariant moments, energies, and pre-readout features."""
    n, d = x.shape
    cx = x.mean(dim=0)
    cv = v.mean(dim=0)
    tilde_x = x - cx
    tilde_v = v - cv

    cov_x = (tilde_x.T @ tilde_x) / n
    cov_v = (tilde_v.T @ tilde_v) / n

    eig_x = torch.linalg.eigvalsh(cov_x).flip(0)
    eig_v = torch.linalg.eigvalsh(cov_v).flip(0)

    kinetic_en = 0.5 * float(v.square().sum(-1).mean().item())
    potential_en = 0.5 * float(x.square().sum(-1).mean().item())
    trace_x = float(cov_x.diag().sum().item())
    trace_v = float(cov_v.diag().sum().item())

    z = torch.cat((x, v), dim=-1)
    mean_feat = model.core.features(z).mean(dim=0)
    logits = F.linear(mean_feat, model.core.decoder.weight, model.core.decoder.bias)

    return {
        'centroid_x': cx.cpu(),
        'centroid_v': cv.cpu(),
        'cov_diag_x': cov_x.diag().cpu(),
        'cov_diag_v': cov_v.diag().cpu(),
        'spec_x': eig_x.cpu(),
        'spec_v': eig_v.cpu(),
        'mean_feat': mean_feat.cpu(),
        'logits': logits.cpu(),
        'kinetic_energy': kinetic_en,
        'potential_energy': potential_en,
        'trace_x': trace_x,
        'trace_v': trace_v,
    }


def compute_bootstrap_ci(site_values: list[float], n_resamples: int = 10000, seed: int = 42) -> tuple[float, float, float]:
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
def run_gamma_scaling_sweep(
    checkpoint_path: Path,
    data_path: Path,
    gamma_scale: float,
    temperature: float = 0.1,
    device: str = 'cuda',
    n_sites: int = 8,
    history_len: int = 512,
    horizon_len: int = 256,
) -> dict:
    """Run paired retention and reset measurement for a single gamma scale factor."""
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = saved['config']
    n_particles = cfg['particles']
    hidden = cfg['hidden']

    model = GammaScaledRetentionWindow(
        vocab=50257,
        hidden=hidden,
        particles=n_particles,
        steps=4,
        collision_hidden=32,
        temperature=temperature,
        gamma_scale=gamma_scale,
    ).to(device).eval()
    model.load_state_dict(saved['model'])
    model.requires_grad_(False)

    data = np.load(data_path / 'validation.npy', mmap_mode='r')

    # Establish 8 non-overlapping sites beyond the 4224 token boundary
    site_configs = []
    for s in range(n_sites):
        start_a = 8192 + s * 11000
        start_b = start_a + 4096
        span_a = (start_a, start_a + history_len + horizon_len)
        span_b = (start_b, start_b + history_len)
        assert span_a[1] <= len(data) and span_b[1] <= len(data)
        site_configs.append({
            'site': s,
            'start_A': start_a,
            'start_B': start_b,
        })

    # Effective damping statistics across state space
    sample_x = torch.randn(512, 4, device=device)
    raw_gamma = model.core.force.damping(sample_x)
    eff_gamma = (raw_gamma * gamma_scale).cpu().numpy().flatten()
    h_substep = (1.0 / model.steps) / 4.0  # h = dt / 4 = 1/16
    ou_noise_std = float(np.sqrt(np.clip(-temperature * np.expm1(-2 * eff_gamma * h_substep), 0, None)).mean())

    gamma_stats = {
        'scale_factor': gamma_scale,
        'temperature': temperature,
        'mean': float(eff_gamma.mean()),
        'std': float(eff_gamma.std()),
        'min': float(eff_gamma.min()),
        'max': float(eff_gamma.max()),
        'theoretical_tau_tokens': float(2.0 / eff_gamma.mean()) if eff_gamma.mean() > 0 else float('inf'),
        'substep_duration_h': h_substep,
        'ou_fluctuation_std_mean': ou_noise_std,
    }

    site_results = []

    for s_cfg in site_configs:
        site = s_cfg['site']
        start_a = s_cfg['start_A']
        start_b = s_cfg['start_B']
        t_site_start = time.time()

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

        # Reset-each-token state (starts from zero for continuation)
        state_R1 = (torch.zeros_like(birth_1.x), torch.zeros_like(birth_1.v))

        # Precompute sequence projections
        force = model.core.force
        layer = force.net[0]

        prev_a_list = [50256 if t == 0 else int(data[start_a + t - 1]) for t in range(history_len)]
        shared_a = F.linear(force.embedding(torch.tensor(prev_a_list, device=device, dtype=torch.long)), layer.weight[:, 4:-2], layer.bias)
        clocks_a = clock_inputs(0, history_len, model.steps, device)
        clock_proj_a = F.linear(clocks_a, layer.weight[:, -2:])

        prev_b_list = [50256 if t == 0 else int(data[start_b + t - 1]) for t in range(history_len)]
        shared_b = F.linear(force.embedding(torch.tensor(prev_b_list, device=device, dtype=torch.long)), layer.weight[:, 4:-2], layer.bias)
        clocks_b = clock_inputs(0, history_len, model.steps, device)
        clock_proj_b = F.linear(clocks_b, layer.weight[:, -2:])

        prev_s_list = [int(data[start_a + history_len + k - 1]) for k in range(horizon_len)]
        shared_s = F.linear(force.embedding(torch.tensor(prev_s_list, device=device, dtype=torch.long)), layer.weight[:, 4:-2], layer.bias)
        clocks_s = clock_inputs(history_len, horizon_len, model.steps, device)
        clock_proj_s = F.linear(clocks_s, layer.weight[:, -2:])

        # Advance history
        def advance_seq(state, shared_seq, clock_proj_seq, tables_seq, noise_seq):
            x, v = state
            for t in range(len(shared_seq)):
                sh = shared_seq[t]
                cp = clock_proj_seq[t]
                tab = tables_seq[t]
                ns = noise_seq[t]
                for s in range(model.steps):
                    x, v, _, _ = model.evolve(x, v, sh, cp[s], ns[s * 4:s * 4 + 4], tab[s])
            return x, v

        state_A1 = advance_seq(state_A1, shared_a, clock_proj_a, tables_h1, noise_h1)
        state_A2 = advance_seq(state_A2, shared_a, clock_proj_a, tables_h2, noise_h2)
        state_B1 = advance_seq(state_B1, shared_b, clock_proj_b, tables_h1, noise_h1)
        state_B2 = advance_seq(state_B2, shared_b, clock_proj_b, tables_h2, noise_h2)

        # Continuation phase
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

            # Reset arm: zero state at start of each token step
            xR1, vR1 = torch.zeros_like(xA1), torch.zeros_like(vA1)

            for s in range(model.steps):
                xA1, vA1, _, _ = model.evolve(xA1, vA1, sh, cp[s], ns_1[s * 4:s * 4 + 4], tab_1[s])
                xB1, vB1, _, _ = model.evolve(xB1, vB1, sh, cp[s], ns_1[s * 4:s * 4 + 4], tab_1[s])
                xA2, vA2, _, _ = model.evolve(xA2, vA2, sh, cp[s], ns_2[s * 4:s * 4 + 4], tab_2[s])
                xB2, vB2, _, _ = model.evolve(xB2, vB2, sh, cp[s], ns_2[s * 4:s * 4 + 4], tab_2[s])
                xR1, vR1, _, _ = model.evolve(xR1, vR1, sh, cp[s], ns_1[s * 4:s * 4 + 4], tab_1[s])

            state_A1 = (xA1, vA1)
            state_B1 = (xB1, vB1)
            state_A2 = (xA2, vA2)
            state_B2 = (xB2, vB2)

            obs_A1 = extract_state_observables(*state_A1, model)
            obs_A2 = extract_state_observables(*state_A2, model)
            obs_B1 = extract_state_observables(*state_B1, model)
            obs_B2 = extract_state_observables(*state_B2, model)
            obs_R1 = extract_state_observables(xR1, vR1, model)

            tgt_t = torch.tensor([tgt_s], dtype=torch.long)
            ce_A1 = float(F.cross_entropy(obs_A1['logits'].unsqueeze(0), tgt_t).item())
            ce_A2 = float(F.cross_entropy(obs_A2['logits'].unsqueeze(0), tgt_t).item())
            ce_B1 = float(F.cross_entropy(obs_B1['logits'].unsqueeze(0), tgt_t).item())
            ce_B2 = float(F.cross_entropy(obs_B2['logits'].unsqueeze(0), tgt_t).item())
            ce_R1 = float(F.cross_entropy(obs_R1['logits'].unsqueeze(0), tgt_t).item())

            ce_A = 0.5 * (ce_A1 + ce_A2)
            ce_B = 0.5 * (ce_B1 + ce_B2)
            gain_k = ce_B - ce_A
            delta_reset = ce_R1 - ce_A1  # > 0 means resetting hurts (history is helpful)

            js_A1_B1 = compute_js_divergence(obs_A1['logits'], obs_B1['logits'])
            js_A2_B2 = compute_js_divergence(obs_A2['logits'], obs_B2['logits'])
            js_hist = 0.5 * (js_A1_B1 + js_A2_B2)
            js_noise = compute_js_divergence(obs_A1['logits'], obs_A2['logits'])

            diff_feat_1 = float((obs_A1['mean_feat'] - obs_B1['mean_feat']).square().sum().item())
            diff_feat_2 = float((obs_A2['mean_feat'] - obs_B2['mean_feat']).square().sum().item())
            d_feat_hist = 0.5 * (diff_feat_1 + diff_feat_2)
            d_feat_noise = float((obs_A1['mean_feat'] - obs_A2['mean_feat']).square().sum().item())

            obs_timeline.append({
                'k': k,
                'ce_A': ce_A,
                'ce_B': ce_B,
                'ce_R': ce_R1,
                'gain': gain_k,
                'delta_reset': delta_reset,
                'js_history': js_hist,
                'js_noise': js_noise,
                'D_feat_history': d_feat_hist,
                'D_feat_noise': d_feat_noise,
                'kinetic_energy': obs_A1['kinetic_energy'],
                'potential_energy': obs_A1['potential_energy'],
                'trace_x': obs_A1['trace_x'],
                'trace_v': obs_A1['trace_v'],
            })

        t_elapsed = time.time() - t_site_start
        g1_4 = float(np.mean([obs_timeline[k]['gain'] for k in range(min(4, len(obs_timeline)))]))
        d_rst = float(np.mean([obs_timeline[k]['delta_reset'] for k in range(len(obs_timeline))]))
        print(f"  [gamma_scale={gamma_scale}] Site {site+1}/{len(site_configs)} completed in {t_elapsed:.1f}s | G[1-4]={g1_4:+.5f} | delta_reset={d_rst:+.5f} | k=0: JS_hist={obs_timeline[0]['js_history']:.5f}, JS_noise={obs_timeline[0]['js_noise']:.5f}", flush=True)

        site_results.append({
            'site': site,
            'timeline': obs_timeline,
        })

    # Intervals summary for G(k)
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

    # Reset effect summary across 256 tokens
    reset_site_means = [float(np.mean([s_res['timeline'][k]['delta_reset'] for k in range(horizon_len)])) for s_res in site_results]
    mean_rst, ci_rst_low, ci_rst_high = compute_bootstrap_ci(reset_site_means)
    reset_summary = {
        'mean_delta_reset': mean_rst,
        'ci_95_low': ci_rst_low,
        'ci_95_high': ci_rst_high,
        'per_site': reset_site_means,
    }

    # Aggregate timeline averages
    mean_timeline = []
    for k in range(horizon_len):
        mean_timeline.append({
            'k': k,
            'ce_A': float(np.mean([s['timeline'][k]['ce_A'] for s in site_results])),
            'ce_B': float(np.mean([s['timeline'][k]['ce_B'] for s in site_results])),
            'gain': float(np.mean([s['timeline'][k]['gain'] for s in site_results])),
            'delta_reset': float(np.mean([s['timeline'][k]['delta_reset'] for s in site_results])),
            'js_history': float(np.mean([s['timeline'][k]['js_history'] for s in site_results])),
            'js_noise': float(np.mean([s['timeline'][k]['js_noise'] for s in site_results])),
            'D_feat_history': float(np.mean([s['timeline'][k]['D_feat_history'] for s in site_results])),
            'D_feat_noise': float(np.mean([s['timeline'][k]['D_feat_noise'] for s in site_results])),
            'kinetic_energy': float(np.mean([s['timeline'][k]['kinetic_energy'] for s in site_results])),
            'potential_energy': float(np.mean([s['timeline'][k]['potential_energy'] for s in site_results])),
            'trace_x': float(np.mean([s['timeline'][k]['trace_x'] for s in site_results])),
            'trace_v': float(np.mean([s['timeline'][k]['trace_v'] for s in site_results])),
        })

    # Decay measurements at checkpoints
    k0 = mean_timeline[0]['D_feat_history']
    idx10 = min(10, horizon_len - 1)
    idx20 = min(20, horizon_len - 1)
    idx50 = min(50, horizon_len - 1)
    idx100 = min(100, horizon_len - 1)
    decay_k10 = mean_timeline[idx10]['D_feat_history'] / k0 if k0 > 0 else 0.0
    decay_k20 = mean_timeline[idx20]['D_feat_history'] / k0 if k0 > 0 else 0.0
    decay_k50 = mean_timeline[idx50]['D_feat_history'] / k0 if k0 > 0 else 0.0
    decay_k100 = mean_timeline[idx100]['D_feat_history'] / k0 if k0 > 0 else 0.0

    retention_decay_profile = {
        'D_feat_k0': k0,
        'retention_ratio_k10': decay_k10,
        'retention_ratio_k20': decay_k20,
        'retention_ratio_k50': decay_k50,
        'retention_ratio_k100': decay_k100,
        'decay_rate_k10_fold': (1.0 / decay_k10) if decay_k10 > 0 else float('inf'),
        'theoretical_damping_decay_k10': float(np.exp(-eff_gamma.mean() * 10)),
    }

    return {
        'gamma_scale': gamma_scale,
        'gamma_stats': gamma_stats,
        'retention_decay_profile': retention_decay_profile,
        'gain_summary': gain_summary,
        'reset_summary': reset_summary,
        'mean_timeline': mean_timeline,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('results/ib_fork_arm_b_3250/age_003250.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, default=Path('results/published/ib_gamma_causal_fork_b_3250.json'))
    parser.add_argument('--gamma-scales', type=float, nargs='+', default=[1.0, 0.5, 0.25])
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--sites', type=int, default=8)
    parser.add_argument('--history', type=int, default=512)
    parser.add_argument('--horizon', type=int, default=256)
    args = parser.parse_args()

    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.35)

    ckpt_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(f"=== Starting Single-Factor Gamma Causal Scaling Measurement ===", flush=True)
    print(f"Checkpoint: {args.checkpoint} (SHA256: {ckpt_hash[:16]}...)", flush=True)
    print(f"Gamma Scales: {args.gamma_scales}, Temperature: {args.temperature}", flush=True)
    print(f"Sites: {args.sites}, History: {args.history}, Horizon: {args.horizon}", flush=True)

    all_scales_results = {}
    for scale in args.gamma_scales:
        print(f"\n--- Running evaluation for Gamma Scale = {scale} ---", flush=True)
        res = run_gamma_scaling_sweep(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            gamma_scale=scale,
            temperature=args.temperature,
            n_sites=args.sites,
            history_len=args.history,
            horizon_len=args.horizon,
        )
        all_scales_results[f"scale_{scale}"] = res

        gs = res['gamma_stats']
        rd = res['retention_decay_profile']
        rs = res['reset_summary']
        print(f"Results for gamma_scale={scale} (mean gamma = {gs['mean']:.4f}, tau = {gs['theoretical_tau_tokens']:.1f} tok):", flush=True)
        print(f"  OU Fluctuation Std: {gs['ou_fluctuation_std_mean']:.5f}", flush=True)
        print(f"  Signal Retention (D_feat): k0={rd['D_feat_k0']:.6f}, k10_ratio={rd['retention_ratio_k10']:.4f} ({rd['decay_rate_k10_fold']:.1f}x decay, theory={1.0/rd['theoretical_damping_decay_k10']:.1f}x), k50_ratio={rd['retention_ratio_k50']:.6f}", flush=True)
        print(f"  Historical Gain G(k):", flush=True)
        for inv, stats in res['gain_summary'].items():
            print(f"    [{inv}]: mean={stats['mean']:+.5f} (95% CI: [{stats['ci_95_low']:+.5f}, {stats['ci_95_high']:+.5f}])", flush=True)
        print(f"  Delta NLL (reset_each_token): mean={rs['mean_delta_reset']:+.5f} (95% CI: [{rs['ci_95_low']:+.5f}, {rs['ci_95_high']:+.5f}])", flush=True)

    report = {
        'protocol': 'Single-factor causal measurement of gamma scaling on Arm B step 3250 checkpoint (docs/INFORMATION_BOLTZMANN_MEMORY_HANDOFF.md)',
        'checkpoint_path': str(args.checkpoint),
        'checkpoint_sha256': ckpt_hash,
        'n_sites': args.sites,
        'history_len': args.history,
        'horizon_len': args.horizon,
        'temperature': args.temperature,
        'gamma_scales': args.gamma_scales,
        'results': all_scales_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\n[DONE] Successfully saved gamma causal measurement report to {args.output}", flush=True)


if __name__ == '__main__':
    main()
