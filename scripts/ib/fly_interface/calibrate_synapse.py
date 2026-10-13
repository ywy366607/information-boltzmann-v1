"""Calibrate the birth per-synapse efficacy (audit 2026-10-13): one presynaptic spike through one synapse should give a
peak depolarisation of 0.62% of the rest-to-threshold distance, as in the validated whole-brain LIF of Shiu et al. 2024
(0.275 mV per synapse into a 5 ms synaptic and 20 ms membrane filter -> peak 0.043 mV, threshold 7 mV above rest).
Zero-shot: from rest, kick 40 random presynaptic cells once and record every target's peak membrane / threshold,
divided by its synapse count (graph built with --per-synapse-weight W0, so count = |w| / W0).  Prints the factor to
apply.  Usage: python calibrate_synapse.py <coba graph npz> <W0>
"""
import sys, numpy as np, torch
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core import fly_bptt_learning as fl
G, W0 = sys.argv[1], float(sys.argv[2])
TARGET = 0.275 * 0.157 / 7.0                       # Shiu et al. 2024 peak PSP per synapse / threshold distance
g = np.load(G, allow_pickle=True)
N = len(g['neuron_body_ids'])
model = FlyReservoirLM(G, d_model=768, read_surface='output', injection='topographic', synapse_model='coba', use_alif=True,
                       use_stp=True, detach_reset=True, surrogate_mode='threshold', transmission_mode='incoming').cuda().eval().requires_grad_(False)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
print('tau_m by class (ms):', np.round(np.exp(model.log_tau_m.detach().cpu().numpy()), 1)[:6], '...')
theta = model.get_thresholds().detach().expand(1, N)[0]
z = torch.zeros(1, N, device='cuda')
pre_e, post_e, w_e = g['edge_pre_e'], g['edge_post_e'], np.abs(g['edge_weight_e'])
rng = np.random.default_rng(1)
ratios = []
for j in rng.choice(np.unique(pre_e), 40, replace=False):
    s = fl.FlyPhysicalState(h=z.clone(), ring=tuple(z.clone() for _ in range(4)), ge=z.clone(), gi=z.clone(), b=z.clone(),
                            x=torch.ones_like(z), u=opts['stp_params'][0].detach().expand(1, N).clone(), baseline=z.clone())
    kick = z.clone(); kick[0, j] = 10.0 * float(theta[j])
    peak = torch.full((N,), -1e9, device='cuda')
    for t in range(40):
        model.external_drive = kick if t == 0 else None
        s = fl.step_fly_physical_tick(model, s, None, z, s.baseline, opts, base_rates=opts['base_rates'])
        peak = torch.maximum(peak, s.h[0])
    model.external_drive = None
    m = pre_e == j
    tg = torch.as_tensor(post_e[m], device='cuda')
    pk = (peak[tg] / theta[tg]).cpu().numpy(); cnt = w_e[m] / W0
    ok = (cnt >= 0.5) & (pk > 0) & (pk < 0.9)
    ratios.extend(list(pk[ok] / cnt[ok]))
r = np.array(ratios)
med = float(np.median(r))
print(f'peak PSP / threshold per synapse at W0={W0}: median {med:.5f} (p10 {np.percentile(r, 10):.5f}, p90 {np.percentile(r, 90):.5f}, n={len(r)})')
print(f'target {TARGET:.5f} -> W = {W0 * TARGET / med:.6f} (factor x{TARGET / med:.3f})')
