"""Is the birth synapse scale the reason senses die after two hops?  Zero-shot, on the BIRTH connectome (no training,
no text): the conversion set every edge to count x g / sqrt(K_in) with g fixed by a spectral radius of 1, which makes
one synapse ~0.35% of a threshold (median; ~286 coincident synapse-spikes to fire), against ~3.9% in a validated
whole-brain LIF (Shiu et al. 2024: 0.275 mV per synapse, 7 mV to threshold).  Here the edges are rescaled to a
uniform efficacy per synapse, w = count x e x theta (count = w_birth sqrt(K_in) / g), for e = 0.35% (as born, with
sqrt(K) undone), 1%, 2%, 3.9%, and the birth weights as they are.  For each: spontaneous activity (no input, 300
ticks: mean rate, silent / saturated fractions), then sugar GRNs driven (1.5 x threshold, 200 ticks) vs control:
extra spikes per cell by hop (BFS) and of PAM / PPL1 / MBON / Kenyon cells / motor and descending neurons.
Usage: python birth_scale_probe.py [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core import fly_bptt_learning as fl

ROOT = Path('D:/information-boltzmann-v1')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'scratch/fly_stream/birth_scale.json'
import os
GRAPH = Path(os.environ.get('GRAPH', ROOT / 'data/malecns_v1/fly_reservoir_coba.npz'))
VARIANTS = os.environ.get('VARIANTS', 'all')
graph = np.load(GRAPH, allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values.astype(str), ann['class'].astype(str).values.astype(str)
N = len(typ)
sc_names = np.array(graph['superclass_names'])[graph['superclass_id']]
dan = np.zeros(N, bool); dan[graph['dan_indices']] = True
groups = {'PAM': dan & np.char.startswith(typ, 'PAM'), 'PPL1': dan & np.char.startswith(typ, 'PPL1'), 'MBON': cls == 'MBON',
          'KC': cls == 'Kenyon_Cell', 'motor': np.isin(sc_names, ['cb_motor', 'vnc_motor']), 'descending': sc_names == 'descending_neuron'}
sugar = np.isin(typ, ['LB3b', 'LB3c', 'dorsal_tpGRN'])
SOURCES = {'sugar': sugar, 'bitter': np.isin(typ, ['LB1a', 'LB1b', 'LB1c', 'LB1d']), 'olfactory': cls == 'olfactory'}
sensory = np.isin(sc_names, [n for n in np.unique(sc_names) if 'sensory' in n])
K = np.maximum(np.bincount(graph['edge_post'], minlength=N), 1).astype(np.float64)
G_CONV = 0.0058                                    # count x g / sqrt(K): g measured from the connectome (birth_scale notes)

model = FlyReservoirLM(str(GRAPH), d_model=768, read_surface='output', injection='topographic', synapse_model='coba',
                       use_alif=True, use_stp=True, detach_reset=True, surrogate_mode='threshold',
                       transmission_mode='incoming').cuda().eval().requires_grad_(False)
theta = float(model.get_thresholds().flatten()[0])
w0 = {sfx: getattr(model, f'edge_weight_{sfx}').detach().clone() for sfx in ('e', 'i')}
post = {sfx: getattr(model, f'edge_post_{sfx}').long() for sfx in ('e', 'i')}
Kt = torch.as_tensor(K, dtype=torch.float32, device='cuda')
pre_e = graph['edge_pre_e']; post_e = graph['edge_post_e']


def hops(src, k_max=5):
    seen = src.copy(); frontier = src.copy(); res = []
    for _ in range(k_max):
        nxt = np.zeros(N, bool); nxt[post_e[frontier[pre_e]]] = True
        nxt &= ~seen; res.append(nxt); seen |= nxt; frontier = nxt
    return res


H = {k: hops(m) for k, m in SOURCES.items()}
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
quiet = torch.zeros(1, N, device='cuda')


def fresh():
    z = torch.zeros(1, N, device='cuda')
    return fl.FlyPhysicalState(h=z.clone(), ring=tuple(z.clone() for _ in range(4)), ge=z.clone(), gi=z.clone(), b=z.clone(),
                               x=torch.ones_like(z), u=opts['stp_params'][0].detach().expand(1, N).clone(), baseline=z.clone())


def run(s, drive, T):
    c = torch.zeros(N, device='cuda')
    for _ in range(T):
        model.external_drive = drive
        s = fl.step_fly_physical_tick(model, s, None, quiet, s.baseline, opts, base_rates=opts['base_rates'])
        c += (s.ring[0][0] > 0).float()
    model.external_drive = None
    return s, c.cpu().numpy()


rep = {'theta': theta, 'g_conversion': G_CONV, 'K_in_median': float(np.median(K))}
thr_all = opts['thresholds'].detach().expand(1, N)[0]
drives = {}
for k, m in SOURCES.items():
    d = torch.zeros(1, N, device='cuda'); idx = torch.as_tensor(np.nonzero(m)[0], device='cuda')
    d[0, idx] = 1.5 * thr_all[idx]; drives[k] = d
sens_idx = torch.as_tensor(np.nonzero(sensory)[0], device='cuda')


def run_background(s, T, frac=0.05, seed=0):
    # continuous input: each tick a random 5% of all sensory cells driven at 1.5 x threshold
    gen = torch.Generator(device='cuda').manual_seed(seed)
    c = torch.zeros(N, device='cuda')
    for _ in range(T):
        d = torch.zeros(1, N, device='cuda')
        on = sens_idx[torch.rand(sens_idx.numel(), device='cuda', generator=gen) < frac]
        d[0, on] = 1.5 * thr_all[on]
        model.external_drive = d
        s = fl.step_fly_physical_tick(model, s, None, quiet, s.baseline, opts, base_rates=opts['base_rates'])
        c += (s.ring[0][0] > 0).float()
    model.external_drive = None
    return s, c.cpu().numpy()


variants = (('birth_as_is', None), ('uniform_0.35pct', 0.0035), ('uniform_1pct', 0.01), ('uniform_2pct', 0.02), ('uniform_3.9pct', 0.039))
if VARIANTS == 'asis':
    variants = (('graph_as_is', None),)
for label, eff in variants:
    with torch.no_grad():
        for sfx in ('e', 'i'):
            w = getattr(model, f'edge_weight_{sfx}')
            if eff is None:
                w.copy_(w0[sfx])
            else:
                count = w0[sfx].abs() * Kt[post[sfx]].sqrt() / G_CONV
                w.copy_(count * eff * theta)
        s, spont = run(fresh(), None, 300)
        rate = spont / 300
        sb, bg = run_background(s, 300)
        bg_rate = bg / 300
        base = run(s, None, 200)[1]
        r = {'spont_rate_mean': float(rate.mean()), 'spont_silent_frac': float((rate == 0).mean()), 'spont_saturated_frac': float((rate > 0.5).mean()),
             'background_rate_mean': float(bg_rate.mean()), 'background_saturated_frac': float((bg_rate > 0.5).mean()),
             'background_silent_frac': float((bg_rate == 0).mean()),
             'background_groups': {k: float(bg_rate[m].mean()) for k, m in groups.items()}}
        for src, m in SOURCES.items():
            extra = run(s, drives[src], 200)[1] - base
            r[src] = {'hops': [{'n': int(h.sum()), 'extra_per_cell': float(extra[h].mean()), 'frac_up': float((extra[h] > 0).mean())} for h in [m] + H[src]],
                      'groups': {k: {'extra_per_cell': float(extra[mm].mean()), 'frac_up': float((extra[mm] > 0).mean())} for k, mm in groups.items()}}
    rep[label] = r
    print(label, f"spont rate {r['spont_rate_mean']:.4f} silent {r['spont_silent_frac']:.2f} | background (5% sensory/tick): rate {r['background_rate_mean']:.4f} "
          f"silent {r['background_silent_frac']:.2f} saturated {r['background_saturated_frac']:.4f}", {k: round(v, 4) for k, v in r['background_groups'].items()}, flush=True)
    for src in SOURCES:
        q = r[src]
        print(f'   {src:>9}:', ' '.join(f"h{k}:{h['extra_per_cell']:+.2f}/{h['frac_up']:.2f}" for k, h in enumerate(q['hops'])), '|',
              ' '.join(f"{k}:{g['extra_per_cell']:+.1f}/{g['frac_up']:.2f}" for k, g in q['groups'].items()), flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(out, 'w'), indent=1)
