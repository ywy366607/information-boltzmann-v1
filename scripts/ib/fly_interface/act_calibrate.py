"""Initial act head and the outcome check, on the individual as the active arm starts it: Kenyon fixes (axo-axonal
input cut, birth normalization over dendritic partners), dopamine modulation and compartment memory attached, words
still written into every sense.  Text streams at the world's pace (geometric, mean 8, pulses).
1  mu, sigma of every output cell's membrane over 2,000 ticks (sigma floored at the median sigma).
2  W ~ N(0, 1/n) scaled so the leaky integrator (lambda = e^-1/8, no bias) has sd 0.15 over 3,000 ticks; then the
   shared bias b for which acts (first integrator at the bound 1 and above the other, reset after) occur once per
   ~300 ticks on text alone.  Saved to scratch/fly_stream/act_init.pt.
3  Outcome: an odour (8 receptor types, 32 rheobase, whiffs p = 0.5) for 24 ticks, contact at tick 12 (taste 4 ticks,
   dopamine cells 3 ticks later for 4 ticks at 8 rheobase), against the odour alone: dopamine peak at the MBONs and
   the change of phi on the edges of the Kenyon cells the odour drove (fired in the odour window, not in a matched
   control window).
Usage: python act_calibrate.py <checkpoint dir>
"""
import sys, json, math, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core import fly_odour_world as ow

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']; ann = graph.parent / 'body-annotations.feather'; dend = graph.parent / 'kc_kc_dendritic_synapses.feather'
if not cfg.get('kc_dendritic_only'):
    fl.attach_kenyon_dendritic_only(model, graph, ann, dend, rescale=True)
if not cfg.get('kc_renorm'):
    fl.renormalize_kenyon_inputs(model, graph, ann, dend)
if getattr(model, 'dopamine_sensitivity', None) is None:
    fl.attach_dopamine_modulation(model, site='pre')
import os
COMP_RATE = float(os.environ.get('COMP_RATE', '0.02'))
if getattr(model, 'comp_pre', None) is None:
    # eta = -kappa (phi stays in (-1, 1)); COMP_RATE sets how much one pairing writes
    fl.attach_compartment_memory(model, graph, ann, eta=-COMP_RATE, forget=min(COMP_RATE, 0.999))
SKIP_HEAD = bool(os.environ.get('SKIP_HEAD'))
model.eval().requires_grad_(False)
world = ow.OdourWorld(model, graph, ann, seed=0)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
g = np.load(graph, allow_pickle=True)
cls = pd.read_feather(ann).set_index('bodyId').reindex(g['neuron_body_ids'])['class'].astype(str).values
mbon = torch.as_tensor(np.flatnonzero(cls == 'MBON'), device='cuda')
writer = model.topographic_writer; quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); text = np.array(val[70_000:90_000], dtype=np.int64)
rng = np.random.default_rng(0)
st = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0, 'left': 0}


def tick(drive=None):
    s = st['s']; model.external_drive = drive
    if st['left'] == 0:
        st['left'] = int(rng.geometric(1 / 8))
        tok = torch.tensor([int(text[st['i'] % len(text)])], device='cuda'); st['i'] += 1
        src, base = fl.write_input(model, s, tok)
        s = fl.step_fly_physical_tick(model, s, tok, src, base, opts, base_rates=opts['base_rates'])
    else:
        base = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, base, opts, base_rates=opts['base_rates'])
    st['left'] -= 1; model.external_drive = None; st['s'] = s
    return s, (s.ring[0][0] > 0)


def current(cells, level):
    d = torch.zeros(1, N, device='cuda'); idx = torch.as_tensor(np.asarray(cells), device='cuda')
    d[0, idx] = level * world.rheobase[idx]
    return d


rep = {'checkpoint': str(fork), 'comp_rate': COMP_RATE}
with torch.no_grad():
    for _ in range(500):
        tick()
    if SKIP_HEAD:
        n = 0
    n = model.read_indices.numel() if not SKIP_HEAD else 0
    s1 = torch.zeros(n, device='cuda', dtype=torch.float64); s2 = torch.zeros_like(s1)
    for _ in range(2000 if not SKIP_HEAD else 0):
        h = tick()[0].h[0, model.read_indices].double(); s1 += h; s2 += h * h
    if not SKIP_HEAD:
        mu = (s1 / 2000).float(); sigma = ((s2 / 2000 - (s1 / 2000) ** 2).clamp_min(1e-12).sqrt()).float()
        # a cell is normalized by its own scale but never amplified beyond a typical cell's (near-silent cells would
        # otherwise turn tiny membrane noise into large act drive and gradients)
        sigma = sigma.clamp_min(float(sigma.median()))
        W = torch.randn(2, n, device='cuda', generator=torch.Generator('cuda').manual_seed(7)) / math.sqrt(n)
        m0 = []
        for _ in range(3000):
            x = (tick()[0].h[:, model.read_indices] - mu) / sigma
            m0.append((x @ W.T)[0])
        m0 = torch.stack(m0).cpu().numpy().astype(np.float64)
        lam = math.exp(-1 / 8)

        def integrate(m, b, reset):
            G = np.zeros(2); acts = 0; Gs = []
            for t in range(len(m)):
                G = lam * G + m[t] + b
                Gs.append(G.copy())
                if reset and ((G[0] >= 1 and G[0] > G[1]) or (G[1] >= 1 and G[1] > G[0])):
                    acts += 1; G = np.zeros(2)
            return acts / len(m), np.array(Gs)
        _, Gs = integrate(m0, 0.0, False)
        k = 0.15 / Gs[200:].std()
        m0 = m0 * k; W = W * k
        lo, hi = -0.5, 0.5
        for _ in range(40):
            b = (lo + hi) / 2
            r, _ = integrate(m0, b, True)
            lo, hi = (b, hi) if r < 1 / 300 else (lo, b)
        rate, _ = integrate(m0, b, True)
        rep['act_init'] = {'W_scale_k': float(k), 'bias': float(b), 'false_act_rate': float(rate), 'n_read': int(n)}
        torch.save({'act_mu': mu.cpu(), 'act_sigma': sigma.cpu(), 'act_w': W.cpu(), 'act_b': torch.full((2,), float(b))},
                   'scratch/fly_stream/act_init.pt')
        print('act head', rep['act_init'], flush=True)
    # 3 outcome -> dopamine -> phi
    kc_idx = torch.as_tensor(np.flatnonzero(cls == 'Kenyon_Cell'), device='cuda')
    # the compartments the outcome reaches: MBONs receiving modulatory synapses from the driven dopamine cells
    dpre, dpost = g['dan_edge_pre'], g['dan_edge_post']
    reached = {v: torch.as_tensor(np.isin(np.arange(N), dpost[np.isin(dpre, world.dopamine[v])]) & (cls == 'MBON'),
                                  device='cuda') for v in ('sweet', 'bitter')}
    out = {}
    for valence in ('sweet', 'bitter'):
        for with_outcome in (True, False):
            dphi, da, nkc = [], [], []
            for trial in range(6):
                odour = list(rng.choice(world.pool, 8, replace=False))
                cells = np.concatenate([world.receptor[x] for x in odour])
                for _ in range(60):
                    tick()
                ctrl = torch.zeros(N, dtype=torch.bool, device='cuda')
                for _ in range(24):
                    ctrl |= tick()[1]
                for _ in range(40):
                    tick()
                phi0 = st['s'].comp_memory.clone()
                drove = torch.zeros(N, dtype=torch.bool, device='cuda'); peak = 0.0
                for t in range(40):
                    d = current(cells, 32.0) if (t < 24 and rng.random() < 0.5) else None
                    if with_outcome and 12 <= t < 16:
                        d = (d if d is not None else 0) + current(world.taste[valence], 8.0)
                    if with_outcome and 15 <= t < 19:
                        d = (d if d is not None else 0) + current(world.dopamine[valence], 8.0)
                    s, sp = tick(d)
                    if t < 24:
                        drove |= sp
                    peak = max(peak, float(s.dopamine[0, mbon].mean()))
                odour_kc = drove & ~ctrl
                on = odour_kc[model.comp_kc][model.comp_edge_kc] & reached[valence][model.comp_post.long()]
                dphi.append(float((st['s'].comp_memory - phi0)[0, on].mean()) if bool(on.any()) else 0.0)
                da.append(peak); nkc.append(float(odour_kc[kc_idx].float().mean()))
            key = f'{valence}_{"outcome" if with_outcome else "odour_only"}'
            out[key] = {'dphi_odour_kc_edges_in_reached_compartments': float(np.mean(dphi)), 'mbon_dopamine_peak': float(np.mean(da)),
                        'odour_only_kc_fraction': float(np.mean(nkc))}
            print('outcome', key, out[key], flush=True)
    rep['outcome'] = out
open('results/published/fly_consequence_20261012/act_calibrate.jsonl', 'a').write(json.dumps(rep) + chr(10))
print('saved')
