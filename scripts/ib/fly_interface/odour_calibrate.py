"""Calibration of the odour world on the adult (fixed weights; no training).  Text streams at the world's pace
(geometric, mean 8, pulses) throughout.
A  act bound: the act pools' (approach: DNa02/DNa03/DNp09; avoid: MDN/DNp42) spike counts over 8 ticks during text
   alone; for each bound B the rate of race crossings per tick (target about 1/300).
B  odour strength: one whiff on the receptor cells of two types (6 random pairs from the pool) at c rheobase units;
   the fraction of Kenyon cells spiking within 6 ticks after the whiff, against the same 6 ticks without a whiff
   (target 5-10%).  Also the pools' counts after the whiff (innate act).
C  outcome: an odour whiff train (c = 4, p = 0.5) for 24 ticks, contact at tick 12: taste cells for 4 ticks and the
   dopamine cells (sweet: PAM01/02/11; bitter: PPL101/103/106) 3 ticks later for 4 ticks at 8 rheobase; the change
   of the dopamine concentration at the MBONs and of the compartment memory (1 + phi) on the edges from the Kenyon
   cells the odour activated, compared with the same odour without the outcome.
Usage: python odour_calibrate.py <checkpoint dir> [out .json]
"""
import sys, json, math, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core import fly_odour_world as ow

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/published/fly_consequence_20261012/odour_calibration.json')
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']; ann_path = graph.parent / 'body-annotations.feather'
if getattr(model, 'dopamine_sensitivity', None) is None:
    fl.attach_dopamine_modulation(model, site='pre')
if getattr(model, 'comp_pre', None) is None:
    fl.attach_compartment_memory(model, graph, ann_path)
model.eval().requires_grad_(False)
world = ow.OdourWorld(model, graph, ann_path, seed=0)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
import pandas as pd
g = np.load(graph, allow_pickle=True)
cls = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])['class'].astype(str).values
kenyon = torch.as_tensor(cls == 'Kenyon_Cell', device='cuda')
mbon = torch.as_tensor(np.flatnonzero(cls == 'MBON'), device='cuda')
app, av = (torch.as_tensor(p, device='cuda') for p in world.pools)
writer = model.topographic_writer
quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[70_000:80_000], dtype=np.int64)
rng = np.random.default_rng(0)
state = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0, 'left': 0}


def tick(drive=None):
    """One tick of the text stream (word pulses at the world's pace) with an optional external current."""
    s = state['s']
    model.external_drive = drive
    if state['left'] == 0:
        state['left'] = int(rng.geometric(1 / 8))
        tok = torch.tensor([int(text[state['i'] % len(text)])], device='cuda'); state['i'] += 1
        src, base = fl.write_input(model, s, tok)
        s = fl.step_fly_physical_tick(model, s, tok, src, base, opts, base_rates=opts['base_rates'])
    else:
        base = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, base, opts, base_rates=opts['base_rates'])
    state['left'] -= 1
    model.external_drive = None
    state['s'] = s
    spikes = (s.ring[0][0] > 0)
    return s, spikes


def current(cells, level):
    d = torch.zeros(1, N, device='cuda')
    idx = torch.as_tensor(np.asarray(cells), device='cuda')
    d[0, idx] = level * world.rheobase[idx]
    return d


rep = {'checkpoint': str(fork)}
with torch.no_grad():
    for _ in range(500):
        tick()
    # A: act bound
    counts = []
    for _ in range(6000):
        _, sp = tick()
        counts.append([int(sp[app].sum()), int(sp[av].sum())])
    c = np.array(counts)
    win = np.lib.stride_tricks.sliding_window_view(c, 8, axis=0).sum(-1)
    rows = {}
    for B in range(1, 13):
        a = (win[:, 0] >= B) & (win[:, 0] > win[:, 1]); v = (win[:, 1] >= B) & (win[:, 1] > win[:, 0])
        rows[B] = {'approach_per_tick': float(a.mean()), 'avoid_per_tick': float(v.mean())}
    rep['A_pool_rate_per_tick'] = [float(c[:, 0].mean()), float(c[:, 1].mean())]
    rep['A_crossings'] = rows
    print('A', rep['A_pool_rate_per_tick'], {B: round(r['approach_per_tick'] + r['avoid_per_tick'], 5) for B, r in rows.items()}, flush=True)
    # B: odour strength
    pairs = [tuple(rng.choice(ow.POOL, 2, replace=False)) for _ in range(6)]
    res = {}
    for level in (1.0, 2.0, 4.0, 8.0, 16.0):
        frac, pool_after = [], []
        for pair in pairs:
            cells = np.concatenate([world.receptor[x] for x in pair])
            for trial in range(4):
                for _ in range(40):
                    tick()
                base = torch.zeros(N, dtype=torch.bool, device='cuda')
                for _ in range(6):
                    _, sp = tick(); base |= sp
                hit = torch.zeros(N, dtype=torch.bool, device='cuda'); pa = 0
                tick(current(cells, level))
                for _ in range(6):
                    _, sp = tick(); hit |= sp; pa += int(sp[app].sum()) - int(sp[av].sum())
                frac.append(float(hit[kenyon].float().mean() - base[kenyon].float().mean()))
                pool_after.append(pa)
        res[level] = {'kenyon_extra_fraction_mean': float(np.mean(frac)), 'kenyon_extra_min_max': [float(np.min(frac)), float(np.max(frac))],
                      'approach_minus_avoid_spikes_after': float(np.mean(pool_after))}
        print('B', level, res[level], flush=True)
    rep['B_odour'] = res
    # C: outcome
    comp_kc = model.comp_kc
    outc = {}
    for valence in ('sweet', 'bitter'):
        for with_outcome in (True, False):
            deltas, da = [], []
            for trial in range(6):
                pair = pairs[trial % len(pairs)]
                cells = np.concatenate([world.receptor[x] for x in pair])
                for _ in range(60):
                    tick()
                s0 = state['s']
                phi0 = s0.comp_memory.clone() if s0.comp_memory.numel() else None
                active = torch.zeros(N, dtype=torch.bool, device='cuda')
                da_peak = 0.0
                for t in range(40):
                    d = current(cells, 4.0) if (t < 24 and rng.random() < 0.5) else None
                    if with_outcome and 12 <= t < 16:
                        d = (d if d is not None else 0) + current(world.taste[valence], 8.0)
                    if with_outcome and 15 <= t < 19:
                        d = (d if d is not None else 0) + current(world.dopamine[valence], 8.0)
                    s, sp = tick(d)
                    if t < 24:
                        active |= sp
                    if s.dopamine.numel():
                        da_peak = max(da_peak, float(s.dopamine[0, mbon].mean()))
                phi1 = state['s'].comp_memory
                if phi0 is not None and phi1.numel():
                    on = active[comp_kc][model.comp_edge_kc]
                    deltas.append(float((phi1 - phi0)[0, on].mean()) if bool(on.any()) else 0.0)
                da.append(da_peak)
            outc[f'{valence}_{"outcome" if with_outcome else "odour_only"}'] = {
                'delta_phi_on_odour_edges_mean': float(np.mean(deltas)) if deltas else None,
                'mbon_dopamine_peak_mean': float(np.mean(da))}
            print('C', valence, with_outcome, outc[f'{valence}_{"outcome" if with_outcome else "odour_only"}'], flush=True)
    rep['C_outcome'] = outc
out.write_text(json.dumps(rep, indent=1))
print('saved', out)
