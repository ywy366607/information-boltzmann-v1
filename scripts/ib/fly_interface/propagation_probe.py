"""How far does a sensory signal travel through this brain?  Zero-shot, fixed weights (loader env interventions apply).
For a source set (sugar or bitter GRNs, olfactory receptors), the hop sets by breadth-first search over excitatory
edges (hop k = cells first reached at k; edges with at least `min_syn` synapses-equivalent of birth weight).  From
the reading state (4 places), the set is driven (1.5 x threshold) for 200 ticks vs a control fork; per hop set: the
extra spikes per cell (stimulus - control) and the fraction of its cells with any extra spike.  Signal dies after hop
k = the brain cannot carry that sense beyond k synapses (the cause then is the coupling at that hop).
Usage: python propagation_probe.py <checkpoint dir> [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'propagation.json'
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values.astype(str), ann['class'].astype(str).values.astype(str)
N = len(typ)
w_now = model.edge_weight_e.detach().abs().cpu().numpy()
pre, post = graph['edge_pre_e'], graph['edge_post_e']
live = w_now > 0
sets = {'sugar': np.isin(typ, ['LB3b', 'LB3c', 'dorsal_tpGRN']), 'bitter': np.isin(typ, ['LB1a', 'LB1b', 'LB1c', 'LB1d']),
        'olfactory': cls == 'olfactory'}
thr = opts['thresholds'].detach().expand(1, N)[0]
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')


def hops(src, k_max=5):
    seen = src.copy(); frontier = src.copy(); out = []
    for k in range(k_max):
        nxt = np.zeros(N, bool); nxt[post[live & frontier[pre]]] = True
        nxt &= ~seen
        out.append(nxt); seen |= nxt; frontier = nxt
    return out


def tick(s, drive=None):
    model.external_drive = drive
    baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
    s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
    model.external_drive = None
    return s


def run(s, drive, T=200):
    c = torch.zeros(N, device='cuda')
    for _ in range(T):
        s = tick(s, drive)
        c += (s.ring[0][0] > 0).float()
    return c


rep = {'fork': str(fork)}
H = {name: hops(m) for name, m in sets.items()}
rows = {name: [] for name in sets}
with torch.no_grad():
    for place in range(4):
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        seq = np.array(val[80_000 + place * 3_000:80_000 + place * 3_000 + 64], dtype=np.int64)
        for w in seq:
            token = torch.tensor([int(w)], device='cuda')
            dr, baseline = fl.write_input(model, s, token)
            s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            for t in range(7):
                s = tick(s)
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        base = run(s, None).cpu().numpy()
        for name, m in sets.items():
            drive = torch.zeros(1, N, device='cuda')
            idx = torch.as_tensor(np.nonzero(m)[0], device='cuda')
            drive[0, idx] = 1.5 * thr[idx]
            extra = run(s, drive).cpu().numpy() - base
            rows[name].append([{'n': int(h.sum()), 'extra_per_cell': float(extra[h].mean()) if h.any() else 0.0,
                                'frac_cells_changed': float((np.abs(extra[h]) > 0).mean()) if h.any() else 0.0,
                                'frac_cells_up': float((extra[h] > 0).mean()) if h.any() else 0.0}
                               for h in [m] + H[name]])
for name, rs in rows.items():
    per = []
    for k in range(len(rs[0])):
        per.append({key: float(np.mean([r[k][key] for r in rs])) for key in rs[0][k]})
    rep[name] = per
    print(name, ' | '.join(f'hop{k}: n {p["n"]:.0f} extra/cell {p["extra_per_cell"]:+.2f} up {p["frac_cells_up"]:.2f} changed {p["frac_cells_changed"]:.2f}'
                           for k, p in enumerate(per)), flush=True)
json.dump(rep, open(out, 'w'), indent=1)
