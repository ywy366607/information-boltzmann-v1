"""Earliest sensory -> read-surface arrival, (1) from the connectome (delay-weighted shortest path, a guarantee) and
(2) measured on the champion by swapping the injected token and finding the first tick the read surface changes."""
import sys, json, numpy as np, torch, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state

ROOT = Path('D:/information-boltzmann-v1')
saved = torch.load('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                       read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                       decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                       surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                       transmission_mode=cfg.get('transmission_mode', 'atomic'))
inj = model.topographic_writer.injection_index.numpy()
read = model.read_indices.numpy()
N = model.n_neurons
print('injection', len(inj), 'read', len(read), 'overlap', len(np.intersect1d(inj, read)))
g = np.load(ROOT / cfg['graph'])
pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]); post = np.concatenate([g['edge_post_e'], g['edge_post_i']])
delay = np.concatenate([g['edge_delay_e'], g['edge_delay_i']]).astype(np.float64)
print('delay values', np.unique(delay))
# parallel edges: keep the shortest delay per (pre, post)
A = sp.coo_matrix((delay, (pre, post)), shape=(N, N)).tocsr()
A.sum_duplicates()  # sums duplicates; recompute min explicitly below if duplicates exist
dup = len(pre) - A.nnz
if dup:
    order = np.lexsort((delay, post, pre)); keep = np.ones(len(order), bool)
    keep[1:] = (pre[order][1:] != pre[order][:-1]) | (post[order][1:] != post[order][:-1])
    o = order[keep]; A = sp.csr_matrix((delay[o], (pre[o], post[o])), shape=(N, N))
print('duplicate pairs', dup)
dist = dijkstra(A, directed=True, indices=inj, min_only=True)
dr = dist[read]
finite = np.isfinite(dr)
print('read neurons reachable', int(finite.sum()), 'of', len(read))
vals, counts = np.unique(dr[finite], return_counts=True)
print('delay-weighted distance to read neurons (value: count):', dict(zip(vals[:12].tolist(), counts[:12].tolist())))
d_min = float(dr[finite].min())
print('d_min =', d_min, '-> read surface can first change at tick', 1 + d_min, '(tick 1 = injection tick)')

# empirical: swap the injected token and find the first tick the read surface (membranes) changes
dev = 'cuda'
edges = ('edge_weight_e', 'edge_weight_i')
model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
model = model.to(dev).eval().requires_grad_(False)
bufs = dict(model.named_buffers())
for k in edges: bufs[k].copy_(saved['model'][k])
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
cursor = int(saved['val_cursor'])
toks = np.array(val[cursor:cursor + 400], dtype=np.int64)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
            conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
            stp_params=model.get_stp_params())
state = move_state(fl.FlyPhysicalState(**L['physical']), dev)
first_change = []
rng = np.random.default_rng(0)
with torch.no_grad():
    for i, tok in enumerate(toks):
        if i >= 64 and i % 3 == 0:          # probe from this live state: actual token vs a random other token
            other = int(rng.integers(0, 50257))
            runs = []
            for t in (int(tok), other):
                s = state.detached()
                trace = []
                for tick in range(8):
                    s, _ = fl.advance_fly_token_stream(model, s, torch.tensor([t], device=dev), ticks=1,
                                                       writer_baseline_clock='physical', **opts) if tick == 0 else \
                        (fl.step_fly_physical_tick(model, s, None, torch.zeros_like(s.h),
                                                   model.topographic_writer.lambda_adapt * s.baseline, opts), None)
                    trace.append(s.h[:, model.read_indices].clone())
                runs.append(trace)
            diff = [not torch.equal(a, b) for a, b in zip(*runs)]
            first_change.append(1 + diff.index(True) if any(diff) else None)
        state, _ = fl.advance_fly_token_stream(model, state, torch.tensor([int(tok)], device=dev), ticks=7,
                                               writer_baseline_clock='physical', **opts)
fc = [f for f in first_change if f is not None]
print('empirical first tick the read membranes change after a token swap (tick 1 = injection):',
      {k: fc.count(k) for k in sorted(set(fc))}, 'never within 8 ticks:', first_change.count(None), 'of', len(first_change))
json.dump({'d_min': d_min, 'distance_histogram': dict(zip(map(str, vals.tolist()), counts.tolist())),
           'empirical_first_change_tick': {str(k): fc.count(k) for k in sorted(set(fc))}},
          open(ROOT / 'scratch/fly_stream/arrival.json', 'w'), indent=1)
