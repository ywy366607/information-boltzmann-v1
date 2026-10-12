"""Membrane statistics while reading (fixed weights, 160 words from the living state): for each cell class, the
per-neuron distribution of the distance to threshold d = (theta_eff - v) / theta over all ticks (v = membrane before
reset), summarized by its median, its 1st percentile (how close the closest approaches come) and the tail ratio
(median - p1) / (p84 - p16) * 2 (about 2.3 for a Gaussian; much larger = rare large excursions drive the spikes).
Usage: python membrane_tails.py <fork dir> [...]"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
original_step = fl.step_fly_physical_tick
rec = {'idx': None, 'rows': []}
def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    if rec['idx'] is not None:
        rec['rows'].append(out.h[0, rec['idx']].float().cpu())
    return out
fl.step_fly_physical_tick = step
res = {}
for fork in map(Path, sys.argv[1:]):
    model, cfg, L = load(fork)
    opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
                alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
    N = model.n_neurons; graph = ROOT / cfg['graph']; g = np.load(graph, allow_pickle=True)
    ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
    cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
    is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
    rng = np.random.default_rng(0)
    groups = {'ORN': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON', 'dopamine': is_dan, 'core': sup == 'cb_intrinsic'}
    pick = {k: rng.choice(np.flatnonzero(m), min(60, m.sum()), replace=False) for k, m in groups.items()}
    idx = np.concatenate(list(pick.values())); rec['idx'] = torch.as_tensor(idx, device='cuda'); rec['rows'] = []
    val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); text = np.array(val[71_000:71_160], dtype=np.int64)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'); torch.manual_seed(0)
    with torch.no_grad():
        for tok in text:
            s, _, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=32, writer_baseline_clock='physical', **opts)
    rec['idx'] = None
    H = torch.stack(rec['rows']).numpy()                       # ticks x cells
    theta = model.get_thresholds().expand(1, N)[0, rec_idx := torch.as_tensor(idx, device='cuda')].float().cpu().numpy()
    D = (theta[None, :] - H) / np.abs(theta[None, :])
    out, start = {}, 0
    for k, ix in pick.items():
        d = D[:, start:start + len(ix)]; start += len(ix)
        med, p1, p16, p84 = (np.percentile(d, q, axis=0) for q in (50, 1, 16, 84))
        tail = (med - p1) / np.maximum(p84 - p16, 1e-9) * 2
        out[k] = {'median_distance': float(np.median(med)), 'p1_distance': float(np.median(p1)), 'tail_ratio(gauss~2.3)': float(np.median(tail)), 'ticks': int(D.shape[0])}
        print(fork.name, f'{k:9s}', {a: round(b, 3) for a, b in out[k].items()}, flush=True)
    res[fork.name] = out
    del model; torch.cuda.empty_cache()
json.dump(res, open('scratch/fly_stream/membrane_tails.json', 'w'), indent=1)
