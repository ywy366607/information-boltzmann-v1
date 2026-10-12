"""Are the Kenyon cells a self-exciting pool because Kenyon -> Kenyon synapses drive their somata?  Zero-shot, fixed
weights (pre segment 1), self-paced reading of the same text.

Arms: intact; Kenyon -> Kenyon excitatory synapses removed (weights 0; 642,932 synapses in the graph).  Per arm:
NLL; Kenyon spikes per cell per tick while reading and in the last 64 ticks of a 96-tick silence;
per word the fraction of Kenyon cells that fire at least once (sparseness); the fraction of Kenyon cells active in
over 90% of words (tonic); mean Jaccard overlap of the active sets of random word pairs (content specificity: real
Kenyon codes overlap little); the same for MBONs' and dopamine cells' rates.
Usage: python kc_recurrence.py <fork dir>
"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
N = model.n_neurons
graph = ROOT / cfg['graph']
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls = ann['class'].astype(str).values
kc_np = cls == 'Kenyon_Cell'
kc = torch.as_tensor(kc_np, device='cuda'); mbon = torch.as_tensor(cls == 'MBON', device='cuda')
dan = model.dan_indices
pre, post = model.edge_pre_e.long(), model.edge_post_e.long()
kk = kc[pre] & kc[post]
print('Kenyon -> Kenyon excitatory synapses:', int(kk.sum()), flush=True)
original_w = model.edge_weight_e.detach().clone()
acc = {}
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    s = out.ring[0][0] > 0
    acc['kc_any'] = s[kc] if 'kc_any' not in acc else (acc['kc_any'] | s[kc])
    acc['kc_sum'] = acc.get('kc_sum', 0.0) + float(s[kc].float().mean())
    acc['mbon_sum'] = acc.get('mbon_sum', 0.0) + float(s[mbon].float().mean())
    acc['dan_sum'] = acc.get('dan_sum', 0.0) + float(s[dan].float().mean())
    acc['ticks'] = acc.get('ticks', 0) + 1
    if 'kc_list' in acc:
        acc['kc_list'].append(float(s[kc].float().mean()))
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 768
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))


def run():
    torch.manual_seed(0)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    ce, sets, kc_r, mb_r, da_r = [], [], 0.0, 0.0, 0.0
    ticks = 0
    with torch.no_grad():
        for i, tok in enumerate(inputs):
            acc.clear()
            s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=max_ticks,
                                                         writer_baseline_clock='physical', **opts)
            if i < BURN:
                continue
            ce.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), torch.tensor([int(targets[i])], device='cuda'))))
            sets.append(acc['kc_any'].cpu().numpy())
            kc_r += acc['kc_sum']; mb_r += acc['mbon_sum']; da_r += acc['dan_sum']; ticks += acc['ticks']
        acc.clear()
        acc['kc_list'] = []
        s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(targets[1])], device='cuda'), [], max_ticks=max_ticks,
                                        writer_baseline_clock='physical', silence_after=96, **opts)
        silence_rate = float(np.mean(acc['kc_list'][-64:]))
    S = np.array(sets)
    frac = S.mean(1)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(S), 2000), rng.integers(0, len(S), 2000)
    keep = a != b
    inter = (S[a[keep]] & S[b[keep]]).sum(1); union = (S[a[keep]] | S[b[keep]]).sum(1).clip(min=1)
    return {'nll': float(np.mean(ce)), 'kenyon_spikes_per_cell_tick_reading': kc_r / ticks,
            'kenyon_spikes_per_cell_tick_silence_last64': silence_rate,
            'kenyon_fraction_active_per_word': {'min': float(frac.min()), 'median': float(np.median(frac)),
                                                'mean': float(frac.mean()), 'max': float(frac.max())},
            'kenyon_fraction_tonic_over_90pct_of_words': float((S.mean(0) > 0.9).mean()),
            'kenyon_fraction_never_active': float((S.mean(0) == 0).mean()),
            'jaccard_random_word_pairs': float((inter / union).mean()),
            'mbon_spikes_per_cell_tick': mb_r / ticks, 'dopamine_cell_spikes_per_cell_tick': da_r / ticks}


report = {'fork': str(fork), 'kenyon_to_kenyon_synapses': int(kk.sum())}
report['intact'] = run()
print('intact', json.dumps(report['intact']), flush=True)
with torch.no_grad():
    model.edge_weight_e[kk] = 0.0
report['kenyon_to_kenyon_removed'] = run()
print('KC->KC removed', json.dumps(report['kenyon_to_kenyon_removed']), flush=True)
with torch.no_grad():
    model.edge_weight_e.copy_(original_w)
report['dNLL'] = report['kenyon_to_kenyon_removed']['nll'] - report['intact']['nll']
# precise: keep only the calyx (dendritic) share of every Kenyon -> Kenyon connection (neuPrint roiInfo, male-cns v1.0)
dend = pd.read_feather(graph.parent / 'kc_kc_dendritic_synapses.feather')
pos = {int(b): i for i, b in enumerate(g['neuron_body_ids'])}
key = lambda a, b: a.long() * N + b.long()
edge_key = key(pre, post)
dk = torch.as_tensor([pos[int(a)] * N + pos[int(b)] for a, b in zip(dend.body_pre, dend.body_post)], device='cuda')
dv = torch.as_tensor((dend.dendritic_synapses / dend.total_synapses).values, dtype=torch.float32, device='cuda')
order = torch.argsort(dk); dk, dv = dk[order], dv[order]
idx = torch.searchsorted(dk, edge_key).clamp(max=len(dk) - 1)
hit = dk[idx] == edge_key
share = torch.where(hit, dv[idx], torch.zeros_like(dv[idx]))
with torch.no_grad():
    model.edge_weight_e[kk] = original_w[kk] * share[kk]
print('kept dendritic share of KC->KC weight:', float(model.edge_weight_e[kk].sum() / original_w[kk].sum()), flush=True)
report['kenyon_to_kenyon_axonal_removed'] = run()
print('KC->KC axonal removed', json.dumps(report['kenyon_to_kenyon_axonal_removed']), flush=True)
with torch.no_grad():
    model.edge_weight_e.copy_(original_w)
report['dNLL_axonal_removed'] = report['kenyon_to_kenyon_axonal_removed']['nll'] - report['intact']['nll']
print('dNLL axonal removed', round(report['dNLL_axonal_removed'], 4), flush=True)
print('dNLL', round(report['dNLL'], 4), flush=True)
json.dump(report, open(fork / 'kc_recurrence.json', 'w'), indent=1)
