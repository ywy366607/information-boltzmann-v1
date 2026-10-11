"""Why are the dopamine cells locked to the clock?  Fixed weights, self-paced reading.

1. Synaptic drive onto the 392 dopamine cells (presynaptic spikes x learned weight, excitatory and inhibitory apart) by
   source group: its mean share, its share of the tick-to-tick variance of the net drive, and how much the per-word
   drive from each group follows word rarity (the signal the core does carry).
2. Same weight sums at birth (graph file) vs learned: has learning moved the drive onto the dopamine cells?
3. The dopamine cells themselves: firing-rate distribution (tonic / silent / responsive), dominant period of their
   population spike train, and how strongly their spikes follow the net drive.
4. Size of the dopamine signal that reaches the targets: temporal sd of the phasic part relative to the tonic level
   (what the threshold modulation sees) vs its spread across targets (what the plasticity rule mostly sees).
Usage: python dan_input.py <fork dir>
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import spearmanr
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
cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
dan_np = model.dan_indices.cpu().numpy()
is_dan = np.zeros(N, bool); is_dan[dan_np] = True
core = sup == 'cb_intrinsic'
group = np.full(N, 5)
group[core] = 0                                   # clock: core cells other than the named classes
group[core & (cls == 'Kenyon_Cell')] = 1
group[core & (cls == 'MBON')] = 2
group[core & (cls == 'CX')] = 3
group[is_dan] = 4
names = ['core: other (clock)', 'Kenyon cells', 'MB output neurons', 'central complex', 'dopamine cells', 'rest of the brain']
G = len(names)
sets = {}
for sign, suffix in (('exc', '_e'), ('inh', '_i')):
    pre, post = getattr(model, 'edge_pre' + suffix).long(), getattr(model, 'edge_post' + suffix).long()
    m = torch.as_tensor(is_dan, device='cuda')[post]
    sets[sign] = (pre[m], post[m], getattr(model, 'edge_weight' + suffix)[m].float(),
                  torch.as_tensor(group, device='cuda')[pre[m]])
report = {'fork': str(fork), 'groups': names}

# 2. birth vs learned weight onto the dopamine cells, by source group
static = {}
for sign, suffix in (('exc', '_e'), ('inh', '_i')):
    post = g['edge_post' + suffix]; pre = g['edge_pre' + suffix]; m = is_dan[post]
    birth = np.bincount(group[pre[m]], weights=np.abs(g['edge_weight' + suffix][m]), minlength=G)
    learned = torch.zeros(G, device='cuda').index_add_(0, sets[sign][3], sets[sign][2].abs()).cpu().numpy()
    static[sign] = {n: {'birth': float(b), 'learned': float(l), 'ratio': float(l / max(b, 1e-12))} for n, b, l in zip(names, birth, learned)}
report['weights_onto_dopamine_cells'] = static
for sign in static:
    print(sign, 'weight onto DA cells, learned / birth:', {k: round(v['ratio'], 2) for k, v in static[sign].items()}, flush=True)

tick_log = []
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    s = (out.ring[0][0] > 0).float()
    row = []
    for sign in ('exc', 'inh'):
        pre, _, w, grp = sets[sign]
        row.append(torch.zeros(G, device='cuda').index_add_(0, grp, s[pre] * w))
    dop = out.dopamine
    tick_log.append((torch.cat(row), s[model.dan_indices], dop[0].clone() if dop.numel() else None, dop[1].clone() if dop.numel() else None))
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 1024
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
counts = np.bincount(np.load(ROOT / cfg['data'] / 'train.npy').astype(np.int64), minlength=50257) + 1.0
rarity = -np.log(counts / counts.sum())
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
drive_ticks, dan_ticks, word_drive, word_of_tick = [], [], [], []
phasic_t, conc_t = [], []
with torch.no_grad():
    for i, tok in enumerate(inputs):
        tick_log.clear()
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=max_ticks,
                                                     writer_baseline_clock='physical', **opts)
        if i < BURN:
            continue
        d = torch.stack([x[0] for x in tick_log]).cpu().numpy()
        drive_ticks.append(d); dan_ticks.append(torch.stack([x[1] for x in tick_log]).cpu().numpy())
        word_drive.append(d.mean(0))
        if tick_log[-1][2] is not None:
            fast, slow = tick_log[-1][2], tick_log[-1][3]
            tgt = slow > 0
            level = slow[tgt].mean()
            phasic = ((fast - slow) / level)[tgt]
            phasic_t.append(float(phasic.mean())); conc_t.append((fast[tgt] / fast[tgt].mean()).cpu().numpy())
D = np.concatenate(drive_ticks)                    # ticks x 2G (exc then inh)
S = np.concatenate(dan_ticks)                      # ticks x 392
W = np.array(word_drive)
rar = rarity[inputs[BURN:]]
exc, inh = D[:, :G], D[:, G:]
net = exc.sum(1) - inh.sum(1)
total = exc.sum(0) + inh.sum(0)
report['drive'] = {}
for j, n in enumerate(names):
    contrib = exc[:, j] - inh[:, j]
    report['drive'][n] = {
        'share_of_mean_drive_exc': float(exc[:, j].mean() / exc.sum(1).mean()),
        'share_of_mean_drive_inh': float(inh[:, j].mean() / max(inh.sum(1).mean(), 1e-12)),
        'share_of_net_drive_variance': float(np.cov(contrib, net)[0, 1] / net.var()),
        'spearman_word_drive_with_rarity': float(spearmanr(W[:, j] - W[:, G + j], rar).correlation)}
    print(f'{n:22s}', {k: round(v, 3) for k, v in report['drive'][n].items()}, flush=True)
rate = S.mean(0)
pop = S.sum(1)
ac = np.correlate(pop - pop.mean(), pop - pop.mean(), 'full')[len(pop) - 1:]
ac = ac / ac[0]
peak = int(np.argmax(ac[2:200]) + 2)
lagged = [float(np.corrcoef(net[:-k or None], pop[k:])[0, 1]) for k in range(0, 5)]
report['dopamine_cells'] = {'rate_min': float(rate.min()), 'rate_median': float(np.median(rate)), 'rate_mean': float(rate.mean()),
                            'rate_max': float(rate.max()), 'share_silent(<1%)': float((rate < 0.01).mean()),
                            'share_tonic(>50%)': float((rate > 0.5).mean()),
                            'population_period_ticks': peak, 'autocorr_at_period': float(ac[peak]),
                            'corr_net_drive_with_spikes_lag0..4': lagged,
                            'spearman_word_spikes_with_rarity': float(spearmanr(np.array([d.sum(1).mean() for d in dan_ticks]), rar).correlation)}
print('dopamine cells:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report['dopamine_cells'].items()}, flush=True)
if phasic_t:
    C = np.array(conc_t)
    report['signal_at_targets'] = {'phasic_temporal_sd_rel_tonic': float(np.std(phasic_t)),
                                   'concentration_spread_across_targets_sd': float(C.std(1).mean()),
                                   'concentration_temporal_sd_per_target_mean': float(C.std(0).mean())}
    print('signal at targets:', {k: round(v, 4) for k, v in report['signal_at_targets'].items()}, flush=True)
json.dump(report, open(fork / 'dan_input.json', 'w'), indent=1)
