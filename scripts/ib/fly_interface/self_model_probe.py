"""Is the self-sustained core an inner model of the brain itself?  Fixed weights, self-paced reading.

1. What each region represents: cross-validated ridge decoding (5 contiguous blocks, ridge strength chosen inside the
   training blocks) from the region's state when the prediction is read (membrane at the read + spikes over the word),
   size-matched (2000 random neurons, smaller groups whole), of
     self variables  ticks thought, output entropy, error just met, written innovation, dopamine-cell spikes
     content         embedding of the current word, of the previous word, of the next word
2. Metacognition: decoding of the error the brain is about to make (CE of this prediction against the word that comes
   next), raw and after removing what the output's own entropy and top probability already say.
3. Where the self-sustained activity lives: spikes per neuron per tick and energy share (spike x outgoing synapses) by
   class of the central-brain intrinsic cells (and the rest) while reading and in ticks 33..64 of a 64-tick silence.
Usage: python self_model_probe.py <fork dir>
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
g = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather((ROOT / cfg['graph']).parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
core = sup == 'cb_intrinsic'
read = np.zeros(N, bool); read[model.read_indices.cpu().numpy()] = True
classes = {'core: central complex': core & (cls == 'CX'), 'core: Kenyon cells': core & (cls == 'Kenyon_Cell'),
           'core: MB output neurons': core & (cls == 'MBON'), 'core: dopamine cells': np.isin(np.arange(N), model.dan_indices.cpu().numpy()),
           'core: other': core & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN'])}
classes['core: other'] &= ~classes['core: dopamine cells']
rest = ~core & ~read & ~classes['core: dopamine cells']
rng = np.random.default_rng(0)
pick = lambda m, k=2000: np.sort(rng.choice(np.flatnonzero(m), min(k, int(m.sum())), replace=False))
decode_groups = {'motor read surface': pick(read), 'core (all classes)': pick(core), **{k: pick(v) for k, v in classes.items()},
                 'rest of the brain': pick(rest)}
S = np.unique(np.concatenate(list(decode_groups.values())))
pos = {k: np.searchsorted(S, v) for k, v in decode_groups.items()}
S_t = torch.as_tensor(S, device='cuda')
print({k: len(v) for k, v in decode_groups.items()}, flush=True)

pre_all = torch.cat([model.edge_pre_e.long(), model.edge_pre_i.long()])
outdeg = torch.bincount(pre_all, minlength=N).float()
class_masks = {**{k: torch.as_tensor(v, device='cuda') for k, v in classes.items()},
               'motor read surface': torch.as_tensor(read, device='cuda'), 'rest of the brain': torch.as_tensor(rest, device='cuda')}
dan = model.dan_indices
spike_sum, tick_rows = None, []
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    global spike_sum
    out = original_step(model_, current, *a, **k)
    s = (out.ring[0][0] > 0).float()
    spike_sum = s if spike_sum is None else spike_sum + s
    if tick_rows is not None:
        tick_rows.append(torch.stack([torch.stack([s[m].sum(), (s * outdeg)[m].sum()]) for m in class_masks.values()]))
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 1024
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
H, SP, rows = [], [], []
reading_ticks = []
with torch.no_grad():
    for i, tok in enumerate(inputs):
        t = torch.tensor([int(tok)], device='cuda')
        drive, _ = fl.write_input(model, s, t)
        innov = float(drive[0, model.topographic_writer.injection_index].norm())
        spike_sum, tick_rows = None, ([] if i >= BURN else None)
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, t, [], max_ticks=max_ticks, writer_baseline_clock='physical', **opts)
        logits = model.decoder(model.read_norm(lat[-1:])).float()
        logp = torch.log_softmax(logits, -1)
        ce = float(-logp[0, int(targets[i])])
        ent = float(-(logp.exp() * logp).sum())
        if i >= BURN:
            reading_ticks.extend(tick_rows)
            H.append(s.h[0, S_t].half().cpu()); SP.append(spike_sum[S_t].half().cpu())
            rows.append((lat.shape[0], ent, float(logp.max()), innov, float(spike_sum[dan].sum()), ce))
    # 3. silence: from the state at the end of the text, 16 words each followed by 64 silent ticks
    silent_ticks = []
    for i in range(16):
        tick_rows = []
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(targets[BURN + i])], device='cuda'), [],
                                                     max_ticks=max_ticks, writer_baseline_clock='physical', silence_after=64, **opts)
        silent_ticks.extend(tick_rows[-32:])                   # ticks 33..64 of the silence
a = np.array(rows)
ticks, ent, maxlogp, innov, da, ce_next = a.T
ce_met = np.r_[np.nan, ce_next[:-1]]
emb = model.embedding.weight.float()
E = lambda toks: emb[torch.as_tensor(toks, device='cuda')].cpu().numpy()
cur, nxt = inputs[BURN:], targets[BURN:]
prev = inputs[BURN - 1:-1]
# spikes as a rate per tick: summed counts grow with the number of ticks and would make everything decode the tick count
X = {'h': torch.stack(H).float(), 'spikes': torch.stack(SP).float() / torch.as_tensor(ticks, dtype=torch.float32)[:, None]}
# control: remove from features and targets everything a (one-hot) tick count explains
T1 = np.eye(int(ticks.max()) + 1)[ticks.astype(int)][:, np.unique(ticks.astype(int))]
proj = lambda M: M - T1 @ np.linalg.lstsq(T1, M, rcond=None)[0]

# metacognition target: the upcoming error beyond what the output itself already says (entropy, top probability)
D = np.c_[np.ones_like(ent), ent, maxlogp]
resid = ce_next - D @ np.linalg.lstsq(D, ce_next, rcond=None)[0]
targets_self = {'ticks thought': ticks, 'output entropy': ent, 'error just met': ce_met, 'written innovation': innov,
                'dopamine-cell spikes': da}
targets_meta = {'upcoming error (raw)': ce_next, 'upcoming error beyond output confidence': resid}
targets_content = {'current word': E(cur), 'previous word': E(prev), 'next word': E(nxt)}
# control for word frequency: a rare word is met with a large error whatever the brain is doing
counts = np.bincount(np.load(ROOT / cfg['data'] / 'train.npy').astype(np.int64), minlength=50257) + 1.0
rarity = -np.log(counts / counts.sum())
targets_content['current word rarity'] = rarity[cur]
ok_m = np.isfinite(ce_met)
Dm = np.c_[np.ones(ok_m.sum()), rarity[cur][ok_m]]
met_beyond = np.full_like(ce_met, np.nan)
met_beyond[ok_m] = ce_met[ok_m] - Dm @ np.linalg.lstsq(Dm, ce_met[ok_m], rcond=None)[0]
targets_self['error just met beyond word rarity'] = met_beyond
print('Spearman(error just met, current word rarity) =', round(float(pd.Series(ce_met).corr(pd.Series(rarity[cur]), method='spearman')), 3), flush=True)


def ridge_cv(Xg, Y, folds=5, alphas=(1e-2, 1e-1, 1, 10, 100, 1e3)):
    """Out-of-fold R^2 (averaged over target dimensions), contiguous blocks, ridge strength chosen on the last fifth
    of the training blocks."""
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    ok = np.isfinite(Y).all(1)
    Xg, Y = Xg[torch.as_tensor(ok)], Y[ok]
    n = len(Y)
    pred = np.zeros_like(Y)
    edges = np.linspace(0, n, folds + 1).astype(int)

    def fit(tr, te, alpha):
        x = Xg[tr].cuda().double(); mu, sd = x.mean(0), x.std(0) + 1e-6
        x = (x - mu) / sd; xt = (Xg[te].cuda().double() - mu) / sd
        y = torch.as_tensor(Y[tr], device='cuda'); ym = y.mean(0)
        K = x @ x.T
        lam = alpha * K.diagonal().mean()
        c = torch.linalg.solve(K + lam * torch.eye(len(tr), device='cuda', dtype=K.dtype), y - ym)
        return ((xt @ x.T) @ c + ym).cpu().numpy()

    for f in range(folds):
        te = np.arange(edges[f], edges[f + 1]); tr = np.setdiff1d(np.arange(n), te)
        cut = int(len(tr) * 0.8); itr, iva = tr[:cut], tr[cut:]
        best = min(alphas, key=lambda al: ((fit(itr, iva, al) - Y[iva]) ** 2).sum())
        pred[te] = fit(tr, te, best)
    r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return float(r2.mean())


report = {'fork': str(fork), 'group_sizes': {k: int(len(v)) for k, v in decode_groups.items()}, 'decoding': {},
          'decoding_tick_controlled': {},
          'spearman_ticks_vs': {k: float(pd.Series(ticks).corr(pd.Series(v), method='spearman'))
                                for k, v in {**targets_self, **targets_meta}.items() if k != 'ticks thought'}}
print('Spearman of ticks with:', {k: round(v, 3) for k, v in report['spearman_ticks_vs'].items()}, flush=True)
alltargets = {**targets_self, **targets_meta, **targets_content}
for gname, idx in pos.items():
    feats = torch.cat([X['h'][:, idx], X['spikes'][:, idx]], 1)
    res = {tname: ridge_cv(feats, y) for tname, y in alltargets.items()}
    report['decoding'][gname] = res
    print(f'{gname:28s}', {k: round(v, 3) for k, v in res.items()}, flush=True)
print('-- tick count removed from features and targets --', flush=True)
for gname, idx in pos.items():
    feats = torch.as_tensor(proj(torch.cat([X['h'][:, idx], X['spikes'][:, idx]], 1).double().numpy())).float()
    res = {}
    for tname, y in alltargets.items():
        if tname == 'ticks thought':
            continue
        y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
        ok = np.isfinite(y).all(1)
        yc = np.full_like(y, np.nan); yc[ok] = y[ok] - T1[ok] @ np.linalg.lstsq(T1[ok], y[ok], rcond=None)[0]
        res[tname] = ridge_cv(feats, yc)
    report['decoding_tick_controlled'][gname] = res
    print(f'{gname:28s}', {k: round(v, 3) for k, v in res.items()}, flush=True)

rt, st = torch.stack(reading_ticks).cpu().numpy(), torch.stack(silent_ticks).cpu().numpy()
sizes = np.array([float(m.sum()) for m in class_masks.values()])
report['activity'] = {}
for j, name in enumerate(class_masks):
    report['activity'][name] = {'neurons': int(sizes[j]),
                                'spikes_per_neuron_tick_reading': float(rt[:, j, 0].mean() / sizes[j]),
                                'spikes_per_neuron_tick_silence': float(st[:, j, 0].mean() / sizes[j]),
                                'energy_share_reading': float(rt[:, j, 1].mean() / rt[:, :, 1].sum(1).mean()),
                                'energy_share_silence': float(st[:, j, 1].mean() / st[:, :, 1].sum(1).mean())}
    print(f'{name:28s}', {k: round(v, 4) if isinstance(v, float) else v for k, v in report['activity'][name].items()}, flush=True)
json.dump(report, open(fork / 'self_model_probe.json', 'w'), indent=1)
