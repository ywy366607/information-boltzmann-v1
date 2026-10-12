"""Does the birth connectome route a coherent odour into the mushroom body?  (Design: subagent discussion of
2026-10-12; FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html.)  Fixed weights, regime homeostasis on.

For r in RATIOS the inhibitory conductance gain is multiplied by r and the brain re-settles for SETTLE ticks of
reading.  Then, at POSITIONS states along the text: for each of the 10 largest olfactory receptor types, all ORNs of
that type are driven together for TICKS ticks at (max(0, 1 - b) + K) rheobase (nominal: divided by the cell's input gain) (coherent, like an odour); as a control,
the same number of ORNs drawn at random across types (scattered, like the writer).  Against a quiet run from the same
state: the response of the projection neurons matched to the type (their largest ORN input is that type, share >
0.5), of all PNs, and the set of Kenyon cells whose spike count rises by >= 2 (responding set); Jaccard of responding
sets between different types in the same state.  Also per r: membrane tail ratio (core, KC, MBON), read-latent
effective rank on text and silence/text variance.
Usage: python orn_coherent.py <fork dir>
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

RATIOS, SETTLE, POSITIONS, TICKS, K = (1.0, 2.0, 4.0), 20_000, 4, 150, 3.0
fork = Path(sys.argv[1])
model, cfg, L = load(fork)
N = model.n_neurons
graph = ROOT / cfg['graph']; g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, typ, sup = ann['class'].astype(str).values, ann['type'].astype(str).values, ann['superclass'].astype(str).values
orn, pn, kc, mbon = cls == 'olfactory', cls == 'ALPN', cls == 'Kenyon_Cell', cls == 'MBON'
core = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN'])
types = pd.Series(typ[orn]).value_counts()
types = [t for t in types.index if t not in ('None', 'nan')][:10]
pre, post, w = g['edge_pre'], g['edge_post'], np.abs(g['edge_weight']).astype(np.float64)
e = orn[pre] & pn[post]
type_of = np.full(N, '', dtype=object); type_of[orn] = typ[orn]
W = pd.DataFrame({'t': type_of[pre[e]], 'pn': post[e], 'w': w[e]}).groupby(['pn', 't']).w.sum().unstack(fill_value=0.0)
share = W.div(W.sum(1), axis=0)
matched = {t: np.array([p for p in share.index if share.loc[p].idxmax() == t and share.loc[p, t] > 0.5], dtype=np.int64)
           for t in types}
T = lambda m: torch.as_tensor(m, device='cuda')
type_mask = {t: T(orn & (typ == t)) for t in types}
rng = np.random.default_rng(0)
orn_idx = np.flatnonzero(orn)
scatter_mask = {}
for t in types:
    m = np.zeros(N, bool); m[rng.choice(orn_idx, int((orn & (typ == t)).sum()), replace=False)] = True
    scatter_mask[t] = T(m)
pn_m, kc_m = T(pn), T(kc)
print('types', {t: (int(type_mask[t].sum()), len(matched[t])) for t in types}, flush=True)

rates0 = model.get_decay_rates()[0].expand(1, N).detach()
rheobase = (model.get_thresholds().expand(1, N).detach() * -torch.log(rates0.clamp(1e-5, 1 - 1e-7)))[0]
box = {'extra': None, 'count': None, 'rec': None, 'idx': None}
original_step = fl.step_fly_physical_tick


def step(model_, current, token, source, *a, **k):
    if box['extra'] is not None:
        source = box['extra'] if source is None else source + box['extra']
    out = original_step(model_, current, token, source, *a, **k)
    if box['count'] is not None:
        box['count'] += (out.ring[0][0] > 0).float()
    if box['rec'] is not None:
        box['rec'].append(out.h[0, box['idx']].float().cpu())
    return out


fl.step_fly_physical_tick = step
writer = model.topographic_writer
quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[72_000:80_000], dtype=np.int64)
g_e0, g_i0 = model.get_conductance_gains()
idx_tail = {k: rng.choice(np.flatnonzero(m), 60, replace=False) for k, m in (('core', core), ('Kenyon', kc), ('MBON', mbon))}
theta_all = model.get_thresholds().expand(1, N)[0].float().cpu().numpy()
report = {'fork': str(fork), 'types': {t: [int(type_mask[t].sum()), int(len(matched[t]))] for t in types}}

for r in RATIOS:
    opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=(g_e0, g_i0 * r),
                alif_params=model.get_alif_params(), stp_params=model.get_stp_params())

    def read(s, tok):
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=32,
                                                     writer_baseline_clock='physical', **opts)
        return s, lat

    def ticks(s, n, mask=None):
        hb = s.homeostasis.reshape(-1)
        off = (1.0 - hb).clamp_min(0.0)
        # the tick multiplies every source by the cell's input gain s; divide it out so the drive is the nominal one
        gain = torch.exp(s.regime[3]) if s.regime.numel() == 4 * N else torch.ones(N, device='cuda')
        box['extra'] = None if mask is None else (((off + K) * rheobase / gain) * mask.float()).unsqueeze(0)
        box['count'] = torch.zeros(N, device='cuda')
        for _ in range(n):
            baseline = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
            s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
        c = box['count']; box.update(extra=None, count=None)
        return s, c

    torch.manual_seed(0)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    pos, settled = 0, 0
    with torch.no_grad():
        while settled < (SETTLE if r != 1.0 else 2000):
            s, lat = read(s, text[pos]); pos += 1; settled += lat.shape[0]
        # tails and read-latent structure
        idx = np.concatenate(list(idx_tail.values())); box.update(rec=[], idx=torch.as_tensor(idx, device='cuda'))
        lats = []
        for _ in range(128):
            s, lat = read(s, text[pos]); pos += 1; lats.append(lat[-1].float().cpu())
        H = torch.stack(box['rec']).numpy(); box.update(rec=None, idx=None)
        D = (theta_all[idx][None, :] - H) / np.abs(theta_all[idx][None, :])
        tails, start = {}, 0
        for k, ix in idx_tail.items():
            d = D[:, start:start + len(ix)]; start += len(ix)
            med, p1, p16, p84 = (np.percentile(d, q, axis=0) for q in (50, 1, 16, 84))
            tails[k] = {'distance': float(np.median(med)), 'tail_ratio': float(np.median((med - p1) / np.maximum(p84 - p16, 1e-9) * 2))}
        z = torch.stack([x.reshape(-1) for x in lats]); z = z - z.mean(0)
        ev = torch.linalg.eigvalsh(z @ z.T).clamp_min(0); p = ev / ev.sum().clamp_min(1e-30)
        var_text = float((z ** 2).sum(1).mean()); rank_text = float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))
        st, sil = s, []
        for _ in range(128):
            st, _c = ticks(st, 10); sil.append(fl.read_fly_motor(model, st).reshape(-1).float().cpu())
        zs = torch.stack(sil); zs = zs - zs.mean(0)
        var_sil = float((zs ** 2).sum(1).mean())
        # coherent vs scattered odours
        rows = []
        for p_i in range(POSITIONS):
            for _ in range(40):
                s, _l = read(s, text[pos]); pos += 1
            _, c0 = ticks(s, TICKS)
            resp = {}
            for t in types:
                for kind, masks in (('coherent', type_mask), ('scattered', scatter_mask)):
                    _, c = ticks(s, TICKS, masks[t])
                    d = c - c0
                    mp = torch.as_tensor(matched[t], device='cuda')
                    resp[(kind, t)] = {'driven ORN Hz': float(c[masks[t]].mean()) / TICKS * 1000, 'matched PN dHz': float(d[mp].mean()) / TICKS * 1000 if len(matched[t]) else float('nan'),
                                       'all PN dHz': float(d[pn_m].mean()) / TICKS * 1000,
                                       'KC set': (d[kc_m] >= 2).cpu().numpy()}
            for kind in ('coherent', 'scattered'):
                sets = [resp[(kind, t)]['KC set'] for t in types]
                jac = [float((a & b).sum() / max((a | b).sum(), 1)) for i, a in enumerate(sets) for b in sets[i + 1:] if (a | b).sum()]
                rows.append({'position': p_i, 'kind': kind, 'driven ORN Hz': float(np.mean([resp[(kind, t)]['driven ORN Hz'] for t in types])),
                             'matched PN dHz': float(np.nanmean([resp[(kind, t)]['matched PN dHz'] for t in types])),
                             'all PN dHz': float(np.mean([resp[(kind, t)]['all PN dHz'] for t in types])),
                             'KC responding fraction': float(np.mean([x.mean() for x in sets])),
                             'KC Jaccard between types': float(np.mean(jac)) if jac else float('nan')})
    R = pd.DataFrame(rows)
    summ = {kind: {c: [float(R[R.kind == kind][c].min()), float(R[R.kind == kind][c].median()), float(R[R.kind == kind][c].mean()),
                       float(R[R.kind == kind][c].max())] for c in ('driven ORN Hz', 'matched PN dHz', 'all PN dHz', 'KC responding fraction', 'KC Jaccard between types')}
            for kind in ('coherent', 'scattered')}
    report[f'r={r:g}'] = {'tails': tails, 'read_rank_text': rank_text, 'silence_over_text_variance': var_sil / max(var_text, 1e-12),
                          'odours_min_med_mean_max': summ}
    print(f'r={r:g}', json.dumps(report[f'r={r:g}']), flush=True)
json.dump(report, open(fork / 'orn_coherent.json', 'w'), indent=1)
print('done', flush=True)
