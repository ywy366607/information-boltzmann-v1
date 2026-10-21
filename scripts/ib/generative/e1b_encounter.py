"""E1b - the encounter term (CGB section 6, derived 2026-10-20): does arbor overlap explain what the matching code
cannot (connection strength), and enter with a mass-action coefficient near 1?

Encounter from skeletons only (no synapse information; skeleton coordinates are 8 nm voxels): per type, cable length (um) per voxel at scale s; for a type
pair E_ab = sum_v rho_a(v) rho_b(v). The skeletons do not separate axon from dendrite, so the first version uses the
whole cable for both sides (direction is left to the degree terms and the code).

Models, same 3000 types and the same 90/5/5 held-out type-pair split as E1:
  degree null | degree + encounter | code k | code k + encounter
Targets: exists (>= 5 synapses, logistic) and strength (log synapses per cell pair on connected pairs, squared error).
Encounter features: log(E_ab / (n_a n_b) + eps) and [E_ab == 0]; for strength the coefficient of log E is reported
(mass action predicts ~1).
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from e1_genomic_bottleneck import D, MIN_SYN, loss_sum, auc

SKELETONS = 'F:/fly_stream_overflow/whole_brain_skeletons_v1'


def type_density(names, scale_um, skeletons=SKELETONS):
    a = pd.read_feather(f'{D}/body-annotations.feather', columns=['bodyId', 'type'])
    index = {t: i for i, t in enumerate(names)}
    body_type = dict(zip(a.bodyId.values.astype(np.int64), [index.get(t, -1) for t in a.type.values]))
    s = scale_um * 1000.0 / 8.0          # skeleton coordinates are 8 nm voxels (the shard field says nm)
    keys, weights = [], []
    shards = sorted(glob.glob(f'{skeletons}/skeletons_*.npz'))
    for n_shard, path in enumerate(shards):
        z = np.load(path)
        bodies, counts, xyz, parent = z['body_ids'], z['node_count'], z['xyz_nm'], z['parent']
        offset = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        cell_type = np.repeat(np.array([body_type.get(int(b), -1) for b in bodies]), counts)
        ok = (parent >= 0) & (cell_type >= 0)
        child = np.flatnonzero(ok)
        par = parent[child].astype(np.int64) + offset[child]
        seg = xyz[child] - xyz[par]
        length = np.sqrt((seg.astype(np.float64) ** 2).sum(1)) * 8.0 / 1000.0
        v = np.floor((xyz[child] + xyz[par]) * 0.5 / s).astype(np.int64)
        voxel = v[:, 0] + 4096 * (v[:, 1] + 4096 * v[:, 2])
        key = cell_type[child].astype(np.int64) * (1 << 40) + voxel
        uq, inv = np.unique(key, return_inverse=True)
        keys.append(uq); weights.append(np.bincount(inv, length))
        if n_shard % 20 == 0:
            print(f'  scale {scale_um} um: shard {n_shard}/{len(shards)}', flush=True)
    key = np.concatenate(keys); w = np.concatenate(weights)
    uq, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, w)
    t = (uq >> 40).astype(np.int64); vox = uq & ((1 << 40) - 1)
    vu, col = np.unique(vox, return_inverse=True)
    return t, col, w.astype(np.float32), len(vu)


def overlap(t, col, w, n_types, n_vox, dev, chunk=60000):
    order = np.argsort(col, kind='stable')
    t, col, w = t[order], col[order], w[order]
    bounds = np.searchsorted(col, np.arange(0, n_vox + chunk, chunk))
    out = torch.zeros(n_types, n_types, dtype=torch.float64, device=dev)
    for lo, hi, c0 in zip(bounds[:-1], bounds[1:], range(0, n_vox + chunk, chunk)):
        if hi <= lo:
            continue
        dense = torch.zeros(n_types, chunk, dtype=torch.float32, device=dev)
        dense[torch.as_tensor(t[lo:hi], device=dev), torch.as_tensor(col[lo:hi] - c0, device=dev)] = torch.as_tensor(w[lo:hi], device=dev)
        out += (dense @ dense.T).double()
    return out.cpu().numpy()


def fit(target, train, family, feats=None, k=0, l2=0.0, steps=1500, dev='cuda', seed=0):
    torch.manual_seed(seed)
    n = target.shape[0]
    o = torch.zeros(n, device=dev, requires_grad=True); i = torch.zeros(n, device=dev, requires_grad=True)
    b0 = torch.tensor(float(target[train].mean()) if family == 'gauss' else 0.0, device=dev, requires_grad=True)
    params, pen = [o, i, b0], []
    wf = None
    if feats is not None:
        wf = torch.zeros(feats.shape[-1], device=dev, requires_grad=True); params.append(wf)
    if k > 0:
        C = (0.1 * torch.randn(n, k, device=dev)).requires_grad_(); A = torch.eye(k, device=dev).requires_grad_()
        params += [C, A]; pen += [C, A]

    def eta():
        e = b0 + o[:, None] + i[None]
        if k > 0:
            e = e + C @ A @ C.T
        if wf is not None:
            e = e + feats @ wf
        return e
    opt = torch.optim.Adam(params, lr=0.05)
    for s in range(steps):
        opt.zero_grad()
        e = eta()
        loss = (F.binary_cross_entropy_with_logits(e[train], target[train]) if family == 'bern'
                else (e[train] - target[train]).pow(2).mean())
        if l2 and pen:
            loss = loss + l2 * sum(x.pow(2).mean() for x in pen)
        loss.backward(); opt.step()
        if s == steps // 2:
            for g in opt.param_groups:
                g['lr'] = 0.01
    with torch.no_grad():
        return eta(), (wf.detach().cpu().numpy().tolist() if wf is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--scales', default='4,8,16')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--l2s', default='0.0001,0.001,0.01,0.1')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/e1b_encounter_20261020.json')
    args = ap.parse_args()
    t0 = time.time()
    z = np.load(args.cache, allow_pickle=True)
    W_np, sizes, names = z['W'], z['sizes'], list(z['names'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = len(names)
    E = {}
    for sc in [float(x) for x in args.scales.split(',')]:
        path = f'{args.work}/e1b_overlap_true_{sc:g}um.npy'
        if os.path.exists(path):
            E[sc] = np.load(path)
        else:
            t, col, w, nv = type_density(names, sc)
            print(f'scale {sc:g} um: {nv} occupied voxels, {len(w)} type-voxel entries', flush=True)
            E[sc] = overlap(t, col, w, n, nv, dev)
            np.save(path, E[sc])
    W = torch.tensor(W_np, dtype=torch.float32, device=dev)
    size = torch.tensor(sizes, dtype=torch.float32, device=dev)
    conn = (W >= MIN_SYN).float()
    dens = torch.log(W.clamp_min(1) / (size[:, None] * size[None]))
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand(W.shape, generator=g).to(dev)
    split = {'train': u >= 0.10, 'val': u < 0.05, 'test': (u >= 0.05) & (u < 0.10)}
    tasks = {'exists': ('bern', conn, split),
             'strength': ('gauss', dens, {m: v & (conn > 0) for m, v in split.items()})}
    nulls = {}
    for tname, (fam, y, ms) in tasks.items():
        e, _ = fit(y, ms['train'], fam, steps=args.steps, dev=dev)
        nulls[tname] = {m: loss_sum(e, y, fam, ms[m]) for m in ms}
    zero_share = {f'{sc:g}um': {'all_pairs': round(float((E[sc] == 0).mean()), 4),
                                'connected_pairs': round(float((E[sc][W_np >= MIN_SYN] == 0).mean()), 4)} for sc in E}
    print(f'pairs with zero overlap: {zero_share}', flush=True)

    def features(sc):
        e = torch.tensor(E[sc] / (sizes[:, None] * sizes[None]), dtype=torch.float32, device=dev)
        floor = float(e[e > 0].min()) * 0.5
        return torch.stack([torch.log(e.clamp_min(floor)), (e == 0).float()], -1)

    def best(feats=None, k=0):
        out = {}
        for tname, (fam, y, ms) in tasks.items():
            top = None
            for l2 in ([float(x) for x in args.l2s.split(',')] if k else [0.0]):
                e, wf = fit(y, ms['train'], fam, feats=feats, k=k, l2=l2, steps=args.steps, dev=dev)
                r = {m: round(1 - loss_sum(e, y, fam, ms[m]) / nulls[tname][m], 4) for m in ms}
                if top is None or r['val'] > top['val']:
                    top = {**r, 'l2': l2, 'feature_weights': wf}
                    if fam == 'bern':
                        top['test_auc'] = round(auc(e[ms['test']], y[ms['test']]), 4)
            out[tname] = top
        return out

    def show(r):
        w = r['strength']['feature_weights']
        print(f"{r['model']:34s} | exists test {r['exists']['test']:+.3f} auc {r['exists']['test_auc']:.3f} | "
              f"strength test {r['strength']['test']:+.3f}" + (f" | strength coef log E {w[0]:+.3f}" if w else ''), flush=True)

    rows = []
    for sc in E:
        rows.append({'model': f'degree + encounter {sc:g} um', 'scale_um': sc, **best(features(sc))}); show(rows[-1])
    rows.append({'model': f'code k={args.k}', **best(k=args.k)}); show(rows[-1])
    sc_best = max(E, key=lambda sc: next(r for r in rows if r.get('scale_um') == sc)['strength']['val'])
    rows.append({'model': f'code k={args.k} + encounter {sc_best:g} um', 'scale_um': sc_best,
                 **best(features(sc_best), k=args.k)}); show(rows[-1])
    rep = {'types': n, 'min_syn': MIN_SYN, 'scales_um': list(E), 'zero_overlap_share': zero_share,
           'metric': 'share of loss explained beyond the degree-only null on held-out type pairs (E1 split)',
           'strength_reference': 'left-right shared variance beyond degree 0.84 (e1_noise_ceiling)',
           'rows': rows, 'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print(f"done in {rep['seconds']} s -> {args.out}")


if __name__ == '__main__':
    main()
