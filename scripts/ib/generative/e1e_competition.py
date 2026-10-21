"""E1e - traces of competition in connection strength (2026-10-20).

If synapse formation draws on a finite local resource (trophic support, membrane space, local pools; CGB section 5),
an axon in a crowded voxel gets fewer synapses per unit cable than in a sparse one. Four parameter-free encounter
measures per type pair, from the polarity-split cable of E1c (training-pair labels), crowding counted over ALL cells
(types outside the 3000 lumped into one 'other' row):
  mass action      E_ab      = sum_v out_a(v) in_b(v)
  post competition E_post_ab = sum_v in_b(v) out_a(v) / sum_all out(v)     (dendrite capacity shared by axons present)
  pre competition  E_pre_ab  = sum_v out_a(v) in_b(v) / sum_all in(v)      (axon output shared by dendrites present)
  both             E_both_ab = sum_v out_a(v) in_b(v) / (sum_all out(v) sum_all in(v))
Prediction under competition: a competitive measure explains more of the reproducible strength than mass action and
narrows the gap to the replicate ceiling. Scored like E1d (right-side strength, held-out right-connected pairs) and
on E1's existence target.
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, torch
from e1_genomic_bottleneck import D, MIN_SYN, loss_sum, auc
from e1b_encounter import SKELETONS, fit
from e1c_polarity import BUCKETS, SHIFT, voxel_key, lookup

MIN_SYN_SIDE = 3


def type_index(names):
    a = pd.read_feather(f'{D}/body-annotations.feather', columns=['bodyId', 'type'])
    index = {t: i for i, t in enumerate(names)}
    other = len(names)
    return dict(zip(a.bodyId.values.astype(np.int64), [index.get(t, other) for t in a.type.values])), other


def merge(keys, vals):
    uq, inv = np.unique(np.concatenate(keys), return_inverse=True)
    return uq, np.bincount(inv, np.concatenate(vals)).astype(np.float32)


def cable_all(body_type, other, scale_um):
    s = scale_um * 1000.0 / 8.0
    keys, weights = [], []
    for path in sorted(glob.glob(f'{SKELETONS}/skeletons_*.npz')):
        z = np.load(path)
        bodies, counts, xyz, parent = z['body_ids'], z['node_count'], z['xyz_nm'], z['parent']
        offset = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        cell_type = np.repeat(np.array([body_type.get(int(b), other) for b in bodies]), counts)
        child = np.flatnonzero(parent >= 0)
        par = parent[child].astype(np.int64) + offset[child]
        length = np.sqrt(((xyz[child] - xyz[par]).astype(np.float64) ** 2).sum(1)) * 8.0 / 1000.0
        key = cell_type[child].astype(np.int64) * SHIFT + voxel_key((xyz[child] + xyz[par]) * 0.5, s)
        uq, inv = np.unique(key, return_inverse=True)
        keys.append(uq); weights.append(np.bincount(inv, length))
        if len(keys) >= 12:
            k, w = merge(keys, weights); keys, weights = [k], [w]
    return merge(keys, weights)


def synapse_train_counts(body_type, other, scale_um, eval_pair):
    ids = np.load('data/malecns_v1/fly_reservoir_coba_r2.npz')['neuron_body_ids'].astype(np.int64)
    cell_type = np.array([body_type.get(int(b), other) for b in ids])
    s = scale_um * 1000.0 / 8.0
    acc = {'out': ([], []), 'in': ([], [])}
    for path in sorted(glob.glob(f'{BUCKETS}/post_*.i32')):
        r = np.fromfile(path, dtype=np.int32).reshape(-1, 8)
        tp, tq = cell_type[r[:, 0]], cell_type[r[:, 1]]
        ev = (tp < other) & (tq < other)
        ev[ev] = eval_pair[tq[ev], tp[ev]]
        for side, t, xyz in (('out', tq, r[:, 5:8]), ('in', tp, r[:, 2:5])):
            key = t[~ev].astype(np.int64) * SHIFT + voxel_key(xyz[~ev].astype(np.float64), s)
            uq, cnt = np.unique(key, return_counts=True)
            ks, cs = acc[side]
            ks.append(uq); cs.append(cnt)
            if len(ks) >= 12:
                k, c = merge(ks, cs); ks[:] = [k]; cs[:] = [c]
    return {side: merge(*acc[side]) for side in acc}


def overlaps(ckeys, cw, o, i, n_rows, n, dev, alpha=1.0, chunk=40000):
    t = (ckeys // SHIFT).astype(np.int64)
    g = np.bincount(t, o, n_rows) / np.maximum(np.bincount(t, o + i, n_rows), 1e-9)
    f = (o + alpha * g[t]) / (o + i + alpha)
    w_out, w_in = (cw * f).astype(np.float32), (cw * (1 - f)).astype(np.float32)
    vox = ckeys % SHIFT
    order = np.argsort(vox, kind='stable')
    vox = vox[order]
    col = np.empty(len(vox), np.int32); col[0] = 0
    np.cumsum(vox[1:] != vox[:-1], out=col[1:])
    n_vox = int(col[-1]) + 1
    t, w_out, w_in = t[order].astype(np.int32), w_out[order], w_in[order]
    bounds = np.searchsorted(col, np.arange(0, n_vox + chunk, chunk))
    out = {k: torch.zeros(n, n, dtype=torch.float64, device=dev) for k in ('mass', 'post', 'pre', 'both')}
    for lo, hi, c0 in zip(bounds[:-1], bounds[1:], range(0, n_vox + chunk, chunk)):
        if hi <= lo:
            continue
        ti = torch.as_tensor(t[lo:hi].astype(np.int64), device=dev)
        ci = torch.as_tensor((col[lo:hi] - c0).astype(np.int64), device=dev)
        A = torch.zeros(n_rows, chunk, device=dev); B = torch.zeros(n_rows, chunk, device=dev)
        A[ti, ci] = torch.as_tensor(w_out[lo:hi], device=dev); B[ti, ci] = torch.as_tensor(w_in[lo:hi], device=dev)
        sa, sb = A.sum(0).clamp_min(1e-12), B.sum(0).clamp_min(1e-12)      # crowding over all cells, incl. 'other'
        Ak, Bk = A[:n], B[:n]
        out['mass'] += (Ak @ Bk.T).double()
        out['post'] += ((Ak / sa) @ Bk.T).double()
        out['pre'] += (Ak @ (Bk / sb).T).double()
        out['both'] += ((Ak / sa) @ (Bk / sb).T).double()
        del A, B
    return {k: v.cpu().numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--sides', default='F:/fly_stream_overflow/cgb_v1/e1_types3000_sides.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--scale', type=float, default=4.0)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/e1e_competition_20261020.json')
    args = ap.parse_args()
    t0 = time.time()
    z = np.load(args.cache, allow_pickle=True)
    W_np, sizes, names = z['W'], z['sizes'], list(z['names'])
    n = len(names)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand((n, n), generator=g)
    eval_pair = (u < 0.10).numpy()
    path = f'{args.work}/e1e_overlaps_{args.scale:g}um.npz'
    if os.path.exists(path):
        E = dict(np.load(path))
    else:
        body_type, other = type_index(names)
        ckeys, cw = cable_all(body_type, other, args.scale)
        print(f'cable (all cells): {len(cw)} type-voxel entries', flush=True)
        cnt = synapse_train_counts(body_type, other, args.scale, eval_pair)
        o, i = lookup(ckeys, cnt['out']), lookup(ckeys, cnt['in'])
        del cnt
        E = overlaps(ckeys, cw, o, i, n + 1, n, dev)
        np.savez(path, **E)
    u = u.to(dev)
    sd = np.load(args.sides)
    WL, WR, nl, nr = sd['WL'], sd['WR'], sd['nl'], sd['nr']
    bil = (nl > 0) & (nr > 0)
    # right-side strength, held-out right-connected pairs (E1d footing)
    y_s = torch.tensor(np.log(np.maximum(WR, 1) / (np.maximum(nr, 1)[:, None] * sizes[None])), dtype=torch.float32, device=dev)
    ok = torch.tensor((WR >= MIN_SYN_SIDE) & bil[:, None], device=dev)
    ms_s = {'train': (u >= 0.10) & ok, 'val': (u < 0.05) & ok, 'test': (u >= 0.05) & (u < 0.10) & ok}
    # existence, E1 footing (both sides pooled)
    y_e = torch.tensor(W_np >= MIN_SYN, dtype=torch.float32, device=dev)
    ms_e = {'train': u >= 0.10, 'val': u < 0.05, 'test': (u >= 0.05) & (u < 0.10)}
    twin = np.log(np.maximum(WL, 1) / (np.maximum(nl, 1)[:, None] * sizes[None]))
    twin_f = torch.tensor(np.stack([twin, (WL >= MIN_SYN_SIDE).astype(float)], -1), dtype=torch.float32, device=dev)

    def feat(M):
        e = M / (sizes[:, None] * sizes[None])
        floor = e[e > 0].min() * 0.5
        return torch.tensor(np.stack([np.log(np.maximum(e, floor)), (e == 0).astype(float)], -1), dtype=torch.float32, device=dev)

    nulls = {}
    for task, (y, ms, fam) in {'strength': (y_s, ms_s, 'gauss'), 'exists': (y_e, ms_e, 'bern')}.items():
        e, _ = fit(y, ms['train'], fam, steps=args.steps, dev=dev)
        nulls[task] = {m: loss_sum(e, y, fam, ms[m]) for m in ms}

    def score(feats):
        r = {}
        for task, (y, ms, fam) in {'strength': (y_s, ms_s, 'gauss'), 'exists': (y_e, ms_e, 'bern')}.items():
            e, w = fit(y, ms['train'], fam, feats=feats, steps=args.steps, dev=dev)
            r[task] = {m: round(1 - loss_sum(e, y, fam, ms[m]) / nulls[task][m], 4) for m in ms}
            r[task]['feature_weights'] = w
            if fam == 'bern':
                r[task]['test_auc'] = round(auc(e[ms['test']], y[ms['test']]), 4)
        return r

    rows = []
    def add(name, feats):
        rows.append({'model': name, **score(feats)})
        r = rows[-1]
        print(f"{name:34s} | strength test {r['strength']['test']:+.3f} coef {r['strength']['feature_weights'][0]:+.3f} | "
              f"exists test {r['exists']['test']:+.3f} auc {r['exists']['test_auc']:.3f}", flush=True)
    F_ = {k: feat(E[k]) for k in ('mass', 'post', 'pre', 'both')}
    for k in ('mass', 'post', 'pre', 'both'):
        add(f'{k}', F_[k])
    add('mass + post + pre', torch.cat([F_['mass'], F_['post'], F_['pre']], -1))
    add('replicate (left twin)', twin_f)
    add('mass + post + pre + replicate', torch.cat([F_['mass'], F_['post'], F_['pre'], twin_f], -1))
    rep = {'scale_um': args.scale, 'crowding': 'all cells (types outside the 3000 lumped as other)',
           'strength_target': 'right-side log synapses per cell pair, right-connected held-out pairs (E1d footing)',
           'exists_target': 'E1 pooled >= 5 synapses (replicate row uses left twin; not the same footing for exists)',
           'rows': rows, 'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print(f"done in {rep['seconds']} s -> {args.out}")


if __name__ == '__main__':
    main()
