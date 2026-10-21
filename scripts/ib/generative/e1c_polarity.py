"""E1c - directed encounter: axon (output) cable of a x dendrite (input) cable of b (CGB section 6, 2026-10-20).

E1b used whole-arbor overlap; insect neurons are unipolar and the skeletons carry no polarity, so the log E coefficient
(~0.5) may be diluted. Here each type's skeleton cable in a voxel (4 um) is split by the type's own synapses there:
f_a(v) = (outputs + alpha g_a) / (outputs + inputs + alpha), g_a = the type's overall output share (prior where it
has no synapses). rho_out = cable f, rho_in = cable (1 - f); E_ab = sum_v rho_out_a(v) rho_in_b(v).

Leakage control: the polarity labels use only synapses of TRAINING type pairs (val/test pairs of the E1 split are
excluded); a second version uses all synapses to measure how much leakage would have inflated the result.
Synapses: E:/fly_stream_overflow/whole_brain_contacts_buckets (int32 rows [post_cell, pre_cell, xp, yp, zp, xq, yq, zq],
8 nm voxels, model order of data/malecns_v1/fly_reservoir_coba_r2.npz).
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, torch
from e1_genomic_bottleneck import D, MIN_SYN, loss_sum, auc
from e1b_encounter import SKELETONS, overlap, fit

BUCKETS = 'E:/fly_stream_overflow/whole_brain_contacts_buckets'
SHIFT = 1 << 40


def voxel_key(xyz, s):
    v = np.floor(xyz / s).astype(np.int64)
    return v[:, 0] + 4096 * (v[:, 1] + 4096 * v[:, 2])


def cable(names, scale_um):
    a = pd.read_feather(f'{D}/body-annotations.feather', columns=['bodyId', 'type'])
    index = {t: i for i, t in enumerate(names)}
    body_type = dict(zip(a.bodyId.values.astype(np.int64), [index.get(t, -1) for t in a.type.values]))
    s = scale_um * 1000.0 / 8.0
    keys, weights = [], []
    for path in sorted(glob.glob(f'{SKELETONS}/skeletons_*.npz')):
        z = np.load(path)
        bodies, counts, xyz, parent = z['body_ids'], z['node_count'], z['xyz_nm'], z['parent']
        offset = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        cell_type = np.repeat(np.array([body_type.get(int(b), -1) for b in bodies]), counts)
        child = np.flatnonzero((parent >= 0) & (cell_type >= 0))
        par = parent[child].astype(np.int64) + offset[child]
        length = np.sqrt(((xyz[child] - xyz[par]).astype(np.float64) ** 2).sum(1)) * 8.0 / 1000.0
        key = cell_type[child].astype(np.int64) * SHIFT + voxel_key((xyz[child] + xyz[par]) * 0.5, s)
        uq, inv = np.unique(key, return_inverse=True)
        keys.append(uq); weights.append(np.bincount(inv, length))
        if len(keys) >= 12:                             # merge as we go: bounded memory
            uq, inv = np.unique(np.concatenate(keys), return_inverse=True)
            keys, weights = [uq], [np.bincount(inv, np.concatenate(weights))]
    uq, inv = np.unique(np.concatenate(keys), return_inverse=True)
    return uq, np.bincount(inv, np.concatenate(weights)).astype(np.float32)


def synapse_counts(names, scale_um, eval_pair, keep_eval=True):
    """(type, voxel) output and input synapse counts, split into training-pair synapses and eval-pair synapses."""
    a = pd.read_feather(f'{D}/body-annotations.feather', columns=['bodyId', 'type'])
    index = {t: i for i, t in enumerate(names)}
    body_type = dict(zip(a.bodyId.values.astype(np.int64), [index.get(t, -1) for t in a.type.values]))
    ids = np.load('data/malecns_v1/fly_reservoir_coba_r2.npz')['neuron_body_ids'].astype(np.int64)
    cell_type = np.array([body_type.get(int(b), -1) for b in ids])
    s = scale_um * 1000.0 / 8.0
    acc = {k: ([], []) for k in ('out_train', 'in_train', 'out_eval', 'in_eval')}
    for path in sorted(glob.glob(f'{BUCKETS}/post_*.i32')):
        r = np.fromfile(path, dtype=np.int32).reshape(-1, 8)
        tp, tq = cell_type[r[:, 0]], cell_type[r[:, 1]]
        ev = (tp >= 0) & (tq >= 0)
        ev[ev] = eval_pair[tq[ev], tp[ev]]
        for side, t, xyz in (('out', tq, r[:, 5:8]), ('in', tp, r[:, 2:5])):
            for part, m in ((('train', ~ev & (t >= 0)), ('eval', ev)) if keep_eval else (('train', ~ev & (t >= 0)),)):
                key = t[m].astype(np.int64) * SHIFT + voxel_key(xyz[m].astype(np.float64), s)
                uq, cnt = np.unique(key, return_counts=True)
                ks, cs = acc[f'{side}_{part}']
                ks.append(uq); cs.append(cnt)
                if len(ks) >= 12:                       # merge as we go: bounded memory
                    m_uq, inv = np.unique(np.concatenate(ks), return_inverse=True)
                    m_c = np.bincount(inv, np.concatenate(cs))
                    ks[:] = [m_uq]; cs[:] = [m_c]
    out = {}
    for k, (ks, cs) in acc.items():
        if not ks:
            continue
        uq, inv = np.unique(np.concatenate(ks), return_inverse=True)
        out[k] = (uq, np.bincount(inv, np.concatenate(cs)))
    return out


def lookup(keys, table):
    uq, val = table
    pos = np.searchsorted(uq, keys).clip(0, len(uq) - 1)
    return np.where(uq[pos] == keys, val[pos], 0.0).astype(np.float32)


def directed_overlap(ckeys, cw, out_cnt, in_cnt, n, dev, alpha=1.0):
    t = (ckeys // SHIFT).astype(np.int64)
    o, i = out_cnt, in_cnt
    g = np.bincount(t, o, n) / np.maximum(np.bincount(t, o + i, n), 1e-9)
    f = (o + alpha * g[t]) / (o + i + alpha)
    w_out, w_in = (cw * f).astype(np.float32), (cw * (1 - f)).astype(np.float32)
    del f
    vox = ckeys % SHIFT                                     # voxel columns by sorting (no full unique: memory)
    order = np.argsort(vox, kind='stable')
    vox = vox[order]
    col = np.empty(len(vox), np.int32)
    col[0] = 0
    np.cumsum(vox[1:] != vox[:-1], out=col[1:])
    n_vox = int(col[-1]) + 1
    del vox
    t, w_out, w_in = t[order].astype(np.int32), w_out[order], w_in[order]
    del order
    chunk = 60000
    bounds = np.searchsorted(col, np.arange(0, n_vox + chunk, chunk))
    E = torch.zeros(n, n, dtype=torch.float64, device=dev)
    for lo, hi, c0 in zip(bounds[:-1], bounds[1:], range(0, n_vox + chunk, chunk)):
        if hi <= lo:
            continue
        ti = torch.as_tensor(t[lo:hi].astype(np.int64), device=dev); ci = torch.as_tensor((col[lo:hi] - c0).astype(np.int64), device=dev)
        A = torch.zeros(n, chunk, device=dev); B = torch.zeros(n, chunk, device=dev)
        A[ti, ci] = torch.as_tensor(w_out[lo:hi], device=dev); B[ti, ci] = torch.as_tensor(w_in[lo:hi], device=dev)
        E += (A @ B.T).double()
    return E.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--scale', type=float, default=4.0)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--l2s', default='0.0001,0.001,0.01,0.1')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/e1c_polarity_20261020.json')
    ap.add_argument('--no-leak-check', dest='leak_check', action='store_false',
                    help='skip the all-synapse labelling (measured at 4 um: +2.7 points of strength)')
    args = ap.parse_args()
    t0 = time.time()
    z = np.load(args.cache, allow_pickle=True)
    W_np, sizes, names = z['W'], z['sizes'], list(z['names'])
    n = len(names)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand(W_np.shape, generator=g)                  # the E1 split, drawn identically
    split = {'train': u >= 0.10, 'val': u < 0.05, 'test': (u >= 0.05) & (u < 0.10)}
    eval_pair = (u < 0.10).numpy()
    versions = ('trainlabels', 'alllabels') if args.leak_check else ('trainlabels',)
    paths = {v: f'{args.work}/e1c_overlap_{v}_{args.scale:g}um.npy' for v in versions}
    if all(os.path.exists(p) for p in paths.values()):
        E = {v: np.load(p) for v, p in paths.items()}
    else:
        ckeys, cw = cable(names, args.scale)
        print(f'cable: {len(cw)} type-voxel entries', flush=True)
        cnt = synapse_counts(names, args.scale, eval_pair, keep_eval=args.leak_check)
        print({k: int(v[1].sum()) for k, v in cnt.items()}, flush=True)
        o_tr, i_tr = lookup(ckeys, cnt['out_train']), lookup(ckeys, cnt['in_train'])
        o_ev, i_ev = ((lookup(ckeys, cnt['out_eval']), lookup(ckeys, cnt['in_eval'])) if args.leak_check else (None, None))
        del cnt
        E = {'trainlabels': directed_overlap(ckeys, cw, o_tr, i_tr, n, dev)}
        np.save(paths['trainlabels'], E['trainlabels'])          # saved before the leak check can fail
        if args.leak_check:
            E['alllabels'] = directed_overlap(ckeys, cw, o_tr + o_ev, i_tr + i_ev, n, dev)
            np.save(paths['alllabels'], E['alllabels'])
    W = torch.tensor(W_np, dtype=torch.float32, device=dev)
    size = torch.tensor(sizes, dtype=torch.float32, device=dev)
    conn = (W >= MIN_SYN).float()
    dens = torch.log(W.clamp_min(1) / (size[:, None] * size[None]))
    split = {m: v.to(dev) for m, v in split.items()}
    tasks = {'exists': ('bern', conn, split),
             'strength': ('gauss', dens, {m: v & (conn > 0) for m, v in split.items()})}
    nulls = {}
    for tname, (fam, y, ms) in tasks.items():
        e, _ = fit(y, ms['train'], fam, steps=args.steps, dev=dev)
        nulls[tname] = {m: loss_sum(e, y, fam, ms[m]) for m in ms}

    def features(Em):
        e = torch.tensor(Em / (sizes[:, None] * sizes[None]), dtype=torch.float32, device=dev)
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
        print(f"{r['model']:44s} | exists test {r['exists']['test']:+.3f} auc {r['exists']['test_auc']:.3f} | "
              f"strength test {r['strength']['test']:+.3f}" + (f" | strength coef log E {w[0]:+.3f}" if w else ''), flush=True)

    conn_np = W_np >= MIN_SYN
    zero = {v: {'all_pairs': round(float((E[v] == 0).mean()), 4),
                'connected_pairs': round(float((E[v][conn_np] == 0).mean()), 4)} for v in E}
    print(f'zero directed overlap: {zero}', flush=True)
    rows = []
    for v in versions:
        rows.append({'model': f'degree + directed encounter ({v})', **best(features(E[v]))}); show(rows[-1])
    both = torch.cat([features(E['trainlabels']), features(np.ascontiguousarray(E['trainlabels'].T))], -1)
    rows.append({'model': 'degree + directed both ways (trainlabels)', **best(both)}); show(rows[-1])
    rows.append({'model': f'code k={args.k} + directed encounter (trainlabels)', **best(features(E['trainlabels']), k=args.k)})
    show(rows[-1])
    rep = {'types': n, 'scale_um': args.scale, 'zero_directed_overlap': zero,
           'polarity': 'f = (out + alpha g)/(out + in + alpha) per type-voxel, alpha 1; trainlabels exclude val/test pairs',
           'reference': {'undirected_4um': {'exists': 0.591, 'strength': 0.151, 'coef': 0.507},
                         'code16_plus_undirected_4um': {'exists': 0.657, 'strength': 0.187}},
           'rows': rows, 'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print(f"done in {rep['seconds']} s -> {args.out}")


if __name__ == '__main__':
    main()
