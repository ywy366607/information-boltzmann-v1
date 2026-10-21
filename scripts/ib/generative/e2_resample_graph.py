"""E2 - rewire a region of the fly connectome from the generator and write a graph the newborn can be born from.

Arms (the region's fast edges only; everything else, including the dopamine list, stays real):
  real       the r2 graph unchanged (reference)
  typelevel  ideal type-level generator: every type pair keeps its real edges and synapse counts, but each edge's
             pre and post cells are redrawn uniformly within their types (cell-level specificity removed)
  generator  each type pair's cell-pair connection density and synapses per connected pair are predicted by the
             E1 generator (directed encounter 4 um + k-dim code, fitted on all type pairs); edges are drawn
             Bernoulli per cell pair, counts 1 + geometric
  null       degree-preserving shuffle of post cells within (pre class, post class) blocks: coarse blocks kept,
             type-level structure destroyed
Weights follow the r2 rule (count x unit; unit_i = 3.67 unit_e; Kenyon claw inputs x 1.8); delays follow the
soma-distance rule of the graph (1..4 at 210 / 510 / 810 um). Region 'mb': posts in Kenyon_Cell, MBON, APL, DPM;
pres in Kenyon_Cell, ALPN, MBON, APL, DPM.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1b_encounter import fit

GRAPH = 'data/malecns_v1/fly_reservoir_coba_r2.npz'
REGIONS = {'mb': (('Kenyon_Cell', 'ALPN', 'MBON'), ('Kenyon_Cell', 'MBON'), ('APL', 'DPM'))}   # pre classes, post classes, types in both


def delay_of(coords, pre, post):
    d = np.linalg.norm(coords[pre] - coords[post], axis=1)
    return (1 + (d > 210) + (d > 510) + (d > 810)).astype(np.int32)


def fit_generator(names, sizes, cell_type, pre, post, count, work, k, dev, steps=1500):
    """Per type pair: logit density of connected cell pairs and log mean synapses per connected pair."""
    n = len(names)
    tp, tq = cell_type[pre], cell_type[post]
    ok = (tp >= 0) & (tq >= 0)
    key = tp[ok] * n + tq[ok]
    conn = np.bincount(key, minlength=n * n).reshape(n, n).astype(np.float64)       # connected cell pairs
    syn = np.bincount(key, count[ok], n * n).reshape(n, n)
    pairs = sizes[:, None] * sizes[None]
    E = np.load(f'{work}/e1c_overlap_trainlabels_4um.npy')
    e = E / pairs; floor = e[e > 0].min() * 0.5
    feats = torch.tensor(np.stack([np.log(np.maximum(e, floor)), (e == 0).astype(float)], -1), dtype=torch.float32, device=dev)
    dens = torch.tensor(np.clip(conn / pairs, 1e-6, 1 - 1e-6), dtype=torch.float32, device=dev)
    w = torch.tensor(pairs, dtype=torch.float32, device=dev)
    all_pairs = torch.ones(n, n, dtype=torch.bool, device=dev)
    # density: binomial likelihood on the counts of connected cell pairs (weights = cell pairs); fitted as a weighted
    # logistic regression on the empirical density
    eta_d = fit_weighted(dens, w, feats, k, dev, steps)
    has = conn > 0
    mlog = torch.tensor(np.log(np.where(has, syn / np.maximum(conn, 1), 1.0)), dtype=torch.float32, device=dev)
    eta_m, _ = fit(mlog, torch.tensor(has, device=dev), 'gauss', feats=feats, k=k, l2=0.001, steps=steps, dev=dev)
    # extrapolated pairs can predict absurd means: bound them by the largest mean the data shows
    top = float(np.log((syn / np.maximum(conn, 1))[has].max()))
    return torch.sigmoid(eta_d).cpu().numpy(), np.exp(np.minimum(eta_m.cpu().numpy(), top)), conn, syn


def fit_weighted(y, w, feats, k, dev, steps):
    n = y.shape[0]
    torch.manual_seed(0)
    o = torch.zeros(n, device=dev, requires_grad=True); i = torch.zeros(n, device=dev, requires_grad=True)
    b0 = torch.tensor(-6.0, device=dev, requires_grad=True); wf = torch.zeros(feats.shape[-1], device=dev, requires_grad=True)
    C = (0.1 * torch.randn(n, k, device=dev)).requires_grad_(); A = torch.eye(k, device=dev).requires_grad_()
    opt = torch.optim.Adam([o, i, b0, wf, C, A], lr=0.05)
    wn = w / w.sum()
    eta = lambda: b0 + o[:, None] + i[None] + C @ A @ C.T + feats @ wf
    for s in range(steps):
        opt.zero_grad()
        e = eta()
        loss = (wn * torch.nn.functional.binary_cross_entropy_with_logits(e, y, reduction='none')).sum() \
            + 0.001 * (C.pow(2).mean() + A.pow(2).mean())
        loss.backward(); opt.step()
        if s == steps // 2:
            for g in opt.param_groups:
                g['lr'] = 0.01
    with torch.no_grad():
        return eta()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['real', 'typelevel', 'generator', 'null'], required=True)
    ap.add_argument('--region', default='mb', choices=sorted(REGIONS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    g = dict(np.load(GRAPH, allow_pickle=False))
    ids = g['neuron_body_ids'].astype(np.int64)
    ann = pd.read_feather('data/malecns_v1/body-annotations.feather', columns=['bodyId', 'class', 'type']).drop_duplicates('bodyId').set_index('bodyId').reindex(ids)
    cls, typ = ann['class'].astype(str).values, ann['type'].astype(str).values
    pre_cls, post_cls, both_types = REGIONS[args.region]
    in_pre = np.isin(cls, pre_cls) | np.isin(typ, both_types)
    in_post = np.isin(cls, post_cls) | np.isin(typ, both_types)
    z = np.load(args.cache, allow_pickle=True)
    names, sizes = list(z['names']), z['sizes'].astype(np.float64)
    tindex = {t: i for i, t in enumerate(names)}
    cell_type = np.array([tindex.get(t, -1) for t in typ])
    unit_e, unit_i = float(g['edge_weight_e'].min()), float(g['edge_weight_i'].min())
    kenyon = cls == 'Kenyon_Cell'
    sign = g['nt_sign']
    # pool both fast lists into (pre, post, count, is_inhibitory)
    pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]).astype(np.int64)
    post = np.concatenate([g['edge_post_e'], g['edge_post_i']]).astype(np.int64)
    inh = np.concatenate([np.zeros(len(g['edge_pre_e']), bool), np.ones(len(g['edge_pre_i']), bool)])
    claw = ~inh & kenyon[post] & ~kenyon[pre]
    w = np.concatenate([g['edge_weight_e'], g['edge_weight_i']]).astype(np.float64)
    count = np.round(w / np.where(inh, unit_i, unit_e * np.where(claw, 1.8, 1.0))).astype(np.int64)
    region = in_pre[pre] & in_post[post]
    typed = (cell_type[pre] >= 0) & (cell_type[post] >= 0)
    report = {'arm': args.arm, 'region': args.region, 'seed': args.seed, 'region_edges': int(region.sum()),
              'region_synapses': int(count[region].sum()), 'region_edges_untyped_kept_real': int((region & ~typed).sum())}
    move = region & typed if args.arm != 'null' else region
    keep_pre, keep_post, keep_c, keep_inh = pre[~move], post[~move], count[~move], inh[~move]
    rp, rq, rc = pre[move], post[move], count[move]
    if args.arm == 'real':
        new_pre, new_post, new_c = rp, rq, rc
    elif args.arm == 'typelevel':
        def redraw(cells):
            out = cells.copy()
            tt = cell_type[cells]
            for t in np.unique(tt):
                m = np.flatnonzero(tt == t)
                pool = np.flatnonzero(cell_type == t)
                out[m] = pool[rng.integers(0, len(pool), len(m))]
            return out
        new_pre, new_post, new_c = redraw(rp), redraw(rq), rc
    elif args.arm == 'null':
        new_pre, new_post, new_c = rp.copy(), rq.copy(), rc
        _, cls_code = np.unique(cls, return_inverse=True)
        block = cls_code[rp].astype(np.int64) * (cls_code.max() + 1) + cls_code[rq]
        order = np.argsort(block, kind='stable')
        bounds = np.flatnonzero(np.diff(block[order])) + 1
        for m in np.split(order, bounds):
            new_post[m] = rq[m][rng.permutation(len(m))]
    else:   # generator
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        d_hat, m_hat, conn, syn = fit_generator(names, sizes, cell_type, pre, post, count, args.work, args.k, dev)
        tp, tq = cell_type[rp], cell_type[rq]
        type_pairs = np.unique(tp * len(names) + tq)
        members = {}
        out_pre, out_post, out_c = [], [], []
        for key in type_pairs:
            a, b = divmod(int(key), len(names))
            A = members.setdefault(a, np.flatnonzero(cell_type == a)); B = members.setdefault(b, np.flatnonzero(cell_type == b))
            n_pairs = len(A) * len(B)
            k_edges = rng.binomial(n_pairs, float(d_hat[a, b]))
            if k_edges == 0:
                continue
            flat = rng.choice(n_pairs, size=k_edges, replace=False)
            ii, jj = A[flat // len(B)], B[flat % len(B)]
            ii, jj = ii[ii != jj], jj[ii != jj]
            mean = max(float(m_hat[a, b]), 1.0)
            c = 1 + rng.geometric(1.0 / mean, size=len(ii)) - 1 if mean > 1 else np.ones(len(ii), np.int64)
            out_pre.append(ii); out_post.append(jj); out_c.append(c)
        new_pre, new_post, new_c = np.concatenate(out_pre), np.concatenate(out_post), np.concatenate(out_c).astype(np.int64)
        report['generator_type_pairs'] = int(len(type_pairs))
        real_region_syn = int(rc.sum())
        report['generated_vs_real'] = {'edges': [int(len(new_pre)), int(len(rp))], 'synapses': [int(new_c.sum()), real_region_syn]}
    new_inh = sign[new_pre] < 0
    pre2 = np.concatenate([keep_pre, new_pre]); post2 = np.concatenate([keep_post, new_post])
    c2 = np.concatenate([keep_c, new_c]); inh2 = np.concatenate([keep_inh, new_inh])
    claw2 = ~inh2 & kenyon[post2] & ~kenyon[pre2]
    w2 = c2 * np.where(inh2, unit_i, unit_e * np.where(claw2, 1.8, 1.0))
    d2 = delay_of(g['coords_um'], pre2, post2)
    for tag, m in (('e', ~inh2), ('i', inh2)):
        order = np.argsort(d2[m], kind='stable')
        g[f'edge_pre_{tag}'] = pre2[m][order].astype(np.int32)
        g[f'edge_post_{tag}'] = post2[m][order].astype(np.int32)
        g[f'edge_weight_{tag}'] = w2[m][order].astype(np.float32)
        g[f'edge_delay_{tag}'] = d2[m][order].astype(np.int32)
        dd = g[f'edge_delay_{tag}']
        g[f'delay_splits_{tag}'] = np.array([0] + [int(np.searchsorted(dd, x, side='right')) for x in (1, 2, 3, 4)], dtype=np.int64)
    g['meta'] = np.array(f"{g['meta']} | E2 {args.arm} region={args.region} seed={args.seed}")
    report.update(edges_e=int((~inh2).sum()), edges_i=int(inh2.sum()), seconds=round(time.time() - t0))
    # the trainer reads annotations and other tables next to the graph: the graph must live in data/malecns_v1
    assert args.out.resolve().parent == Path('data/malecns_v1').resolve(), 'write the graph into data/malecns_v1'
    np.savez_compressed(args.out, **g)
    (args.out.with_suffix('.json')).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
