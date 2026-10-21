"""E1 - genomic bottleneck on the maleCNS connectome (2026-10-18).
How few numbers per cell type generate the type-to-type wiring?  A type gets a k-dim code c and a matching matrix A:
    eta_ab = b0 + o_a + i_b + c_a^T A c_b                     (o, i: out/in degree terms)
Two targets, both robust to the heavy-tailed synapse counts (a Poisson count score was dominated by a few huge pairs:
held-out val 0.45 vs test 0.10 in the smoke run):
  exists:   a type pair has >= MIN_SYN synapses (logistic), also AUC
  strength: log synapses per cell pair, on connected pairs (squared error)
Score = share of the loss explained beyond the degree-only null, on type pairs never seen in fitting
(90% train / 5% val picks the L2 / 5% test).  Reference generators with no per-type freedom: one-hot superclass,
hemilineage, hemilineage x superclass with a free matching matrix; and lineage block + learned code.
Prediction (docs/information_boltzmann/BRAIN_GENERATIVE_PRINCIPLE_20261018.md): k ~ 32 explains > 70% held-out, and
with the lineage prior the k needed for the same accuracy halves.
Usage: python e1_genomic_bottleneck.py [--types 3000] [--ks 2,4,8,16,32,64,128] [--cache x.npz] [--out ...json]"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
import torch.nn.functional as F
import pyarrow.feather as pf

D = 'data/malecns_v1'
MIN_SYN = 5


def type_matrix(n_types):
    a = pd.read_feather(f'{D}/body-annotations.feather')[['bodyId', 'type', 'itoleeHl', 'superclass']]
    a = a[a.type.notna()]
    ids = a.bodyId.values.astype(np.int64)
    order = np.argsort(ids)
    ids = ids[order]
    tcode, tnames = pd.factorize(a.type.values[order])
    T = len(tnames)
    t = pf.read_table(f'{D}/connectome-weights.feather', memory_map=True)

    def typed_batches():          # streamed twice (caching 152M rows does not fit beside the rest)
        for b in t.to_batches(max_chunksize=5_000_000):
            p = b.column(0).to_numpy().astype(np.int64); q = b.column(1).to_numpy().astype(np.int64)
            v = b.column(2).to_numpy().astype(np.float64)
            ip = np.searchsorted(ids, p).clip(0, len(ids) - 1); iq = np.searchsorted(ids, q).clip(0, len(ids) - 1)
            ok = (ids[ip] == p) & (ids[iq] == q)
            yield tcode[ip[ok]], tcode[iq[ok]], v[ok]
    # pass 1: total synapses per type (in + out) to keep the busiest n_types
    tot = np.zeros(T)
    for tp, tq, v in typed_batches():
        tot += np.bincount(tp, v, T) + np.bincount(tq, v, T)
    keep = np.argsort(tot)[::-1][:n_types]
    remap = -np.ones(T, np.int64); remap[keep] = np.arange(len(keep))
    n = len(keep)
    W = np.zeros(n * n, np.float64)
    for tp, tq, v in typed_batches():
        a_, b_ = remap[tp], remap[tq]
        ok = (a_ >= 0) & (b_ >= 0)
        W += np.bincount(a_[ok] * n + b_[ok], v[ok], n * n)
    W = W.reshape(n, n)
    names = np.asarray(tnames)[keep]
    sizes = np.bincount(tcode, minlength=T)[keep].astype(np.float64)
    meta = a.iloc[order].assign(tc=tcode).groupby('tc').agg(hl=('itoleeHl', lambda s: s.mode().iloc[0] if s.notna().any() else 'none'),
                                                         sc=('superclass', lambda s: s.mode().iloc[0] if s.notna().any() else 'none'))
    hl = meta.hl.reindex(keep).fillna('none').values
    sc = meta.sc.reindex(keep).fillna('none').values
    return W, sizes, names, hl, sc, float(tot.sum() / 2)


def fit(target, train, family, codes_fixed=None, k=0, l2=0.0, steps=1500, dev='cuda', seed=0):
    """eta_ab = b0 + o_a + i_b [+ x_a^T Af x_b] [+ c_a^T A c_b], fitted on the training pairs.
    family 'bern': logistic on 0/1; 'gauss': squared error."""
    torch.manual_seed(seed)
    n = target.shape[0]
    o = torch.zeros(n, device=dev, requires_grad=True); i = torch.zeros(n, device=dev, requires_grad=True)
    b0 = torch.tensor(float(target[train].mean()) if family == 'gauss' else 0.0, device=dev, requires_grad=True)
    params, terms, pen = [o, i, b0], [], []
    if codes_fixed is not None:
        m = codes_fixed.shape[1]
        Af = torch.zeros(m, m, device=dev, requires_grad=True); params.append(Af); pen.append(Af)
        terms.append(lambda: codes_fixed @ Af @ codes_fixed.T)
    if k > 0:
        C = (0.1 * torch.randn(n, k, device=dev)).requires_grad_(); A = torch.eye(k, device=dev).requires_grad_()
        params += [C, A]; pen += [C, A]
        terms.append(lambda: C @ A @ C.T)
    eta = lambda: b0 + o[:, None] + i[None] + (sum(f() for f in terms) if terms else 0.0)
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
        return eta()


def loss_sum(e, target, family, mask):
    if family == 'bern':
        return float(F.binary_cross_entropy_with_logits(e[mask], target[mask], reduction='sum'))
    return float((e[mask] - target[mask]).pow(2).sum())


def auc(e, y):
    e, y = e.double().cpu().numpy(), y.cpu().numpy() > 0.5
    r = np.argsort(np.argsort(e)) + 1.0
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', type=int, default=3000)
    ap.add_argument('--ks', default='2,4,8,16,32,64,128')
    ap.add_argument('--l2s', default='0.0001,0.001,0.01,0.1')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--cache', default='')
    ap.add_argument('--out', default='results/published/e1_genomic_bottleneck_20261018.json')
    args = ap.parse_args()
    t0 = time.time()
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        W_np, sizes, names, hl, sc, total = z['W'], z['sizes'], z['names'], z['hl'], z['sc'], float(z['total'])
    else:
        W_np, sizes, names, hl, sc, total = type_matrix(args.types)
        if args.cache:
            np.savez(args.cache, W=W_np, sizes=sizes, names=names, hl=hl, sc=sc, total=total)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    W = torch.tensor(W_np, dtype=torch.float32, device=dev)
    size = torch.tensor(sizes, dtype=torch.float32, device=dev)
    conn = (W >= MIN_SYN).float()
    dens = torch.log(W.clamp_min(1) / (size[:, None] * size[None]))
    print(f'{len(names)} types, {W_np.sum():.0f} of {total:.0f} synapses ({W_np.sum() / total:.1%}); '
          f'connected pairs (>= {MIN_SYN}) {float(conn.mean()):.2%}', flush=True)
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand(W.shape, generator=g).to(dev)
    split = {'train': u >= 0.10, 'val': u < 0.05, 'test': (u >= 0.05) & (u < 0.10)}    # unseen type pairs
    tasks = {'exists': ('bern', conn, split),
             'strength': ('gauss', dens, {m: v & (conn > 0) for m, v in split.items()})}
    nulls = {}
    for t, (fam, y, ms) in tasks.items():
        e = fit(y, ms['train'], fam, steps=args.steps, dev=dev)
        nulls[t] = {m: loss_sum(e, y, fam, ms[m]) for m in ms}
        if fam == 'bern':
            null_auc = round(auc(e[ms['test']], y[ms['test']]), 4)

    def best(**kw):          # per task, L2 chosen on val; explained = 1 - loss / degree-only null loss
        out = {}
        for t, (fam, y, ms) in tasks.items():
            top = None
            for l2 in [float(x) for x in args.l2s.split(',')]:
                e = fit(y, ms['train'], fam, l2=l2, steps=args.steps, dev=dev, **kw)
                r = {m: round(1 - loss_sum(e, y, fam, ms[m]) / nulls[t][m], 4) for m in ms}
                if top is None or r['val'] > top['val']:
                    top = {**r, 'l2': l2}
                    if fam == 'bern':
                        top['test_auc'] = round(auc(e[ms['test']], y[ms['test']]), 4)
            out[t] = top
        return out

    def onehot(lab):
        codes, uq = pd.factorize(lab)
        return F.one_hot(torch.tensor(codes), len(uq)).float().to(dev), len(uq)

    def show(r):
        print(f"{r['model']:38s} numbers {r.get('numbers', 0):>8d} | exists test {r['exists']['test']:+.3f} "
              f"auc {r['exists']['test_auc']:.3f} | strength test {r['strength']['test']:+.3f}", flush=True)

    rows = []
    print(f'degree only (null): exists test auc {null_auc}', flush=True)
    hs = np.char.add(np.char.add(hl.astype(str), '|'), sc.astype(str))
    for label, lab in (('superclass block', sc), ('hemilineage block', hl), ('hemilineage x superclass block', hs)):
        X, m = onehot(lab)
        rows.append({'model': label, 'labels': m, 'numbers': m * m, **best(codes_fixed=X)}); show(rows[-1])
    X, m = onehot(hs)
    for k in [int(x) for x in args.ks.split(',')]:
        rows.append({'model': f'learned code k={k}', 'numbers': len(names) * k + k * k, **best(k=k)}); show(rows[-1])
        if k <= 32:
            rows.append({'model': f'lineage block + learned code k={k}', 'numbers': m * m + len(names) * k + k * k,
                         **best(k=k, codes_fixed=X)}); show(rows[-1])
    rep = {'types': len(names), 'synapse_share': round(float(W_np.sum() / total), 4), 'min_syn': MIN_SYN,
           'connected_share': round(float(conn.mean()), 4), 'null_exists_test_auc': null_auc,
           'split': 'type pairs: 90% train / 5% val (picks L2) / 5% test',
           'metric': 'share of loss explained beyond the degree-only null; exists: logistic on >= min_syn synapses; '
                     'strength: squared error of log synapses per cell pair on connected pairs',
           'rows': rows, 'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print(f"done in {rep['seconds']} s -> {args.out}")


if __name__ == '__main__':
    main()
