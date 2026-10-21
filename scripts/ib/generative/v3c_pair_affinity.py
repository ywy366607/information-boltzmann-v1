"""CGB V3c - is the symmetric pair affinity pair-specific or code-like? (pre-registration: CGB_V3_RESULTS_20261020.md)

E1d footing: right-side log synapses per cell pair, right-connected held-out pairs, E1 split, degree-only null.
"""
import argparse, json, time
import numpy as np, torch
from e1_genomic_bottleneck import loss_sum

MIN_SYN_SIDE = 3


def fit(target, train, feats=None, k=0, sym=True, l2=0.0, steps=1500, dev='cuda', seed=0):
    torch.manual_seed(seed)
    n = target.shape[0]
    o = torch.zeros(n, device=dev, requires_grad=True); i = torch.zeros(n, device=dev, requires_grad=True)
    b0 = torch.tensor(float(target[train].mean()), device=dev, requires_grad=True)
    params, pen, wf = [o, i, b0], [], None
    if feats is not None:
        wf = torch.zeros(feats.shape[-1], device=dev, requires_grad=True); params.append(wf)
    if k > 0:
        C = (0.1 * torch.randn(n, k, device=dev)).requires_grad_(); A = torch.eye(k, device=dev).requires_grad_()
        params += [C, A]; pen += [C, A]

    def eta():
        e = b0 + o[:, None] + i[None]
        if k > 0:
            Am = 0.5 * (A + A.T) if sym else A
            e = e + C @ Am @ C.T
        if wf is not None:
            e = e + feats @ wf
        return e
    opt = torch.optim.Adam(params, lr=0.05)
    for s in range(steps):
        opt.zero_grad()
        e = eta()
        loss = (e[train] - target[train]).pow(2).mean()
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
    ap.add_argument('--sides', default='F:/fly_stream_overflow/cgb_v1/e1_types3000_sides.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--l2s', default='0.0001,0.001,0.01,0.1,1,10')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/v3c_pair_affinity_20261020.json')
    args = ap.parse_args()
    t0 = time.time()
    sizes = np.load(args.cache, allow_pickle=True)['sizes']
    z = np.load(args.sides)
    WR, nr, nl = z['WR'].astype(np.float64), z['nr'], z['nl']
    n = len(sizes)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    bil = (nl > 0) & (nr > 0)
    yR_np = np.log(np.maximum(WR, 1) / (np.maximum(nr, 1)[:, None] * sizes[None]))
    y = T(yR_np)
    ok = T((WR >= MIN_SYN_SIDE) & bil[:, None]).bool()
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand((n, n), generator=g).to(dev)
    ms = {'train': (u >= 0.10) & ok, 'val': (u < 0.05) & ok, 'test': (u >= 0.05) & (u < 0.10) & ok}
    out = []
    for s in (4, 2):
        E = np.load(f'{args.work}/e1c_overlap_trainlabels_{s:g}um.npy')
        for M in (E, np.ascontiguousarray(E.T)):
            e = M / (sizes[:, None] * sizes[None])
            floor = e[e > 0].min() * 0.5
            out += [np.log(np.maximum(e, floor)), (e == 0).astype(float)]
    G = T(np.stack(out, -1))
    conn_rev = (WR.T >= MIN_SYN_SIDE)
    REV = T(np.stack([np.where(conn_rev, yR_np.T, 0.0), conn_rev.astype(float)], -1))
    test_np = ms['test'].cpu().numpy()
    rev_share = float(conn_rev[test_np].mean())
    print(f'test pairs {int(test_np.sum())}, reverse direction connected in {rev_share:.1%}', flush=True)

    null_e, _ = fit(y, ms['train'], steps=args.steps, dev=dev)
    null = {m: loss_sum(null_e, y, 'gauss', ms[m]) for m in ms}
    rows = []

    def add(name, feats=None, k=0, sym=True):
        top = None
        for l2 in ([float(x) for x in args.l2s.split(',')] if k else [0.0]):
            e, w = fit(y, ms['train'], feats=feats, k=k, sym=sym, l2=l2, steps=args.steps, dev=dev)
            r = {m: round(1 - loss_sum(e, y, 'gauss', ms[m]) / null[m], 4) for m in ms}
            if top is None or r['val'] > top['val']:
                top = {**r, 'l2': l2, 'feature_weights': w}
            if k:
                print(f'   {name} l2 {l2:g}: val {r["val"]:+.4f} test {r["test"]:+.4f}', flush=True)
        rows.append({'model': name, **top})
        print(f"{name:40s} test {top['test']:+.4f} val {top['val']:+.4f} l2 {top['l2']}", flush=True)
    add('geometry', G)
    add('geometry + reverse', torch.cat([G, REV], -1))
    for k in (16, 64, 256):
        add(f'geometry + symmetric code {k}', G, k=k)
    add('geometry + general code 64', G, k=64, sym=False)
    add('geometry + symmetric code 64 + reverse', torch.cat([G, REV], -1), k=64)
    r = {x['model']: x['test'] for x in rows}
    base = r['geometry']
    rev_gain = r['geometry + reverse'] - base
    best_sym = max(r[f'geometry + symmetric code {k}'] for k in (16, 64, 256)) - base
    keep = r['geometry + symmetric code 64 + reverse'] - r['geometry + symmetric code 64']
    verdict = {'reverse_gain': round(rev_gain, 4), 'best_symmetric_code_gain': round(best_sym, 4),
               'reverse_gain_beyond_code64': round(keep, 4),
               'Pb1 reverse >= +0.03': rev_gain >= 0.03,
               'Pb2 best symmetric code < 1/3 of reverse gain': best_sym < rev_gain / 3,
               'Pb3 reverse keeps >= 2/3 beyond code 64': keep >= 2 * rev_gain / 3}
    rep = {'preregistration': 'docs/information_boltzmann/CGB_V3_RESULTS_20261020.md (V3c)',
           'reverse_connected_share_test': rev_share, 'rows': rows, 'verdict': verdict,
           'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print('verdict', verdict, f"\ndone in {rep['seconds']} s -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
