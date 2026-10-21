"""E1d - decompose connection strength on one footing (2026-10-20).

Target: right-side strength, log(W_R[a, b] / (n_R(a) n(b))), on held-out type pairs connected on the right
(>= MIN_SYN_SIDE), with the E1 split. All models are scored as the share of squared error explained beyond the
degree-only null on the same test pairs:
  replicate             + the left twin's strength of the same pair (the reproducible ceiling, half data)
  geometry s            + directed encounter at scale s (axon of a x dendrite of b, both directions, E1c caches)
  code                  + learned k-dim matching code
  code + geometry       both
  geometry + replicate  what the twin still adds beyond geometry = reproducible strength geometry misses
"""
import argparse, json, time
import numpy as np, torch
from e1_genomic_bottleneck import loss_sum
from e1b_encounter import fit

MIN_SYN_SIDE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--sides', default='F:/fly_stream_overflow/cgb_v1/e1_types3000_sides.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--scales', default='4,2')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--l2s', default='0.0001,0.001,0.01,0.1')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/e1d_strength_ceiling_20261020.json')
    args = ap.parse_args()
    t0 = time.time()
    sizes = np.load(args.cache, allow_pickle=True)['sizes']
    z = np.load(args.sides)
    WL, WR, nl, nr = z['WL'], z['WR'], z['nl'], z['nr']
    n = len(sizes)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bil = (nl > 0) & (nr > 0)
    y = torch.tensor(np.log(np.maximum(WR, 1) / (np.maximum(nr, 1)[:, None] * sizes[None])), dtype=torch.float32, device=dev)
    ok = torch.tensor((WR >= MIN_SYN_SIDE) & bil[:, None], device=dev)
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand((n, n), generator=g).to(dev)
    ms = {'train': (u >= 0.10) & ok, 'val': (u < 0.05) & ok, 'test': (u >= 0.05) & (u < 0.10) & ok}
    twin = np.log(np.maximum(WL, 1) / (np.maximum(nl, 1)[:, None] * sizes[None]))
    twin_f = torch.tensor(np.stack([twin, (WL >= MIN_SYN_SIDE).astype(float)], -1), dtype=torch.float32, device=dev)

    def geo(scale):
        E = np.load(f'{args.work}/e1c_overlap_trainlabels_{scale:g}um.npy')
        out = []
        for M in (E, np.ascontiguousarray(E.T)):
            e = M / (sizes[:, None] * sizes[None])
            floor = e[e > 0].min() * 0.5
            out += [np.log(np.maximum(e, floor)), (e == 0).astype(float)]
        return torch.tensor(np.stack(out, -1), dtype=torch.float32, device=dev)

    null_e, _ = fit(y, ms['train'], 'gauss', steps=args.steps, dev=dev)
    null = {m: loss_sum(null_e, y, 'gauss', ms[m]) for m in ms}

    def score(feats=None, k=0):
        top = None
        for l2 in ([float(x) for x in args.l2s.split(',')] if k else [0.0]):
            e, w = fit(y, ms['train'], 'gauss', feats=feats, k=k, l2=l2, steps=args.steps, dev=dev)
            r = {m: round(1 - loss_sum(e, y, 'gauss', ms[m]) / null[m], 4) for m in ms}
            if top is None or r['val'] > top['val']:
                top = {**r, 'l2': l2, 'feature_weights': w}
        return top

    rows = []
    def add(name, **kw):
        rows.append({'model': name, **score(**kw)})
        print(f"{name:40s} strength explained (test) {rows[-1]['test']:+.3f}", flush=True)
    print(f'right-side connected test pairs {int(ms["test"].sum())}', flush=True)
    add('replicate (left twin)', feats=twin_f)
    scales = [float(s) for s in args.scales.split(',')]
    geos = {}
    for s in scales:
        try:
            geos[s] = geo(s)
        except FileNotFoundError:
            print(f'no directed overlap cache at {s:g} um yet', flush=True)
    for s, G in geos.items():
        add(f'geometry {s:g} um', feats=G)
    if len(geos) > 1:
        add('geometry ' + '+'.join(f'{s:g}' for s in geos) + ' um', feats=torch.cat(list(geos.values()), -1))
    add(f'code k={args.k}', k=args.k)
    Gall = torch.cat(list(geos.values()), -1)
    add(f'code k={args.k} + geometry', feats=Gall, k=args.k)
    add('geometry + replicate', feats=torch.cat([Gall, twin_f], -1))
    add(f'code k={args.k} + geometry + replicate', feats=torch.cat([Gall, twin_f], -1), k=args.k)
    rep = {'target': 'right-side log synapses per cell pair, right-connected held-out pairs (E1 split)',
           'metric': 'share of squared error explained beyond the degree-only null', 'rows': rows,
           'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print(f"done in {rep['seconds']} s -> {args.out}")


if __name__ == '__main__':
    main()
