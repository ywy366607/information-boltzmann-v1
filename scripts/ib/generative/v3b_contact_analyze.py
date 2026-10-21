"""CGB V3b - does symmetric contact (stereology) explain the symmetric pair affinity? (pre-registration: Pa1-Pa4 in
docs/information_boltzmann/CGB_V3_RESULTS_20261020.md). Overlaps from v3b_contact_overlap.py (two 1/8 volume samples).
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
    ap.add_argument('--residues', default='0,4')
    ap.add_argument('--period', type=int, default=8)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/v3b_contact_20261020.json')
    args = ap.parse_args()
    t0 = time.time()
    sizes = np.load(args.cache, allow_pickle=True)['sizes']
    z = np.load(args.sides)
    WL, WR, nl, nr = z['WL'].astype(np.float64), z['WR'].astype(np.float64), z['nl'], z['nr']
    n = len(sizes)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    bil = (nl > 0) & (nr > 0)
    yR = T(np.log(np.maximum(WR, 1) / (np.maximum(nr, 1)[:, None] * sizes[None])))
    yL = T(np.log(np.maximum(WL, 1) / (np.maximum(nl, 1)[:, None] * sizes[None])))
    okR, okL = T((WR >= MIN_SYN_SIDE) & bil[:, None]).bool(), T((WL >= MIN_SYN_SIDE) & bil[:, None]).bool()
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand((n, n), generator=g).to(dev)
    ms = {'train': (u >= 0.10) & okR, 'val': (u < 0.05) & okR, 'test': (u >= 0.05) & (u < 0.10) & okR}
    conn = (WR >= MIN_SYN_SIDE)

    out = []
    for s in (4, 2):
        E = np.load(f'{args.work}/e1c_overlap_trainlabels_{s:g}um.npy')
        for M in (E, np.ascontiguousarray(E.T)):
            e = M / (sizes[:, None] * sizes[None])
            floor = e[e > 0].min() * 0.5
            out += [np.log(np.maximum(e, floor)), (e == 0).astype(float)]
    G = T(np.stack(out, -1))

    res = [int(x) for x in args.residues.split(',')]
    M, sampling = {}, {}
    for name in ('U', 'K'):
        for s in (2, 1):
            parts = [np.load(f'{args.work}/v3b_{name}_{s:g}um_r{q}of{args.period}.npy').astype(np.float64) for q in res]
            a, b = parts[0][conn], parts[1][conn]
            both = (a > 0) & (b > 0)
            sampling[f'{name}{s}'] = {'log_r_between_samples_connected': round(float(np.corrcoef(np.log(a[both]), np.log(b[both]))[0, 1]), 4),
                                      'connected_pairs_zero_in_sum': round(float(((a + b) == 0).mean()), 4)}
            M[f'{name}{s}'] = sum(parts) * args.period / len(res)
    print('sampling', sampling, flush=True)

    def feat(X):
        e = X / (sizes[:, None] * sizes[None])
        floor = e[e > 0].min() * 0.5
        return T(np.stack([np.log(np.maximum(e, floor)), (e == 0).astype(float)], -1))
    F_ = {k: feat(v) for k, v in M.items()}

    null_e, _ = fit(yR, ms['train'], 'gauss', steps=args.steps, dev=dev)
    null = {m: loss_sum(null_e, yR, 'gauss', ms[m]) for m in ms}
    rows = []
    def add(name, F):
        e, w = fit(yR, ms['train'], 'gauss', feats=F, steps=args.steps, dev=dev)
        rows.append({'model': name, **{m: round(1 - loss_sum(e, yR, 'gauss', ms[m]) / null[m], 4) for m in ms},
                     'feature_weights': w})
        print(f"{name:34s} test {rows[-1]['test']:+.4f} val {rows[-1]['val']:+.4f} w {np.round(w, 3).tolist()}", flush=True)
    add('geometry 4+2 um', G)
    for k in ('U2', 'K2', 'U1', 'K1'):
        add(f'geometry + {k}', torch.cat([G, F_[k]], -1))
    add('geometry + K1 + U1', torch.cat([G, F_['K1'], F_['U1']], -1))
    add('geometry + K1 + K2', torch.cat([G, F_['K1'], F_['K2']], -1))
    add('K1 alone', F_['K1'])
    base = rows[0]['test']
    gains = {r['model']: round(r['test'] - base, 4) for r in rows[1:]}

    def recip(F):
        eR, _ = fit(yR, okR, 'gauss', feats=F, steps=args.steps, dev=dev)
        eL, _ = fit(yL, okL, 'gauss', feats=F, steps=args.steps, dev=dev)
        rR, rL = (yR - eR).cpu().numpy(), (yL - eL).cpu().numpy()
        oR, oL = okR.cpu().numpy(), okL.cpu().numpy()
        a, b = np.nonzero(np.triu(oR & oR.T & oL & oL.T, 1))
        x, y = np.r_[rR[a, b], rR[b, a]], np.r_[rL[b, a], rL[a, b]]
        sx, sy = np.r_[rR[a, b], rR[b, a]], np.r_[rL[a, b], rL[b, a]]
        return {'pairs': int(len(a)), 'reverse_r': round(float(np.corrcoef(x, y)[0, 1]), 4),
                'same_r': round(float(np.corrcoef(sx, sy)[0, 1]), 4)}
    pa3 = {'geometry': recip(G), 'geometry + K1': recip(torch.cat([G, F_['K1']], -1)),
           'geometry + U2': recip(torch.cat([G, F_['U2']], -1))}
    print('Pa3', pa3, flush=True)
    verdict = {'Pa1 K1 >= +0.03': gains['geometry + K1'] >= 0.03,
               'Pa2 K1 gain > K2 gain': gains['geometry + K1'] > gains['geometry + K2'],
               'Pa3 reverse r <= 0.22 after K1': pa3['geometry + K1']['reverse_r'] <= 0.22,
               'Pa4 K beats U at 1 um and 2 um': (gains['geometry + K1'] > gains['geometry + U1'])
                                                 and (gains['geometry + K2'] > gains['geometry + U2'])}
    rep = {'preregistration': 'docs/information_boltzmann/CGB_V3_RESULTS_20261020.md (V3b)',
           'sampling': sampling, 'rows': rows, 'gain_over_geometry_test': gains, 'Pa3': pa3, 'verdict': verdict,
           'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print('gains', gains, '\nverdict', verdict, f"\ndone in {rep['seconds']} s -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
