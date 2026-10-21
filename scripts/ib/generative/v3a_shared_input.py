"""CGB V3a - does shared input (a proxy for correlated activity) explain the strength geometry misses? (2026-10-20)

Pre-registration: docs/information_boltzmann/CGB_V3_RESULTS_20261020.md (P1-P4). Footing = E1d: right-side log
synapses per cell pair, right-connected held-out pairs, E1 split, degree-only null. Shared input / output profiles are
built from the LEFT hemisphere only, with the pair's own partners removed from both profiles.
"""
import argparse, json, time
import numpy as np, torch
from e1_genomic_bottleneck import loss_sum
from e1b_encounter import fit

MIN_SYN_SIDE = 3


def shared(W):
    """cosine of the Hellinger profiles in the columns of W (rows = partner types), partners a and b removed."""
    X = np.sqrt(W / np.maximum(W.sum(0, keepdims=True), 1e-12))
    X2, d = X ** 2, np.diag(X).copy()
    G = X.T @ X - d[:, None] * X - X.T * d[None, :]
    N = X2.sum(0)
    na = N[:, None] - d[:, None] ** 2 - X2.T
    nb = N[None, :] - X2 - d[None, :] ** 2
    S = G / np.sqrt(np.maximum(na, 1e-12) * np.maximum(nb, 1e-12))
    S[(na <= 1e-12) | (nb <= 1e-12)] = 0.0
    np.fill_diagonal(S, 0.0)
    return np.clip(S, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--sides', default='F:/fly_stream_overflow/cgb_v1/e1_types3000_sides.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/v3a_shared_input_20261020.json')
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

    def geo():
        out = []
        for s in (4, 2):
            E = np.load(f'{args.work}/e1c_overlap_trainlabels_{s:g}um.npy')
            for M in (E, np.ascontiguousarray(E.T)):
                e = M / (sizes[:, None] * sizes[None])
                floor = e[e > 0].min() * 0.5
                out += [np.log(np.maximum(e, floor)), (e == 0).astype(float)]
        return T(np.stack(out, -1))

    def feat(S):
        floor = S[S > 0].min() * 0.5 if (S > 0).any() else 1e-6
        return T(np.stack([S, np.log(np.maximum(S, floor)), (S == 0).astype(float)], -1))

    SI = shared(WL)                       # columns of WL = each type's input profile
    SO = shared(WL.T)                     # columns of WL.T = each type's output profile
    perm = np.random.default_rng(7).permutation(n)
    SI_perm = SI[perm][:, perm]
    G = geo()
    stats = {k: {'mean_connected_right': float(S[(WR >= MIN_SYN_SIDE)].mean()), 'mean_all': float(S.mean())}
             for k, S in (('SI', SI), ('SO', SO))}
    stats['corr_SI_SO_connected'] = float(np.corrcoef(SI[WR >= MIN_SYN_SIDE], SO[WR >= MIN_SYN_SIDE])[0, 1])
    print('feature stats', stats, flush=True)

    null_e, _ = fit(yR, ms['train'], 'gauss', steps=args.steps, dev=dev)
    null = {m: loss_sum(null_e, yR, 'gauss', ms[m]) for m in ms}
    rows = []
    def add(name, F):
        e, w = fit(yR, ms['train'], 'gauss', feats=F, steps=args.steps, dev=dev)
        rows.append({'model': name, **{m: round(1 - loss_sum(e, yR, 'gauss', ms[m]) / null[m], 4) for m in ms},
                     'feature_weights': w})
        print(f"{name:34s} test {rows[-1]['test']:+.4f} val {rows[-1]['val']:+.4f}", flush=True)
        return e
    twin = np.log(np.maximum(WL, 1) / (np.maximum(nl, 1)[:, None] * sizes[None]))
    twin_f = T(np.stack([twin, (WL >= MIN_SYN_SIDE).astype(float)], -1))
    add('geometry 4+2 um', G)
    add('geometry + SI', torch.cat([G, feat(SI)], -1))
    add('geometry + SO', torch.cat([G, feat(SO)], -1))
    add('geometry + SI_perm', torch.cat([G, feat(SI_perm)], -1))
    add('geometry + SI + SO', torch.cat([G, feat(SI), feat(SO)], -1))
    add('SI alone', feat(SI))
    add('geometry + replicate', torch.cat([G, twin_f], -1))
    add('geometry + replicate + SI', torch.cat([G, twin_f, feat(SI)], -1))
    base = rows[0]['test']
    gains = {r['model']: round(r['test'] - base, 4) for r in rows[1:]}

    # P4: cross-side residuals of the geometry model on reciprocal pairs
    eR, _ = fit(yR, okR, 'gauss', feats=G, steps=args.steps, dev=dev)
    eL, _ = fit(yL, okL, 'gauss', feats=G, steps=args.steps, dev=dev)
    rR, rL = (yR - eR).cpu().numpy(), (yL - eL).cpu().numpy()
    oR, oL = okR.cpu().numpy(), okL.cpu().numpy()
    recip = oR & oR.T & oL & oL.T
    a, b = np.nonzero(np.triu(recip, 1))
    rng = np.random.default_rng(3)

    def pear(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    def boot(x, y, k=2000):
        idx = rng.integers(0, len(x), (k, len(x)))
        v = [pear(x[i], y[i]) for i in idx]
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    x = np.r_[rR[a, b], rR[b, a]]; y = np.r_[rL[b, a], rL[a, b]]          # a->b right vs b->a left, and vice versa
    same_x, same_y = np.r_[rR[a, b], rR[b, a]], np.r_[rL[a, b], rL[b, a]]  # same direction across sides: the scale
    shuf = rng.permutation(len(y))
    p4 = {'reciprocal_unordered_pairs': int(len(a)),
          'reverse_direction_cross_side_r': round(pear(x, y), 4), 'reverse_ci95': boot(x, y),
          'same_direction_cross_side_r': round(pear(same_x, same_y), 4), 'same_ci95': boot(same_x, same_y),
          'shuffled_r': round(pear(x, y[shuf]), 4)}
    # within-pair asymmetry: is |r_ab - r_ba| reproducible across sides (one direction wins consistently)?
    dR, dL = rR[a, b] - rR[b, a], rL[a, b] - rL[b, a]
    p4['asymmetry_cross_side_r'] = round(pear(dR, dL), 4)
    p4['asymmetry_ci95'] = boot(dR, dL)
    print('P4', p4, flush=True)

    verdict = {'P1 SI >= +0.03': gains['geometry + SI'] >= 0.03,
               'P2 SI > SO': gains['geometry + SI'] > gains['geometry + SO'],
               'P3 |SI_perm| < 0.005': abs(gains['geometry + SI_perm']) < 0.005}
    rep = {'preregistration': 'docs/information_boltzmann/CGB_V3_RESULTS_20261020.md',
           'target': 'right-side log synapses per cell pair, right-connected held-out pairs (E1d footing)',
           'feature_stats': stats, 'rows': rows, 'gain_over_geometry_test': gains, 'P4': p4,
           'verdict': verdict, 'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print('gains', gains, '\nverdict', verdict, f"\ndone in {rep['seconds']} s -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
