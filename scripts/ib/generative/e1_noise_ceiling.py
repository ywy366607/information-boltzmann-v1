"""E1 noise ceiling (2026-10-20): is the ~62% plateau of the learned codes the reproducible part of the wiring?

The brain is bilateral: cells of a type with somata on the left and on the right are two replicates of the same
developmental program. W_L[a, b] = synapses from left-soma cells of type a onto type b; W_R likewise.
Target: exists_R = W_R[a, b] >= MIN_SYN_SIDE on held-out type pairs (same 90/5/5 split as E1, same 3000 types).
  replicate:  degree null + the twin pair's left-side wiring (a few-parameter logistic on log(1 + W_L per cell))
              = how much of the right side is predictable at all from the same program (noise ceiling, half data)
  code k:     degree null + learned k-dim code fitted on the right side's training pairs (E1's generator)
If code ~ replicate, the low-dimensional generator captures the reproducible wiring; the rest is developmental noise.
If replicate >> code, there is reproducible pair-specific wiring the code misses (exact-matching genes).
"""
import argparse, json, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import pyarrow.feather as pf
from e1_genomic_bottleneck import D, fit, loss_sum, auc

MIN_SYN_SIDE = 3


def side_matrices(names):
    a = pd.read_feather(f'{D}/body-annotations.feather')[['bodyId', 'type', 'somaSide']]
    a = a[a.type.isin(set(names))]
    index = {t: i for i, t in enumerate(names)}
    ids = a.bodyId.values.astype(np.int64)
    order = np.argsort(ids)
    ids, tcode = ids[order], np.array([index[t] for t in a.type.values[order]])
    side = a.somaSide.values[order]
    left = np.asarray(side == 'L')
    right = np.asarray(side == 'R')
    n = len(names)
    WL, WR = np.zeros(n * n), np.zeros(n * n)
    t = pf.read_table(f'{D}/connectome-weights.feather', memory_map=True)
    for b in t.to_batches(max_chunksize=5_000_000):
        p = b.column(0).to_numpy().astype(np.int64); q = b.column(1).to_numpy().astype(np.int64)
        v = b.column(2).to_numpy().astype(np.float64)
        ip = np.searchsorted(ids, p).clip(0, len(ids) - 1); iq = np.searchsorted(ids, q).clip(0, len(ids) - 1)
        ok = (ids[ip] == p) & (ids[iq] == q)
        ip, iq, v = ip[ok], iq[ok], v[ok]
        key = tcode[ip] * n + tcode[iq]
        WL += np.bincount(key[left[ip]], v[left[ip]], n * n)
        WR += np.bincount(key[right[ip]], v[right[ip]], n * n)
    nl = np.bincount(tcode[left], minlength=n).astype(float)
    nr = np.bincount(tcode[right], minlength=n).astype(float)
    return WL.reshape(n, n), WR.reshape(n, n), nl, nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--side-cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000_sides.npz')
    ap.add_argument('--ks', default='4,8,16,32')
    ap.add_argument('--l2s', default='0.0001,0.001,0.01,0.1')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--out', default='results/published/e1_noise_ceiling_20261020.json')
    args = ap.parse_args()
    t0 = time.time()
    names = np.load(args.cache, allow_pickle=True)['names']
    try:
        z = np.load(args.side_cache)
        WL, WR, nl, nr = z['WL'], z['WR'], z['nl'], z['nr']
    except FileNotFoundError:
        WL, WR, nl, nr = side_matrices(list(names))
        np.savez(args.side_cache, WL=WL, WR=WR, nl=nl, nr=nr)
    bilateral = (nl > 0) & (nr > 0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    y = torch.tensor(WR >= MIN_SYN_SIDE, dtype=torch.float32, device=dev)
    rows_ok = torch.tensor(bilateral, device=dev)[:, None].expand_as(y)
    g = torch.Generator(device='cpu').manual_seed(1)
    u = torch.rand(y.shape, generator=g).to(dev)
    ms = {'train': (u >= 0.10) & rows_ok, 'val': (u < 0.05) & rows_ok, 'test': (u >= 0.05) & (u < 0.10) & rows_ok}
    print(f'{int(bilateral.sum())} bilateral types of {len(names)}; right-side connected (>= {MIN_SYN_SIDE}) '
          f'{float(y[rows_ok].mean()):.2%}', flush=True)
    null = fit(y, ms['train'], 'bern', steps=args.steps, dev=dev)
    nulls = {m: loss_sum(null, y, 'bern', ms[m]) for m in ms}
    rows = [{'model': 'degree only (null)', 'test_auc': round(auc(null[ms['test']], y[ms['test']]), 4)}]
    # replicate: the twin's left wiring per left cell, as a few fixed features with free weights
    x = torch.tensor(np.log1p(WL / np.maximum(nl, 1)[:, None]), dtype=torch.float32, device=dev)
    feats = torch.stack([x, x ** 2, (torch.tensor(WL, device=dev) >= MIN_SYN_SIDE).float()], -1)
    top = None
    for steps in (args.steps,):
        o = torch.zeros(len(names), device=dev, requires_grad=True); i = torch.zeros(len(names), device=dev, requires_grad=True)
        b0 = torch.zeros((), device=dev, requires_grad=True); w = torch.zeros(3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([o, i, b0, w], lr=0.05)
        for s in range(steps):
            opt.zero_grad()
            e = b0 + o[:, None] + i[None] + feats @ w
            F.binary_cross_entropy_with_logits(e[ms['train']], y[ms['train']]).backward(); opt.step()
        with torch.no_grad():
            e = b0 + o[:, None] + i[None] + feats @ w
            top = {m: round(1 - loss_sum(e, y, 'bern', ms[m]) / nulls[m], 4) for m in ms}
            top['test_auc'] = round(auc(e[ms['test']], y[ms['test']]), 4)
    rows.append({'model': 'replicate (left twin of the same pair)', **top})
    print(f"replicate (left twin)            exists test {top['test']:+.3f} auc {top['test_auc']:.3f}", flush=True)
    for k in [int(v) for v in args.ks.split(',')]:
        best = None
        for l2 in [float(v) for v in args.l2s.split(',')]:
            e = fit(y, ms['train'], 'bern', k=k, l2=l2, steps=args.steps, dev=dev)
            r = {m: round(1 - loss_sum(e, y, 'bern', ms[m]) / nulls[m], 4) for m in ms}
            if best is None or r['val'] > best['val']:
                best = {**r, 'l2': l2, 'test_auc': round(auc(e[ms['test']], y[ms['test']]), 4)}
        rows.append({'model': f'learned code k={k}', 'numbers': len(names) * k + k * k, **best})
        print(f"learned code k={k:<3d}               exists test {best['test']:+.3f} auc {best['test_auc']:.3f}", flush=True)
    rep = {'types': len(names), 'bilateral_types': int(bilateral.sum()), 'min_syn_side': MIN_SYN_SIDE,
           'metric': 'share of logistic loss explained beyond the degree-only null on held-out right-side type pairs',
           'rows': rows, 'seconds': round(time.time() - t0)}
    json.dump(rep, open(args.out, 'w'), indent=1)
    print(f"done in {rep['seconds']} s -> {args.out}")


if __name__ == '__main__':
    main()
