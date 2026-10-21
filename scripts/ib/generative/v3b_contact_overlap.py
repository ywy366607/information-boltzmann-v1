"""CGB V3b - symmetric contact overlap from skeletons (stereology; pre-registration in CGB_V3_RESULTS_20261020.md).

Per type and voxel: L = cable length (um), LR = sum of piece length x radius (um^2). Segments are cut into pieces
<= 0.5 um first. Outputs, for scales 2 and 1 um, the type x type matrices
  U = sum_v L_a L_b                      (undirected cable overlap)
  K = sum_v L_a LR_b + LR_a L_b          (expected touches of random cylinders, up to a constant)
Memory (whole-brain 1 um has ~2e8 type-voxel entries, too many to hold): systematic uniform random sampling, the
stereological standard - keep 2 um z slabs whose index mod PERIOD equals one of the residues (default 0 and 4 of 8:
two independent samples of 1/8 of the volume each, one pass). Each residue is saved separately; their agreement is
the sampling error, and their sum x PERIOD / len(residues) is the estimate.
"""
import argparse, glob, os, time
import numpy as np, pandas as pd, torch
from e1_genomic_bottleneck import D
from e1b_encounter import SKELETONS

SCALES = (2.0, 1.0)
PIECE_UM = 0.5
NM8 = 8.0 / 1000.0                     # skeleton units (8 nm voxels) -> um


def body_types(names):
    a = pd.read_feather(f'{D}/body-annotations.feather', columns=['bodyId', 'type'])
    index = {t: i for i, t in enumerate(names)}
    return dict(zip(a.bodyId.values.astype(np.int64), [index.get(t, -1) for t in a.type.values]))


def pieces(path, body_type):
    z = np.load(path)
    bodies, counts, xyz, rad, parent = z['body_ids'], z['node_count'], z['xyz_nm'], z['radius_nm'], z['parent']
    offset = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    cell_type = np.repeat(np.array([body_type.get(int(b), -1) for b in bodies]), counts)
    child = np.flatnonzero((parent >= 0) & (cell_type >= 0))
    par = parent[child].astype(np.int64) + offset[child]
    p0, p1 = xyz[child].astype(np.float64) * NM8, xyz[par].astype(np.float64) * NM8
    L = np.sqrt(((p1 - p0) ** 2).sum(1))
    r = 0.5 * (rad[child] + rad[par]).astype(np.float64) * NM8
    k = np.maximum(np.ceil(L / PIECE_UM).astype(np.int64), 1)
    seg = np.repeat(np.arange(len(L)), k)
    j = np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)
    frac = (j + 0.5) / k[seg]
    mid = p0[seg] + (p1[seg] - p0[seg]) * frac[:, None]
    lp = L[seg] / k[seg]
    return cell_type[child][seg].astype(np.int64), mid, lp, lp * r[seg]


def vkey(mid, s):
    v = np.floor(mid / s).astype(np.int64)
    return v[:, 0] + 8192 * (v[:, 1] + 8192 * v[:, 2])


def merge(parts):
    k = np.concatenate([p[0] for p in parts])
    uq, inv = np.unique(k, return_inverse=True)
    return uq, np.bincount(inv, np.concatenate([p[1] for p in parts])).astype(np.float32), np.bincount(inv, np.concatenate([p[2] for p in parts])).astype(np.float32)


def accumulate(keys, L, LR, n, out_U, out_K, dev, chunk=40000):
    t = keys >> 42
    vox = keys & ((1 << 42) - 1)
    order = np.argsort(vox, kind='stable')
    vox, t, L, LR = vox[order], t[order], L[order].astype(np.float32), LR[order].astype(np.float32)
    col = np.empty(len(vox), np.int64); col[0] = 0
    np.cumsum(vox[1:] != vox[:-1], out=col[1:])
    n_vox = int(col[-1]) + 1
    bounds = np.searchsorted(col, np.arange(0, n_vox + chunk, chunk))
    for lo, hi, c0 in zip(bounds[:-1], bounds[1:], range(0, n_vox + chunk, chunk)):
        if hi <= lo:
            continue
        ti = torch.as_tensor(t[lo:hi], device=dev); ci = torch.as_tensor(col[lo:hi] - c0, device=dev)
        A = torch.zeros(n, chunk, device=dev); B = torch.zeros(n, chunk, device=dev)
        A[ti, ci] = torch.as_tensor(L[lo:hi], device=dev); B[ti, ci] = torch.as_tensor(LR[lo:hi], device=dev)
        out_U += (A @ A.T).double()
        AB = (A @ B.T).double()
        out_K += AB + AB.T
        del A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='F:/fly_stream_overflow/cgb_v1/e1_types3000.npz')
    ap.add_argument('--work', default='F:/fly_stream_overflow/cgb_v1')
    ap.add_argument('--period', type=int, default=8)
    ap.add_argument('--residues', default='0,4')
    ap.add_argument('--limit', type=int, default=0, help='shards (test)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    t0 = time.time()
    names = list(np.load(args.cache, allow_pickle=True)['names'])
    n = len(names)
    body_type = body_types(names)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    shards = sorted(glob.glob(f'{SKELETONS}/skeletons_*.npz'))
    if args.limit:
        shards = shards[:args.limit]
    residues = [int(x) for x in args.residues.split(',')]
    parts = {(s, q): [] for s in SCALES for q in residues}
    total_len = kept_len = 0.0
    for i, path in enumerate(shards):
        t, mid, lp, lr = pieces(path, body_type)
        total_len += lp.sum()
        slab = np.floor(mid[:, 2] / SCALES[0]).astype(np.int64) % args.period
        for q in residues:
            keep = slab == q
            tq, mq, lq, rq = t[keep], mid[keep], lp[keep].astype(np.float32), lr[keep].astype(np.float32)
            kept_len += lq.sum()
            for s in SCALES:
                key = (tq << 42) + vkey(mq, s)
                uq, inv = np.unique(key, return_inverse=True)
                parts[s, q].append((uq, np.bincount(inv, lq).astype(np.float32), np.bincount(inv, rq).astype(np.float32)))
                if len(parts[s, q]) >= 24:
                    parts[s, q] = [merge(parts[s, q])]
        if i % 20 == 0:
            sizes_now = {f'{s:g}um/r{q}': sum(len(p[0]) for p in v) for (s, q), v in parts.items()}
            print(f'shard {i}/{len(shards)} {time.time() - t0:.0f} s entries {sizes_now}', flush=True)
    for (s, q) in list(parts):
        keys, L, LR = merge(parts[s, q]); parts[s, q] = None
        U = torch.zeros(n, n, dtype=torch.float64, device=dev); K = torch.zeros_like(U)
        accumulate(keys, L, LR, n, U, K, dev)
        np.save(f'{args.work}/v3b_U_{s:g}um_r{q}of{args.period}{args.tag}.npy', U.cpu().numpy().astype(np.float32))
        np.save(f'{args.work}/v3b_K_{s:g}um_r{q}of{args.period}{args.tag}.npy', K.cpu().numpy().astype(np.float32))
        print(f'scale {s:g} um residue {q}: {len(keys)} type-voxel entries, {time.time() - t0:.0f} s', flush=True)
        del keys, L, LR, U, K
    print(f'cable in the 3000 types: {total_len:.0f} um, sampled {kept_len / total_len:.3f}; '
          f'done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
