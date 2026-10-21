"""V3c extension: the V3c code fits chose l2 = 10, the edge of the grid; refit codes with l2 up to 1000 (same footing,
same split, l2 chosen on validation) to get the converged code gains. Also the code alone (no geometry)."""
import json, sys, time
import numpy as np, torch
from e1_genomic_bottleneck import loss_sum
from v3c_pair_affinity import fit

t0 = time.time()
W = 'F:/fly_stream_overflow/cgb_v1'
sizes = np.load(f'{W}/e1_types3000.npz', allow_pickle=True)['sizes']
z = np.load(f'{W}/e1_types3000_sides.npz')
WR, nr, nl = z['WR'].astype(float), z['nr'], z['nl']
n = len(sizes); dev = 'cuda'; T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
bil = (nl > 0) & (nr > 0)
yR_np = np.log(np.maximum(WR, 1) / (np.maximum(nr, 1)[:, None] * sizes[None])); y = T(yR_np)
ok = T((WR >= 3) & bil[:, None]).bool()
g = torch.Generator(device='cpu').manual_seed(1); u = torch.rand((n, n), generator=g).to(dev)
ms = {'train': (u >= 0.10) & ok, 'val': (u < 0.05) & ok, 'test': (u >= 0.05) & (u < 0.10) & ok}
out = []
for s in (4, 2):
    E = np.load(f'{W}/e1c_overlap_trainlabels_{s}um.npy')
    for M in (E, np.ascontiguousarray(E.T)):
        e = M / (sizes[:, None] * sizes[None]); fl = e[e > 0].min() * 0.5
        out += [np.log(np.maximum(e, fl)), (e == 0).astype(float)]
G = T(np.stack(out, -1))
conn_rev = WR.T >= 3
REV = T(np.stack([np.where(conn_rev, yR_np.T, 0.0), conn_rev.astype(float)], -1))
null_e, _ = fit(y, ms['train'], dev=dev); null = {m: loss_sum(null_e, y, 'gauss', ms[m]) for m in ms}
rows = []
for name, feats, k, sym in (('geometry + symmetric code 16', G, 16, True), ('geometry + symmetric code 64', G, 64, True),
                            ('geometry + symmetric code 256', G, 256, True), ('geometry + general code 16', G, 16, False),
                            ('geometry + general code 64', G, 64, False), ('geometry + general code 256', G, 256, False),
                            ('general code 64 alone', None, 64, False),
                            ('geometry + general code 64 + reverse', torch.cat([G, REV], -1), 64, False)):
    top = None
    for l2 in (10, 30, 100, 300, 1000):
        e, w = fit(y, ms['train'], feats=feats, k=k, sym=sym, l2=l2, dev=dev)
        r = {m: round(1 - loss_sum(e, y, 'gauss', ms[m]) / null[m], 4) for m in ms}
        print(f'   {name} l2 {l2}: val {r["val"]:+.4f} test {r["test"]:+.4f}', flush=True)
        if top is None or r['val'] > top['val']:
            top = {**r, 'l2': l2}
    rows.append({'model': name, **top, 'l2_at_grid_edge': top['l2'] in (10, 1000)})
    print(f"{name:40s} test {top['test']:+.4f} val {top['val']:+.4f} l2 {top['l2']}", flush=True)
json.dump({'note': __doc__, 'geometry_test': 0.316, 'geometry_plus_reverse_test': 0.3418, 'rows': rows,
           'seconds': round(time.time() - t0)}, open('results/published/v3c_code_l2_extend_20261020.json', 'w'), indent=1)
print(f'done in {time.time() - t0:.0f} s')
