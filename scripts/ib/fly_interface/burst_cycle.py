"""What starts and what ends a population burst?  Fixed weights, 8-tick streaming clock, text.  Every tick, per
group (Kenyon, ALPN, MBON, core excitatory, core inhibitory, optic-lobe inhibitory, all): spike fraction, mean
transmitter resource x (STP), mean release u, mean ALIF adaptation b, mean excitatory and inhibitory conductance,
mean membrane distance to threshold.  Burst onsets = ticks where the Kenyon spike fraction first exceeds 1% after at
least 50 quiet ticks; every variable is averaged in a window aligned to the onsets (-150 .. +300 ticks), and the
recovery time of x after a burst is fitted.
Usage: python burst_cycle.py <fork dir> [ticks]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
src = open('D:/information-boltzmann-v1/scripts/ib/fly_interface/taste_episodes.py', encoding='utf-8').read().split('rng = np.random.default_rng(0)')[0]
fork_arg = sys.argv[1]; TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
sys.argv = [sys.argv[0], fork_arg, '1']
exec(src)
sup = ann['superclass'].astype(str).values
sign = np.asarray(g['nt_sign']).reshape(-1)[:N]
core = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan
groups_np = {'Kenyon': cls == 'Kenyon_Cell', 'ALPN': cls == 'ALPN', 'MBON': cls == 'MBON', 'core E': core & (sign > 0),
             'core I': core & (sign < 0), 'optic I': (sup == 'ol_intrinsic') & (sign < 0), 'all': np.ones(N, bool)}
gm = {k: torch.as_tensor(v, device='cuda') for k, v in groups_np.items()}
theta = model.get_thresholds().expand(1, N)[0]
original = fl.step_fly_physical_tick
rows = []


def step(model_, current, *a, **k):
    out = original(model_, current, *a, **k)
    s0 = out.ring[0][0] > 0
    d = (theta - out.h[0]) / theta
    row = []
    for m in gm.values():
        row += [float(s0[m].float().mean()), float(out.x[0, m].mean()), float(out.u[0, m].mean()),
                float(out.b[0, m].mean()) if out.b.numel() else 0.0, float(out.ge[0, m].mean()), float(out.gi[0, m].mean()),
                float(d[m].median())]
    rows.append(row)
    return out


fl.step_fly_physical_tick = step
with torch.no_grad():
    s, _ = run_tokens(s, text[:TOTAL // TICKS])
fl.step_fly_physical_tick = original
names = ['spikes', 'x', 'u', 'alif', 'ge', 'gi', 'distance']
X = np.array(rows).reshape(len(rows), len(gm), len(names))
kc = X[:, 0, 0]
on = [t for t in range(50, len(kc)) if kc[t] > 0.01 and kc[t - 50:t].max() <= 0.01]
print('ticks', len(kc), 'burst onsets', len(on), on[:20], flush=True)
W0, W1 = 150, 300
valid = [t for t in on if t - W0 >= 0 and t + W1 < len(kc)]
avg = np.mean([X[t - W0:t + W1] for t in valid], 0) if valid else None
report = {'fork': str(fork), 'ticks': len(kc), 'onsets': on, 'groups': list(gm), 'variables': names}
if avg is not None:
    lags = np.arange(-W0, W1)
    summary = {}
    for gi_, gname in enumerate(gm):
        row = {}
        for vi, vname in enumerate(names):
            tr = avg[:, gi_, vi]
            pre = tr[:W0 - 20].mean()
            row[vname] = {'pre': float(pre), 'at_onset': float(tr[W0]), 'peak_or_trough_lag': int(lags[np.argmax(np.abs(tr - pre))]),
                          'extreme': float(tr[np.argmax(np.abs(tr - pre))])}
        summary[gname] = row
        print(gname, {v: (round(r['pre'], 4), round(r['extreme'], 4), r['peak_or_trough_lag']) for v, r in row.items()}, flush=True)
    report['onset_aligned_summary'] = summary
    report['onset_aligned_mean'] = avg.tolist()
    # recovery of the Kenyon and core-E transmitter resource after the burst: time to recover 63% of the drop
    for gname in ('Kenyon', 'core E', 'ALPN'):
        gi_ = list(gm).index(gname); xs = avg[:, gi_, 1]
        pre = xs[:W0 - 20].mean(); tmin = int(np.argmin(xs[W0:])) + W0; drop = pre - xs[tmin]
        rec = next((t - tmin for t in range(tmin, len(xs)) if xs[t] >= xs[tmin] + 0.63 * drop), None)
        report[f'x_recovery_63pct_ticks_{gname}'] = rec
        print(gname, 'x drop', round(float(drop), 4), 'recovery 63% (ticks)', rec, flush=True)
    intervals = np.diff(on)
    report['inter_burst_interval_min_med_mean_max'] = [int(intervals.min()), float(np.median(intervals)), float(intervals.mean()), int(intervals.max())] if len(intervals) else None
    print('inter-burst', report['inter_burst_interval_min_med_mean_max'], flush=True)
json.dump(report, open(fork / 'burst_cycle.json', 'w'))
