"""Kenyon-cell activity over 4,096 ticks of text on the 8-tick streaming clock: fraction of KCs spiking per tick and
per 32-tick block, and the same for MBONs, PNs and core; burst statistics (blocks with > 1% KCs active)."""
import sys, json, numpy as np, torch
sys.path.insert(0, 'D:/information-boltzmann-v1')
src = open('D:/information-boltzmann-v1/scripts/ib/fly_interface/taste_episodes.py', encoding='utf-8').read().split('rng = np.random.default_rng(0)')[0]
fork_arg = sys.argv[1]; R_INH = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; sys.argv = [sys.argv[0], fork_arg, '1']
exec(src)
g_e0, g_i0 = model.get_conductance_gains()
opts['conductance_gains'] = (g_e0, g_i0 * R_INH)
with torch.no_grad():
    s, _ = run_tokens(s, text[5000:5000 + (2500 if R_INH != 1.0 else 0)])   # re-settle 20k ticks under the new inhibition
groups = {'Kenyon': kc_idx, 'MBON': mb_idx, 'ALPN': torch.as_tensor(np.flatnonzero(cls == 'ALPN'), device='cuda')}
original = fl.step_fly_physical_tick
rows = []
def step(model_, current, *a, **k):
    out = original(model_, current, *a, **k)
    r0 = out.ring[0][0]
    rows.append([float((r0[ix] > 0).float().mean()) for ix in groups.values()])
    return out
fl.step_fly_physical_tick = step
with torch.no_grad():
    s, _ = run_tokens(s, text[:512])
fl.step_fly_physical_tick = original
X = np.array(rows)
kc = X[:, 0]
blocks = kc.reshape(-1, 32).max(1)
burst_ticks = np.flatnonzero(kc > 0.01)
print('r_inh', R_INH, 'ticks', len(kc), 'KC per-tick fraction mean', kc.mean(), 'p50', np.median(kc), 'p99', np.percentile(kc, 99), 'max', kc.max())
print('ticks with >1% KCs', len(burst_ticks), 'first 20', burst_ticks[:20].tolist())
print('intervals between bursts', np.diff(burst_ticks[np.r_[True, np.diff(burst_ticks) > 5]])[:30].tolist())
print('MBON per-tick mean', X[:, 1].mean(), 'p99', np.percentile(X[:, 1], 99), ' ALPN mean', X[:, 2].mean(), 'p99', np.percentile(X[:, 2], 99))
json.dump({'kc_per_tick': kc.tolist(), 'mbon_per_tick': X[:, 1].tolist(), 'pn_per_tick': X[:, 2].tolist()}, open(fork / f'kc_bursts_r{R_INH:g}.json', 'w'))
