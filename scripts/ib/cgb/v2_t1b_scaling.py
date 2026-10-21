"""V2 / T1b - pre-registered test of the scaling law found post hoc in T1 (c), with the physical sodium pool (0.95).

Derivation: at high input rates a synapse delivers g U f / (1 + U f tau_v) -> g / tau_v, so a recurrent network's
operating point depends on g / tau_v only.
Predictions (written before this run):
 P1  no depletion (tau_v 0.25 ms): the rate runs to the sodium-pool ceiling, roughly 1 / (tau_r ln(1/(1-kappa)))-ish,
     i.e. hundreds of Hz, and stays there for every g >= 1
 P2  with depletion: (g, tau_v) pairs with equal g / tau_v give equal rates (within 5%) on NEW pairs:
     ratio 1/225: (1.5, 337.5), (3, 675), (6, 1350); ratio 1/75: (2, 150), (4, 300), (8, 600)
 P3  the rate rises monotonically with g / tau_v
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass


def network(g_scale, tau_v, seed=0, N=400, p=0.1, T=3000.0):
    rng = np.random.default_rng(seed)
    exc = CellClass(E_out=5.0, U=0.4, tau_v=tau_v); inh = CellClass(E_out=-0.3, U=0.4, tau_v=tau_v)
    cls = np.r_[np.zeros(int(0.8 * N), int), np.ones(N - int(0.8 * N), int)]
    A = rng.random((N, N)) < p; np.fill_diagonal(A, False)
    pre, post = np.nonzero(A)
    m = Medium([exc, inh], cls, pre, post, (np.where(cls[pre] == 0, 0.05, 0.1) * g_scale).astype(float), seed=seed)
    m.plastic = False
    spikes = 0; warm = 500.0
    for k in range(int(T / m.dt)):
        s = m.step(I_ext=0.06 + 0.05 * rng.standard_normal(N))
        if m.t > warm:
            spikes += s.sum()
    return spikes / N / ((T - warm) / 1000)

out = {'P1_no_depletion': [], 'P2_equal_ratio': {}}
for g in (1, 2, 4, 8):
    out['P1_no_depletion'].append({'g': g, 'rate_hz': round(network(g, 0.25), 1)}); print(out['P1_no_depletion'][-1], flush=True)
for ratio, pairs in (('1/225', [(1.5, 337.5), (3, 675), (6, 1350)]), ('1/75', [(2, 150), (4, 300), (8, 600)])):
    out['P2_equal_ratio'][ratio] = [{'g': g, 'tau_v': t, 'rate_hz': round(network(g, t), 2)} for g, t in pairs]
    print(ratio, out['P2_equal_ratio'][ratio], flush=True)
ratios = {'1/225': 1 / 225, '1/75': 1 / 75}
for ratio, rows in out['P2_equal_ratio'].items():
    r = np.array([x['rate_hz'] for x in rows]); out[f'spread_{ratio}'] = round(float((r.max() - r.min()) / r.mean()), 4)
Path('F:/fly_stream_overflow/cgb_v2/t1b_scaling.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
