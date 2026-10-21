"""V2 / T1 - conserved pools only (E1-E3, plasticity off): what appears without being written?

Predictions written before running (docs/information_boltzmann/CGB_CORE_EQUATIONS_20261020.md section 5):
 (a) paired-pulse ratio of release = 1 - U exp(-dt / tau_v)            (closed form of the vesicle pool)
 (b) Poisson input at rate f: transmitted release per ms = U f v_bar, v_bar = 1 / (1 + U f tau_v); it saturates at
     1 / tau_v however fast the input (the pool, not the input, sets the throughput)
 (c) a strongly recurrent excitatory network: with tau_v -> 0 (no depletion) raising the coupling g runs the network
     away to the refractory limit; with depletion the rate saturates as g grows, at a level set by tau_v (slower
     replenishment, lower rate), not by g
 (d) the sodium pool alone gives a refractory period and spike-rate adaptation: under a constant strong drive the
     first interval is shorter than the steady one
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass

out = {}

# (a) paired pulses
U, tv = 0.4, 300.0
res = []
for gap in (10, 25, 50, 100, 200, 400, 800):
    m = Medium([CellClass(U=U, tau_v=tv)], np.zeros(2, int), np.array([0]), np.array([1]), np.array([0.1]))
    m.plastic = False
    rel = []
    for k in range(int((gap + 20) / m.dt)):
        f = np.zeros(2, bool); f[0] = k in (0, int(gap / m.dt))
        v0 = m.v[0]; m.step(forced=f)
        if f[0]:
            rel.append(U * v0)
    res.append({'gap_ms': gap, 'ppr_sim': round(rel[1] / rel[0], 4), 'ppr_theory': round(1 - U * np.exp(-gap / tv), 4)})
out['a_paired_pulse'] = res

# (b) throughput under Poisson input
res = []
for f_hz in (1, 5, 20, 50, 100, 200):
    m = Medium([CellClass(U=U, tau_v=tv)], np.zeros(2, int), np.array([0]), np.array([1]), np.array([0.1]))
    m.plastic = False
    rng = np.random.default_rng(1); p = f_hz / 1000 * m.dt; total = 0.0; T = 20000.0
    for k in range(int(T / m.dt)):
        f = np.zeros(2, bool); f[0] = rng.random() < p
        v0 = m.v[0]; m.step(forced=f)
        total += U * v0 if f[0] else 0.0
    fpm = f_hz / 1000
    res.append({'input_hz': f_hz, 'release_per_s_sim': round(total / T * 1000, 3),
                'theory': round(U * fpm / (1 + U * fpm * tv) * 1000, 3), 'ceiling_1_over_tau_v': round(1000 / tv, 3)})
out['b_throughput'] = res

# (c) recurrent network: runaway without depletion, rate set by tau_v with it
def network(g_scale, tau_v, seed=0, N=400, p=0.1, T=3000.0):
    rng = np.random.default_rng(seed)
    exc = CellClass(E_out=5.0, U=0.4, tau_v=tau_v)
    inh = CellClass(E_out=-0.3, U=0.4, tau_v=tau_v)
    cls = np.r_[np.zeros(int(0.8 * N), int), np.ones(N - int(0.8 * N), int)]
    A = rng.random((N, N)) < p; np.fill_diagonal(A, False)
    pre, post = np.nonzero(A)
    g = np.where(cls[pre] == 0, 0.05, 0.1) * g_scale
    m = Medium([exc, inh], cls, pre, post, g.astype(float), seed=seed)
    m.plastic = False
    spikes = 0; warm = 500.0
    for k in range(int(T / m.dt)):
        I = 0.06 + 0.05 * rng.standard_normal(N)                 # weak noisy background drive
        s = m.step(I_ext=I)
        if m.t > warm:
            spikes += s.sum()
    return spikes / N / ((T - warm) / 1000)

res = []
for tau_v in (0.25, 150.0, 300.0, 600.0):                       # 0.25 ms ~ no depletion (recovers within a step)
    for g_scale in (1, 2, 4, 8):
        res.append({'tau_v_ms': tau_v, 'g_scale': g_scale, 'rate_hz': round(network(g_scale, tau_v), 2)})
        print(res[-1], flush=True)
out['c_recurrent'] = res

# (d) refractoriness and adaptation from the sodium pool
m = Medium([CellClass()], np.zeros(1, int), np.zeros(0, int), np.zeros(0, int), np.zeros(0))
m.plastic = False
times = []
for k in range(int(500 / m.dt)):
    if m.step(I_ext=np.array([0.3]))[0]:
        times.append(m.t)
isi = np.diff(times)
out['d_sodium_pool'] = {'spikes': len(times), 'first_isi_ms': round(float(isi[0]), 2) if len(isi) else None,
                        'last_isi_ms': round(float(isi[-1]), 2) if len(isi) else None,
                        'min_isi_ms': round(float(isi.min()), 2) if len(isi) else None}
print(json.dumps(out, indent=1))
Path('F:/fly_stream_overflow/cgb_v2').mkdir(parents=True, exist_ok=True)
Path('F:/fly_stream_overflow/cgb_v2/t1_pools.json').write_text(json.dumps(out, indent=1))
