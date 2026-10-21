"""V2 / T4c - trophic selection isolated: constitutive supply, capture by plasticity tags vs by raw activity.

One output cell, 200 excitatory inputs of equal initial strength:
  correlated  100 inputs firing in volleys (a common Poisson event at 5 Hz, each input joins with p 0.8, 1 ms jitter):
              the volleys drive the output, so these synapses see pre-before-post coincidences -> positive tags
  random      100 independent Poisson inputs at rate r (4 Hz = the correlated inputs' own rate, or 12 Hz = 3x)
Only trophic support selects: tags are computed (E5) but not converted into structure (E6 off, eta 0), no inhibitory
plasticity; structure changes only by turnover lambda0 / (1 + T / T*). Supply is constitutive (q0 from the output).
Capture: 'tag' (revised E9: [z]+ x factor) or 'activity' (control: presynaptic trace x factor).
Calibration blind to the groups: per capture mode, T* = f x the mean support of all 200 synapses in a pilot without
turnover (20 s at equal rates).
Predictions written before running:
 P1c tag capture, equal rates: correlated keep >= 2x the structure of random
 P2c activity capture: ratio ~1 at equal rates; at 3x random rate the random group wins (ratio < 1)
 P3c tag capture at 3x random rate: correlated still >= 2x
 P4c structure settles on a nonzero plateau (last three samples within 10%); halving q0 roughly halves what survives
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass, SynapseParams, TrophicParams

N_C, N_R = 100, 100


def run(capture, rand_hz, T_ms, f=None, T_pilot=None, q0=1.0, seed=0, lambda0=1 / 20000.0):
    rng = np.random.default_rng(seed)
    n_in = N_C + N_R
    cls = np.r_[np.zeros(n_in, int), np.ones(1, int)]
    pre, post = np.arange(n_in), np.full(n_in, n_in)
    syn = SynapseParams(m0=0.02, aK=0.2, thP=0.7, nP=4.0, tau_bap=10.0, bap_amp=2.0, gmax=0.04, eta=0.0)   # tags only
    m = Medium([CellClass(E_out=5.0, U=0.4, tau_v=300.0), CellClass()], cls, pre, post, np.full(n_in, 0.01), syn=syn, seed=seed)
    m.trophic_edges[:] = True
    m.trophic = TrophicParams(rule='constitutive', capture=capture, q0=q0,
                              lambda0=(0.0 if f is None else lambda0), T_star=(1.0 if f is None else f * T_pilot))
    p_event, p_rand = 5.0 / 1000 * m.dt, rand_hz / 1000 * m.dt
    pending = []                                   # (time, input) of jittered volley spikes
    traj, out_spikes = [], 0
    for k in range(int(T_ms / m.dt)):
        forced = np.zeros(m.N, bool)
        if rng.random() < p_event:
            join = np.flatnonzero(rng.random(N_C) < 0.8)
            for i, d in zip(join, rng.uniform(0, 1.0, len(join))):
                pending.append((m.t + d, i))
        if pending:
            due = [x for x in pending if x[0] <= m.t]
            pending = [x for x in pending if x[0] > m.t]
            for _, i in due:
                forced[i] = True
        forced[N_C:n_in] = rng.random(N_R) < p_rand
        s = m.step(forced=forced)
        out_spikes += s[n_in]
        if k % int(10000 / m.dt) == 0:
            traj.append(round(float(m.g.sum()), 5))
    c, r = slice(0, N_C), slice(N_C, n_in)
    return m, {'corr_g': float(m.g[c].mean()), 'rand_g': float(m.g[r].mean()),
               'corr_over_rand': float(m.g[c].mean() / max(m.g[r].mean(), 1e-12)),
               'corr_T': float(m.T[c].mean()), 'rand_T': float(m.T[r].mean()),
               'corr_alive': float((m.g[c] > 0.001).mean()), 'rand_alive': float((m.g[r] > 0.001).mean()),
               'output_hz': out_spikes / (T_ms / 1000), 'total_every_10s': traj}


out = {}
for capture in ('tag', 'activity'):
    m, _ = run(capture, 4.0, 20000.0)
    T_pilot = float(m.T.mean()); out[f'{capture}_T_pilot'] = T_pilot
    for rand_hz in (4.0, 12.0):
        for f in (0.3, 0.1):
            _, s = run(capture, rand_hz, 120000.0, f, T_pilot)
            out[f'{capture}_r{rand_hz:g}_f{f}'] = s
            print(capture, rand_hz, f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    if capture == 'tag':
        _, s = run(capture, 4.0, 120000.0, 0.1, T_pilot, q0=0.5)
        out['tag_r4_f0.1_halfq0'] = s
        print('tag half q0', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
Path('F:/fly_stream_overflow/cgb_v2/t4c_capture.json').write_text(json.dumps(out, indent=1))
