"""V2 / T3 - prediction as reverse flow (E1 + E8): does inhibitory plasticity learn to cancel the predictable input?

Circuit: 200 input cells carry a repeating pattern (each input fires at its own phase of a 500 ms cycle, a "song");
they excite one output cell and also drive 50 inhibitory interneurons (fixed random input -> interneuron wiring).
The interneurons inhibit the output; only those inhibitory synapses are plastic (Vogels rule, target rate rho0).
Nothing computes an error: the output's membrane current is the residual (E1).
Predictions written before running:
 P1 during training on song A the output rate falls toward rho0 (5 Hz) from its initial value (detailed balance)
 P2 after training, presenting a new song B (same inputs, phases shuffled) gives a clearly higher output rate than A
    (ratio B/A >= 2): the learned inhibition cancels A's temporal structure, not just the mean drive
 P3 control without inhibitory plasticity: no fall, and A and B give similar rates (ratio ~1)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass


def build(seed=0, n_in=200, n_inh=50):
    rng = np.random.default_rng(seed)
    exc = CellClass(E_out=5.0, U=0.4, tau_v=300.0)
    inh = CellClass(E_out=-0.3, U=0.4, tau_v=300.0, inhibitory_plastic=True)
    out_c = CellClass(E_out=5.0)
    # cells: inputs 0..n_in-1, inhibitory n_in..n_in+n_inh-1, output last
    cls = np.r_[np.zeros(n_in, int), np.ones(n_inh, int), np.full(1, 2)]
    out = n_in + n_inh
    pre, post, g = [], [], []
    for i in range(n_in):                                   # inputs -> output
        pre.append(i); post.append(out); g.append(0.05)      # calibrated: ~70 Hz output before learning
    for j in range(n_inh):                                  # inputs -> interneurons (fixed, random 20%)
        for i in np.flatnonzero(rng.random(n_in) < 0.2):
            pre.append(i); post.append(n_in + j); g.append(0.08)   # ~7 Hz interneurons
    for j in range(n_inh):                                  # interneurons -> output (plastic)
        pre.append(n_in + j); post.append(out); g.append(0.001)
    m = Medium([exc, inh, out_c], cls, np.array(pre), np.array(post), np.array(g, float), seed=seed)
    m.inh_rho0 = 0.005                                      # 5 Hz target (per ms)
    return m, out


def song(n_in, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 500.0, n_in)                      # each input's phase in the 500 ms cycle


def play(m, out, phases, T, learn_inh, jitter=2.0, rate_in_cycle=None):
    n_in = len(phases); rng = m.rng
    m.inh_eta = 1e-4 if learn_inh else 0.0
    m.plastic = True
    e = m.plastic_exc.copy(); m.plastic_exc[:] = False       # excitatory synapses fixed in this test
    counts, t0 = 0, m.t
    for k in range(int(T / m.dt)):
        t = (m.t % 500.0)
        f = np.zeros(m.N, bool)
        f[:n_in] = np.abs(((t - phases + 250) % 500) - 250) < m.dt / 2
        f[:n_in] &= rng.random(n_in) > 0.1                  # 10% failures: not a perfect clock
        s = m.step(I_ext=None, forced=f)
        counts += s[out]
    m.plastic_exc[:] = e
    return counts / ((m.t - t0) / 1000.0)


res = {}
for label, learn in (('plastic', True), ('control', False)):
    m, out = build()
    A, B = song(200, 1), song(200, 2)
    curve = [round(play(m, out, A, 2000.0, learn), 2) for _ in range(30)]      # 60 s of song A
    rate_A = play(m, out, A, 10000.0, False)
    rate_B = play(m, out, B, 10000.0, False)
    res[label] = {'training_rate_hz_every_2s': curve, 'after_A_hz': round(rate_A, 2), 'after_B_hz': round(rate_B, 2),
                  'B_over_A': round(rate_B / max(rate_A, 1e-9), 2)}
    print(label, res[label], flush=True)
Path('F:/fly_stream_overflow/cgb_v2/t3_prediction.json').write_text(json.dumps(res, indent=1))
