"""V2 / T3b - pre-registered after T3's P2 failed: prediction of temporal structure needs a basis (CGB L3).

T3: 50 interneurons with 40 random inputs each fire all over the cycle, so plastic inhibition could only cancel the
mean drive (B/A 0.89). Nature's cancellers (electric-fish ELL, cerebellum) put the plastic inhibition / negative image
on top of an expansion of sparse, time-selective granule cells. Here: 200 interneurons with 3 strong inputs each
(each fires at its inputs' phases) = a temporal basis; everything else as in T3.
Prediction written before running: P2b after 60 s on song A, song B gives >= 1.5x the output rate of A; the control
without inhibitory plasticity gives ~1.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass
src = open(Path(__file__).with_name('v2_t3_prediction.py'), encoding='utf-8').read().split('res = {}')[0]
ns = {'__file__': str(Path(__file__).with_name('v2_t3_prediction.py'))}; exec(src, ns)
song, play = ns['song'], ns['play']


def build_basis(seed=0, n_in=200, n_inh=200, fan_in=3, g_ii=0.4):
    rng = np.random.default_rng(seed)
    exc = CellClass(E_out=5.0, U=0.4, tau_v=300.0)
    inh = CellClass(E_out=-0.3, U=0.4, tau_v=300.0, inhibitory_plastic=True)
    cls = np.r_[np.zeros(n_in, int), np.ones(n_inh, int), np.full(1, 2)]
    out = n_in + n_inh
    pre, post, g = list(range(n_in)), [out] * n_in, [0.05] * n_in
    for j in range(n_inh):
        for i in rng.choice(n_in, fan_in, replace=False):
            pre.append(i); post.append(n_in + j); g.append(g_ii)
    for j in range(n_inh):
        pre.append(n_in + j); post.append(out); g.append(0.0005)
    m = Medium([exc, inh, CellClass(E_out=5.0)], cls, np.array(pre), np.array(post), np.array(g, float), seed=seed)
    m.inh_rho0 = 0.005
    return m, out


res = {}
for label, learn in (('plastic', True), ('control', False)):
    m, out = build_basis()
    A, B = song(200, 1), song(200, 2)
    curve = [round(float(play(m, out, A, 2000.0, learn)), 2) for _ in range(30)]
    rA, rB = float(play(m, out, A, 10000.0, False)), float(play(m, out, B, 10000.0, False))
    res[label] = {'training_rate_hz_every_2s': curve, 'after_A_hz': round(rA, 2), 'after_B_hz': round(rB, 2),
                  'B_over_A': round(rB / max(rA, 1e-9), 2)}
    print(label, res[label], flush=True)
Path('F:/fly_stream_overflow/cgb_v2/t3b_basis.json').write_text(json.dumps(res, indent=1))
