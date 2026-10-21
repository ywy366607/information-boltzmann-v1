"""V2 / T2 - the plasticity template (E4-E6): do learning rules appear from calcium and two enzyme pools?

One presynaptic and one postsynaptic cell, one plastic excitatory synapse. Predictions written before running:
 P1 STDP: 60 pairings at 1 Hz, pre spike then post spike at lag dt (post driven to fire by a brief current):
    dt > 0 (pre before post, within ~20 ms) -> potentiation; dt < 0 (post before pre) -> depression; |dt| large
    (> 100 ms) -> ~0. (The calcium template is known to also give a second depression window at large positive dt;
    report it if it appears.)
 P2 BCM-like frequency dependence: 60 s of paired Poisson pre and post at rate f: low f depresses, high f potentiates,
    with a crossover between.
 P3 Bounded weights: 300 strong pairings saturate g below gmax with no clipping (g never exceeds gmax).
Learning needs a third factor; here the tonic m0 (cortex-like) is raised so that tags convert within the protocol.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass, SynapseParams


def SYN():
    # regime from a small manual scan (scratchpad t2_scan.py): kinase stronger than phosphatase (Graupner-Brunel),
    # phosphatase threshold above the calcium of an unpaired spike's slow part, dendritic afterdepolarisation 10 ms;
    # the synapse is small (EPSP ~1% of threshold) and its slots are bounded at 2x the start
    return SynapseParams(m0=0.02, aK=0.2, thP=0.7, nP=4.0, tau_bap=10.0, bap_amp=2.0, gmax=0.04, eta=4e-5)


def pair(syn, g0=0.02):
    m = Medium([CellClass()], np.zeros(2, int), np.array([0]), np.array([1]), np.array([g0]), syn=syn)
    return m


def run_pairings(dt_lag, n=60, period=1000.0, syn=None, g0=0.02):
    m = pair(syn or SYN(), g0)
    steps = int(period / m.dt)
    for rep in range(n):
        for k in range(steps):
            t = k * m.dt
            f = np.zeros(2, bool)
            f[0] = abs(t - 100.0) < m.dt / 2
            f[1] = abs(t - (100.0 + dt_lag)) < m.dt / 2
            m.step(forced=f)
    for k in range(int(5000 / m.dt)):                      # let the tags convert / decay
        m.step()
    return float(m.g[0] / g0 - 1), float(m.z[0])


def run_rates(f_hz, T=60000.0, syn=None, g0=0.02, seed=0):
    m = pair(syn or SYN(), g0)
    rng = np.random.default_rng(seed); p = f_hz / 1000 * m.dt
    for k in range(int(T / m.dt)):
        m.step(forced=rng.random(2) < p)
    for k in range(int(5000 / m.dt)):
        m.step()
    return float(m.g[0] / g0 - 1)


out = {'P1_stdp': [], 'P2_rate': [], 'P3_bound': {}}
for lag in (-100, -60, -40, -25, -15, -10, -5, 5, 10, 15, 25, 40, 60, 100):
    dg, z = run_pairings(lag)
    out['P1_stdp'].append({'lag_ms': lag, 'dg': round(dg, 5)}); print(out['P1_stdp'][-1], flush=True)
for f in (1, 5, 10, 20, 40):
    out['P2_rate'].append({'rate_hz': f, 'dg': round(run_rates(f), 5)}); print(out['P2_rate'][-1], flush=True)
syn = SYN()
m = pair(syn, 0.02); traj = []                              # dg reported relative to the start (+1.0 = doubled)
for rep in range(300):
    for k in range(int(200 / m.dt)):
        t = k * m.dt
        f = np.zeros(2, bool); f[0] = abs(t - 20) < m.dt / 2; f[1] = abs(t - 30) < m.dt / 2
        m.step(forced=f)
    traj.append(float(m.g[0]))
out['P3_bound'] = {'gmax': syn.gmax, 'g_after_100': round(traj[99], 4), 'g_after_300': round(traj[-1], 4),
                   'max_g': round(max(traj), 4)}
print(json.dumps(out, indent=1))
Path('F:/fly_stream_overflow/cgb_v2/t2_plasticity.json').write_text(json.dumps(out, indent=1))
