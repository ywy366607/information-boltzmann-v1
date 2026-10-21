"""V2 / T4 - trophic retention (E9): do the terminals that explain the residual survive?

Circuit = T3b (song on 200 inputs -> output; 200 phase-selective inhibitory interneurons = the predictors, plastic by
the Vogels rule) plus 100 unrelated inhibitory cells firing Poisson. The 300 inhibitory terminals onto the output
depend on trophic support (E9); their structure turns over at lambda0 / (1 + T / T*) without it.
Rules: 'residual' (E9: release when the output's input persists and its activity just fell; active terminals take it
up by mass action) vs 'flux' (control: release follows the output's recent firing; same total production).
Calibration is blind to the groups: a pilot sets T* to the mean support of all 300 terminals, and q_flux so that both
rules release the same total.
Predictions written before running:
 P1 residual, random cells at the predictors' rate: predictors keep >= 2x the structure of the random terminals
 P2 flux, same rates: the predictors' advantage vanishes or reverses (they fire when the output is being silenced)
 P3 residual, random cells at 3x the rate: reported without a direction (mass action favours rate, correlation
    favours the predictors) - the sharpest test of E9
 P4 the total structure of the 300 terminals settles (not growing without bound, not all gone)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass, TrophicParams

N_IN, N_BASIS, N_RAND = 200, 200, 100


def build(seed=0):
    rng = np.random.default_rng(seed)
    exc = CellClass(E_out=5.0, U=0.4, tau_v=300.0)
    inh = CellClass(E_out=-0.3, U=0.4, tau_v=300.0, inhibitory_plastic=True)
    n_cells = N_IN + N_BASIS + N_RAND + 1
    out = n_cells - 1
    cls = np.r_[np.zeros(N_IN, int), np.ones(N_BASIS + N_RAND, int), np.full(1, 2)]
    pre, post, g = list(range(N_IN)), [out] * N_IN, [0.05] * N_IN
    for j in range(N_BASIS):
        for i in rng.choice(N_IN, 3, replace=False):
            pre.append(i); post.append(N_IN + j); g.append(0.4)
    first_inh_edge = len(pre)
    for j in range(N_BASIS + N_RAND):
        pre.append(N_IN + j); post.append(out); g.append(0.0005)
    m = Medium([exc, inh, CellClass(E_out=5.0)], cls, np.array(pre), np.array(post), np.array(g, float), seed=seed)
    m.inh_rho0 = 0.005; m.inh_eta = 1e-4
    m.trophic_edges[first_inh_edge:] = True
    basis_edges = np.arange(first_inh_edge, first_inh_edge + N_BASIS)
    rand_edges = np.arange(first_inh_edge + N_BASIS, first_inh_edge + N_BASIS + N_RAND)
    return m, out, basis_edges, rand_edges


def run(rule, rand_hz, T_ms, trophic, seed=0, record_T=False):
    m, out, be, re_ = build(seed)
    m.trophic = trophic
    rng = np.random.default_rng(seed + 7)
    phases = np.random.default_rng(1).uniform(0, 500.0, N_IN)
    p_rand = rand_hz / 1000 * m.dt
    g0 = m.g.copy()
    traj = []
    for k in range(int(T_ms / m.dt)):
        t = m.t % 500.0
        f = np.zeros(m.N, bool)
        f[:N_IN] = (np.abs(((t - phases + 250) % 500) - 250) < m.dt / 2) & (rng.random(N_IN) > 0.1)
        f[N_IN + N_BASIS:N_IN + N_BASIS + N_RAND] = rng.random(N_RAND) < p_rand
        m.step(forced=f)
        if k % int(10000 / m.dt) == 0:
            traj.append(round(float(m.g[m.trophic_edges].sum()), 5))
    basis_rate = None
    return m, be, re_, g0, traj


def summary(m, be, re_, g0, traj):
    gb, gr = m.g[be], m.g[re_]
    return {'basis_g_mean': float(gb.mean()), 'random_g_mean': float(gr.mean()),
            'basis_over_random': float(gb.mean() / max(gr.mean(), 1e-12)),
            'basis_T_mean': float(m.T[be].mean()), 'random_T_mean': float(m.T[re_].mean()),
            'basis_alive_share': float((gb > 0.1 * g0[be]).mean()), 'random_alive_share': float((gr > 0.1 * g0[re_]).mean()),
            'total_structure_every_10s': traj, 'released_total': float(m.released)}


out = {}
# blind pilot: residual rule, no turnover yet, 20 s -> mean support of all terminals and the release total
pilot_p = TrophicParams(rule='residual', lambda0=0.0, T_star=1.0)
m, be, re_, g0, _ = run('residual', 5.4, 20000.0, pilot_p)
T_star = float(m.T[m.trophic_edges].mean())
rel_res = m.released
pf = TrophicParams(rule='flux', q_flux=1.0, lambda0=0.0)
m2, *_ = run('flux', 5.4, 20000.0, pf)
q_flux = rel_res / max(m2.released, 1e-12)
out['calibration'] = {'T_star': T_star, 'q_flux': q_flux, 'basis_rate_hz': None}
print('calibration', out['calibration'], flush=True)

for name, rule, rand_hz in (('C1_residual_equal_rate', 'residual', 5.4), ('C2_flux_equal_rate', 'flux', 5.4),
                            ('C3_residual_3x_rate', 'residual', 16.2), ('C4_flux_3x_rate', 'flux', 16.2)):
    tp = TrophicParams(rule=rule, T_star=T_star, q_flux=q_flux)
    out[name] = summary(*run(rule, rand_hz, 120000.0, tp))
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
Path('F:/fly_stream_overflow/cgb_v2/t4_trophic.json').write_text(json.dumps(out, indent=1))
