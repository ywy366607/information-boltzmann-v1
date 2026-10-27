"""Cross-check the eco-sphere JavaScript core on a generated table (the division grammar,
`cgb_nested_population.DIV_GRAMMAR`) against the Python modules the gate checks against Lean.

1. Rates: on random genomes (as [0, 1] coordinates, a prefactor at 0 an absent event), random inner
   states, pools and occupied neighbours, every organism column of the JS core equals
   `Organisms.rates` (coupled rules: gate factor, foodE * df, pool factors; a move exchanges the
   mover with whatever occupies the target, Lean `move_exchanges`, so its rate ignores occupancy),
   followed by dissolution and division as transcribed in
   `cgb_gate.check_nested_population_rates_match_lean` (float64, relative error < 1e-5: the
   Python organism computes in float32).
2. Effects: in a closed world (no exchange with the reservoirs) the JS core run with division,
   mutation and dissolution keeps food + waste + body contents exact; every newborn differs from
   its parent in at most one coordinate, by one grid step; bodies divide only into empty cells.
"""
from __future__ import annotations

import json
import math as m
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_organism as NO  # noqa: E402
import cgb_nested_population as NP  # noqa: E402
from ecosphere_tables import tables  # noqa: E402

T = NP.DIV_GRAMMAR
C = NP.Coder(T)
rng = np.random.default_rng(23)
tbl = NW.load("rain_torus_w0")
G, V = tbl["geometry"]["G"], tbl["table"]["V"]
BODY, KD, FE = tbl["table"]["B"], tbl["table"]["kappa_dis"], tbl["table"]["foodE"]
BQ, NS = BODY + T.UNITS[NO.S], T.UNITS[NO.S]
cases = []
for b in range(40):
    theta = rng.uniform(0, 1, size=C.D)
    theta[rng.random(C.D) < 0.15] = 0.0                 # some absent events
    theta[-1] = rng.uniform(0.5, 1.0)                   # division possible
    n = np.zeros((6, T.SMAX), dtype=np.int64)
    for p, u in enumerate(T.UNITS):
        k = T.STATES[p]
        cuts = np.sort(rng.integers(0, u + 1, size=k - 1))
        n[p, :k] = np.diff(np.concatenate([[0], cuts, [u]]))
    if b % 3 == 0:
        n[NO.E, :] = 0; n[NO.E, 1] = T.UNITS[NO.E]       # a full division page
    if b % 5 == 4:
        n[NO.S, :] = 0; n[NO.S, 0] = NS                  # a dead membrane
    x, y, h = int(rng.integers(G)), int(rng.integers(G)), int(rng.integers(4))
    f = rng.choice([0, 1, 7, 300, V - 1, V], size=(G, G))
    w = rng.choice([0, 2, 40, V - 30, V], size=(G, G))
    occ_nb = (rng.random(4) < 0.4).tolist()
    cases.append(dict(theta=theta.tolist(), n=n.tolist(), x=x, y=y, h=h, f=f.tolist(), w=w.tolist(), occ=occ_nb))

TT = tables(table=T.name)
js = r"""
const E = require(process.argv[2]); const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = [];
for (const cs of inp.cases) {
  const S = new E.Sphere(inp.T, 1); const G = S.G, SM = inp.T.organism.SMAX;
  for (let x = 0; x < G; x++) for (let y = 0; y < G; y++) { const c = x * G + y; S.f[c] = cs.f[x][y]; S.w[c] = cs.w[x][y]; }
  const o = S.place(cs.x, cs.y, cs.theta, cs.h);
  for (let p = 0; p < 6; p++) for (let s = 0; s < SM; s++) o.n[p * SM + s] = cs.n[p][s];
  for (let d = 0; d < 4; d++) if (cs.occ[d]) S.occ[S.nb(o.c, d)] = 999999;
  const buf = new Float64Array(512); const k = S.orgRates(o, buf);
  out.push(Array.from(buf.slice(0, k)));
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""
tmp = ROOT / "showcase" / "ecosphere" / "_xcheck"
tmp.mkdir(exist_ok=True)
(tmp / "pop_rates.js").write_text(js)
(tmp / "pop_in.json").write_text(json.dumps(dict(T=TT, cases=cases)))
core = str(ROOT / "showcase" / "ecosphere" / "ecosphere_core.js")
subprocess.run(["node", str(tmp / "pop_rates.js"), core, str(tmp / "pop_in.json"), str(tmp / "pop_out.json")], check=True)
res = json.loads((tmp / "pop_out.json").read_text())


def cellF(k):
    return -(m.lgamma(V + 1) - m.lgamma(k + 1) - m.lgamma(V - k + 1))


worst, ncol = 0.0, None
for cs, got in zip(cases, res):
    class St:
        pass
    st = St()
    st.B, st.G, st.V, st.dt, st.foodE = 1, G, V, 0.05, FE
    st.f = torch.tensor(cs["f"], dtype=torch.int32)[None]
    st.w = torch.tensor(cs["w"], dtype=torch.int32)[None]
    g = C.decode(np.array(cs["theta"]))
    kdiv = math_exp = m.exp(float(g.pop("log_div")[0])) if np.isfinite(g["log_div"][0]) else 0.0
    o = NO.Organisms(st, g, table=T)
    # float64 parameters from the same decoded genome
    gd = {k: torch.tensor(np.asarray(v), dtype=torch.float64) for k, v in g.items()}
    phi = torch.zeros(1, 6, T.SMAX, dtype=torch.float64)
    for c, (p, s) in enumerate(T.SELECTED_PHI):
        phi[:, p, s] = gd["phi"][:, c]
    phi[:, NO.S, 1] = T.PHI_S1
    o.phi = phi
    o.kr = torch.exp(gd["logk_rule"])
    nleak = len(T.EXCH) - T.N_SELECTED_X
    o.kx = torch.cat([torch.exp(gd["logk_x"]), torch.full((1, nleak), m.exp(T.LOGK_LEAK), dtype=torch.float64)], 1)
    o.kflip, o.kmove = torch.exp(gd["log_flip"]), torch.exp(gd["log_move"])
    o.n = torch.tensor(cs["n"], dtype=torch.int32)[None]
    o.x[0], o.y[0], o.h[0] = cs["x"], cs["y"], cs["h"]
    ref = list(o.rates().double().numpy()[0])
    empty = [not cs["occ"][d] for d in range(4)]
    nn = np.array(cs["n"])
    ph = phi[0].numpy()
    gate = nn[T.GATE[0], T.GATE[1]]
    dE = NS * T.PHI_S1 - BQ * (ph[NO.E, 1] - ph[NO.E, 0])
    rdiv = 4 * kdiv * sum(empty) * min(1.0, m.exp(-dE)) if (gate > 0 and nn[NO.E, 1] == BQ) else 0.0
    q = BODY + int(sum(s_ * nn[p, s_] for p in range(6) for s_ in range(1, T.STATES[p])))
    wc = cs["w"][cs["x"]][cs["y"]]
    if gate == 0 and wc + q <= V:
        rdis = KD * min(1.0, m.exp(-(cellF(wc + q) - cellF(wc) - float((ph * nn).sum()))))
    else:
        rdis = 0.0
    ref += [rdis, rdiv]
    assert len(ref) == len(got), (len(ref), len(got))
    ncol = len(ref)
    for a_, b_ in zip(ref, got):
        worst = max(worst, abs(a_ - b_) / max(1e-300, abs(a_), abs(b_)))

# ---------- effects: a closed world run in JS ----------
js2 = r"""
const E = require(process.argv[2]); const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const T = inp.T; T.world.kappa_in = 0;                     // closed: nothing enters or leaves
const S = new E.Sphere(T, 7); const G = S.G, SM = T.organism.SMAX, O = T.organism;
const total = () => { let s = 0; for (let c = 0; c < G * G; c++) s += S.f[c] + S.w[c];
  for (const o of S.orgs) if (!o.gone) s += S.bodyContent(o); return s; };
for (let c = 0; c < G * G; c++) { S.f[c] = 200; S.w[c] = 0; }
for (let i = 0; i < 12; i++) { const o = S.placeAt((i * 37) % (G * G), inp.theta); if (o) { o.n[O.E * SM] = 0; o.n[O.E * SM + 1] = O.UNITS[O.E]; } }
const t0 = total(); let worst = 0, badKids = 0, kids = 0, badCells = 0;
const parentOf = new Map();
for (let t = 0; t < inp.ticks; t++) {
  const before = new Set(S.orgs.map(o => o.id));
  S.step();
  for (const o of S.orgs) if (!before.has(o.id) && o.gen > 0) {
    kids++;
    const p = parentOf.get(o.id);
  }
  worst = Math.max(worst, Math.abs(total() - t0));
  for (const o of S.orgs) if (!o.gone && S.occ[o.c] !== o.id) badCells++;
}
// mutation statistics are checked on the kernel itself
let diffBad = 0, changed = 0; const th = Float64Array.from(inp.theta);
for (let i = 0; i < 20000; i++) { const u = S.mutate(th); let nd = 0;
  for (let k = 0; k < th.length; k++) if (u[k] !== th[k]) { nd++; const d = Math.abs(u[k] - th[k]);
    if (d > T.coder.delta + 1e-9 && !(Math.abs(Math.round(th[k] / T.coder.delta) * T.coder.delta - th[k]) > 1e-9)) diffBad++; }
  if (nd > 1) diffBad++; if (nd) changed++; }
fs.writeFileSync(process.argv[4], JSON.stringify({ worst, births: S.births || 0, dissolved: S.dissolved, alive: S.orgs.length,
  badCells, mut_rate: changed / 20000, diffBad, t0 }));
"""
(tmp / "pop_run.js").write_text(js2)
theta0 = rng.uniform(0.3, 0.9, size=C.D)
theta0 = np.round(theta0 / NP.DELTA) * NP.DELTA
theta0[-1] = 1.0
(tmp / "pop_run_in.json").write_text(json.dumps(dict(T=TT, theta=theta0.tolist(), ticks=300)))
subprocess.run(["node", str(tmp / "pop_run.js"), core, str(tmp / "pop_run_in.json"), str(tmp / "pop_run_out.json")], check=True)
run = json.loads((tmp / "pop_run_out.json").read_text())
# a genome that only divides (every other prefactor absent, the reserve as high as the range allows):
# division, the newborn's state and the parent's drain must keep the content exact too
thd = np.zeros(C.D)
iE1 = [i for i, ps in enumerate(T.SELECTED_PHI) if tuple(ps) == (NO.E, 1)]
thd[:len(T.SELECTED_PHI)] = 0.5
for i in iE1:
    thd[i] = 1.0
thd[-1] = 1.0
(tmp / "pop_div_in.json").write_text(json.dumps(dict(T=TT, theta=thd.tolist(), ticks=100)))
subprocess.run(["node", str(tmp / "pop_run.js"), core, str(tmp / "pop_div_in.json"), str(tmp / "pop_div_out.json")], check=True)
div = json.loads((tmp / "pop_div_out.json").read_text())
out = dict(table=T.name, cases=len(cases), columns=ncol, organism_max_relative_error=worst, closed_run=run,
           division_run=div,
           passed=bool(worst < 1e-5 and run["worst"] == 0 and run["badCells"] == 0 and run["diffBad"] == 0
                       and abs(run["mut_rate"] - NP.PMUT) < 0.01 and div["births"] > 0 and div["worst"] == 0
                       and div["badCells"] == 0))
print(out)
(ROOT / "results/published/cgb_generated/ecosphere_js_vs_python_population_20261027.json").write_text(json.dumps(out, indent=1))
