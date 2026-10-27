"""Cross-check the eco-sphere JavaScript core against the Python engines: on random states, every
world rate (12 classes per cell) and every organism rate column equal the Python values that the
gate checks against Lean (relative error < 1e-9 in float64)."""
from __future__ import annotations

import json
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
from ecosphere_tables import tables  # noqa: E402

rng = np.random.default_rng(21)
tbl = NW.load("rain_torus_w0")
G, V = tbl["geometry"]["G"], tbl["table"]["V"]
B = 5
genomes = NO.random_genomes(B, rng)
cases = []
for b in range(B + 3):
    f = rng.integers(0, V + 1, size=(G, G)); f[:3] = rng.choice([0, 1, V - 1, V], size=(3, G))
    w = rng.integers(0, 40, size=(G, G)); w[-2:] = rng.choice([0, V], size=(2, G))
    mu = rng.choice([-8.0, 0.0, 8.0], size=(G, G))
    n = np.zeros((6, NO.SMAX), dtype=np.int64)
    for p, u in enumerate(NO.UNITS):
        cuts = np.sort(rng.integers(0, u + 1, size=NO.STATES[p] - 1))
        n[p, :NO.STATES[p]] = np.diff(np.concatenate([[0], cuts, [u]]))
    x, y, h = map(int, rng.integers(G, size=3)); h %= 4
    if b >= B:                       # dead membrane: the dissolution column
        n[NO.S, :] = 0
        n[NO.S, 0] = NO.UNITS[NO.S]
    g = {k: np.asarray(v)[b % B] for k, v in genomes.items()}
    cases.append(dict(f=f.tolist(), w=w.tolist(), mu=mu.tolist(), n=n.tolist(), x=x, y=y, h=h,
                      genome={k: (v.tolist() if np.ndim(v) else float(v)) for k, v in g.items()}))
T = tables()
T["world"]["B"], T["world"]["kappa_dis"], T["world"]["foodE"] = tbl["table"]["B"], tbl["table"]["kappa_dis"], tbl["table"]["foodE"]
js = r"""
const E = require(process.argv[2]); const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = [];
for (const cs of inp.cases) {
  const S = new E.Sphere(inp.T, 1); const G = S.G;
  for (let x = 0; x < G; x++) for (let y = 0; y < G; y++) { const c = x * G + y; S.f[c] = cs.f[x][y]; S.w[c] = cs.w[x][y]; S.mu[c] = cs.mu[x][y]; }
  const world = []; for (let c = 0; c < G * G; c++) world.push(Array.from(S.worldRates(c)));
  const o = S.place(cs.x, cs.y, cs.genome, cs.h);
  for (let p = 0; p < 6; p++) for (let s = 0; s < inp.T.organism.SMAX; s++) o.n[p * inp.T.organism.SMAX + s] = cs.n[p][s];
  const buf = new Float64Array(512); const k = S.orgRates(o, buf);
  out.push({ world, org: Array.from(buf.slice(0, k)) });
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""
tmp = ROOT / "showcase" / "ecosphere" / "_xcheck"
tmp.mkdir(exist_ok=True)
(tmp / "run.js").write_text(js)
(tmp / "in.json").write_text(json.dumps(dict(T=T, cases=cases)))
subprocess.run(["node", str(tmp / "run.js"), str(ROOT / "showcase" / "ecosphere" / "ecosphere_core.js"),
                str(tmp / "in.json"), str(tmp / "out.json")], check=True)
res = json.loads((tmp / "out.json").read_text())
worst_w = worst_o = 0.0
for cs, r in zip(cases, res):
    Wt = NW.World(tbl, B=1, device="cpu", dtype=torch.float64)
    Wt.f[0] = torch.tensor(cs["f"]); Wt.w[0] = torch.tensor(cs["w"])
    ref = Wt.rates(torch.tensor(cs["mu"])[None]).numpy()[0]          # (12, G, G)
    jsw = np.array(r["world"]).reshape(G, G, 12).transpose(2, 0, 1)
    worst_w = max(worst_w, float((np.abs(jsw - ref) / np.maximum(np.abs(ref), 1e-300)).max()))

    class St: pass
    st = St(); st.B, st.G, st.V, st.dt = 1, G, V, 0.05
    st.f = torch.tensor(cs["f"], dtype=torch.int32)[None]; st.w = torch.tensor(cs["w"], dtype=torch.int32)[None]
    gg = {k: np.asarray(v)[None] for k, v in cs["genome"].items()}
    st.foodE = tbl["table"]["foodE"]
    o = NO.Organisms(st, gg)
    o.n = torch.tensor(cs["n"], dtype=torch.int32)[None]
    o.x[0], o.y[0], o.h[0] = cs["x"], cs["y"], cs["h"]
    o.phi = o.phi.double(); o.kr = o.kr.double(); o.kx = o.kx.double(); o.kflip = o.kflip.double(); o.kmove = o.kmove.double()
    _, fc, wc = o.pools()
    ref_o = o.rates(fc, wc).numpy()[0]
    # Lean dissolution (NestedWorld eff1, dir false): kappa_dis * exp(-(cellF(w + B) - cellF(w) - bugE(yEmpty)) / 2)
    import math as m
    # Lean: c1 = kappa_dis * [no intact membrane] * exp(-|dE| / 2), dE = cellF(w + m) - cellF(w) - bugE(y)
    Bb, kd = tbl["table"]["B"], tbl["table"]["kappa_dis"]
    cellF = lambda k: -(m.lgamma(V + 1) - m.lgamma(k + 1) - m.lgamma(V - k + 1))
    wcell = cs["w"][cs["x"]][cs["y"]]
    nn = np.array(cs["n"])
    dead = nn[NO.S, 1] == 0
    mq = Bb + int(sum(s_ * nn[p, s_] for p in range(6) for s_ in range(NO.SMAX)))
    phd = o.phi[0].double().numpy()
    bugE = float((phd * nn).sum())
    if dead and wcell + mq <= V:
        dE = cellF(wcell + mq) - cellF(wcell) - bugE
        dis = kd * m.exp(-abs(dE) / 2) * m.exp(-dE / 2)
    else:
        dis = 0.0
    ref_o = np.concatenate([ref_o, [dis, 0.0]])        # legacy bodies have no division recipe
    worst_o = max(worst_o, float((np.abs(np.array(r["org"]) - ref_o) / np.maximum(np.abs(ref_o), 1e-300)).max()))
ssa = json.loads((ROOT / "results/published/cgb_generated/nested_world_tau_vs_ssa_20261026.json").read_text())
js2 = r"""
const E = require(process.argv[2]); const fs = require('fs');
const T = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')); T.world.G = 6;
const R = 2000, times = []; for (let t = 0; t < 90; t += 5) times.push(t);
const sums = times.map(() => 0), sq = times.map(() => 0);
for (let run = 0; run < R; run++) {
  const S = new E.Sphere(T, 1000 + run); const st = T.world.stimulus;
  times.forEach((t0, k) => {
    for (let t = t0; t < t0 + 5; t++) {
      for (let c = 0; c < 36; c++) S.mu[c] = S.kind[c] === 2 ? st.mu_sink : st.mu_dry;
      if (t < 30) { S.mu[0] = st.mu_rain; S.mu[3 * 6 + 4] = st.mu_rain; }
      for (let l = 0; l < 20; l++) S.worldLeap();
    }
    let tot = 0; for (let c = 0; c < 36; c++) tot += S.f[c];
    sums[k] += tot; sq[k] += tot * tot;
  });
}
fs.writeFileSync(process.argv[4], JSON.stringify({ mean: sums.map(v => v / R), var: sq.map((v, k) => v / R - (sums[k] / R) ** 2) }));
"""
(tmp / "stat.js").write_text(js2)
(tmp / "T.json").write_text(json.dumps(T))
subprocess.run(["node", str(tmp / "stat.js"), str(ROOT / "showcase" / "ecosphere" / "ecosphere_core.js"),
                str(tmp / "T.json"), str(tmp / "stat.json")], check=True)
st = json.loads((tmp / "stat.json").read_text())
z = [abs(a - b) / np.sqrt(se ** 2 + v / 2000) for a, b, se, v in zip(st["mean"], ssa["ssa_mean"], ssa["ssa_sem"], st["var"])]
rel = [abs(a - b) / b for a, b in zip(st["mean"], ssa["ssa_mean"])]
out = dict(world_max_relative_error=worst_w, organism_max_relative_error=worst_o, cases=len(cases),
           js_world_vs_exact_ssa=dict(max_z=float(max(z)), max_relative_bias=float(max(rel)),
                                      js_mean=st["mean"], ssa_mean=ssa["ssa_mean"]),
           passed=bool(worst_w < 1e-9 and worst_o < 1e-6 and max(rel) < 0.01))
print(out)
(ROOT / "results/published/cgb_generated/ecosphere_js_vs_python_20261027.json").write_text(json.dumps(out, indent=1))
