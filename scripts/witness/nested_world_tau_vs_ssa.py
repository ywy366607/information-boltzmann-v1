"""Tau-leaping (torch) against exact Gillespie on a 6 x 6 rain torus: the same stimulus (two cells
raining for 30 time units, then dry for 60), mean food per cell over time, 200 runs each."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW

tbl = NW.load("rain_torus_w0"); tbl["geometry"]["G"] = G = 6
kind = NW.kinds(G); s = tbl["stimulus"]
def mu_at(t):
    m = np.where(kind == 2, s["mu_sink"], s["mu_dry"]).astype(np.float64)
    if t < 30: m[0, 0] = m[3, 4] = s["mu_rain"]
    return m
times = list(range(0, 90, 5))
R = 200
ssa = np.zeros((R, len(times)))
for r in range(R):
    f = np.zeros((G, G), np.int64); w = np.zeros((G, G), np.int64)
    for k, t in enumerate(times):
        NW.ssa_world(f, w, mu_at(t), kind, tbl, 5.0, 1000 * r + k)
        ssa[r, k] = f.sum()
tau = {}
for dt in (0.05, 0.2):
    W = NW.World(tbl, B=R, device="cuda", seed=1, dt=dt)
    out = np.zeros((R, len(times)))
    for k, t in enumerate(times):
        W.step(torch.tensor(mu_at(t), device="cuda")[None].expand(R, G, G).contiguous(), T=5.0)
        out[:, k] = W.f.sum((1, 2)).cpu().numpy()
    tau[dt] = out
res = dict(times=[t + 5 for t in times], ssa_mean=ssa.mean(0).tolist(), ssa_sem=(ssa.std(0) / np.sqrt(R)).tolist(),
           **{f"tau_dt{dt}_mean": v.mean(0).tolist() for dt, v in tau.items()},
           **{f"tau_dt{dt}_max_z": float(np.max(np.abs(v.mean(0) - ssa.mean(0)) / np.sqrt(ssa.var(0) / R + v.var(0) / R + 1e-9))) for dt, v in tau.items()})
for k in range(len(times)):
    print(res["times"][k], round(res["ssa_mean"][k], 1), *[round(tau[dt].mean(0)[k], 1) for dt in tau])
print({k: v for k, v in res.items() if "max_z" in k})
(ROOT / "results/published/cgb_generated/nested_world_tau_vs_ssa_20261026.json").write_text(json.dumps(res, indent=1))
