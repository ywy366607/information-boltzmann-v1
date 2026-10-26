"""Niche declaration for the nested organism (principle fixed before any organism lives there):
the leanest rain at which (1) the ideal local forager (senses only its cell; runs at one cell per
tick; tumbles when the food is below its memory of recent food; stays while food >= f*) spends at least
half of its time on food >= f*, and (2) a blind walker reaches less than half of the local forager's
share. "Ideal" = best of the class: memory time constants TAUS, the best one counts (tau = 1 is the
first version, which compared with the last tick only and tumbled on decay noise).

Revision (2026-10-27, from mass action): intake is proportional to the food under the body
(`mass_action_down`), and the reserve (20 units, ~3000 ticks of starvation) smooths fluctuations of
a few hundred ticks, so maintenance needs the TIME-AVERAGED food under the forager >= f*, not a
share of time above f*. The criteria become: (1) local mean food >= f*, (2) blind mean food < half
the local mean. The time-share numbers are still reported. f* = the maintenance
threshold of the body (nested_maintenance_threshold). World and ideal policies only."""
import copy, json, math, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW, cgb_nested_world_triton as NT
fstar = json.loads((ROOT / "results/published/cgb_generated/nested_maintenance_threshold_20261027.json").read_text())["f_star"]
base = NW.load("rain_torus_w0"); G = base["geometry"]["G"]
B, T = 32, 10000
TAUS = (1.0, 3.0, 10.0, 30.0)
import os
SPEEDS = tuple(float(v) for v in os.environ.get("NICHE_SPEEDS", "1").split(","))   # cells per tick of the local forager
# arguments: rain rates, or rate@mu_rain pairs (the rain's size is part of the niche too)
ARGS = [a.split("@") for a in sys.argv[1:]] or [["0.05"], ["0.1"], ["0.2"], ["0.4"]]
OUT = ("nested_niche_sweep_fine_20261027.json" if all(len(a) == 1 for a in ARGS) else
       "nested_niche_sweep_2d_memory_20261027.json" if all(len(a) == 2 for a in ARGS) else "nested_niche_sweep_mean_20261027.json")
rows = []
for a in ARGS:
    rate = float(a[0]); mu_rain = float(a[1]) if len(a) > 1 else base["stimulus"]["mu_rain"]
    dmul = float(a[2]) if len(a) > 2 else 1.0          # diffusion: hop rate times dmul
    life = float(a[3]) if len(a) > 3 else 500.0        # dry food lifetime (ticks); the rain's fill rate is kept
    tbl = copy.deepcopy(base); tbl["stimulus"]["rain_rate"] = rate
    tbl["table"]["kappa_diff"] = base["table"]["kappa_diff"] * dmul
    tbl["table"]["kappa_in"] = base["table"]["kappa_in"] * 500.0 / life
    tbl["stimulus"]["mu_rain"] = mu_rain + 2.0 * math.log(life / 500.0)
    tbl["stimulus"]["mu_sink"] = base["stimulus"]["mu_sink"] - 2.0 * math.log(life / 500.0)   # waste leaves at the same rate
    W = NT.FusedWorld(tbl, B=B, seed=21); wea = NW.Weather(tbl, B=B, device="cuda", seed=21)
    buf, rep = W.captured()
    for _ in range(1500):
        buf.copy_(wea.tick()); rep()
    rng = np.random.default_rng(5)
    st = {k: dict(P=rng.uniform(0, G, (B, 2)), ang=rng.uniform(0, 6.283, B), prev=np.zeros(B), on=np.zeros(B), sum=np.zeros(B))
          for k in [("local", tau, v) for tau in TAUS for v in SPEEDS] + ["blind"]}
    rich = 0.0
    for t in range(T):
        buf.copy_(wea.tick()); rep()
        f = W.f.cpu().numpy()
        rich += (f >= fstar).mean()
        for k, a in st.items():
            P = a["P"]
            cur = f[np.arange(B), P[:, 0].astype(int) % G, P[:, 1].astype(int) % G].astype(float)
            a["on"] += cur >= fstar
            a["sum"] += cur
            if k != "blind":
                stay = cur >= fstar
                tumble = (cur < a["prev"]) & ~stay
                a["prev"] = a["prev"] + (cur - a["prev"]) / k[1]   # memory of recent food
            else:
                stay = np.zeros(B, bool)
                tumble = rng.random(B) < 0.1
            a["ang"] = np.where(tumble, rng.uniform(0, 6.283, B), a["ang"])
            v = 1.0 if k == "blind" else k[2]
            P[:, 0] = np.where(~stay, (P[:, 0] + v * np.cos(a["ang"])) % G, P[:, 0])
            P[:, 1] = np.where(~stay, (P[:, 1] + v * np.sin(a["ang"])) % G, P[:, 1])
            if k == "blind":
                a["prev"] = cur
    row = dict(rain_rate=rate, mu_rain=mu_rain, diff_mult=dmul, food_life=life, food_total=float(W.f.float().sum(dim=(1, 2)).mean()), cells_rich=rich / T, local_by_tau={f"{k[1]:g},{k[2]:g}": float(np.median(st[k]["on"] / T)) for k in st if k != "blind"},
               local_mean_by_tau={f"{k[1]:g},{k[2]:g}": float(np.median(st[k]["sum"] / T)) for k in st if k != "blind"},
               local_mean=max(float(np.median(st[k]["sum"] / T)) for k in st if k != "blind"),
               blind_mean=float(np.median(st["blind"]["sum"] / T)),
               local=max(float(np.median(st[k]["on"] / T)) for k in st if k != "blind"),
               blind=float(np.median(st["blind"]["on"] / T)), food_mean=float(W.f.float().mean()))
    rows.append(row)
    print(json.dumps(row), flush=True)
ok = sorted([r for r in rows if r["local_mean"] >= fstar and r["blind_mean"] < 0.5 * r["local_mean"]], key=lambda r: r["food_mean"])
out = dict(f_star=fstar, rows=rows, chosen=([ok[0]["rain_rate"], ok[0]["mu_rain"], ok[0]["diff_mult"], ok[0]["food_life"]] if ok else None),
           rule="feasible (local mean food >= f*), sensing useful (blind mean < local mean / 2), leanest (least standing food)")
print("chosen", out["chosen"])
(ROOT / "results/published/cgb_generated" / os.environ.get("NICHE_OUT", OUT)).write_text(json.dumps(out, indent=1))
