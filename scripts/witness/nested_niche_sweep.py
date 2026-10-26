"""Niche declaration for the nested organism (principle fixed before any organism lives there):
the leanest rain rate at which (1) an ideal local forager (senses only its cell; runs at one cell per
tick; tumbles when food fell; stays while food >= f*) spends at least half of its time on food >= f*,
and (2) a blind walker reaches less than half of the local forager's share. f* = the maintenance
threshold of the body (nested_maintenance_threshold). World and ideal policies only."""
import copy, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW, cgb_nested_world_triton as NT
fstar = json.loads((ROOT / "results/published/cgb_generated/nested_maintenance_threshold_20261027.json").read_text())["f_star"]
base = NW.load("rain_torus_w0"); G = base["geometry"]["G"]
B, T = 32, 10000
RATES = tuple(float(x) for x in sys.argv[1:]) or (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
rows = []
for rate in RATES:
    tbl = copy.deepcopy(base); tbl["stimulus"]["rain_rate"] = rate
    W = NT.FusedWorld(tbl, B=B, seed=21); wea = NW.Weather(tbl, B=B, device="cuda", seed=21)
    buf, rep = W.captured()
    for _ in range(1500):
        buf.copy_(wea.tick()); rep()
    rng = np.random.default_rng(5)
    st = {k: dict(P=rng.uniform(0, G, (B, 2)), ang=rng.uniform(0, 6.283, B), prev=np.zeros(B), on=np.zeros(B)) for k in ("local", "blind")}
    rich = 0.0
    for t in range(T):
        buf.copy_(wea.tick()); rep()
        f = W.f.cpu().numpy()
        rich += (f >= fstar).mean()
        for k, a in st.items():
            P = a["P"]
            cur = f[np.arange(B), P[:, 0].astype(int) % G, P[:, 1].astype(int) % G].astype(float)
            a["on"] += cur >= fstar
            if k == "local":
                stay = cur >= fstar
                tumble = (cur < a["prev"]) & ~stay
            else:
                stay = np.zeros(B, bool)
                tumble = rng.random(B) < 0.1
            a["ang"] = np.where(tumble, rng.uniform(0, 6.283, B), a["ang"])
            P[:, 0] = np.where(~stay, (P[:, 0] + np.cos(a["ang"])) % G, P[:, 0])
            P[:, 1] = np.where(~stay, (P[:, 1] + np.sin(a["ang"])) % G, P[:, 1])
            a["prev"] = cur
    row = dict(rain_rate=rate, cells_rich=rich / T, local=float(np.median(st["local"]["on"] / T)),
               blind=float(np.median(st["blind"]["on"] / T)), food_mean=float(W.f.float().mean()))
    rows.append(row)
    print(json.dumps(row), flush=True)
ok = [r for r in rows if r["local"] >= 0.5 and r["blind"] < 0.5 * r["local"]]
out = dict(f_star=fstar, rows=rows, chosen=(ok[0]["rain_rate"] if ok else None))
print("chosen", out["chosen"])
(ROOT / "results/published/cgb_generated/nested_niche_sweep_fine_20261027.json").write_text(json.dumps(out, indent=1))
