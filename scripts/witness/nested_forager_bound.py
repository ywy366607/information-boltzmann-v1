"""World bound for foraging in rain_torus_w0: how often does a cell hold food >= q, and how much
of its time can an oracle forager (sees the whole field, walks at most `speed` cells per tick
toward the nearest cell with food >= q, then stays while it lasts) spend on such cells? If even
the oracle rarely stands on rich food, the world, not the search, limits life."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW, cgb_nested_world_triton as NT

tbl = NW.load("rain_torus_w0"); G = tbl["geometry"]["G"]
B, T = 8, 6000
W = NT.FusedWorld(tbl, B=B, seed=11); wea = NW.Weather(tbl, B=B, device="cuda", seed=11)
buf, rep = W.captured()
for _ in range(1500):
    buf.copy_(wea.tick()); rep()
res = {}
qs = (10, 30, 60)
frac = {q: [] for q in qs}
on = {(q, s): np.zeros(B) for q in qs for s in (0.25, 1.0)}
# local foragers (sense only their own cell): run at 1 cell/tick; "local" tumbles when food fell
# since the last tick and stays while food >= q; "blind" tumbles at random (rate 0.1) and never stays
loc = {k: dict(P=np.random.default_rng(1).uniform(0, G, size=(B, 2)), ang=np.random.default_rng(2).uniform(0, 6.283, B),
               prev=np.zeros(B), on={q: np.zeros(B) for q in qs}) for k in ("local", "blind")}
lrng = np.random.default_rng(3)
pos = {(q, s): np.random.default_rng(0).integers(G, size=(B, 2)).astype(float) for q in qs for s in (0.25, 1.0)}
ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
for t in range(T):
    buf.copy_(wea.tick()); rep()
    f = W.f.cpu().numpy()
    for k, a in loc.items():
        P = a["P"]
        cur = f[np.arange(B), P[:, 0].astype(int) % G, P[:, 1].astype(int) % G].astype(float)
        for q in qs:
            a["on"][q] += cur >= q
        if k == "local":
            stay = cur >= 30
            tumble = (cur < a["prev"]) & ~stay
        else:
            stay = np.zeros(B, bool)
            tumble = lrng.random(B) < 0.1
        a["ang"] = np.where(tumble, lrng.uniform(0, 6.283, B), a["ang"])
        mv = ~stay
        P[:, 0] = np.where(mv, (P[:, 0] + np.cos(a["ang"])) % G, P[:, 0])
        P[:, 1] = np.where(mv, (P[:, 1] + np.sin(a["ang"])) % G, P[:, 1])
        a["prev"] = cur
    for q in qs:
        rich = f >= q
        frac[q].append(rich.mean())
        for s in (0.25, 1.0):
            P = pos[(q, s)]
            for b in range(B):
                x, y = int(P[b, 0]) % G, int(P[b, 1]) % G
                if rich[b, x, y]:
                    on[(q, s)][b] += 1
                    continue
                cand = np.argwhere(rich[b])
                if len(cand) == 0:
                    continue
                dx = (cand[:, 0] - P[b, 0] + G / 2) % G - G / 2
                dy = (cand[:, 1] - P[b, 1] + G / 2) % G - G / 2
                k = np.argmin(np.abs(dx) + np.abs(dy))
                d = np.hypot(dx[k], dy[k]) + 1e-9
                step = min(s, d)
                P[b, 0] = (P[b, 0] + step * dx[k] / d) % G
                P[b, 1] = (P[b, 1] + step * dy[k] / d) % G
for q in qs:
    res[f"cells_with_food_ge_{q}"] = float(np.mean(frac[q]))
    for s in (0.25, 1.0):
        res[f"oracle_speed{s}_time_on_food_ge_{q}"] = [float(x) for x in np.percentile(on[(q, s)] / T, [0, 50, 100])]
for k, a in loc.items():
    for q in qs:
        res[f"{k}_time_on_food_ge_{q}"] = [float(x) for x in np.percentile(a["on"][q] / T, [0, 50, 100])]
print(json.dumps(res, indent=1))
(ROOT / "results/published/cgb_generated/nested_forager_bound_20261027.json").write_text(json.dumps(res, indent=1))
