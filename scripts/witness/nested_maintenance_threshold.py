"""Maintenance threshold f* of the nested body under the declared physics: the lowest constant
food level at which the existence witness (reserve energy 3, fast uptake and repair, still) keeps
>= 90% alive over 8000 ticks. Computed from physics only, before any world niche is chosen."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW, cgb_nested_organism as NO, cgb_nested_organism_triton as NX
tbl = NW.load("rain_torus_w0")
wit = dict(phi=np.array([3.0, 6.0, 3.0, 3.0, 3.0, 1.0]), logk_rule=np.array([-6.0, -9.0, -9.0, -2.0]),
           logk_x=np.array([-3.0, -9.0, -1.0, -1.0]) - 4.7, log_flip=np.array(-9.0), log_move=np.array(-9.0))
class Stub: pass
foods = [4, 6, 8, 10, 12, 15, 20]
L = 256 * len(foods)
st = Stub(); st.B, st.G, st.V, st.dt = L, 3, tbl["table"]["V"], 0.05; st.foodE = tbl["table"]["foodE"]
level = torch.tensor(np.repeat(foods, 256), dtype=torch.int32, device="cuda")[:, None, None].expand(L, 3, 3).contiguous()
st.f = level.clone(); st.w = torch.zeros((L, 3, 3), dtype=torch.int32, device="cuda")
o = NX.ExactOrganisms(st, {k: np.repeat(np.asarray(v)[None], L, axis=0) for k, v in wit.items()}, seed=4, still=True)
for t in range(8000):
    for _ in range(20):
        st.f.copy_(level); st.w.zero_(); o.leap()
    o.end_tick()
alive = (o.death < 0).float().view(len(foods), 256).mean(1).cpu().numpy()
res = {str(f): float(a) for f, a in zip(foods, alive)}
fstar = next((f for f, a in zip(foods, alive) if a >= 0.9), None)
out = dict(alive_fraction_8000=res, f_star=fstar)
print(json.dumps(out))
(ROOT / "results/published/cgb_generated/nested_maintenance_threshold_20261027.json").write_text(json.dumps(out, indent=1))
