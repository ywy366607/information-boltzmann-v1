"""Can metabolism alone keep the membrane? Organisms held still in a cell whose food is a reservoir
at a fixed level (no foraging needed), 8000 ticks. Compares a selected genome with a hand-made
existence witness (reserve energy 3: between the membrane, 2, and rain-spot food, ~4.6; fast
uptake and repair). Diagnostic only: the witness never enters selection."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW, cgb_nested_organism as NO, cgb_nested_organism_triton as NX

tbl = NW.load("rain_torus_w0")
rec = json.loads(Path(sys.argv[1]).read_text())
sel = {k: np.asarray(v, float) for k, v in rec["best"]["genome"].items()}
wit = dict(phi=np.array([3.0, 6.0, 3.0, 3.0, 3.0, 1.0]), logk_rule=np.array([-6.0, -9.0, -9.0, -2.0]),
           logk_x=np.array([-3.0, -9.0, -1.0, -1.0]) - 4.7, log_flip=np.array(-9.0), log_move=np.array(-9.0))
class Stub: pass
out = {}
for name, g in (("selected", sel), ("witness", wit)):
    for food in (5, 30, 100):
        L = 256
        st = Stub(); st.B, st.G, st.V, st.dt = L, 3, tbl["table"]["V"], 0.05; st.foodE = tbl["table"]["foodE"]
        st.f = torch.full((L, 3, 3), food, dtype=torch.int32, device="cuda"); st.w = torch.zeros((L, 3, 3), dtype=torch.int32, device="cuda")
        o = NX.ExactOrganisms(st, {k: np.repeat(np.asarray(v)[None], L, axis=0) for k, v in g.items()}, seed=1, still=True)
        alive_at = []
        for t in range(8000):
            for _ in range(20):
                st.f.fill_(food); st.w.zero_(); o.leap()
            o.end_tick()
            if t % 2000 == 1999:
                alive_at.append(float((o.death < 0).float().mean()))
        s1 = o.n32[:, NO.S, 1].float().mean().item(); e1 = o.n32[:, NO.E, 1].float().mean().item()
        out[f"{name}_food{food}"] = dict(alive_fraction_at_2k_4k_6k_8k=alive_at, membrane_mean=s1, reserve_mean=e1)
        print(name, food, out[f"{name}_food{food}"], flush=True)
(ROOT / "results/published/cgb_generated/nested_metabolism_witness_20261027.json").write_text(json.dumps(out, indent=1))
