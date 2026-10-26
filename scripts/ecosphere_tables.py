"""Tables for the eco-sphere page (world + organism structure + genome), from the same Python
modules the gate checks (cgb_nested_world, cgb_nested_organism)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_organism as NO  # noqa: E402


def tables(genome=None, world="rain_torus_w0", dt=0.05):
    w = NW.load(world)
    T = dict(world=dict(G=w["geometry"]["G"], V=w["table"]["V"], kappa_in=w["table"]["kappa_in"],
                        kappa_diff=w["table"]["kappa_diff"], dt=dt, stimulus=w["stimulus"], name=world,
                        B=w["table"]["B"], kappa_dis=w["table"]["kappa_dis"], foodE=w["table"]["foodE"]),
             organism=dict(UNITS=list(NO.UNITS), STATES=list(NO.STATES), SMAX=NO.SMAX, RULES=[list(r) for r in NO.RULES],
                           EXCH=[[p, a, b, bool(f), bool(g)] for (p, a, b, f, g) in NO.EXCH], FUEL=list(NO.FUEL),
                           GATE=list(NO.GATE), RUN=NO.RUN, M=NO.M, S=NO.S, E=NO.E, F=NO.F, Y=NO.Y, D=NO.D,
                           SELECTED_PHI=[list(x) for x in NO.SELECTED_PHI], PHI_S1=NO.PHI_S1, LOGK_LEAK=NO.LOGK_LEAK))
    if genome is not None:
        T["genome"] = {k: (list(map(float, v)) if hasattr(v, "__len__") else float(v)) for k, v in genome.items()}
    return T
