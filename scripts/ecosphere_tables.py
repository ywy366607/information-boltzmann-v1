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


def tables(genome=None, world="rain_torus_w0", dt=0.05, table=None, theta=None):
    """`table` None: the legacy organism, `genome` a decoded dict. `table` a name (e.g. the
    division grammar): its structure, the genome coder (bodies carry [0, 1] coordinates, a
    prefactor at 0 is an absent event), division and the mutation kernel; `theta` the founders'
    coordinates."""
    w = NW.load(world)
    T = dict(world=dict(G=w["geometry"]["G"], V=w["table"]["V"], kappa_in=w["table"]["kappa_in"],
                        kappa_diff=w["table"]["kappa_diff"], dt=dt, stimulus=w["stimulus"], name=world,
                        B=w["table"]["B"], kappa_dis=w["table"]["kappa_dis"], foodE=w["table"]["foodE"]))
    if table is None:
        T["organism"] = dict(UNITS=list(NO.UNITS), STATES=list(NO.STATES), SMAX=NO.SMAX, RULES=[list(r) for r in NO.RULES],
                             EXCH=[[p, a, b, bool(f), bool(g)] for (p, a, b, f, g) in NO.EXCH], FUEL=list(NO.FUEL),
                             GATE=list(NO.GATE), RUN=NO.RUN, M=NO.M, S=NO.S, E=NO.E, F=NO.F, Y=NO.Y, D=NO.D,
                             SELECTED_PHI=[list(x) for x in NO.SELECTED_PHI], PHI_S1=NO.PHI_S1, LOGK_LEAK=NO.LOGK_LEAK)
        if genome is not None:
            T["genome"] = {k: (list(map(float, v)) if hasattr(v, "__len__") else float(v)) for k, v in genome.items()}
        return T
    import cgb_nested_population as NP
    Tb = NO.table_of(table)
    rules = [[int(i), int(j), int(a1), int(a2), int(b1), int(b2), int(df), int(dw), bool(g)]
             for (i, j, a1, a2, b1, b2, df, dw, g) in Tb.RULES]
    T["organism"] = dict(name=Tb.name, UNITS=list(Tb.UNITS), STATES=list(Tb.STATES), SMAX=Tb.SMAX, RULES=rules,
                         EXCH=[[p, a, b, bool(f), bool(g)] for (p, a, b, f, g) in Tb.EXCH], FUEL=list(Tb.FUEL),
                         GATE=list(Tb.GATE), RUN=Tb.RUN, M=NO.M, S=NO.S, E=NO.E, F=NO.F, Y=NO.Y, D=NO.D,
                         SELECTED_PHI=[list(x) for x in Tb.SELECTED_PHI], PHI_S1=Tb.PHI_S1, LOGK_LEAK=Tb.LOGK_LEAK,
                         DIV=dict(BQ=w["table"]["B"] + Tb.UNITS[NO.S], NS=Tb.UNITS[NO.S]))
    assert Tb.UNITS[NO.E] == T["organism"]["DIV"]["BQ"], "the division page must hold exactly one newborn"
    C = NP.Coder(Tb)
    T["coder"] = dict(lo=[float(x) for x in C.lo], hi=[float(x) for x in C.hi], sizes=[[k, n] for k, n in C.sizes],
                      nphi=len(Tb.SELECTED_PHI), pmut=NP.PMUT, delta=NP.DELTA)
    if theta is not None:
        T["theta0"] = [float(x) for x in theta]
    return T
