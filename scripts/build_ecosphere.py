"""Build the eco-sphere page: template + JS core + tables (world, organism structure, genome).

Usage: python scripts/build_ecosphere.py <selection record .json> <out.html> [arms eval .json|-] [persist eval .json]
A legacy record: the genome is the validated best (`best.genome`). A population record (with
`table`, e.g. the division grammar): the founders carry the validated best's [0, 1] coordinates
(`best.theta`), bodies divide and mutate, and the world has the record's SL body slots. An
optional held-out evaluation (scripts/eval_population_species.py, arms mode and --persist) is shown
as is."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from ecosphere_tables import tables  # noqa: E402


def main(record, out, evaluation=None, persist=None):
    rec = json.loads(Path(record).read_text(encoding="utf-8"))
    world = rec.get("world", "rain_torus_w0")
    if rec.get("table"):
        T = tables(world=world, table=rec["table"], theta=rec["best"]["theta"])
        T["world"]["SL"] = int(rec["args"]["SL"])
    else:
        T = tables(rec["best"]["genome"], world=world)
    T["record"] = dict(path=Path(record).resolve().relative_to(ROOT).as_posix(), gen=rec["best"]["gen"],
                       val=rec["best"]["val"], val_mean=rec["best"]["val_mean"], table=rec.get("table"),
                       ticks=rec.get("args", {}).get("ticks"),
                       val_living=rec["validation"][-1].get("living") if rec.get("validation") else None)
    if persist:
        ev = json.loads(Path(persist).read_text(encoding="utf-8"))
        T["persist"] = dict(path=Path(persist).resolve().relative_to(ROOT).as_posix(), ticks=ev["ticks"], pmut=ev["pmut"],
                            worlds=ev["held_out_seeds"], extinct=ev["extinct"], n=len(ev["living_per_world"]),
                            living=ev["living"], last=ev["last"])
    if evaluation and evaluation != "-":
        ev = json.loads(Path(evaluation).read_text(encoding="utf-8"))
        T["evaluation"] = dict(path=Path(evaluation).resolve().relative_to(ROOT).as_posix(), ticks=ev["ticks"],
                               worlds=ev["held_out_seeds"], arms={k: dict(living=v["living"], extinct=v["extinct"])
                                                                 for k, v in ev["arms"].items()})
    d = ROOT / "showcase" / "ecosphere"
    html = (d / "ecosphere_template.html").read_text(encoding="utf-8")
    core = (d / "ecosphere_core.js").read_text(encoding="utf-8")
    html = html.replace("/*__CORE__*/", core).replace("/*__TABLES__*/", json.dumps(T, ensure_ascii=False))
    Path(out).write_text(html, encoding="utf-8")
    print("wrote", out, len(html), "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:5])
