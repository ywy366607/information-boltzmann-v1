"""Build the eco-sphere page: template + JS core + tables (world, organism structure, genome).

Usage: python scripts/build_ecosphere.py <selection record .json> <out.html>
The genome is the record's validated best (`best.genome`)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW  # noqa: E402
from ecosphere_tables import tables  # noqa: E402


def main(record, out):
    rec = json.loads(Path(record).read_text(encoding="utf-8"))
    g = rec["best"]["genome"]
    T = tables(g)
    w = NW.load(rec.get("world", "rain_torus_w0"))
    T["world"]["B"], T["world"]["kappa_dis"] = w["table"]["B"], w["table"]["kappa_dis"]
    T["record"] = dict(path=Path(record).resolve().relative_to(ROOT).as_posix(), gen=rec["best"]["gen"],
                       val=rec["best"]["val"], val_mean=rec["best"]["val_mean"])
    d = ROOT / "showcase" / "ecosphere"
    html = (d / "ecosphere_template.html").read_text(encoding="utf-8")
    core = (d / "ecosphere_core.js").read_text(encoding="utf-8")
    html = html.replace("/*__CORE__*/", core).replace("/*__TABLES__*/", json.dumps(T, ensure_ascii=False))
    Path(out).write_text(html, encoding="utf-8")
    print("wrote", out, len(html), "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
