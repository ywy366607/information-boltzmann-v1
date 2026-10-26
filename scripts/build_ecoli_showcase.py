"""Assemble the E. coli showcase page from exported colony runs (`export_ecoli_viz.py`) and the
page text (`showcase/ecoli_meta.json`) into one self-contained HTML file.

usage: python scripts/build_ecoli_showcase.py <meta json> <out html> <viz json> [<viz json> ...]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta_path, out, vizs = sys.argv[1], sys.argv[2], sys.argv[3:]
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    worlds = []
    for p in vizs:
        w = json.loads(Path(p).read_text(encoding="utf-8"))
        w.update(meta["worlds"][str(w["world_seed"])])
        worlds.append(w)
    lns = [math.log(v) for w in worlds for v in w["field"]]
    data = dict(meta=meta, worlds=worlds, lnLo=min(lns), lnHi=max(lns))
    html = (ROOT / "showcase" / "ecoli_template.html").read_text(encoding="utf-8")
    # `</` cannot appear inside the JSON script block
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    Path(out).write_text(html, encoding="utf-8")
    print(out, f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
