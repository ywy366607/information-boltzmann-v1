"""Generate the Lean instance of an organism from its rule table (single source).

Usage: python scripts/gen_lean_organism.py ecoli_v1 EcoliV1
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from cgb_table_medium import load, lean_source  # noqa: E402


def lean_path(module):
    return ROOT / "lean" / "CGBAxioms" / "CGBAxioms" / f"GenerationOrganism{module}.lean"


def main():
    name, module = sys.argv[1], sys.argv[2]
    t = load(ROOT / "information_boltzmann" / "organisms" / f"{name}.json")
    lean_path(module).write_text(lean_source(t, module), encoding="utf-8")
    print(lean_path(module), t["sha256"])


if __name__ == "__main__":
    main()
