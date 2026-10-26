"""Write the organism file and architecture manifest of a selected nested organism, for the gate.

Usage: python scripts/admit_nested_organism.py <name> <selection record> <eval record>
Then: python scripts/cgb_gate.py <name>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_organism as NO  # noqa: E402


def main(name, record, evalrec):
    rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()  # noqa: E731
    rec = json.loads(Path(record).read_text(encoding="utf-8"))
    org = dict(name=name, status="candidate", world=rec.get("world", "rain_torus_w0"),
               structure=dict(pages=list(NO.PAGES), units=list(NO.UNITS), states=list(NO.STATES),
                              rules=[list(r) for r in NO.RULES], exchanges=[list(x) for x in NO.EXCH],
                              fuel=list(NO.FUEL), gate=list(NO.GATE), run=NO.RUN,
                              physics=dict(phi_S1=NO.PHI_S1, logk_leak=NO.LOGK_LEAK,
                                           phi_per_quantum=list(NO.PHI_PER_QUANTUM))),
               genome=rec["best"]["genome"], selection=dict(record=rel(record), gen=rec["best"]["gen"],
                                                           validation=rec["best"]["val"]))
    (ROOT / "information_boltzmann" / "organisms" / f"{name}.json").write_text(json.dumps(org, indent=1), encoding="utf-8")
    man = {
        "name": name,
        "role": "nested_organism",
        "implementation": (f"information_boltzmann/organisms/{name}.json (selected numbers) on the structure of "
                           "information_boltzmann/core/cgb_nested_organism.py, run by cgb_nested_organism_triton.py "
                           "(exact events, generated from the table) inside the world rain_torus_w0; the eco-sphere page "
                           "runs the same table in JavaScript (showcase/ecosphere/ecosphere_core.js)"),
        "lean_medium": "CGB.NestedWorld.lattice",
        "mechanisms": [
            {"id": "joint_medium", "kind": "structure",
             "what": "organism nested in the rain torus: inner pair rules, exchanges with the cell's pools, turning, "
                     "fuel-driven movement, dissolution; one Lean table with the world",
             "theorems": ["CGB.NestedWorld.axioms_torus", "CGB.NestedWorld.axioms"],
             "checks": ["nested_world_rates_match_lean:rain_torus_w0", "nested_organism_rates_match_lean:rain_torus_w0",
                        "nested_organism_kernel_matches:rain_torus_w0"]},
            {"id": "membrane", "kind": "structure",
             "what": "every exchange through the boundary goes through intact membrane units (conductance proportional "
                     "to them); without them the organism takes nothing in (death)",
             "theorems": ["CGB.NestedWorld.intake_is_exchange", "CGB.NestedWorld.dead_intake_is_physics"],
             "checks": ["nested_organism_rates_match_lean:rain_torus_w0"]},
            {"id": "leaks", "kind": "structure",
             "what": "every content-holding state of every page leaks one quantum to the waste pool through no membrane, "
                     "with bounded energies: no hold the organism can make permanent",
             "theorems": ["CGB.NestedWorld.leak_bound", "CGB.NestedWorld.partner_exit_blocked",
                          "CGB.NestedWorld.gated_exit_blocked"],
             "checks": ["nested_organism_rates_match_lean:rain_torus_w0"]},
            {"id": "dissolution", "kind": "structure",
             "what": "a body without intact membrane dissolves with all it holds; the only event that frees a place",
             "theorems": ["CGB.Generation.corpses_must_go", "CGB.NestedWorld.frees_only_by_dissolving",
                          "CGB.NestedWorld.dead_output_is_physics"],
             "checks": ["nested_world_rates_match_lean:rain_torus_w0"]},
            {"id": "selected_numbers", "kind": "selected",
             "what": "energies and prefactors in the physics ranges, found by selection on survival in fresh worlds "
                     "each generation, validated on seeds 500-515",
             "evidence": rel(record)},
            {"id": "physics_constants", "kind": "constant",
             "what": "membrane energy 2, leak prefactor 1e-5, per-quantum energy range [0, 8], structure counts; the "
                     "world's foodE, B, kappa_dis: declared physics (prereg section 5), not selected"},
        ],
        "life": {"world": "rain_torus_w0", "organism": f"information_boltzmann/organisms/{name}.json",
                 "record": rel(record), "eval": rel(evalrec)},
    }
    man["mechanisms"][0]["checks"].append(f"nested_table_is_record:{name}")
    (ROOT / "information_boltzmann" / "architectures" / f"{name}.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", name)


if __name__ == "__main__":
    main(*sys.argv[1:4])
