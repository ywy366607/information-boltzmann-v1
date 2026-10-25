"""Keep the Lean theory auditable as the model grows.

This is a maintenance guard, not a capability test.  It reads the inventory
written by ``lean/CGBAxioms/scripts/audit.py`` and fails before a theory change
quietly turns into an unbounded theorem/module expansion.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "lean" / "CGBAxioms" / ".lake" / "audit_inventory.json"
LEAN_ROOT = INVENTORY.parents[1]

# These are review budgets, deliberately above the current inventory but close
# enough to force consolidation before maintenance cost grows materially.
# Lead decision 2026-10-25: the contract/witness layer is frozen at its 10-25 size
# (FROZEN_LINES, every source except GENERATION); the headroom above the old caps
# (72 modules, 4000 declarations, 800 theorems, 15600 lines) is reserved for the
# generative chain axioms -> forced structure -> mechanism in GENERATION.
CAPS = {
    "modules": 73,
    "declarations": 4300,
    "source_declared_theorems": 900,
    "source_lines": 17600,
    "largest_source_file": 900,
}
GENERATION = "CGBAxioms/Generation.lean"
FROZEN_LINES = 15600

# Active entry points stay short; detailed evidence lives in existing reports.
FILE_CAPS = {
    "docs/information_boltzmann/CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md": 220,
    "information_boltzmann/core/cgb_language_kernel.py": 180,
    "scripts/validate_cgb_language_kernel.py": 140,
    "scripts/bench_algebraic_coefficients.py": 300,
    "information_boltzmann/core/cgb_prequential_bridge.py": 260,
    "information_boltzmann/core/cgb_low_resource_life.py": 300,
    "scripts/validate_cgb_contract.py": 360,
    "scripts/validate_cgb_prequential_bridge.py": 180,
    "scripts/measure_cgb_prequential_heldout.py": 230,
    "information_boltzmann/core/cgb_observation_stream.py": 300,
    "scripts/validate_cgb_observation_stream.py": 180,
    "information_boltzmann/core/cgb_sensory_port.py": 150,
    "scripts/validate_cgb_sensory_port.py": 150,
    "scripts/check_lean_complexity_budget.py": 125,
}


def budget_report(data: dict) -> dict:
    if not (data.get("module_coverage_passed", False)
            and data.get("proof_trust_passed", False)):
        raise SystemExit("Lean audit trust/coverage did not pass")
    sources = sorted((LEAN_ROOT / "CGBAxioms").rglob("*.lean")) + [LEAN_ROOT / "CGBAxioms.lean"]
    current_hashes = {p.relative_to(LEAN_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sources}
    if current_hashes != data.get("source_hashes"):
        raise SystemExit("Lean inventory is stale; build and audit current sources first")
    lines = [len(p.read_text(encoding="utf-8").splitlines()) for p in sources]
    values = {
        "modules": len(data["summary"]["modules"]),
        "declarations": data["summary"]["declarations"],
        "source_declared_theorems": data["source_declared_theorem_count"],
        "source_lines": sum(lines),
        "largest_source_file": max(lines),
    }
    active_lines = {name: len((ROOT / name).read_text(encoding="utf-8").splitlines())
                    for name in FILE_CAPS}
    violations = [
        f"{key}={values[key]} exceeds cap {CAPS[key]}"
        for key in CAPS
        if values[key] > CAPS[key]
    ]
    frozen = sum(n for p, n in zip(sources, lines) if p.relative_to(LEAN_ROOT).as_posix() != GENERATION)
    if frozen > FROZEN_LINES:
        violations.append(f"frozen contract/witness layer={frozen} lines exceeds {FROZEN_LINES}")
    violations.extend(f"{name}={active_lines[name]} lines exceeds cap {cap}"
                      for name, cap in FILE_CAPS.items() if active_lines[name] > cap)
    if violations:
        raise SystemExit("Lean complexity budget exceeded: " + "; ".join(violations))
    return {"passed": True, "source_hashes_current": True, "values": values,
            "caps": CAPS, "active_lines": active_lines, "active_line_caps": FILE_CAPS}


def main() -> int:
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    print(json.dumps(budget_report(data), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
