"""Build and audit all CGB declarations, recording signatures and premise dependencies.

This checks proof trust and import coverage. It does not certify the physical truth,
necessity, or consistency of arbitrary local hypotheses in a theorem statement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
EXCLUDED_MODULES = {"CGBAxioms.Audit", "CGBAxioms.AuditAll"}


def code_only(source: str) -> str:
    """Mask nested Lean comments and strings, preserving source line numbers."""
    out = list(source)
    pos, depth, quoted, line_comment = 0, 0, False, False
    while pos < len(source):
        pair = source[pos:pos + 2]
        char = source[pos]
        if line_comment:
            if char == "\n":
                line_comment = False
            else:
                out[pos] = " "
        elif depth:
            if pair in {"/-", "-/"}:
                depth += 1 if pair == "/-" else -1
                out[pos:pos + 2] = [" ", " "]
                pos += 1
            elif char != "\n":
                out[pos] = " "
        elif quoted:
            if char == "\\" and pos + 1 < len(source):
                out[pos:pos + 2] = [" ", " " if source[pos + 1] != "\n" else "\n"]
                pos += 1
            elif char == '"':
                quoted = False
                out[pos] = " "
            elif char != "\n":
                out[pos] = " "
        elif pair == "--":
            line_comment = True
            out[pos:pos + 2] = [" ", " "]
            pos += 1
        elif pair == "/-":
            depth = 1
            out[pos:pos + 2] = [" ", " "]
            pos += 1
        elif char == '"':
            quoted = True
            out[pos] = " "
        pos += 1
    if depth or quoted:
        raise ValueError("Unterminated Lean comment or string")
    return "".join(out)


def source_errors(source: str) -> list[str]:
    code = code_only(source)
    return [f"line {code.count(chr(10), 0, m.start()) + 1}: {m.group()}"
            for m in re.finditer(r"\b(?:axiom|sorry|admit|skipKernelTC)\b", code)]


def parse_inventory(output: str, expected_modules: set[str]) -> dict:
    records, summaries = [], []
    for line in output.splitlines():
        if line.startswith("CGB_AUDIT_RECORD "):
            records.append(json.loads(line.removeprefix("CGB_AUDIT_RECORD ")))
        elif line.startswith("CGB_AUDIT_SUMMARY "):
            summaries.append(json.loads(line.removeprefix("CGB_AUDIT_SUMMARY ")))
    if not records or len(summaries) != 1:
        raise ValueError("Missing inventory or unique audit summary")
    summary = summaries[0]
    if summary["declarations"] != len(records):
        raise ValueError("Declaration inventory is incomplete")
    if len({record["name"] for record in records}) != len(records):
        raise ValueError("Duplicate declaration records")
    if summary["theorems"] != sum(r["kind"] == "theorem" for r in records):
        raise ValueError("Theorem count mismatch")
    missing = expected_modules.difference(summary["modules"])
    if missing:
        raise ValueError(f"Unimported project modules: {sorted(missing)}")
    for record in records:
        bad = set(record["axioms"]).difference(ALLOWED_AXIOMS)
        if bad or record["kind"] == "axiom":
            raise ValueError(f"Untrusted declaration {record['name']}: {sorted(bad)}")
        if not record["signature"]:
            raise ValueError(f"Missing signature: {record['name']}")
    return {"summary": summary, "declarations": records}


def self_test() -> int:
    checks = 0
    for fixture in ["theorem bad : True := by sorry", "private axiom bad : False",
                    "theorem bad : True := by admit", "set_option debug.skipKernelTC true"]:
        if not source_errors(fixture):
            raise AssertionError(f"Forbidden source accepted: {fixture}")
        checks += 1
    if source_errors('/- axiom /- sorry -/ admit -/\n-- sorry\ndef text := "axiom sorry"'):
        raise AssertionError("Comments/strings incorrectly rejected")
    checks += 1
    record = {"name": "CGB.guard", "module": "CGBAxioms.Guard", "kind": "theorem",
              "signature": "True", "axioms": []}
    def output(r: dict, modules: list[str], count: int = 1) -> str:
        return "CGB_AUDIT_RECORD " + json.dumps(r) + "\nCGB_AUDIT_SUMMARY " + json.dumps(
            {"declarations": count, "theorems": 1, "modules": modules})
    parse_inventory(output(record, ["CGBAxioms.Guard"]), {"CGBAxioms.Guard"})
    checks += 1
    failures = ["", output(record, [], 1), output(record, ["CGBAxioms.Guard"], 2),
                output(dict(record, axioms=["sorryAx"]), ["CGBAxioms.Guard"])]
    for fixture in failures:
        try:
            parse_inventory(fixture, {"CGBAxioms.Guard"})
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid audit inventory accepted")
    return checks


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=PROJECT, text=True, encoding="utf-8",
                            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stdout}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / ".lake/audit_inventory.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-index", action="store_true",
                        help="check the generated theorem index instead of refreshing it")
    args = parser.parse_args()
    guards = self_test()
    if args.self_test:
        print(json.dumps({"audit_guard_checks": guards, "passed": True}))
        return
    source_files = sorted((PROJECT / "CGBAxioms").rglob("*.lean")) + [PROJECT / "CGBAxioms.lean"]
    hashes, declarations, expected = {}, {}, set()
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        violations = source_errors(source)
        if violations:
            raise ValueError(f"Forbidden source in {path}: {violations}")
        relative = path.relative_to(PROJECT).as_posix()
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        module = relative.removesuffix(".lean").replace("/", ".")
        if module.startswith("CGBAxioms.") and module not in EXCLUDED_MODULES:
            expected.add(module)
        code = code_only(source)
        names = re.findall(r"\b(?:theorem|lemma|def|abbrev|opaque)\s+([\w'.]+)", code)
        declarations[module] = names
    build = run(["lake", "build"])
    # Resolve Lake's library path once; do not retain a large Lake parent for every audit.
    lean_path = run(["lake", "env", sys.executable, "-c",
                     "import os; print(os.environ['LEAN_PATH'])"]).stdout.strip()
    os.environ["LEAN_PATH"] = lean_path
    audit_output = run(["lean", "CGBAxioms/AuditAll.lean"]).stdout
    inventory = parse_inventory(audit_output, expected)
    authored = []
    for record in inventory["declarations"]:
        record["source_declared"] = any(record["name"] == name or record["name"].endswith("." + name)
            for name in declarations.get(record["module"], []))
        if record["kind"] == "theorem" and record["source_declared"]:
            authored.append(record["name"])
    for module in sorted(expected):
        for name in declarations[module]:
            if not any(r["module"] == module and (r["name"] == name or r["name"].endswith("." + name))
                       for r in inventory["declarations"]):
                raise ValueError(f"Declared item missing from inventory: {module}.{name}")
    for relative, expected_hash in hashes.items():
        if hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Source changed during audit: {relative}")
    inventory.update({"source_hashes": hashes, "source_declared_theorems": authored,
                      "source_declared_theorem_count": len(authored), "audit_guard_checks": guards,
                      "proof_trust_passed": True, "module_coverage_passed": True,
                      "scope": "Trust, imports and explicit signatures; not physical validity or minimality."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".build.log").write_text(build.stdout, encoding="utf-8")
    args.output.with_suffix(".audit.log").write_text(audit_output, encoding="utf-8")
    index_command = [sys.executable, str(PROJECT.parent.parent / "scripts/build_proven_index.py"),
                     "--inventory", str(args.output.resolve())]
    if args.check_index:
        index_command.append("--check")
    print(run(index_command).stdout, end="")
    basis_command = [sys.executable, str(PROJECT.parent.parent / "scripts/build_theorem_basis.py"),
                     "--inventory", str(args.output.resolve())]
    if args.check_index:
        basis_command.append("--check")
    print(run(basis_command).stdout, end="")
    catalog_command = [sys.executable, str(PROJECT.parent.parent / "scripts/build_theorem_catalog.py"),
                       "--inventory", str(args.output.resolve())]
    if args.check_index:
        catalog_command.append("--check")
    print(run(catalog_command).stdout, end="")
    print(json.dumps({"passed": True, **inventory["summary"], "source_declared_theorems": len(authored),
                      "audit_guard_checks": guards, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
