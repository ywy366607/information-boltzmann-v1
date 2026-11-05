"""Validate and assemble human/model-authored N-theorem use reviews.

This script never assigns semantic labels. Each review is written after reading
the declaration; only coverage, provenance, schema and counts are automated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "results/cgb_theorem_traversal_20261103/n_review"
OUT = ROOT / "results/published/cgb_theorem_n_review_20261103.jsonl"
SUMMARY = OUT.with_suffix(".summary.json")
ROLES = {"joint_witness", "local_witness", "counterexample", "measurement",
         "helper", "meta", "duplicate", "side"}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def collect(complete: bool) -> tuple[list[dict], dict]:
    manifest = json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / "lean/CGBAxioms/.lake/audit_inventory.json").read_text(encoding="utf-8"))
    known = {row["name"] for row in inventory["declarations"]}
    for path, digest in manifest["sources"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed since review preparation: {path}")
    merged, groups, seen = [], [], set()
    for group in manifest["groups"]:
        i = group["group"]
        original = WORK.parent / f"traverse_out_{i}.jsonl"
        if hashlib.sha256(original.read_bytes()).hexdigest() != group["original_sha256"]:
            raise ValueError(f"Original traversal changed: {original}")
        inputs = {row["name"]: row for row in read_rows(ROOT / group["input"])}
        path = WORK / f"review_{i}.jsonl"
        reviews = read_rows(path) if path.exists() else []
        local = set()
        for row in reviews:
            name = row["name"]
            if name not in inputs or name in seen or name in local:
                raise ValueError(f"Unexpected/duplicate review: {name}")
            src = inputs[name]
            assert row["signature_sha256"] == src["signature_sha256"], name
            assert row["reviewed"] is True, name
            assert row["decision"] in {"R", "M", "W", "N"}, name
            assert row["confidence"] in {"high", "medium", "low"}, name
            assert row["roles"] and set(row["roles"]) <= ROLES, name
            for key in ("premise_zh", "use_zh"):
                assert isinstance(row[key], str) and row[key].strip(), (name, key)
            for key, upper in (("family", 9), ("questions", 5)):
                assert isinstance(row[key], list), (name, key)
                assert all(type(n) is int and 1 <= n <= upper for n in row[key]), (name, key)
            assert isinstance(row["related"], list), name
            related = []
            for reference in row["related"]:
                matches = [reference] if reference in known else [
                    item for item in known if item.endswith("." + reference)]
                if len(matches) != 1:
                    raise ValueError(f"Unresolved related declaration: {name} -> {reference}")
                related.append(matches[0])
            merged.append({**row, "related": related, "group": i, "source": src["source"],
                           "line": src["line"], "source_sha256": src["source_sha256"],
                           "original_verdict": "N"})
            local.add(name)
        missing = sorted(set(inputs) - local)
        groups.append({"group": i, "expected": len(inputs), "reviewed": len(local),
                       "missing": missing})
        if complete and missing:
            raise ValueError(f"Group {i}: {len(missing)} unreviewed declarations")
        seen.update(local)
    assert sum(g["expected"] for g in groups) == 1403
    stats = {
        "schema_version": 1,
        "expected": 1403, "reviewed": len(merged),
        "pending": 1403 - len(merged), "groups": groups,
        "decisions": dict(Counter(row["decision"] for row in merged)),
        "roles_multi_label": dict(Counter(role for row in merged for role in set(row["roles"]))),
        "confidence": dict(Counter(row["confidence"] for row in merged)),
        "family_links": dict(Counter(str(n) for row in merged for n in set(row["family"]))),
        "question_links": dict(Counter(str(n) for row in merged for n in set(row["questions"]))),
        "scope": "N-use review only; labels are review proposals, not new body obligations",
        "original_classifications_changed": False, "lean_sources_changed": False,
        "source_count": len(manifest["sources"]),
        "source_sha256": manifest["sources"],
        "original_traversal_sha256": {str(g["group"]): g["original_sha256"]
                                      for g in manifest["groups"]},
    }
    return sorted(merged, key=lambda row: (row["group"], row["name"])), stats


def frontier_report() -> dict:
    """Read-only declaration/dependency coverage; never a semantic certificate."""
    core = json.loads((ROOT / "results/published/cgb_generator_causal_core_20261103.json").read_text(encoding="utf-8"))
    frontier = core["inside_out_frontier"]
    seed_keys = ["mechanical_review_declarations", "additional_parent_read_declarations",
                 "context_review_declarations"]
    # Additional source-reviewed rings need data, not a new hard-coded module list.
    seed_keys += sorted(k for k in frontier
                        if k.endswith("_parent_read_declarations") and k not in seed_keys)
    seed_groups = [frontier.get(k, []) for k in seed_keys]
    seeds = [name for group in seed_groups for name in group]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate frontier seed names across seed groups")
    inventory = json.loads((ROOT / "lean/CGBAxioms/.lake/audit_inventory.json").read_text(encoding="utf-8"))
    declarations = {row["name"]: row for row in inventory["declarations"]}
    for source, expected in inventory["source_hashes"].items():
        path = ROOT / "lean/CGBAxioms" / source
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Audit dependency inventory is stale: {source}")
    missing_seeds = sorted(set(seeds) - declarations.keys())
    if missing_seeds:
        raise ValueError(f"Unresolved frontier seeds: {missing_seeds}")
    labels, traversal_names = {}, set()
    for i in range(6):
        for row in read_rows(ROOT / f"results/cgb_theorem_traversal_20261103/traverse_out_{i}.jsonl"):
            if row["name"] in traversal_names:
                raise ValueError(f"Duplicate traversal declaration: {row['name']}")
            traversal_names.add(row["name"])
            labels[row["name"]] = row.get("verdict")
    untraversed_seeds = sorted(set(seeds) - traversal_names)
    if untraversed_seeds:
        raise ValueError(f"Frontier seeds absent from traverse_out_0..5: {untraversed_seeds}")
    n_reviews = {row["name"]: row for row in read_rows(OUT) if row.get("original_verdict") == "N"}
    for name, row in n_reviews.items():
        labels[name] = row["decision"]
    closure, unresolved_dependencies, todo = set(), set(), list(seeds)
    while todo:
        name = todo.pop()
        if name in closure:
            continue
        if name not in declarations:
            unresolved_dependencies.add(name)
            continue
        closure.add(name)
        todo.extend(declarations[name].get("project_dependencies", []))
    theorems = {n for n in closure if declarations[n].get("source_declared")
                and declarations[n].get("kind") in {"theorem", "lemma"}}
    definitions = {n for n in closure if declarations[n].get("kind") == "definition"}
    related, semantic_edges = set(), []
    for name, row in n_reviews.items():
        refs = set(row.get("related", []))
        for ref in refs:
            if name in closure or ref in closure:
                semantic_edges.append([name, ref])
                related.update({name, ref} - closure)
    role_counts = Counter(role for name, row in n_reviews.items()
                          if name in theorems
                          for role in set(row.get("roles", [])))
    return {
        "scope": "mechanical project-dependency closure only; not semantic coverage or an admission certificate",
        "dependency_closure_complete": not unresolved_dependencies,
        "structural_support_is_premise_satisfaction": False,
        "seed_counts": {"total": len(seeds), "unique": len(set(seeds)), "present": len(set(seeds) & declarations.keys()),
                        "traversal_present": len(set(seeds) & traversal_names), "groups": [len(g) for g in seed_groups],
                        "group_names": seed_keys, "names": seeds},
        "dependency_theorem_counts": {"total": len(theorems), "by_label": dict(sorted(Counter(labels.get(n, "unknown") for n in theorems).items())),
                                      "pending_or_unknown": sorted(n for n in theorems if labels.get(n) not in {"R", "M", "W", "N"})},
        "dependency_theorem_names": sorted(theorems),
        "project_dependency_edges": sorted([name, dep] for name in closure
            for dep in declarations[name].get("project_dependencies", [])),
        "support_declaration_count": len(closure), "definition_count": len(definitions),
        "unresolved_project_dependency_names": sorted(unresolved_dependencies),
        "N_use_role_counts": dict(sorted(role_counts.items())),
        "N_use_role_scope": "only reviewed original-N declarations inside the dependency theorem closure",
        "semantic_neighbor_names": sorted(related),
        "review_related_edges": sorted(semantic_edges),
        "semantic_neighbors_are_proof_edges": False,
        "input_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [ROOT / "lean/CGBAxioms/.lake/audit_inventory.json", OUT,
                                      ROOT / "results/published/cgb_generator_causal_core_20261103.json",
                                      *[ROOT / f"results/cgb_theorem_traversal_20261103/traverse_out_{i}.jsonl"
                                        for i in range(6)]]},
        "pending_unknown": {"unreviewed_original_N_in_closure": sorted(n for n in theorems if n in traversal_names and labels.get(n) == "N" and n not in n_reviews),
                            "closure_names_without_traversal_label": sorted(n for n in theorems if n not in labels)},
        "satisfied": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check published artifacts against reviewed rows")
    parser.add_argument("--name", help="Show matching reviewed declarations")
    parser.add_argument("--role", choices=sorted(ROLES))
    parser.add_argument("--frontier", action="store_true", help="Print read-only dependency frontier report")
    parser.add_argument("--frontier-out", type=Path, help="Optionally save --frontier JSON report")
    args = parser.parse_args()
    if args.frontier:
        if args.progress or args.build or args.check or args.name or args.role:
            parser.error("--frontier is a separate report; do not combine it with review modes")
        report = frontier_report()
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.frontier_out:
            target = args.frontier_out if args.frontier_out.is_absolute() else ROOT / args.frontier_out
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return
    if args.frontier_out:
        parser.error("--frontier-out requires --frontier")
    if args.progress and (args.build or args.check):
        parser.error("Partial progress cannot be published or accepted as complete")
    rows, stats = collect(complete=not args.progress)
    output = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    summary = json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    if args.build:
        OUT.write_text(output, encoding="utf-8")
        SUMMARY.write_text(summary, encoding="utf-8")
    if args.check:
        for path, expected in ((OUT, output), (SUMMARY, summary)):
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"Published review missing or stale: {path}")
    if args.name or args.role:
        selected = [row for row in rows if (not args.name or args.name in row["name"])
                    and (not args.role or args.role in row["roles"])]
        print(json.dumps(selected, ensure_ascii=False, indent=2))
    else:
        compact = {k: v for k, v in stats.items()
                   if k not in ("groups", "source_sha256", "original_traversal_sha256")}
        compact["groups"] = [{k: v for k, v in g.items() if k != "missing"}
                             for g in stats["groups"]]
        print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
