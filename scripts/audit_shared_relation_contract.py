"""Audit the running relation contract using logs, source AST and read-only SQL.

No torch import, checkpoint deserialization, world call or training mutation.
Structural observations are separate from measured prediction performance.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def function_source(root, filename, name):
    path = root / filename
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)
    calls = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]
    return dict(file=filename, line=node.lineno, calls=calls,
                source=ast.get_source_segment(text, node),
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prediction_summary(rows):
    scored = [r for r in rows if r.get("self_action_observer", {}).get("available")]
    result = dict(events=len(rows), scored=len(scored))
    for key in ("relation_mse", "legacy_relation_mse", "original_mse",
                "relation_log_score", "legacy_relation_log_score", "predicted_variance"):
        result[key] = mean(r["self_action_observer"]["options"][key][
            r["self_action_observer"]["options"]["chosen"]] for r in scored)
    result["relation_wins"] = sum(r["self_action_observer"]["options"]["relation_wins"]
                                  for r in scored)
    result["original_wins"] = sum(r["self_action_observer"]["options"]["original_wins"]
                                  for r in scored)
    result["comparisons"] = sum(len(r["self_action_observer"]["options"][
        "different_alternatives"]) for r in scored)
    return result


def audit(root, run):
    metrics = run / "learning/body/metrics.jsonl"
    raw = metrics.read_bytes()
    # Ignore only an unfinished final write. A malformed complete line is an error.
    lines = raw.splitlines(keepends=True)
    partial_tail = bool(lines and not lines[-1].endswith(b"\n"))
    rows = [json.loads(line) for line in (lines[:-1] if partial_tail else lines)
            if line.strip()]
    if not rows:
        raise ValueError("No complete real event records")
    config=json.loads((run/'learning/body/config.json').read_text(encoding='utf-8'))
    joint=config.get('joint_relation_update',{}).get('schema')==1
    base = "scripts/ib_local/"
    functions = {label: function_source(root, base + filename, name) for label, filename, name in (
        ("organization_features", "generative_relations.py", "dataset"),
        ("organization_evidence", "generative_relations.py", "marginal"),
        ("organization", "generative_relations.py", "organize"),
        ("prediction_features", "causal_relation_program.py", "relation_features"),
        ("revision", "causal_relation_program.py", "revise"),
        ("composition", "generative_relations.py", "execute_programs"),
        ("unknown_commit", "shared_unknown_bindings.py", "commit"),
    )}
    dataset = functions["organization_features"]
    legacy_organization = not joint
    basis_alive = [all(v > 0 for v in r["shared_relation_basis"][
        "parameter_update_norms"].values()) for r in rows]
    failures = [dict(cycle=r["cycle"], revision=r["shared_unknown"]["revision"])
                for r in rows if not r["shared_unknown"]["revision"]["committed"]]
    counters = [r["generative_relation"]["reorganizations"] for r in rows]
    increases = sum(b[1] > a[1] for a, b in zip(counters, counters[1:]))
    decreases = sum(b[1] < a[1] for a, b in zip(counters, counters[1:]))
    db_path = next((run / "learning/body").glob("*.sqlite"))
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as db:
        db.execute("BEGIN")
        database = dict(
            tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")],
            unknown_versions=dict(db.execute("SELECT version,COUNT(*) FROM unknown_bindings GROUP BY version")),
            support_counts=list(db.execute("SELECT COUNT(*),COUNT(DISTINCT unknown_id),"
                                           "COUNT(DISTINCT episode_id) FROM unknown_evidence").fetchone()),
            unknown_evidence_foreign_keys=list(db.execute("PRAGMA foreign_key_list(unknown_evidence)")),
            question_payload_fields=list(json.loads(db.execute(
                "SELECT payload FROM unknown_bindings LIMIT 1").fetchone()[0])),
        )
        if 'relation_variables' in database['tables']:
            database['variable_uses']=db.execute('SELECT COUNT(*) FROM relation_variables').fetchone()[0]
            database['variables_used_by_multiple_relations']=db.execute('SELECT COUNT(*) FROM '
                '(SELECT unknown_id FROM relation_variables GROUP BY unknown_id HAVING COUNT(*)>1)').fetchone()[0]
    report = dict(
        schema=1, generated_at=datetime.now(timezone.utc).isoformat(),
        scope="read-only whole-contract audit; no optimizer, checkpoint load or world calls",
        run=str(run.resolve()), metrics_sha256=hashlib.sha256(raw).hexdigest(),
        ignored_unfinished_tail=partial_tail,
        status=json.loads((run / "learning/status.json").read_text(encoding="utf-8")),
        first_cycle=rows[0]["cycle"], last_cycle=rows[-1]["cycle"],
        actual_updates=sum(r["updates"] for r in rows),
        prediction=prediction_summary(rows),
        first_32=prediction_summary(rows[:32]), last_32=prediction_summary(rows[-32:]),
        choices=dict(Counter(r["choice"] for r in rows)),
        question_roles=dict(Counter(r["shared_unknown"]["role"] for r in rows)),
        old_questions_recalled=sum(r["shared_unknown"]["recalled"] for r in rows),
        basis_all_groups_updated_events=sum(basis_alive),
        basis_group_zero_updates=[dict(cycle=r["cycle"], groups=[k for k, v in
            r["shared_relation_basis"]["parameter_update_norms"].items() if v == 0])
            for r in rows if not all(v > 0 for v in r["shared_relation_basis"][
                "parameter_update_norms"].values())],
        eight_old_facts_replayed_events=sum(r["causal_program_replay"] == 8 for r in rows),
        both_real_commits=sum(r["causal_program"]["evidence_committed"] and
                             r["shared_unknown"]["revision"]["committed"] for r in rows),
        rejected_commits=failures,
        maximum_frozen_score_reproduction_error=max(r["self_action_observer"][
            "actual_score_reproduction_error"] for r in rows),
        question_initial_byte_tv=mean(r["question_context"]["decision_byte_probability_tv"] for r in rows),
        four_draft_choice_tv=mean(.5 * sum(abs(p - .25) for p in r["joint_language_search"][
            "distribution"]) for r in rows if r.get("joint_language_search")),
        wood_policy_tv=mean(r["wood_policy_tv"] for r in rows),
        fixed_validation=[dict(cycle=r["cycle"], **r["fixed_validation"])
                          for r in rows if "fixed_validation" in r],
        initial_validation=json.loads((run / "learning/body/initial_validation.json").read_text(encoding="utf-8")),
        joint_relation_update=config.get('joint_relation_update'),
        organization=dict(legacy_features=legacy_organization,
            unit_noise_prior=not joint,
            hot_slot_counters_first=counters[0], hot_slot_counters_last=counters[-1],
            transitions_with_share_counter_increase=increases,
            transitions_with_share_counter_decrease=decreases,
            counter_scope="resident slots; hydration/eviction changes sums; not lifetime event counts"),
        database=database, source=functions,
        major_contracts=[
            dict(contract="prediction, evidence retention and relation organization use the same mechanism",
                 status="connected" if joint else "incomplete",
                 finding="explicit config selects shared coordinates, prior and scoped U" if joint else
                         "organization uses legacy normalized role features and unit prior"),
            dict(contract="one unknown can be used as an argument of several persisted relations",
                 status="connected" if joint and database.get('variables_used_by_multiple_relations',0)>0 else "partial",
                 implemented="versioned bounded relation uses" if joint else "persistent question identity and bounded factual supports",
                 boundary="conditional mechanism questions; not an arbitrary algebraic proof graph"),
            dict(contract="finite reconciliation of legal overlaps in the active relation graph",
                 status="connected" if joint else "partial",
                 implemented="finite precision-weighted boundary solve and re-evaluated continuation" if joint else
                             "typed execution, two path mixture and real-label supervision",
                 boundary="two legal program forms, no general H1 solver"),
            dict(contract="whole-loop useful learning and action improvement",
                 status="not_accepted", finding="prospective average prediction improves; action discrimination does not improve; matched training control unfinished"),
        ],
        interpretation="Paired frozen forecasts share actual labels. This is not a randomized training-arm comparison or a world counterfactual intervention.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("results/cgb_shared_relation_basis_20261114"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_root if args.run_root.is_absolute() else root / args.run_root
    output = args.output or run / "design_contract_audit.json"
    report = audit(root, run)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(dict(report=str(output), cycle=report["last_cycle"],
                         events=report["prediction"]["events"],
                         legacy_organization=report["organization"]["legacy_features"])))


if __name__ == "__main__":
    main()
