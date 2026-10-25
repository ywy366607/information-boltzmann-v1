"""Exact finite-state interface validation, not a learning/capability study.

Enumerate the original body/world's supported transitions with natural sensing
or declared sensory loss. Only the evaluator holds the hidden world state.
Recorder candidates, action matching, all operation costs and physical marginals
are checked against independent Lean exports and a finite protocol oracle.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "CGBAxioms"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def export(name: str) -> dict:
    result = subprocess.run(["lake", "env", "lean", "reviews/" + name], cwd=LEAN,
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    lines = [json.loads(s) for s in result.stdout.splitlines() if s.startswith("{")]
    assert len(lines) == 1
    return lines[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core = ROOT / "information_boltzmann" / "core"
    sources = [Path(__file__).resolve(), core / "cgb_causal_record.py",
               core / "cgb_sensory_record.py"] + [LEAN / "CGBAxioms" / name for name in
               ("LanguageLifeKernel.lean", "CausalLearning.lean", "SensoryMaintenance.lean",
                "RenewableLearning.lean", "GraceRecovery.lean", "EventChannels.lean",
                "Interaction.lean")] + [LEAN / "reviews" / name for name in
                ("export_causal_record_20261025.lean", "export_sensory_record_20261025.lean")]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    body_table = export("export_sensory_record_20261025.lean")
    world_table = export("export_causal_record_20261025.lean")
    body = module("observer_body_reference", core / "cgb_sensory_record.py")
    causal = module("observer_causal_reference", core / "cgb_causal_record.py")
    weights, sensors = world_table["weights"], world_table["world_sensors"]
    supports = world_table["source_supports"]
    labels = world_table["signal_labels"]
    raw, decode = body_table["rows"], body_table["record_decode"]
    stimulus_indices = {s: i for i, s in enumerate(body_table["stimulus_labels"])}

    def screen(prior, a, o):
        oi = labels.index(o)
        return sum(1 << w for w in range(9) if prior & (1 << w) and supports[w][a][oi])

    def own(carrier):
        return (carrier.reserve, carrier.spent, carrier.status,
                carrier._pending, carrier._receipt)

    def oracle(before, source, stimulus, deliver):
        """Uncompressed protocol, using Lean supports for all inference."""
        reserve, spent, status, pending, receipt = before
        prior, a, live = source
        if status in ("awaiting", "waiting"):
            if reserve == 0:
                status, pending, receipt = "expired", None, None
            else:
                reserve, spent = reserve - 1, spent + 1
        if status == "awaiting":
            if stimulus == "missing":
                status, pending = "unobserved", None
            elif reserve == 0:
                status, pending = "expired", None
            else:
                reserve, spent = reserve - 1, spent + 1
                receipt, pending, status = (*pending, stimulus), None, "waiting"
        delivered = None
        if deliver and status == "waiting":
            if reserve == 0:
                receipt, status = None, "expired"
            else:
                reserve, spent = reserve - 1, spent + 1
                old, original_a, observed = receipt
                refined = screen(old, original_a, observed)
                if reserve > 0 and refined != old:
                    old, reserve, spent = refined, reserve - 1, spent + 1
                delivered = (old, original_a, observed)
                receipt, status = None, "consumed"
        if live and status not in ("awaiting", "waiting") and reserve > 0:
            reserve, spent = reserve - 1, spent + 1
            pending, status = (prior, a), "awaiting"
        return (reserve, spent, status, pending, receipt), delivered

    def key(observer, world, origin, initial):
        return (observer.record, world, own(observer.carrier), origin, initial)

    def carrier(reserve):
        return causal.DelayedSourceCarrier(reserve, capture_fee=1, holding_fee=1, read_fee=1)

    # Least fixed point: all sixteen body codes, all nine initial regimes,
    # four declared endowments, all sensing/delivery decisions and supported
    # stochastic branches. No time counter or unexecuted experience is added.
    pending = deque()
    seen = set()
    for record in range(16):
        for world in range(9):
            for initial in (0, 2, 5, 9):
                observer = body.SensoryRecordObserver(record, carrier(initial))
                item = key(observer, world, None, initial)
                seen.add(item)
                pending.append((observer, world, None, initial))
    boundaries = edges = delivered_count = expired_count = 0
    while pending:
        observer, world, origin, initial = pending.popleft()
        row = raw[decode[observer.record]]
        a = row["action"]
        live = observer.record != 0
        original_world = weights[world][a] if live else [0] * 8 + [2]
        for missing in (False, True):
            stimulus = "missing" if missing else labels[sensors[world]]
            si = stimulus_indices[stimulus]
            prior = 511 if missing else sum(1 << w for w in range(9)
                                            if sensors[w] == sensors[world])
            for deliver in (False, True):
                next_observer = copy.deepcopy(observer)
                before = own(observer.carrier)
                expect_state, expect_result = oracle(before, (prior, a, live), stimulus, deliver)
                action, alive, result = next_observer.advance(stimulus, deliver=deliver)
                assert own(next_observer.carrier) == expect_state
                assert result == expect_result
                assert action == a and alive == live
                assert decode[next_observer.record] == row["transitions"][si]
                assert next_observer.carrier.reserve + next_observer.carrier.spent == initial
                if result is not None:
                    assert origin is not None
                    true_source, original_action = origin
                    assert result[0] & (1 << true_source)
                    assert result[1] == original_action
                    delivered_count += 1
                phase = next_observer.carrier.status
                if phase == "awaiting":
                    next_origin = (world, a)
                elif phase == "waiting":
                    assert origin is not None
                    next_origin = origin
                else:
                    next_origin = None
                if phase == "expired":
                    expired_count += 1
                # Original body write budget plus the COMPLETE carrier charge.
                body_credit = 0 if not live else body.CREDIT[(observer.record - 1) // 3]
                new_credit = (0 if not next_observer.record else
                              body.CREDIT[(next_observer.record - 1) // 3])
                _, _, writes, topups, lost = row["ports"][si]
                charged = next_observer.carrier.spent - observer.carrier.spent
                assert new_credit + next_observer.carrier.reserve + writes + lost + charged == (
                    body_credit + observer.carrier.reserve + topups)
                # Exact one-step physical marginal; augmented metadata does not
                # remove or reweight a world branch when recording fails.
                actual_world = weights[world][action] if alive else [0] * 8 + [2]
                assert actual_world == original_world and sum(actual_world) == 2
                boundaries += 1
                for successor, mass in enumerate(original_world):
                    if mass:
                        edges += 1
                        item = key(next_observer, successor, next_origin, initial)
                        if item not in seen:
                            seen.add(item)
                            pending.append((copy.deepcopy(next_observer), successor, next_origin, initial))

    # End-to-end cost includes both context capture and outcome append, plus
    # the interval BEFORE the first next sensor. Legacy d+3 starts after pairing.
    budget_cases = 0
    for budget in range(14):
        for delay in range(7):
            item = carrier(budget)
            begun = item.begin(3, 0)
            item.tick()
            item.complete("miss")
            for _ in range(delay):
                item.tick()
            result = item.read()
            assert item.reserve + item.spent == budget
            assert (result == (2, 0, "miss")) == (budget >= delay + 5)
            if budget == delay + 4:
                assert result == (3, 0, "miss")
            if not begun:
                assert result is None
            budget_cases += 1

    # Negative controls catch the three interface errors at issue.
    assert screen(3, 0, "probe") == 0  # Bridging a missing sensor to an old feed event.
    assert screen(511, 0, "probe") & (1 << 3)  # Correct immediate source missB.
    zero_reserve_search = body.encode_record("search", 0)
    assert body.write_record(zero_reserve_search, "missing") == 0
    assert weights[4][2] != [0] * 8 + [2]  # Pre-live action vs post-update death.
    skipped = carrier(5)
    assert skipped.begin(3, 0) and skipped.complete("miss")
    assert skipped.read() == (2, 0, "miss") and skipped.spent == 4
    # Four credits would appear sufficient ONLY if the mandatory interval was omitted.
    assert budget_cases == 98 and delivered_count and expired_count
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in sources}
    report = {
        "status": "finite_observer_joint_interface_verified_not_capability_evidence",
        "source_hashes": hashes, "reachable_augmented_states": len(seen),
        "checked_boundary_updates": boundaries, "supported_joint_edges": edges,
        "delivered_source_retention_checks": delivered_count,
        "expired_boundary_checks": expired_count, "end_to_end_budget_cases": budget_cases,
        "scope": "normal_world_natural_sense_or_loss_all_delivery_choices_endowments_0_2_5_9",
        "reachability_complete_fixed_point": True,
        "pre_update_action_used": True, "hidden_world_input_to_observer": False,
        "source_provenance_from_supported_joint_transition": True,
        "full_pending_and_receipt_holding_costs_checked": True,
        "original_physical_marginal_preserved": True,
        "negative_controls_detected": ["stale_action_after_missing_feedback",
            "post_update_body_used_for_world_move", "omitted_pending_interval_cost"],
        "prepaired_receipt_cost_unit_tariffs": "d+3",
        "context_to_refinement_cost_unit_tariffs": "d+5",
        "cut_causal_attribution_certified": False, "tariffs_calibrated_as_joules": False,
        "learned_cognition_or_multistep_reward_attribution_verified": False,
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "reachable_augmented_states",
        "checked_boundary_updates", "supported_joint_edges", "delivered_source_retention_checks",
        "end_to_end_budget_cases")}))


if __name__ == "__main__":
    main()
