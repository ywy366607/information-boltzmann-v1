"""Exhaustive semantics/interface audit of an existing finite world model.

No optimizer or capability study is run. Lean certifies that the exported
integer weights equal the registered live-body world probabilities times two.
"""

from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations, product
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "CGBAxioms"
EXPORT = LEAN / "reviews" / "export_causal_record_20261025.lean"
KERNEL = ROOT / "information_boltzmann" / "core" / "cgb_causal_record.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refine(weights: list, sensors: list[int]) -> list[int]:
    """Weighted probabilistic partition refinement, all actuator settings."""
    labels = sensors[:]
    counts = [len(set(labels))]
    while True:
        groups = sorted(set(labels))
        classes: dict[tuple, int] = {}
        updated = []
        for world, rows in enumerate(weights):
            key = (labels[world], tuple(tuple(
                sum(row[v] for v in range(9) if labels[v] == group)
                for group in groups) for row in rows))
            updated.append(classes.setdefault(key, len(classes)))
        if len(classes) == counts[-1]:
            return counts
        counts.append(len(classes))
        labels = updated


def word_weight(weights: list, sensors: list[int], world: int,
                actions: tuple[int, ...], outcomes: tuple[int, ...]) -> int:
    """Numerator of an actual observation word, denominator 2**len(actions)."""
    law = [int(v == world) for v in range(9)]
    for action, outcome in zip(actions, outcomes):
        law = [sum(law[w] * weights[w][action][v] for w in range(9))
               if sensors[v] == outcome else 0 for v in range(9)]
    return sum(law)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [Path(__file__).resolve(), KERNEL, EXPORT,
               ROOT / "information_boltzmann" / "core" / "cgb_sensory_record.py",
               LEAN / "CGBAxioms" / "LanguageLifeKernel.lean",
               LEAN / "CGBAxioms" / "CausalLearning.lean",
               LEAN / "CGBAxioms" / "GraceRecovery.lean",
               LEAN / "CGBAxioms" / "RenewableLearning.lean",
               LEAN / "CGBAxioms" / "SensoryMaintenance.lean"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    result = subprocess.run(
        ["lake", "env", "lean", str(EXPORT.relative_to(LEAN))], cwd=LEAN,
        capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    payloads = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(payloads) == 1, result.stdout[-2000:]
    table = payloads[0]
    assert table["format"] == "CGB.CausalRecord.v1"
    spec = importlib.util.spec_from_file_location("cgb_causal_record", KERNEL)
    assert spec is not None and spec.loader is not None
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel
    spec.loader.exec_module(kernel)
    assert list(kernel.WORLD_NAMES) == table["world_labels"]
    assert list(kernel.SIGNALS) == table["signal_labels"]
    sensors, weights = table["world_sensors"], table["weights"]
    assert [kernel.SIGNALS.index(o) for o in kernel.WORLD_SENSORS] == sensors
    for w, world in enumerate(kernel.WORLD_NAMES):
        for a in range(3):
            assert list(kernel.live_world_weights(world, a)) == weights[w][a]
            assert sum(weights[w][a]) == 2

    transitions = 0
    for prior in range(512):
        for a in range(3):
            for o, sensor in enumerate(kernel.SIGNALS):
                got = kernel.update_live_support(prior, a, sensor)
                assert got == table["updates"][prior][a][o]
                # Independent relational support oracle, not the Python branch rule.
                oracle = sum(1 << v for v in range(9) if sensors[v] == o and any(
                    prior & (1 << w) and weights[w][a][v] > 0 for w in range(9)))
                assert got == oracle
                transitions += 1

    # The only data entering this recursion are action and actual next sensor.
    initial_masks = {kernel.observe_support(511, o) for o in kernel.SIGNALS} - {0}
    seen = set(initial_masks)
    pending = deque(sorted(seen))
    while pending:
        prior = pending.popleft()
        for a, sensor in product(range(3), kernel.SIGNALS):
            successor = kernel.update_live_support(prior, a, sensor)
            if successor and successor not in seen:
                seen.add(successor)
                pending.append(successor)

    witnesses = []
    for left, right in combinations(range(9), 2):
        found = None
        if sensors[left] != sensors[right]:
            found = {"actions": [], "observations": [], "current_sensor_differs": True}
        for horizon in range(1, 3):
            if found is not None:
                break
            for actions in product(range(3), repeat=horizon):
                if found is not None:
                    break
                for outcomes in product(range(7), repeat=horizon):
                    lp = word_weight(weights, sensors, left, actions, outcomes)
                    rp = word_weight(weights, sensors, right, actions, outcomes)
                    if lp != rp:
                        found = {"actions": list(actions), "observations": list(outcomes),
                                 "numerators": [lp, rp], "denominator": 2 ** horizon}
                        break
        assert found is not None, (left, right)
        witnesses.append({"worlds": [left, right], **found})

    assert refine(weights, sensors) == [6, 8, 9]
    assert len(seen) == 12
    prior = (1 << 0) | (1 << 1)
    left = kernel.update_live_support(prior, 0, "miss")
    right = kernel.update_live_support(prior, 1, "miss")
    assert left == 1 << 3 and right == 1 << 2
    # Equal support does not determine probabilities, hence not expected value.
    response = [Fraction(sum(weights[w][0][v] for v in range(9) if sensors[v] == 3), 2)
                for w in range(2)]
    probabilities = [sum(p * Fraction(r) for p, r in zip(prior_law, response))
                     for prior_law in ((Fraction(1, 4), Fraction(3, 4)),
                                       (Fraction(3, 4), Fraction(1, 4)))]
    assert probabilities == [Fraction(1, 4), Fraction(3, 4)]
    for w, a, o in product(range(9), range(3), range(7)):
        supported = bool(kernel.source_screen(1 << w, a, kernel.SIGNALS[o]))
        assert supported == table["source_supports"][w][a][o]
    for prior, reserve, a, o, overhead, expected, remaining, fee in table["paid_cases"]:
        got = kernel.paid_source_update(prior, reserve, a, kernel.SIGNALS[o], overhead)
        assert got == (expected, remaining, fee, overhead <= reserve)
        assert remaining + fee == reserve
        assert expected & ~prior == 0
        if overhead > reserve:
            assert got == (prior, reserve, 0, False)

    # A moving-world support cannot simply be frozen when funding ends.
    frozen = 1 << kernel.WORLD_NAMES.index("missA")
    moved = kernel.update_live_support(frozen, 0, "probe")
    assert moved == 1 << kernel.WORLD_NAMES.index("probeA") and not frozen & moved
    # Delayed source inference remains about the receipt's origin, not `moved`.
    assert kernel.source_screen(frozen, 0, "probe") == frozen

    def carrier(reserve: int):
        return kernel.DelayedSourceCarrier(reserve, capture_fee=1, holding_fee=1, read_fee=1)

    first, second = carrier(8), carrier(8)
    assert first.capture(3, 0, "miss") and second.capture(3, 1, "miss")
    assert weights[1][0][3] > 0 and weights[0][1][2] > 0  # Actual paired events.
    assert not first.capture(511, 2, "answerA")  # Busy storage never overwrites.
    for _ in range(2):
        assert first.tick() and second.tick()
    second_result = second.read()  # Reverse delivery order, self-contained pairs.
    assert first.tick()
    first_result = first.read()
    assert second_result == (1, 1, "miss") and first_result == (2, 0, "miss")
    assert first.spent == 6 and second.spent == 5
    assert first.reserve + first.spent == second.reserve + second.spent == 8
    assert first.read() is None and second.read() is None  # Consume once.
    starved = carrier(2)
    assert starved.capture(3, 0, "miss") and starved.tick()
    assert not starved.tick() and starved.status == "expired" and starved.read() is None
    read_only = carrier(2)
    assert read_only.capture(3, 0, "miss")
    assert read_only.read() == (3, 0, "miss")  # Read paid; no free candidate write.
    no_read = carrier(1)
    assert no_read.capture(3, 0, "miss") and no_read.read() is None
    assert no_read.status == "expired" and no_read.reserve + no_read.spent == 1
    lifetime_cases = 0
    for reserve, delay in product(range(13), range(10)):
        delayed = carrier(reserve)
        captured = delayed.capture(3, 0, "miss")
        for _ in range(delay):
            delayed.tick()
        result_delayed = delayed.read()
        assert delayed.reserve + delayed.spent == reserve
        if reserve >= delay + 3:
            assert result_delayed == (2, 0, "miss")
        elif reserve == delay + 2:
            assert result_delayed == (3, 0, "miss")
        else:
            assert result_delayed is None
        assert captured == (reserve >= 1)
        lifetime_cases += 1

    rejected = 0
    for args_bad in ((-1, 0, "feed"), (512, 0, "feed"), (True, 0, "feed"),
                     (3, -1, "feed"), (3, 3, "feed"), (3, True, "feed"),
                     (3, 0, "missing"), (3, 0, "unknown")):
        try:
            kernel.update_live_support(*args_bad)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f"invalid input accepted: {args_bad}")
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in sources}
    report = {
        "status": "known_model_support_record_verified_not_capability_evidence",
        "source_hashes": hashes,
        "scope": "existing_GraceRecovery_live_body_world_port_before_body_update",
        "hidden_true_world_input_to_record_updater": False,
        "world_probability_weight_cases": 9 * 3 * 9,
        "weight_denominator": 2,
        "all_subset_action_sensor_updates": transitions,
        "weighted_response_partition_counts": [6, 8, 9],
        "world_pair_count": len(witnesses),
        "world_pair_witnesses": witnesses,
        "maximum_future_witness_steps": max(len(w["actions"]) for w in witnesses),
        "reachable_nonempty_support_masks": sorted(seen),
        "support_reachability_domain": "unknown_initial_world_current_sensor_then_all_live_actuator_settings",
        "action_feedback_witness": {"prior": prior, "sensor": "miss",
                                    "action_0_mask": left, "action_1_mask": right},
        "equal_support_different_feed_probabilities": [str(p) for p in probabilities],
        "invalid_inputs_rejected": rejected,
        "historical_source_support_cases": 9 * 3 * 7,
        "lean_paid_settlement_cases": len(table["paid_cases"]),
        "funded_carrier_lifetime_cases": lifetime_cases,
        "reverse_delivery_original_action_pairing_verified": True,
        "busy_capture_and_duplicate_read_rejected": True,
        "unfunded_holding_or_read_invalidates_receipt": True,
        "read_without_write_credit_retains_source_uncertainty": True,
        "current_state_freeze_counterexample_verified": True,
        "paid_historical_reference_contract_verified": True,
        "recorder_reserve_separately_endowed_not_reused_body_credit": True,
        "carrier_scope": "one_step_action_sensor_paired_before_delivery_delay",
        "carrier_provenance_and_tariffs": "explicit_unreviewed_H_not_measured_physics",
        "unknown_multistep_reward_attribution_verified": False,
        "arbitrary_delayed_feedback_pairing_discovered": False,
        "logical_support_mask_bits": 9,
        "minimum_standalone_byte_container": 2,
        "python_object_memory_or_gpu_memory_measured": False,
        "response_probability_record": False,
        "world_model_learned_from_data": False,
        "paid_physical_learning_or_delayed_feedback_matching_integrated": False,
        "joint_kernel_refinement_or_cut_certified": False,
        "general_axioms_added": False,
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "world_probability_weight_cases",
        "all_subset_action_sensor_updates", "weighted_response_partition_counts",
        "maximum_future_witness_steps", "lean_paid_settlement_cases",
        "funded_carrier_lifetime_cases", "invalid_inputs_rejected")}))


if __name__ == "__main__":
    main()
