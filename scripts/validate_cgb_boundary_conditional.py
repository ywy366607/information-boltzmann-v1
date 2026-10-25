"""Exact boundary-kernel/finite-time audits; no training or capability experiment."""

import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import struct
import subprocess

from validate_cgb_unified_reference import ROOT, LEAN, model, life, sha


def kernel(weights, fee=1):
    return model.BoundaryKernel(struct.pack(f"<{len(weights)}H", *weights), fee)


def stationary(weights):
    return [F(w, sum(weights)) for w in weights]


def matrix(weights):
    rows, size = [], len(weights)
    for i, old in enumerate(weights):
        row = [F(min(old, new), size * old) if i != j else F(0)
               for j, new in enumerate(weights)]
        row[i] = 1 - sum(row)
        rows.append(row)
    return rows


def tv(left, right):
    return sum(abs(a - b) for a, b in zip(left, right)) / 2


def advance(law, rows):
    return [sum(p * row[j] for p, row in zip(law, rows)) for j in range(len(law))]


def validate(vectors):
    entries = tickets = tracking = 0
    for record in vectors:
        weights = record["weights"]
        rows = matrix(weights)
        assert rows == [[F(n, d) for n in row] for row, d in
                        zip(record["rows"], record["denominators"])]
        if max(weights) <= 7:
            k = kernel(weights)
            for i, weight in enumerate(weights):
                counts = Counter(k.hop(i, j, u) for j in range(len(weights)) for u in range(weight))
                assert [F(counts[j], len(weights) * weight) for j in range(len(weights))] == rows[i]
                tickets += len(weights) * weight
    families = [[1, 3], [2, 5, 7], [1] * 8, list(range(1, 9)),
                [1] * 256, [65535] + [1] * 255]
    for weights in families:
        rows, pi = matrix(weights), stationary(weights)
        alpha = F(sum(weights), len(weights) * max(weights))
        for i, row in enumerate(rows):
            assert sum(row) == 1 and min(row) >= 0
            for j, value in enumerate(row):
                assert pi[i] * value == pi[j] * rows[j][i]
                assert value >= alpha * pi[j]
                entries += 1
        if len(weights) <= 8:
            law = [F(i == 0) for i in range(len(weights))]
            initial = tv(law, pi)
            for step in range(1, 7):
                law = advance(law, rows)
                assert tv(law, pi) <= (1 - alpha) ** step * initial
            # Externally registered target changes; NOT endogenous feedback.
            following = stationary(list(reversed(weights)))
            assert tv(advance(law, rows), following) <= (1 - alpha) * tv(law, pi) + tv(pi, following)
            # Expected sampled-delta error, L1: 2*eta*|p|*TV(mu,pi).
            eta_p = F(3, 10)
            assert sum(abs(eta_p * (a - b)) for a, b in zip(law, pi)) == 2 * eta_p * tv(law, pi)
            tracking += 1
    # Independent binary occupancies with weights (1,3), including the vacuum.
    weights = [F(1), F(1), F(3), F(3)]  # (00,10,01,11)
    mean = [(weights[1] + weights[3]) / sum(weights),
            (weights[2] + weights[3]) / sum(weights)]
    normalized_mean = [m / sum(mean) for m in mean]
    assert mean == [F(1, 2), F(3, 4)] and normalized_mean == [F(2, 5), F(3, 5)]
    assert normalized_mean != stationary([1, 3])
    # A legal one-attempt output can remain far from equilibrium.
    weights = [65535] + [1] * 255
    pi, rows = stationary(weights), matrix(weights)
    gap = tv(rows[1], pi)
    assert gap > F(99, 100)
    validate_payment()
    return dict(lean_transition_vectors=len(vectors), enumerated_random_tickets=tickets,
                detailed_balance_and_minorization_entries=entries,
                finite_time_and_exogenous_tracking_cases=tracking,
                binary_normalized_mean=[str(x) for x in normalized_mean],
                one_particle_target=["1/4", "3/4"],
                one_attempt_256_label_tv=str(gap),
                paid_rejections_self_proposals_and_refusals=True,
                one_attempt_per_driver_publication=True, active_sample_json_restore=True,
                single_sample_is_not_a_probability_vector=True,
                sampled_delta_l1_bias="2*eta*abs(p)*TV(actual_sample_law,target)",
                mixing_scope="Fixed frames or specified exogenous frame sequences; funded attempts only",
                locality_scope="Two frozen weights in one declared clique; full LG1 field not realized",
                rng_scope="Uniform proposal and acceptance ticket are explicit external primitives")


def validate_payment():
    k = kernel([3, 1])
    world = life.PrefixReplayWorld(b"ABCDE", source_id="boundary-contract", reserve=8, capacity=8)
    label = 0
    for proposal, ticket in ((1, 2), (0, 0)):  # rejection, then self-proposal
        old = world.state_dict()["state"]["counts"]["write_spent"]
        label, result = k.tick(label, proposal, ticket, world)
        assert result["funded"] and label == 0
        assert world.state_dict()["state"]["counts"]["write_spent"] == old + 1
    label, result = k.tick(label, 1, 0, world)  # funded accepted transfer
    assert result["funded"] and label == 1 and result["predicted_label"] == 0
    assert world.state_dict()["state"]["tick"] == 3
    checkpoint = json.loads(json.dumps(dict(world=world.state_dict(), weights=k.weights.hex(),
                                          attempt_fee=k.attempt_fee, label=label)))
    clone = life.PrefixReplayWorld(b"ABCDE", source_id="boundary-contract", reserve=8, capacity=8)
    clone.load_state_dict(checkpoint["world"])
    cloned_kernel = model.BoundaryKernel(bytes.fromhex(checkpoint["weights"]), checkpoint["attempt_fee"])
    other_label = checkpoint["label"]
    for proposal, ticket in ((0, 0), (1, 0)):
        label, result = k.tick(label, proposal, ticket, world)
        other_label, other = cloned_kernel.tick(other_label, proposal, ticket, clone)
        assert (label, result) == (other_label, other) and world.state_dict() == clone.state_dict()
    assert result["predicted_label"] is None and not result["funded"]  # maintenance failed
    before = world.state_dict()
    for args in ((True, 0, 0), (0, 2, 0), (0, 1, -1), (0, 1, 3), (0, 1, True)):
        try:
            k.tick(*args, world)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid boundary attempt admitted")
        assert world.state_dict() == before
    unfunded = life.PrefixReplayWorld(b"A", source_id="boundary-unfunded", reserve=1, capacity=1)
    label, result = k.tick(0, 1, 0, unfunded)
    assert result["predicted_label"] == 0 and not result["funded"] and label == 0
    assert unfunded.state_dict()["state"]["counts"]["write_spent"] == 0
    assert world.state_dict()["state"]["counts"]["fresh"] == 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default=shutil.which("lake"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/published/cgb_boundary_conditional_20261026.json")
    args = parser.parse_args()
    if not args.lake:
        parser.error("provide --lake")
    result = subprocess.run([args.lake, "env", "lean", "reviews/unified_reference_20261025.lean"],
                            cwd=LEAN, capture_output=True, encoding="utf-8", timeout=240, check=True)
    assert "sorryAx" not in result.stdout
    exported = next(json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"format":"CGB.UnifiedReference.boundary.v1"'))
    report = dict(status="conditional_boundary_sampler_and_finite_time_contract_checked",
                  capability_claim=False, training_performed=False, full_medium_admitted=False,
                  production_lean_modules_added=0, checks=validate(exported["vectors"]),
                  source_hashes={name: sha(ROOT / name) for name in (
                      "information_boltzmann/core/cgb_unified_reference.py",
                      "information_boltzmann/core/cgb_low_resource_life.py",
                      "scripts/validate_cgb_boundary_conditional.py",
                      "scripts/validate_cgb_unified_reference.py",
                      "lean/CGBAxioms/reviews/unified_reference_20261025.lean",
                      "docs/information_boltzmann/CGB_GENERATION_LAYER1_20261025.md",
                      "docs/information_boltzmann/CGB_LG1_ORGANISM_SPEC_20261026.md",
                      "docs/information_boltzmann/CGB_UNIFIED_REFERENCE_20261026.md")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
