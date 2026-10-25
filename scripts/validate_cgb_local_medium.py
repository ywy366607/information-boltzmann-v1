"""Exact shared-energy reaction audit; numerical/interface checks, no training."""

import argparse
from itertools import product
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from validate_cgb_unified_reference import ROOT, LEAN, life, sha

MODULE = "information_boltzmann/core/cgb_local_medium.py"
spec = importlib.util.spec_from_file_location("cgb_local_medium", ROOT / MODULE)
local = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = local
spec.loader.exec_module(local)


def pack(law, states):
    # Independent whole-word encoder, used only in the observer test workspace.
    word = 0
    for edge, (p, q, w) in enumerate(states):
        code = (p * (law.charge_cap + 1) + q) * (law.slot_cap + 1) + w
        word |= code << (edge * law.bits)
    return word.to_bytes(law.state_bytes, "little")


def energy(body):
    f = body.law
    states = [body.get(i) for i in range(f.edges)]
    return (sum(body.unit_energy(s) for s in states)
            + sum(f.potential[sum(states[r * f.labels + b][0] for b in range(f.labels))]
                  for r in range(f.rows))
            + sum(f.potential[sum(states[r * f.labels + b][1] for r in range(f.rows))]
                  for b in range(f.labels)))


def content(body):
    return sum(sum(body.get(i)) for i in range(body.law.edges))


def successor(body, edits):
    result = local.LocalMedium(body.law, bytes(body.payload))
    for edge, state in edits:
        result._put(edge, state)  # observer realization; live writes go through tick
    return result


def validate_reactions():
    rows = enabled = 0
    families = [local.LocalLaw(2, 2, 1, 1, (0, 1, 4)),
                local.LocalLaw(1, 2, 2, 1, (0, -2, 3, 1, 4), epsilon=-2, coupling=-1)]
    for f in families:
        assert len({local.LocalMedium(f, bytes(f.state_bytes)).decode(i)
                    for i in range(f.events)}) == f.events
        base = local.LocalMedium(f, bytes(f.state_bytes))
        assert all(base.reverse(base.reverse(i)) == i for i in range(f.events))
        choices = list(product(range(f.charge_cap + 1), range(f.charge_cap + 1), range(f.slot_cap + 1)))
        for states in product(choices, repeat=f.edges):
            body = local.LocalMedium(f, pack(f, states))
            initial_energy, initial_content = energy(body), content(body)
            for index in range(f.events):
                edits, exponent, bound = body.prepare(index)
                rows += 1
                if exponent is None:
                    assert not edits
                    continue
                after = successor(body, edits)
                back, reverse_exponent, reverse_bound = after.prepare(body.reverse(index))
                assert content(after) == initial_content
                assert successor(after, back).payload == body.payload
                assert bound == reverse_bound and exponent <= bound
                assert exponent - reverse_exponent == -2 * (energy(after) - initial_energy)
                enabled += 1
    return dict(exhaustive_states=4096 + 324, proposal_rows=rows, enabled_reverse_pairs=enabled,
                whole_energy_log_balance_exact=True, content_conservation_exact=True,
                static_reverse_scheduler=True, target_weight="2**(-2*E), within the conserved-content shell")


def validate_reads():
    f = local.LocalLaw(2, 3, 2, 2, (0, 1, 4, 9, 16, 25, 36))
    initial = pack(f, [(1, 1, 1)] * f.edges)
    checks = 0
    for index in range(f.events):
        body = local.LocalMedium(f, initial)
        kind, edge, other, _sign, channel = body.decode(index)
        row, col = divmod(edge, f.labels)
        allowed = ({edge, other} if kind >= 2 else
                   set(range(row * f.labels, (row + 1) * f.labels)) if channel == 0 else
                   set(range(col, f.edges, f.labels)))
        original, seen = body.get, set()
        def record(i):
            seen.add(i)
            return original(i)
        body.get = record
        frame = body.prepare(index)
        assert seen <= allowed
        body.get = original
        for outside in set(range(f.edges)) - allowed:
            changed = local.LocalMedium(f, initial)
            changed._put(outside, (2, 0, 2))
            assert changed.prepare(index) == frame
            checks += 1
    byte_law = local.LocalLaw(3, 256, 1, 1, tuple(0 for _ in range(257)))
    byte_body = local.LocalMedium(byte_law, pack(byte_law, [(1, 0, 1)] * byte_law.edges))
    for index in (0, 4 * 255, 4 * 767 + 3, 4 * 768, 6 * 768, byte_law.events - 1):
        edits, exponent, _bound = byte_body.prepare(index)
        if exponent is not None:
            after = successor(byte_body, edits)
            back, rev, _ = after.prepare(byte_body.reverse(index))
            assert successor(after, back).payload == byte_body.payload
            assert exponent - rev == -2 * (energy(after) - energy(byte_body))
    # Odd bit counts, a singleton dimension, and the full unsigned 4-bit range.
    wide = local.LocalLaw(3, 1, 15, 15, (0,) * 46)
    states = [(15, 0, 15), (1, 14, 2), (15, 15, 15)]
    packed = local.LocalMedium(wide, pack(wide, states))
    assert [packed.get(i) for i in range(3)] == states and len(packed.payload) == 5
    assert all(packed.reverse(packed.reverse(i)) == i for i in range(wide.events))
    bad = packed.state_dict()
    raw = bytearray(packed.payload)
    raw[-1] |= 240
    bad["payload"] = raw.hex()
    before = packed.state_dict()
    try:
        packed.load_state_dict(bad)
    except ValueError:
        assert packed.state_dict() == before
    else:
        raise AssertionError("nonzero padding accepted")
    # At full capacity both old/new encodings use 12 bits, but order differs.
    legacy = packed.state_dict()
    legacy["version"] = 1
    del legacy["encoding"]
    fields = [v for state in states for v in state]
    raw = bytearray((len(fields) + 1) // 2)
    for i, value in enumerate(fields):
        raw[i // 2] |= value << (4 * (i % 2))
    legacy["payload"] = raw.hex()
    try:
        packed.load_state_dict(legacy)
    except ValueError:
        assert packed.state_dict() == before
    else:
        raise AssertionError("legacy nibble layout silently reinterpreted")
    # Eleven bits at bit-offset 22 genuinely spans three bytes.
    three = local.LocalLaw(3, 1, 10, 8, (0,) * 31)
    states = [(10, 0, 8), (1, 9, 2), (10, 10, 8)]
    packed = local.LocalMedium(three, pack(three, states))
    assert three.bits == 11 and [packed.get(i) for i in range(3)] == states
    for replacement in ((0, 0, 0), (7, 2, 5), (10, 10, 8)):
        packed._put(2, replacement)
        states[2] = replacement
        assert [packed.get(i) for i in range(3)] == states
        assert bytes(packed.payload) == pack(three, states)
    return dict(clique_outside_change_checks=checks, actual_state_reads_within_declared_clique=True,
                native_byte_edges=byte_law.edges, native_byte_packed_state_bytes=len(byte_body.payload),
                edge_indices_stored=0, maximum_field_bits=4, fields_per_edge=3,
                native_byte_bits_per_edge=byte_law.bits,
                three_byte_read_write_checked=True,
                encoding="ceil(log2((Q+1)^2*(W+1))) fixed bits per edge; lossless",
                whole_clique_sum_cost_is_degree_dependent=True)


def validate_history_and_payment():
    f = local.LocalLaw(2, 2, 1, 2, (0, 1, 4))
    body = local.LocalMedium(f, pack(f, [(1, 0, 1), (0, 0, 0), (0, 0, 0), (1, 0, 0)]))
    world = life.PrefixReplayWorld(b"ABCDEFG", source_id="closed-reaction-interface", reserve=12, capacity=12)
    old_boundary = [body.get(i)[1] for i in range(f.edges)]
    initial_rate = body.prepare(0)[1]
    initial_content = content(body)
    # A valid transport path relocates a historical charge, with no external state assignment.
    path = [12, 31, 6, 0, 16]
    for tick, event in enumerate(path):
        result = body.tick(event, 0, world)
        assert result["accepted"] and result["tick"] == tick
        assert content(body) == initial_content
        if tick == 2:
            assert [body.get(i)[1] for i in range(f.edges)] == old_boundary
            history_rate = body.prepare(0)[1]
            assert history_rate - initial_rate == 4
            saved = json.loads(json.dumps(dict(body=body.state_dict(), world=world.state_dict())))
            clone = local.LocalMedium(f, bytes(len(body.payload)))
            clone.load_state_dict(saved["body"])
            other_world = life.PrefixReplayWorld(b"ABCDEFG", source_id="closed-reaction-interface", reserve=12, capacity=12)
            other_world.load_state_dict(saved["world"])
        elif tick > 2:
            assert clone.tick(event, 0, other_world) == result
            assert clone.state_dict() == body.state_dict() and other_world.state_dict() == world.state_dict()
    assert body.get(0) == (0, 0, 2)  # new slot came from an existing charge
    frozen = bytes(body.payload)
    blocked = body.tick(0, 0, world)
    assert blocked["funded"] and not blocked["accepted"] and bytes(body.payload) == frozen
    failed = body.tick(0, 0, world)
    assert not failed["active"] and not failed["funded"] and bytes(body.payload) == frozen
    before = body.state_dict(), world.state_dict()
    for event, ticket in ((True, 0), (-1, 0), (0, True), (0, -1), (0, 1)):
        try:
            body.tick(event, ticket, world)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reaction admitted")
        assert (body.state_dict(), world.state_dict()) == before
    bad = json.loads(json.dumps(saved["body"]))
    bad["law"]["charge_cap"] = True
    try:
        body.load_state_dict(bad)
    except ValueError:
        assert body.state_dict() == before[0]
    else:
        raise AssertionError("checkpoint type mismatch accepted")
    bad = json.loads(json.dumps(before[0]))
    raw = bytearray.fromhex(bad["payload"])
    raw[0] |= 15  # S=12, so local codes 12..15 are not states
    bad["payload"] = raw.hex()
    try:
        body.load_state_dict(bad)
    except ValueError:
        assert body.state_dict() == before[0]
    else:
        raise AssertionError("unused local code accepted")
    rejected_body = local.LocalMedium(f, pack(f, [(1, 0, 1), (0, 0, 0), (0, 0, 0), (1, 0, 0)]))
    reject_world = life.PrefixReplayWorld(b"X", source_id="dyadic-rejection", reserve=2, capacity=2)
    old_payload = bytes(rejected_body.payload)
    rejected = rejected_body.tick(0, 1, reject_world)  # valid nonzero ticket, positive-rate event
    assert rejected["funded"] and not rejected["accepted"] and rejected["spent"] == 1
    assert bytes(rejected_body.payload) == old_payload
    return dict(same_boundary_charges_different_history_rate_ratio=16,
                new_slot_from_conserved_charge=True, closed_content=initial_content,
                paid_path_attempts=len(path), active_joint_json_restore=True,
                blocked_attempt_is_paid=True, resource_failure_keeps_publication=True,
                enabled_rejection_is_paid=True,
                observation_conditions_rates=False, capability_claim=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", default=shutil.which("lake"))
    ap.add_argument("--output", type=Path, default=ROOT / "results/published/cgb_local_medium_20261026.json")
    args = ap.parse_args()
    if not args.lake:
        ap.error("provide --lake")
    lean = subprocess.run([args.lake, "env", "lean", "reviews/unified_reference_20261025.lean"], cwd=LEAN,
                          capture_output=True, encoding="utf-8", check=True, timeout=240)
    assert "sorryAx" not in lean.stdout and "local_cross_balance' depends on axioms" in lean.stdout
    report = dict(status="joint_local_reactions_checked", capability_claim=False,
                  full_medium_admitted=False, training_performed=False, production_lean_modules_added=0,
                  per_family_caps_are_registered_conductances=True,
                  reactions=validate_reactions(), locality=validate_reads(), history=validate_history_and_payment(),
                  source_hashes={name: sha(ROOT / name) for name in (MODULE,
                      "scripts/validate_cgb_local_medium.py", "information_boltzmann/core/cgb_low_resource_life.py",
                      "lean/CGBAxioms/reviews/unified_reference_20261025.lean",
                      "docs/information_boltzmann/CGB_LG1_ORGANISM_SPEC_20261026.md",
                      "docs/information_boltzmann/CGB_UNIFIED_REFERENCE_20261026.md")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
