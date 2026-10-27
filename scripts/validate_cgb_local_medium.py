"""Exact shared-energy reaction audit; numerical/interface checks, no training."""

import argparse
from fractions import Fraction
from itertools import product
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

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


def validate_open_exchange():
    checked = enabled = 0
    f = local.LocalLaw(1, 2, 2, 1, (0, -2, 3, 1, 4), epsilon=-2, coupling=-1)
    triples = list(product(range(3), range(3), range(2)))
    for states in product(triples, repeat=f.edges):
        body = local.LocalMedium(f, pack(f, states))
        for stock in (0, 1, 2):
            for cost, base, drive in ((0, 0, 2), (2, -1, 3), (-1, 1, 0)):
                r = local.ReservoirLaw(stock, cost, base, drive)
                opened = local.OpenMedium(body, r)
                for label in range(f.labels):
                    for index in range(f.events, opened.events):
                        checked += 1
                        edits, exponent, bound, sign = opened.prepare(index, label)
                        assert opened.reverse(opened.reverse(index)) == index
                        if exponent is None:
                            continue
                        assert -bound <= exponent <= bound
                        after = local.OpenMedium(successor(body, edits), r)
                        after.stock = stock - sign
                        reverse, rev, rev_bound, rev_sign = after.prepare(opened.reverse(index), label)
                        assert successor(after.body, reverse).payload == body.payload
                        assert rev_sign == -sign and rev_bound == bound
                        col = edits[0][0] % f.labels
                        mu = base + (drive if col == label else 0)
                        delta = energy(after.body) - energy(body) + (cost - mu) * sign
                        assert exponent - rev == -2 * delta
                        assert content(after.body) + after.stock == content(body) + stock
                        enabled += 1
    # An actual read trace, including a nontrivial Q clique.
    f = local.LocalLaw(3, 2, 1, 1, (0, 1, -2, 4))
    body = local.LocalMedium(f, pack(f, [(1, 0, 1)] * f.edges))
    opened = local.OpenMedium(body, local.ReservoirLaw(3))
    reads, original = [], body.get
    body.get = lambda e: (reads.append(e), original(e))[1]
    opened.prepare(f.events, 0)
    assert set(reads) == {0, 2, 4}
    body.get = original
    stimulated = opened.prepare(f.events, 0)[1]
    unstimulated = opened.prepare(f.events, 1)[1]
    assert stimulated - unstimulated == opened.reservoir.stimulus
    for rows, labels in ((1, 1), (1, 256), (3, 256)):
        law = local.LocalLaw(rows, labels, 1, 1, (0,) * (max(rows, labels) + 1))
        op = local.OpenMedium(local.LocalMedium(law, pack(law, [(0, 0, 0)] * law.edges)),
                              local.ReservoirLaw(1))
        families = op.proposal_families
        assert sum(n for _, n in families) == op.events
        assert sum(Fraction(n, len(families) * n) for _, n in families) == 1
        for start, n in families:
            for index in range(start, start + n):
                assert start <= op.reverse(index) < start + n
    return dict(proposal_rows=checked, enabled_reverse_pairs=enabled,
                exact_boundary_log_balance=True, body_plus_reservoir_conserved=True,
                boundary_reads_only_its_Q_clique=True, byte_modulates_boundary_only=True,
                family_proposal_normalized_and_reverse_symmetric=True)


def fresh_open(source, reserve=200, supply=0, hole=None, opportunities_per_tick=1):
    f = local.LocalLaw(1, 256, 1, 1, (0,) * 257)
    states = [(1, 1, 0)] * f.edges
    if hole is not None:
        states[hole] = (1, 0, 0)
    body = local.LocalMedium(f, pack(f, states))
    opened = local.OpenMedium(body, local.ReservoirLaw(8, content_cost=1, stimulus=2),
                              opportunities_per_tick=opportunities_per_tick)
    world = life.PrefixReplayWorld(source, source_id="open-material-interface", reserve=reserve,
                                  capacity=200, supply_per_tick=supply)
    return opened, world


def validate_open_windows():
    """Bounded same-byte reactions, not repeated world publications."""
    zero, zw = fresh_open(b"A", opportunities_per_tick=0)
    initial = zero.state_dict()
    result = zero.tick_window((), zw)
    assert result["tick"] == 0 and result["byte"] == 65
    assert result["scheduled_attempts"] == result["evaluated_attempts"] == 0
    assert result["spent"] == result["planned_random_bits"] == 0
    assert zero.state_dict() == initial and zw.state_dict()["state"]["tick"] == 1
    assert zw.state_dict()["state"]["counts"]["maintenance_spent"] == 1

    one, ow = fresh_open(b"A")
    legacy, lw = fresh_open(b"A")
    index = one.body.law.events + 2 * 65 + 1
    window = one.tick_window(((index, 0),), ow)
    old = legacy.tick(index, 0, lw)
    assert window["funded_attempts"] == int(old["funded"]) == 1
    assert window["accepted_reactions"] == int(old["accepted"]) == 1
    for key in ("tick", "byte", "active", "target_mass", "total_mass", "material_delta", "spent"):
        assert window[key] == old[key]
    assert window["evaluated_random_bits"] == old["random_bits"]
    assert one.state_dict() == legacy.state_dict() and ow.state_dict() == lw.state_dict()

    opened, world = fresh_open(b"BA", reserve=10, hole=66, opportunities_per_tick=3)
    f = opened.body.law
    chain = ((f.events + 2 * 65 + 1, 0), (f.events + 2 * 66, 0), (4 * f.edges + 2 * 66, 0))
    initial_content = content(opened.body) + opened.stock
    result = opened.tick_window(chain, world)
    assert result["target_mass"] == 0  # B was absent before its own publication.
    assert result["scheduled_attempts"] == result["evaluated_attempts"] == 3
    assert result["funded_attempts"] == result["accepted_reactions"] == result["spent"] == 3
    assert opened.body.get(66) == (1, 0, 1) and (opened.intake, opened.released) == (1, 1)
    assert content(opened.body) + opened.stock == initial_content
    counts = world.state_dict()["state"]["counts"]
    assert counts["fresh"] == counts["maintenance_spent"] == 1 and counts["write_spent"] == 3
    assert opened.tick_window((), world)["target_mass"] == 0  # Released A stays absent.

    before = opened.state_dict(), world.state_dict()
    invalid_plans = ([chain[0]], ((True, 0),), ((-1, 0),), ((opened.events, 0),),
                     (chain[0], (chain[1][0], True)), (chain[0], (chain[1][0], -1)),
                     (chain[0], (chain[1][0], 1 << (2 * opened.proposal_bound(chain[1][0])))),
                     chain + (chain[0],), ((0,),), (iter(chain),))
    for plan in invalid_plans:
        try:
            opened.tick_window(plan, world)
        except (ValueError, TypeError):
            assert (opened.state_dict(), world.state_dict()) == before
        else:
            raise AssertionError("invalid or excessive window partly committed")
    for quota in (-1, True, 1.5):
        try:
            fresh_open(b"A", opportunities_per_tick=quota)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("invalid opportunity quota admitted")

    limited, low_world = fresh_open(b"ABC", reserve=3, hole=66, opportunities_per_tick=4)
    plan = chain + (chain[-1],)
    result = limited.tick_window(plan, low_world)
    assert result["scheduled_attempts"] == 4 and result["evaluated_attempts"] == 2
    assert result["funded_attempts"] == result["accepted_reactions"] == result["spent"] == 2
    assert limited.body.get(66) == (1, 1, 0)  # The unfunded third reaction did not form a slot.
    assert result["planned_random_bits"] == sum(2 * limited.proposal_bound(i) for i, _ in plan)
    assert result["evaluated_random_bits"] == sum(2 * limited.proposal_bound(i) for i, _ in plan[:2])
    frozen = limited.state_dict()
    for tick in (1, 2):
        result = limited.tick_window(plan, low_world)
        assert result["tick"] == tick and not result["active"] and result["funded_attempts"] == 0
        assert result["evaluated_attempts"] == result["evaluated_random_bits"] == 0
        assert limited.state_dict() == frozen

    # Paid random rejection and blocked capacity both consume an opportunity.
    rejected, rw = fresh_open(b"A", opportunities_per_tick=2)
    release = rejected.body.law.events + 2 * 65 + 1
    blocked = rejected.body.law.events + 2 * 66
    maximum_ticket = (1 << (2 * rejected.proposal_bound(release))) - 1
    result = rejected.tick_window(((release, maximum_ticket), (blocked, 0)), rw)
    assert result["funded_attempts"] == result["spent"] == 2 and result["accepted_reactions"] == 0

    # Whole-window JSON and random-source continuation retain the actual plan.
    source = b"ABCDEFGH"
    live, live_world = fresh_open(source, supply=5, opportunities_per_tick=4)
    rng = random.Random(20261027)
    for tick in range(len(source)):
        plan = tuple((i, rng.getrandbits(2 * live.proposal_bound(i)))
                     for i in (live.sample_proposal(rng) for _ in range(4)))
        result = live.tick_window(plan, live_world)
        if tick == 2:
            saved = json.loads(json.dumps(dict(body=live.state_dict(), world=live_world.state_dict(),
                                                rng=rng.getstate())))
            clone, clone_world = fresh_open(source, supply=5, opportunities_per_tick=4)
            clone.load_state_dict(saved["body"])
            clone_world.load_state_dict(saved["world"])
            replay_rng = random.Random()
            rv = saved["rng"]
            replay_rng.setstate((rv[0], tuple(rv[1]), rv[2]))
        elif tick > 2:
            replay = tuple((i, replay_rng.getrandbits(2 * clone.proposal_bound(i)))
                           for i in (clone.sample_proposal(replay_rng) for _ in range(4)))
            assert replay == plan and clone.tick_window(replay, clone_world) == result
            assert clone.state_dict() == live.state_dict() and clone_world.state_dict() == live_world.state_dict()
    mismatched, _ = fresh_open(source, supply=5)
    before = mismatched.state_dict()
    try:
        mismatched.load_state_dict(saved["body"])
    except ValueError:
        assert mismatched.state_dict() == before
    else:
        raise AssertionError("snapshot changed the registered opportunity quota")
    legacy_snapshot = legacy.state_dict()
    legacy_snapshot["version"] = 1
    del legacy_snapshot["opportunities_per_tick"]
    legacy.load_state_dict(legacy_snapshot)
    try:
        live.load_state_dict(legacy_snapshot)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy single-opportunity snapshot admitted to a larger window")
    before = live.state_dict()
    for quota in (1, True, -1):
        bad = dict(before, opportunities_per_tick=quota)
        try:
            live.load_state_dict(bad)
        except ValueError:
            assert live.state_dict() == before
        else:
            raise AssertionError("tampered opportunity quota changed the body")
    try:
        live.opportunities_per_tick = 99
    except AttributeError:
        pass
    else:
        raise AssertionError("registered opportunity quota is mutable")
    return dict(zero_window_still_publishes_and_maintains=True, single_attempt_legacy_equivalent=True,
                sequential_same_byte_source_to_slot=True, score_frozen_before_publication=True,
                late_invalid_ticket_and_excess_budget_atomic=True, refusal_truncates_evaluation=True,
                failure_keeps_world_advancing=True, blocked_and_rejected_opportunities_paid=True,
                completed_window_JSON_and_rng_restore=True, quota_snapshot_checked=True,
                finite_registered_opportunities_not_free_internal_time=True, capability_claim=False)


def validate_window_real_bytes():
    """Same real arrivals and supply, with explicit extra paid reaction work."""
    sys.path.insert(0, str(ROOT / "information_boltzmann/core"))
    from cgb_observation_stream import load_owt_sources
    data = ROOT / "data/ib_owt_10000"
    _, validation, _, provenance = load_owt_sources(data / "train.npy", data / "validation.npy", "byte")
    source = bytes(int(v) for v in validation[:256])
    assert len(source) == 256
    class CountedRandom(random.Random):
        def __init__(self, seed):
            super().__init__(seed)
            self.generated_bits = 0
        def getrandbits(self, k):
            self.generated_bits += k
            return super().getrandbits(k)
    rows = []
    for quota in (1, 64):
        opened, world = fresh_open(source, supply=65, opportunities_per_tick=quota)
        start = time.process_time()
        plans, random_bits = [], 0
        for tick in range(len(source)):
            # Reset by tick so the single opportunity is the paired first one;
            # neither the label nor the future source is used to make proposals.
            rng = CountedRandom(20261027 + tick)
            plan = tuple((i, rng.getrandbits(2 * opened.proposal_bound(i)))
                         for i in (opened.sample_proposal(rng) for _ in range(quota)))
            plans.append(plan)
            random_bits += rng.generated_bits
        planning_cpu = time.process_time() - start
        seen = set()
        def storage_bytes(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            return sys.getsizeof(value) + (sum(storage_bytes(v) for v in value)
                                          if isinstance(value, (tuple, list)) else 0)
        plan_bytes = storage_bytes(plans)
        totals = dict.fromkeys(("scheduled_attempts", "evaluated_attempts", "funded_attempts",
                                "accepted_reactions", "planned_random_bits", "evaluated_random_bits", "spent"), 0)
        cpu = wall = 0.0
        for tick, plan in enumerate(plans):
            start_cpu, start_wall = time.process_time(), time.perf_counter()
            result = opened.tick_window(plan, world)
            cpu += time.process_time() - start_cpu
            wall += time.perf_counter() - start_wall
            assert result["tick"] == tick and result["byte"] == source[tick] and result["active"]
            assert content(opened.body) + opened.stock == opened.initial_content + opened.reservoir.source_initial
            for key in totals:
                totals[key] += result[key]
        counts = world.state_dict()["state"]["counts"]
        assert totals["scheduled_attempts"] == totals["funded_attempts"] == quota * len(source)
        assert counts["fresh"] == counts["maintenance_spent"] == len(source)
        assert counts["write_spent"] == totals["spent"] and counts["offered"] == 65 * len(source)
        rows.append(dict(opportunities_per_tick=quota, **totals, actual_generated_random_bits=random_bits,
                         scheduler_and_rejection_random_bits=random_bits - totals["planned_random_bits"],
                         source_intake=opened.intake, source_release=opened.released, source_stock=opened.stock,
                         net_plastic_slots=sum(opened.body.get(i)[2] for i in range(opened.body.law.edges)),
                         body_bytes=len(opened.body.payload), precomputed_python_plan_bytes=plan_bytes,
                         planning_cpu_seconds=planning_cpu, kernel_cpu_seconds=cpu, kernel_wall_seconds=wall,
                         resource_ledger=counts))
    return dict(source_provenance=provenance, split="validation", fresh_bytes=len(source),
                supply_per_tick=65, initial_reserve=200, reserve_capacity=200, arms=rows,
                same_initial_body=True, proposals_precomputed_without_byte_labels=True,
                random_bits_include_randrange_rejection=True, net_slots_not_gross_formation_count=True,
                random_bit_scope="requested getrandbits(k) output bits; PRNG internal words and energy unmeasured",
                CPU_timing_includes_kernel_observer_readout=True, fixed_finite_work_budget=True,
                continuous_OS_arrival_not_implemented=True, training_performed=False,
                capability_claim=False, bits_per_byte_not_used_for_capability=True)


def validate_input_response_bound():
    # Exact one-attempt kernel differences: all closed proposals cancel.
    f = local.LocalLaw(1, 3, 1, 1, (0,) * 4)
    maximum, checked = Fraction(0), 0
    for states in product(product(range(2), repeat=3), repeat=f.edges):
        body = local.LocalMedium(f, pack(f, states))
        for stock in (0, 1):
            op = local.OpenMedium(body, local.ReservoirLaw(stock, 1, 0, 2))
            difference = {}
            old_key = bytes(body.payload), stock
            F = len(op.proposal_families)
            proposal = Fraction(1, F * 2 * f.edges)
            for label, direction in ((0, 1), (1, -1)):
                for index in range(f.events, op.events):
                    edits, exponent, bound, sign = op.prepare(index, label)
                    if exponent is None:
                        continue
                    probability = direction * proposal / (2 ** (bound - exponent))
                    key = bytes(successor(body, edits).payload), stock - sign
                    difference[key] = difference.get(key, 0) + probability
                    difference[old_key] = difference.get(old_key, 0) - probability
            assert sum(difference.values()) == 0
            tv = sum(abs(value) for value in difference.values()) / 2
            assert tv <= Fraction(2, F * f.labels)
            maximum = max(maximum, tv)
            checked += 1
    q, threshold = Fraction(1, 512), Fraction(1, 2)
    windows = []
    for n in (0, 1, 64, 256, 1024):
        envelope = 1 - (1 - q) ** n
        assert 0 <= envelope <= min(1, n * q)
        windows.append(dict(paid_opportunities=n, contact_envelope_upper=float(envelope),
                            minimum_operation_fee=n))
    necessary = 0
    while 1 - (1 - q) ** necessary < threshold:
        necessary += 1
    return dict(exact_body_source_kernel_pairs=checked, maximum_checked_TV=str(maximum),
                bound="TV(K_b(x),K_c(x)) <= min(1,2/(F*B)) for one paid attempt",
                scope="same old body/source; distinct stimuli; uniform family and uniform event",
                native_1x256_TV_upper_bound="1/512",
                finite_windows=windows, half_TV_requires_at_least_opportunities=necessary,
                compound_TV_proof="paper first-contact coupling; Lean checks scalar envelope/budget only",
                fixed_exposure_and_full_funding_required_for_kernel_power=True,
                larger_envelope_is_not_response_or_learning_guarantee=True,
                implication="family balance alone does not establish usable per-byte response")


def validate_open_timing():
    opened, world = fresh_open(b"ABCDE", reserve=10, hole=66)
    f = opened.body.law
    # Release Q[A], take source material into empty Q[B], then form a slot there.
    events = [f.events + 2 * 65 + 1, f.events + 2 * 66, 4 * f.edges + 2 * 66]
    for n, index in enumerate(events):
        masses, total = opened.masses()
        target = b"ABCDE"[n]
        result = opened.tick(index, 0, world)
        assert result["target_mass"] == masses[target] and result["total_mass"] == total
        assert result["accepted"] and result["tick"] == n
        assert content(opened.body) + opened.stock == opened.initial_content + 8
        if n == 0:
            assert result["target_mass"] == 1 and opened.masses()[0][65] == 0
            saved = json.loads(json.dumps(dict(body=opened.state_dict(), world=world.state_dict())))
            clone, twin_world = fresh_open(b"ABCDE", reserve=10, hole=66)
            clone.load_state_dict(saved["body"])
            twin_world.load_state_dict(saved["world"])
        else:
            assert clone.tick(index, 0, twin_world) == result
            assert clone.state_dict() == opened.state_dict() and twin_world.state_dict() == world.state_dict()
    assert opened.body.get(66) == (1, 0, 1)
    assert (opened.intake, opened.released, opened.stock) == (1, 1, 8)
    # A valid nonzero ticket rejects; a subsequent capacity-blocked move costs too.
    index = f.events + 2 * 68 + 1
    bound = opened.proposal_bound(index)
    result = opened.tick(index, (1 << (2 * bound)) - 1, world)
    assert result["funded"] and not result["accepted"]
    result = opened.tick(f.events + 2 * 67, 0, world)
    assert result["funded"] and not result["accepted"]
    before = opened.state_dict(), world.state_dict()
    for index, ticket in ((-1, 0), (opened.events, 0), (0, True), (0, -1), (0, 1 << 100)):
        try:
            opened.tick(index, ticket, world)
        except ValueError:
            assert (opened.state_dict(), world.state_dict()) == before
        else:
            raise AssertionError("invalid open attempt mutated the world")
    for field, value in (("stock", before[0]["stock"] + 1), ("intake", True)):
        bad = json.loads(json.dumps(before[0]))
        bad[field] = value
        try:
            opened.load_state_dict(bad)
        except ValueError:
            assert opened.state_dict() == before[0]
        else:
            raise AssertionError("invalid open ledger admitted")
    dead, no_credit = fresh_open(b"AB", reserve=1)
    for _ in range(2):
        old = dead.state_dict()
        result = dead.tick(dead.body.law.events + 1, 0, no_credit)
        assert not result["funded"] and dead.state_dict() == old
    assert result["tick"] == 1 and not result["active"]
    # A present but zero target mass is preserved, not silently smoothed.
    zero, zw = fresh_open(b"AA")
    zero.tick(zero.body.law.events + 2 * 65 + 1, 0, zw)
    result = zero.tick(zero.body.law.events + 2 * 65, 0, zw)
    assert result["target_mass"] == 0
    # Same old body and ticket, different public byte: boundary chemistry alone
    # changes the accepted reaction (no externally assigned structural update).
    matched, mw = fresh_open(b"A", hole=65)
    other, ow = fresh_open(b"B", hole=65)
    index = matched.body.law.events + 2 * 65
    assert matched.tick(index, 8, mw)["accepted"]
    assert not other.tick(index, 8, ow)["accepted"]
    empty = local.OpenMedium(local.LocalMedium(f, pack(f, [(1, 0, 0)] * f.edges)),
                             local.ReservoirLaw(0))
    assert empty.masses() == ((1,) * 256, 256)
    assert empty.prepare(f.events, 0)[1] is None
    # Intervention check: move only the newly formed slot back to the source.
    # Old P/Q readouts and total material match; later dynamics can differ.
    actual = local.OpenMedium(local.LocalMedium(f, bytes(opened.body.payload)),
                              local.ReservoirLaw(opened.stock, 1, 0, 2))
    control_body = local.LocalMedium(f, bytes(opened.body.payload))
    control_body._put(66, (1, 0, 0))
    control = local.OpenMedium(control_body, local.ReservoirLaw(opened.stock + 1, 1, 0, 2))
    assert actual.masses() == control.masses()
    assert content(actual.body) + actual.stock == content(control.body) + control.stock
    wa = life.PrefixReplayWorld(b"DB", source_id="structural-response", reserve=4, capacity=4)
    wc = life.PrefixReplayWorld(b"DB", source_id="structural-response", reserve=4, capacity=4)
    index = f.events + 2 * 66
    assert actual.tick(index, 8, wa)["accepted"]
    assert not control.tick(index, 8, wc)["accepted"]
    assert actual.tick(f.events + 2 * 67, 0, wa)["target_mass"] == 1
    assert control.tick(f.events + 2 * 67, 0, wc)["target_mass"] == 0
    return dict(pre_target_readout=True, active_open_JSON_restore=True,
                source_intake_to_plastic_slot=True, rejected_and_blocked_paid=True,
                resource_failure_keeps_publication=True, invalid_inputs_atomic=True,
                zero_target_mass_reported=True, empty_readout_fallback_declared=True,
                source_exhaustion_blocks_intake=True, byte_counterfactual_changes_reaction=True,
                conserved_structure_intervention_changes_future_readout=True)


def validate_open_real_bytes():
    # A provenance-checked real-stream INTERFACE audit, not a capability run.
    sys.path.insert(0, str(ROOT / "information_boltzmann/core"))
    from cgb_observation_stream import load_owt_sources
    data = ROOT / "data/ib_owt_10000"
    _, validation, _, provenance = load_owt_sources(data / "train.npy", data / "validation.npy", "byte")
    source = bytes(int(v) for v in validation[:64])
    opened, world = fresh_open(source, supply=2)
    rng = random.Random(20261026)
    accepted = exchanges = zeros = 0
    bits = 0.0
    for t in range(len(source)):
        index = opened.sample_proposal(rng)
        ticket = rng.getrandbits(2 * opened.proposal_bound(index))
        result = opened.tick(index, ticket, world)
        assert result["byte"] == source[t] and result["tick"] == t
        accepted += int(result["accepted"])
        exchanges += int(result["material_delta"] != 0)
        zeros += int(result["target_mass"] == 0)
        if result["target_mass"]:
            bits -= math.log2(result["target_mass"] / result["total_mass"])
        assert content(opened.body) + opened.stock == opened.initial_content + 8
        if t == 31:
            saved = json.loads(json.dumps(dict(body=opened.state_dict(), world=world.state_dict(),
                                                rng=rng.getstate())))
            clone, tw = fresh_open(source, supply=2)
            clone.load_state_dict(saved["body"])
            tw.load_state_dict(saved["world"])
            replay_rng = random.Random()
            rv = saved["rng"]
            replay_rng.setstate((rv[0], tuple(rv[1]), rv[2]))
        elif t > 31:
            replay_index = clone.sample_proposal(replay_rng)
            replay_ticket = replay_rng.getrandbits(2 * clone.proposal_bound(replay_index))
            assert (replay_index, replay_ticket) == (index, ticket)
            assert clone.tick(index, ticket, tw) == result
            assert clone.state_dict() == opened.state_dict() and tw.state_dict() == world.state_dict()
    return dict(source_provenance=provenance, split="validation", fresh_bytes=len(source),
                accepted_reactions=accepted, accepted_exchanges=exchanges,
                zero_probability_targets=zeros, observer_bits_per_byte=None if zeros else bits / len(source),
                body_bytes=len(opened.body.payload), native_alphabet=256,
                boundary_proposal_fraction=f"1/{len(opened.proposal_families)}",
                old_all_event_boundary_fraction=f"{2 * opened.body.law.edges}/{opened.events}",
                observer_global_readout_not_local_body_computation=True,
                sampler_rng_JSON_restore=True,
                capability_claim=False, budget_limited_interface_only=True)


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
    assert "boundary_exchange_log_balance' depends on axioms" in lean.stdout
    assert "boundary_contact_fraction' depends on axioms" in lean.stdout
    assert "boundary_contact_envelope' depends on axioms" in lean.stdout
    assert "boundary_response_budget' depends on axioms" in lean.stdout
    report = dict(status="joint_local_reactions_checked", capability_claim=False,
                  full_medium_admitted=False, training_performed=False, production_lean_modules_added=0,
                  production_theorems_added=0, private_scalar_review_lemmas_added=2,
                  per_family_caps_are_registered_conductances=True,
                  reactions=validate_reactions(), locality=validate_reads(), history=validate_history_and_payment(),
                  open_exchange=validate_open_exchange(), open_timing=validate_open_timing(),
                  open_windows=validate_open_windows(),
                  input_response=validate_input_response_bound(),
                  real_byte_open_interface=validate_open_real_bytes(),
                  real_byte_window_interface=validate_window_real_bytes(),
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
