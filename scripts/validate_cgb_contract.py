"""Bounded delayed-world interface audits; no training or capability experiment."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load_core(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "information_boltzmann/core" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(value):
    return json.loads(json.dumps(value))


def reject(world, operation):
    before = world.state_dict()
    try:
        operation()
    except (ValueError, TypeError, RuntimeError):
        assert world.state_dict() == before, "rejection mutated the world"
        return
    raise AssertionError("invalid operation was admitted")


def balance(world, initial):
    state = world.state_dict()["state"]
    counts = state["counts"]
    spent = sum(counts[key] for key in world.COSTS)
    assert state["reserve"] + spent + counts["spill"] == initial + counts["offered"]
    assert counts["accepted"] == counts["delivered"] + counts["cancelled"] + int(state["pending"] is not None)
    return counts


def fixture_rule(code, hidden, bit, context, depth, prefix):
    """Positive integer arithmetic oracle, deliberately sensitive to delivered bytes."""
    auxiliary = context[-1] if len(context) > 1 else 0
    return ((code + 9 if hidden else 1) * (3 if hidden == bit else 1)
            * (2 if hidden == ((context[0] >> depth) & 1) else 1)
            * (3 if hidden == ((auxiliary >> depth) & 1) else 1))


def fresh_port(module):
    return module.PackedTreePort(256, fixture_rule, context=(0,), seed=20261025,
        learning=False, rule_id="delayed-byte-interface-oracle-v1:" + digest(Path(__file__)))


def validate_world(life, bridge_module):
    source = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def fresh(**kwargs):
        return life.PrefixReplayWorld(source, source_id="interface-bytes", **kwargs)

    world = fresh(origin=5, reserve=100, capacity=100, delay=3, max_block=4,
                  maintenance=1, request_fee=2, holding_fee=3, read_per_byte=4)
    reject(world, world.publish)
    reject(world, lambda: world.request(0, 1))
    frame = world.begin()
    assert frame.tick == 0 and frame.receipt is None and frame.active
    assert world.begin() == frame
    reject(world, lambda: world.request(0, 1))
    assert world.publish().payload == source[5]
    for lag, length in ((-1, 1), (0, 0), (0, 5), (0, 2), (1, 1), (True, 1)):
        reject(world, lambda lag=lag, length=length: world.request(lag, length))
    assert world.request(0, 1)
    before = world.state_dict()
    assert not world.request(0, 1) and world.state_dict() == before
    for tick in (1, 2, 3, 4):
        frame = world.begin()
        assert frame.tick == tick and frame.receipt == (source[5:6] if tick == 3 else None)
        open_state = world.state_dict()
        assert world.begin() == frame and world.state_dict() == open_state
        assert world.publish().payload == source[5 + tick]
        balance(world, 100)
    counts = balance(world, 100)
    assert {k: counts[k] for k in ("fresh", "accepted", "delivered", "replayed_bytes")} == {
        "fresh": 5, "accepted": 1, "delivered": 1, "replayed_bytes": 1}
    assert [counts[k] for k in ("maintenance_spent", "request_spent", "holding_spent", "read_spent")] == [5, 2, 9, 4]
    spill = fresh(reserve=7, capacity=8, maintenance=1, supply_per_tick=5)
    spill.begin()
    assert balance(spill, 7)["spill"] == 4
    for kwargs in (dict(reserve=2, maintenance=1, request_fee=3),
                   dict(reserve=3, maintenance=1, request_fee=1, holding_fee=2, delay=2),
                   dict(reserve=5, maintenance=1, request_fee=1, holding_fee=0, read_per_byte=3)):
        unfunded = fresh(**kwargs)
        unfunded.begin()
        unfunded.publish()
        before = unfunded.state_dict()
        accepted = unfunded.request(0, 1)
        if not accepted:
            assert unfunded.state_dict() == before
        else:
            assert unfunded.begin().receipt is None
            assert balance(unfunded, kwargs["reserve"])["cancelled"] == 1
    failed = fresh(reserve=1, maintenance=2)
    for tick in range(4):
        assert not failed.begin().active
        before = failed.state_dict()
        assert failed.publish().payload == source[tick]
        assert not failed.request(0, 1)
        assert failed.state_dict()["state"]["reserve"] == before["state"]["reserve"] == 0
        balance(failed, 1)
    eof = life.PrefixReplayWorld(b"X", source_id="eof", reserve=1)
    eof.begin()
    eof.publish()
    before = eof.state_dict()
    try:
        eof.begin()
    except StopIteration:
        assert eof.state_dict() == before
    else:
        raise AssertionError("EOF was not explicit")
    validate_restore(life, bridge_module, source)
    left = life.PrefixReplayWorld(b"ABC!future", source_id="left", reserve=20, capacity=20)
    right = life.PrefixReplayWorld(b"ABC?changed", source_id="right", reserve=20, capacity=20)
    for tick in range(3):
        assert left.begin() == right.begin()
        assert left.publish() == right.publish()
        if tick == 0:
            assert left.request(0, 1) == right.request(0, 1)
        a, b = left.state_dict()["state"], right.state_dict()["state"]
        assert a == b
    return dict(atomic_prefix_range_rejection=True, positive_delay_fresh_continuity=True,
                single_pending_slot=True, single_receipt_delivery=True, bounded_read_block=True,
                all_fees_and_spill_conserved=True, absorbing_failure=True, explicit_eof=True,
                pending_and_open_receipt_json_restore=True, illegal_snapshot_atomic_rejection=True,
                public_prefix_nonanticipation=True, auxiliary_and_vocab_atomic_rejection=True)


def validate_restore(life, bridge_module, source):
    def fresh():
        return life.PrefixReplayWorld(source, source_id="restore", reserve=80, capacity=80, delay=2)

    world, port = fresh(), fresh_port(bridge_module)
    for tick in range(6):
        frame = world.begin()
        auxiliary = tuple(frame.receipt or ())
        predicted = port.predict(auxiliary=auxiliary)
        event = world.publish()
        assert math.isclose(port.observe(event.payload, auxiliary=auxiliary), predicted.nll(event.payload))
        if tick == 0:
            assert world.request(0, 1)
        if tick in (0, 1):
            saved_world, saved_port = frozen(world.state_dict()), frozen(port.state_dict())
            clone, cloned_port = fresh(), fresh_port(bridge_module)
            clone.load_state_dict(saved_world)
            cloned_port.load_state_dict(saved_port)
            for field, value in (("due", 0), ("payload", [0])):
                bad = frozen(saved_world)
                bad["state"]["pending"][field] = value
                reject(clone, lambda bad=bad: clone.load_state_dict(bad))
            for _ in range(3):
                a, b = world.begin(), clone.begin()
                assert a == b
                aux = tuple(a.receipt or ())
                assert port.predict(auxiliary=aux) == cloned_port.predict(auxiliary=aux)
                x, y = world.publish(), clone.publish()
                assert x == y and port.observe(x.payload, auxiliary=aux) == cloned_port.observe(y.payload, auxiliary=aux)
                assert world.state_dict() == clone.state_dict() and port.state_dict() == cloned_port.state_dict()
            world.load_state_dict(saved_world)
            port.load_state_dict(saved_port)
    opened = fresh()
    opened.begin()
    opened.publish()
    assert opened.request(0, 1)
    opened.begin()
    opened.publish()
    receipt = opened.begin()
    assert receipt.receipt == source[:1]
    clone = fresh()
    clone.load_state_dict(frozen(opened.state_dict()))
    assert clone.begin() == receipt and clone.publish() == opened.publish()
    for field, value in (("cursor", -1), ("tick", True), ("reserve", 81), ("phase", "invalid")):
        bad = frozen(clone.state_dict())
        bad["state"][field] = value
        reject(clone, lambda bad=bad: clone.load_state_dict(bad))
    bad = frozen(opened.state_dict())
    bad["state"]["counts"]["fresh"] += 1
    reject(opened, lambda: opened.load_state_dict(bad))
    forged = life.PrefixReplayWorld(source, source_id="spill-range", reserve=16)
    bad = forged.state_dict()
    bad["state"].update(phase="open", reserve=0, active=False)
    bad["state"]["counts"]["spill"] = 16
    reject(forged, lambda: forged.load_state_dict(bad))
    a, b = fresh_port(bridge_module), fresh_port(bridge_module)
    ledger_module = sys.modules["cgb_information_ledger"]
    calls = []
    ledgers = [ledger_module.PredictiveInformationLedger() for _ in range(2)]
    bridges = [bridge_module.CGBPrequentialBridge(p, ledger, lambda _: calls.append(1) or math.log(256))
               for p, ledger in zip((a, b), ledgers)]
    before = a.state_dict(), dict(vars(ledgers[0])), bridges[0].last_tick
    for token, auxiliary in ((65, tuple(range(62))), (256, ())):
        try:
            bridges[0].step(life.Observation(0, token), write_units=0, dissipation_units=0,
                            auxiliary=auxiliary)
        except bridge_module.PrequentialBridgeViolation:
            assert not calls and before == (a.state_dict(), dict(vars(ledgers[0])), bridges[0].last_tick)
        else:
            raise AssertionError("invalid context or vocabulary admitted")
    for auxiliary in ((), (65, 66)):
        frame = a.predict(auxiliary=auxiliary)
        assert sum((frame.probability(t) for t in range(256)), Fraction()) == 1
    for bridge in bridges:
        bridge.step(life.Observation(0, 65), write_units=0, dissipation_units=0, auxiliary=(65,))
    assert a.state_dict() == b.state_dict() and vars(ledgers[0]) == vars(ledgers[1])


def run_path(life, bridge_module, ledger_module, source, source_id, events, supply):
    def fresh_world():
        return life.PrefixReplayWorld(source, source_id=source_id, reserve=16, capacity=32,
            maintenance=1, request_fee=2, holding_fee=1, read_per_byte=1,
            delay=3, max_block=4, supply_per_tick=supply)

    world = fresh_world()
    port = fresh_port(bridge_module)
    ledger = ledger_module.PredictiveInformationLedger(reference_name="uniform256")
    bridge = bridge_module.CGBPrequentialBridge(port, ledger, lambda _: math.log(256))
    active_count = fallback_count = information_pairs = 0
    failure_state = None
    snapshot_bytes = 0
    nll_sum = 0.0
    started = time.perf_counter()
    for tick in range(events):
        before_counts = world.state_dict()["state"]["counts"]
        frame = world.begin()
        auxiliary = tuple(frame.receipt or ())
        predicted = port.predict(auxiliary=auxiliary) if frame.active else None
        restored = None
        if tick in (4, 6):
            snapshot = frozen(dict(world=world.state_dict(), port=port.state_dict(),
                                   ledger=vars(ledger), last_tick=bridge.last_tick))
            snapshot_bytes = max(snapshot_bytes, len(json.dumps(snapshot).encode("utf-8")))
            clone, clone_port = fresh_world(), fresh_port(bridge_module)
            clone.load_state_dict(snapshot["world"])
            clone_port.load_state_dict(snapshot["port"])
            clone_ledger = ledger_module.PredictiveInformationLedger()
            vars(clone_ledger).update(snapshot["ledger"])  # Trusted audit snapshot, not a public loader.
            clone_bridge = bridge_module.CGBPrequentialBridge(clone_port, clone_ledger, lambda _: math.log(256))
            assert clone_bridge.last_tick == snapshot["last_tick"]
            assert clone.begin() == frame and clone_port.predict(auxiliary=auxiliary) == predicted
            restored = clone, clone_port, clone_ledger, clone_bridge
        if auxiliary:
            old = port.state_dict()
            cut = port.predict(auxiliary=(0,) * len(auxiliary))
            assert port.state_dict() == old
            information_pairs += int(predicted.probability(0) != cut.probability(0))
        event = world.publish()
        assert event.tick == tick and event.payload == source[tick]
        if tick % 16 == 3:
            world.request(0, 4)  # Fixed audit cadence, never a learned acquisition policy.
        after_counts = world.state_dict()["state"]["counts"]
        paid = sum(after_counts[k] - before_counts[k] for k in world.COSTS)
        if frame.active:
            result = bridge.step(event, write_units=0, dissipation_units=paid, auxiliary=auxiliary)
            assert result.prediction == predicted and result.model_nll_nats == predicted.nll(event.payload)
            nll = result.model_nll_nats
            active_count += 1
        else:
            failure_state = port.state_dict() if failure_state is None else failure_state
            assert port.state_dict() == failure_state
            nll = math.log(256)
            ledger.record(tick=tick, model_nll_nats=nll, reference_nll_nats=nll,
                          write_units=0, dissipation_units=paid)
            assert sum((Fraction(1, 256) for _ in range(256)), Fraction()) == 1
            fallback_count += 1
        nll_sum += nll
        if restored is not None:
            clone, clone_port, clone_ledger, clone_bridge = restored
            other = clone.publish()
            result = clone_bridge.step(other, write_units=0, dissipation_units=paid, auxiliary=auxiliary)
            assert other == event and result.model_nll_nats == nll
            assert clone.state_dict() == world.state_dict() and clone_port.state_dict() == port.state_dict()
            assert vars(clone_ledger) == vars(ledger) and clone_bridge.last_tick == bridge.last_tick
        balance(world, 16)
    counts = balance(world, 16)
    assert port.updates == 0 and active_count + fallback_count == counts["fresh"] == events
    assert ledger.dissipation_units == sum(counts[k] for k in world.COSTS)
    return {"fresh_bytes": events, "active_bridge_bytes": active_count, "uniform_fallback_bytes": fallback_count,
            "fallback_port_frozen": failure_state is None or failure_state == port.state_dict(),
            "pending_and_open_receipt_real_json_restore_equal": True,
            "trusted_whole_loop_json_snapshot_max_bytes": snapshot_bytes,
            "restore_scope": "world, port, ledger fields, bridge counter; trusted audit copy, not generic ledger loader",
            "replay_bytes": counts["replayed_bytes"], "resource_counts": counts,
            "final_reserve": world.state_dict()["state"]["reserve"], "pending_capacity": 1,
            "max_replay_block_bytes": 4, "all_prefix_balances_checked": True,
            "matched_same_old_state_information_cut_pairs": information_pairs,
            "information_cut": "same receipt length and paid world; replace only readout bytes by zeros",
            "actual_updates": port.updates, "head_nibbles": port.depths, "packed_parameter_bytes": port.packed.numel(),
            "nll_nats_sum": nll_sum, "prediction_ledger": ledger.summary(),
            "fee_interpretation": "declared operational units; no joule calibration",
            "wall_seconds": time.perf_counter() - started, "seconds_per_world_tick": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--events", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 8 <= args.events <= 256:
        raise ValueError("interface audit requires 8..256 events")
    modules = {name: load_core(name) for name in ("cgb_low_resource_life", "cgb_information_ledger",
               "cgb_language_kernel", "cgb_prequential_bridge", "cgb_observation_stream")}
    life, bridge = modules["cgb_low_resource_life"], modules["cgb_prequential_bridge"]
    report = {"status": "delayed_public_prefix_interface_audit", "capability_claim": False,
              "training_performed": False, "actual_updates": 0, "model_architecture_selected": False,
              "fixture": "positive-integer numerical oracle; not an identified brain law",
              "unit_checks": validate_world(life, bridge)}
    if args.real_data:
        folder = ROOT / "data/ib_owt_10000"
        _, validation, manifest, hashes = modules["cgb_observation_stream"].load_owt_sources(
            folder / "train.npy", folder / "validation.npy", "byte")
        archive = validation[:].tobytes()
        source_id = "owt-validation:" + hashlib.sha256(archive).hexdigest()
        nominal = run_path(life, bridge, modules["cgb_information_ledger"], archive, source_id, args.events, 2)
        cut = run_path(life, bridge, modules["cgb_information_ledger"], archive, source_id, args.events, 0)
        assert nominal["uniform_fallback_bytes"] == 0 and cut["uniform_fallback_bytes"] > 0
        assert nominal["matched_same_old_state_information_cut_pairs"] > 0
        report.update(source_revision=manifest["revision"], source_hashes=hashes, source_split="validation",
            source_hashes_audit_only=True, alphabet="native_utf8_byte256", source_origin=0,
            external_archive_bytes=len(archive), external_archive_host_size=sys.getsizeof(archive),
            external_archive_sha256=hashlib.sha256(archive).hexdigest(),
            document_boundaries="metadata only; no EOS/BOS mapped to fake bytes",
            replay_scope="already published validation prefix; no future or train-only fiction",
            request_policy="fixed lag=0 length=4 every 16 ticks starting tick 3",
            nominal=nominal, resource_cut=cut, world_tick_seconds_calibrated=False)
    report["implementation_sha256"] = {name: digest(ROOT / "information_boltzmann/core" / f"{name}.py")
                                        for name in modules}
    report["validator_sha256"] = digest(Path(__file__))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
