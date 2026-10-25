"""Consolidate and check the existing finite witness; no capability training."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import struct
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "CGBAxioms"
MODULE = "information_boltzmann/core/cgb_unified_reference.py"
spec = importlib.util.spec_from_file_location("cgb_unified_reference", ROOT / MODULE)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)
world_spec = importlib.util.spec_from_file_location("cgb_reference_world",
    ROOT / "information_boltzmann/core/cgb_low_resource_life.py")
life = importlib.util.module_from_spec(world_spec)
sys.modules[world_spec.name] = life
world_spec.loader.exec_module(life)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(lake, name):
    result = subprocess.run([lake, "env", "lean", "reviews/" + name], cwd=LEAN,
                            capture_output=True, encoding="utf-8", timeout=240)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:] + result.stderr[-1000:])
    assert "sorryAx" not in result.stdout and "[sorry" not in result.stdout
    objects = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    if name == "unified_reference_20261025.lean":
        world = next(x for x in objects if x["format"] == "CGB.UnifiedReference.world.v1")
        world["ticket_vectors"] = next(x["vectors"] for x in objects
            if x["format"] == "CGB.UnifiedReference.tickets.v1")
        return world
    return objects[0]


def validate_families():
    rows = world_rows = 0
    for n in (2, 3, 7, 16, 256):
        for capacity in (0, 1, 2, 5):
            f = model.FiniteFamily(n, capacity)
            cells = (range(f.cells) if n < 256 else
                     [0, 1] + [2 + p * n + h for p in range(3) for h in (0, 1, 127, 255)])
            for q in cells:
                for r in range(capacity + 1):
                    old = model.Body(q, r, f)
                    for signal in range(-1, n + 5):
                        u = model.local_step(old, signal)
                        assert u.body.content + u.outgoing == old.content + u.incoming
                        assert f.credit(u.body.cell) + u.writes + u.lost_credit == f.credit(q) + u.topups
                        assert all(v >= 0 for v in (u.incoming, u.outgoing, u.writes, u.topups, u.lost_credit))
                        carrier = model.Carrier.seed(old).advance(old, signal)
                        assert sum(p.count(0) for p in carrier.owners) == u.body.content
                        assert sum(p.count(1) for p in carrier.owners) == 4 + max(4, capacity)
                        rows += 1
                for w in range(4 * n + 1):
                    kernel = model.world_row(q, w, f)
                    assert sum(kernel.values()) == 1 and min(kernel.values()) > 0
                    assert all(0 <= v <= 4 * n for v in kernel)
                    world_rows += 1
            horizon = 6 if n < 256 else 2
            run = model.ReferenceRun(family=f).extend(horizon)
            run.balance(horizon)
    for n in (0, 1, True):
        try:
            model.FiniteFamily(n)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous search needs at least two situations")
    return dict(situations=[2, 3, 7, 16, 256], capacities=[0, 1, 2, 5],
                local_and_carrier_rows=rows, normalized_world_rows=world_rows,
                exhaustive_up_to_situations=16, largest_family="representative cells, all signals/worlds",
                scope="Finite implementation checks; all-time Lean certificates apply to the default only")


def likelihood_table(likelihood):
    top = max(likelihood)
    return b"".join(struct.pack("<256H", value, top + 1 - value, *([1] * 254))
                    for value in likelihood)


def validate_bytes(vectors):
    checked = 0
    for counts, likelihood, expected in vectors:
        # The final maximal-int vector checks arithmetic separately; a positive
        # 256-byte row has other entries too and must respect the total cap.
        if max(likelihood) > 32000:
            assert sum(counts) ** 2 * max(likelihood) < 2 ** 31
            continue
        frame = model.ByteFrame(model.TicketMemory(bytes(counts)), likelihood_table(likelihood))
        projected, fee = frame.project(0)
        assert list(projected.counts) == expected
        assert fee == sum((a ^ b).bit_count() for a, b in zip(counts, expected))
    for h in (1, 2, 3, 7, 31):
        for tickets in (1, 7, 31, 255):
            counts = bytes([tickets // h + (i < tickets % h) for i in range(h)])
            frame = model.ByteFrame(model.TicketMemory(counts), likelihood_table(list(range(1, h + 1))))
            assert sum(frame.probability(b) for b in range(256)) == 1
            for byte in range(256):
                projected, _fee = frame.project(byte)
                masses = frame.masses(byte)
                posterior = [Fraction(m, sum(masses)) for m in masses]
                errors = [abs(Fraction(c, tickets) - p) for c, p in zip(projected.counts, posterior)]
                assert sum(projected.counts) == tickets
                assert all(e < Fraction(1, tickets) for e in errors)
                assert sum(errors) / 2 <= Fraction(h, 2 * tickets)
                assert tickets * max(masses) < 2 ** 31
                checked += 1
    memory = model.TicketMemory(bytes([50, 50]))
    table = likelihood_table([1, 3])
    def world(reserve=100):
        return life.PrefixReplayWorld(b"\x00\x01\x00\x01", source_id="integer-interface",
                                     reserve=reserve, capacity=reserve)
    funded = world()
    learned, row = model.byte_step(memory, table, funded)
    assert learned != memory and row["probability"] == Fraction(2, 258)
    state = funded.state_dict()["state"]
    assert state["reserve"] + sum(state["counts"][c] for c in funded.COSTS) == 100
    assert state["counts"]["write_spent"] == row["spent"] > 0
    blocked = world(1)
    unchanged, rejected = model.byte_step(memory, table, blocked)
    assert unchanged == memory and not rejected["funded"]
    assert blocked.state_dict()["state"]["counts"]["write_spent"] == 0
    with patch.object(model.ByteFrame, "project", side_effect=AssertionError("inactive projection")), \
         patch.object(model.ByteFrame, "probability", side_effect=AssertionError("inactive inference")):
        unchanged, dead = model.byte_step(memory, table, blocked)
    assert unchanged == memory and not dead["active"] and dead["tick"] == 1
    assert dead["probability"] == Fraction(1, 256)
    saved = json.loads(json.dumps(dict(world=funded.state_dict(),
                                     memory=list(learned.payload), table=table.hex())))
    clone = world()
    clone.load_state_dict(saved["world"])
    cloned_memory = model.TicketMemory(bytes(saved["memory"]))
    cloned_table = bytes.fromhex(saved["table"])
    for _ in range(3):
        a, ra = model.byte_step(learned, table, funded)
        b, rb = model.byte_step(cloned_memory, cloned_table, clone)
        assert a == b and ra == rb and funded.state_dict() == clone.state_dict()
        learned, cloned_memory = a, b
    legacy = world().state_dict()
    legacy["version"] = 1
    del legacy["state"]["counts"]["write_spent"]
    clone = world()
    clone.load_state_dict(legacy)
    assert clone.state_dict() == world().state_dict()
    for bad in (True, -1, 1.0):
        before = funded.state_dict()
        try:
            funded.pay_write(bad)
        except life.ContractViolation:
            pass
        else:
            raise AssertionError("invalid write tariff accepted")
        assert funded.state_dict() == before
    unopened = world()
    for operation in (lambda: unopened.pay_write(0),
                      lambda: model.byte_step(memory, table[:-1], unopened)):
        before = unopened.state_dict()
        try:
            operation()
        except (life.ContractViolation, ValueError):
            pass
        else:
            raise AssertionError("invalid phase/frame accepted")
        assert unopened.state_dict() == before
    for kind in ("missing_new_counter", "legacy_with_write_counter", "balance_tamper"):
        bad = world().state_dict()
        if kind == "missing_new_counter":
            del bad["state"]["counts"]["write_spent"]
        elif kind == "legacy_with_write_counter":
            bad["version"] = 1
        else:
            bad["state"]["counts"]["write_spent"] = 1
        before = clone.state_dict()
        try:
            clone.load_state_dict(bad)
        except life.ContractViolation:
            pass
        else:
            raise AssertionError("invalid write checkpoint accepted")
        assert clone.state_dict() == before
    first = world()
    first.begin()
    bad = first.state_dict()
    bad["state"]["reserve"] -= 1
    bad["state"]["counts"]["write_spent"] = 1  # balances, but before any publication
    before = first.state_dict()
    try:
        first.load_state_dict(bad)
    except life.ContractViolation:
        pass
    else:
        raise AssertionError("write before first publication restored")
    assert first.state_dict() == before
    # Counterexample: the projection may erase a rare but decisive hypothesis.
    absorbed = model.ByteFrame(model.TicketMemory(bytes([0, 100])), likelihood_table([10000, 1]))
    assert absorbed.project(0)[0].counts == bytes([0, 100])
    exact = Fraction(4 * 10000, 4 * 10000 + 996)
    assert exact > Fraction(97, 100)
    return dict(projections=checked, lean_reference_vectors=len(vectors),
                persistent_counts_bytes="H", likelihood_program_bytes="512*H",
                arithmetic_bound=255 * 255 * 32767, signed_int32_safe=True,
                shared_maintenance_and_write_budget=True, score_before_update=True,
                invalid_write_and_checkpoint_atomic_rejection=True,
                full_restore_continuation=True, legacy_world_zero_write_migration=True,
                resource_failure_keeps_world_ticking=True,
                inactive_skips_inference_and_projection=True,
                one_step_tv_bound="H/(2M), relative to the already quantized prior",
                erased_rare_hypothesis_exact_posterior=str(exact),
                zero_count_absorption=True, uniform_infinite_history_error_bound=False,
                generative_likelihood_rule_supplied=False, capability_claim=False,
                local_medium_realization=False, finite_body_language_refinement=False)


def validate_components(body, world):
    assert body["initial"] == 5 and world["denominator"] == 2
    for row in body["rows"]:
        x = model.Body(row["cell"], row["reserve"])
        assert (x.content, x.action) == (row["content"], row["action"])
        for i, signal in enumerate(range(-1, 7)):
            u = model.local_step(x, signal)
            assert u.body.cell * 3 + u.body.reserve == row["transitions"][i]
            ports = [u.incoming, u.outgoing, u.writes, u.topups, u.lost_credit]
            assert ports == row["ports"][i]
            old_carrier = model.Carrier.seed(x)
            carrier = old_carrier.advance(x, signal)
            assert sum(part.count(0) for part in carrier.owners) == u.body.content
            for j, (old, new) in enumerate(zip(old_carrier.owners, carrier.owners)):
                b, intake, release = world["carrier_parts"][row["index"]][i][j]
                assert old.count(0) == b
                assert sum(a == 1 and b == 0 for a, b in zip(old, new)) == intake
                assert sum(a == 0 and b == 2 for a, b in zip(old, new)) == release
                assert sum(a == 2 and b == 1 for a, b in zip(old, new)) == intake
                assert all(a == b or (a, b) in ((1, 0), (0, 2), (2, 1)) for a, b in zip(old, new))
    for q in range(8):
        for w in range(9):
            expected = {v: Fraction(n, 2) for v, n in enumerate(world["rows"][q][w]) if n}
            assert model.world_row(q, w) == expected
            assert sum(expected.values()) == 1
    # Verify the actual Python orchestration on every full joint state/input.
    for row in body["rows"]:
        for w in range(9):
            for i, signal in enumerate(range(-1, 7)):
                z = model.State(model.Body(row["cell"], row["reserve"]), w)
                runner = model.ReferenceRun(lambda _t, _w: signal)
                runner.laws = [{z: Fraction(1)}]
                runner.extend(1)
                q, r = divmod(row["transitions"][i], 3)
                expected = {model.State(model.Body(q, r), v): Fraction(n, 2)
                            for v, n in enumerate(world["rows"][row["cell"]][w]) if n}
                assert runner.laws[1] == expected
    # An incorrect new-world-first schedule would identify A one tick too early.
    runner = model.ReferenceRun().extend(1)
    assert runner.laws[1] == {model.State(model.Body(), 6): Fraction(1)}
    assert model.local_step(model.Body(), model.SENSE[6]).body != model.Body()
    return dict(body_input_rows=192, world_probability_entries=648,
                joint_state_input_rows=1728, carrier_local_rows=192, carrier_component_rows=384,
                world_and_body_both_read_old_state=True)


def validate_ghost(world):
    law = {model.State(model.Body(2, 2), 0): Fraction(1, 3),
           model.State(model.Body(5, 1), 1): Fraction(2, 3)}
    runner = model.ReferenceRun(cut=True)
    runner.laws = [law]
    runner.extend(1)
    expected = {}
    for z, p in law.items():
        new = model.local_step(z.body, model.natural(0, z.world)).body
        for ghost, g in law.items():
            for v, n in enumerate(world["rows"][ghost.body.cell][z.world]):
                if n:
                    target = model.State(new, v)
                    expected[target] = expected.get(target, Fraction(0)) + p * g * Fraction(n, 2)
    assert expected == runner.laws[1]
    staged = model.ReferenceRun(cut=True).extend(3).extend(7).extend(12)
    direct = model.ReferenceRun(cut=True).extend(12)
    assert (staged.laws, staged.ledger) == (direct.laws, direct.ledger)
    for invalid in (True, 8, None):
        failed = model.ReferenceRun(lambda _t, _w: invalid, cut=True)
        before = (list(failed.laws), list(failed.world_kernels), list(failed.ledger))
        try:
            failed.extend(1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid protocol admitted")
        assert (failed.laws, failed.world_kernels, failed.ledger) == before
        failed.protocol = model.natural
        good = model.ReferenceRun(cut=True).extend(1)
        failed.extend(1)
        assert (failed.laws, failed.ledger) == (good.laws, good.ledger)
    return dict(own_cut_body_marginal=True, staged_continuation_equal=True,
                invalid_protocol_atomic_rejection_and_retry=True)


def paid_relearning_path():
    state, carrier = model.State(), model.Carrier.seed()
    probability, inputs, outputs, writes, topups = Fraction(1), 0, 0, 0, 0
    log = []
    # Select supported world branches, without changing the controller or kernel.
    for tick, target_world in enumerate((6, 0, 1, 3, 5, 7, 1, 0, 2, 4, 6, 0, 0)):
        signal = model.natural(tick, state.world)
        u = model.local_step(state.body, signal)
        probability *= model.world_row(state.body.cell, state.world)[target_world]
        carrier = carrier.advance(state.body, signal)
        inputs += u.incoming
        outputs += u.outgoing
        writes += u.writes
        topups += u.topups
        state = model.State(u.body, target_world)
        assert state.body.content + outputs == 4 + inputs
        assert model.CREDIT[state.body.cell] + writes == 1 + topups
        assert sum(o.count(1) for o in carrier.owners) == 8
        log.append(dict(tick=tick + 1, cell=model.CELLS[state.body.cell],
                        reserve=state.body.reserve, world=model.WORLDS[state.world],
                        cumulative_writes=writes))
    assert [log[i]["cell"] for i in (1, 6, 11)] == ["aSpent", "bSpent", "aSpent"]
    assert writes == 5 and topups == 6 and probability > 0
    return dict(probability=str(probability), writes=writes, topups=topups,
                incoming=inputs, outgoing=outputs, final_content=state.body.content,
                fixed_material_identities=24, source_stock=8, recharge_work=inputs,
                state_path=log, scope="One supported path, not typical behavior or capability evidence")


def validate_protocols():
    natural = model.ReferenceRun()
    assert all(natural.survival(n) == 1 for n in range(25))
    checked = 0
    for start in range(7):
        short = model.ReferenceRun(model.interrupted(start, 1))
        long = model.ReferenceRun(model.interrupted(start, 3))
        for n in range(start + 7):
            assert short.survival(n) == 1
            if n >= start + 3:
                assert long.survival(n) == 0
            checked += 1
    two = model.ReferenceRun(model.interrupted(2, 2))
    witness = model.State(model.Body(2, 1), 1)
    assert two.laws[0] == {model.State(): 1}
    assert two.survival(3) == Fraction(1, 2)
    assert two.laws[3][witness] == Fraction(1, 2)
    assert witness.body.content > 0 and two.recovery(3, witness) == 0
    cut = model.ReferenceRun(cut=True)
    scores = [cut.survival(n) for n in range(10)]
    assert all(s <= Fraction(3, 4) for s in scores[7:])
    return dict(natural_checked_times=25, interruption_start_times=list(range(7)),
                interruption_time_pairs=checked, cut_survival=[str(s) for s in scores],
                two_loss=dict(start=2, time=3, survival="1/2", body="aSpent", reserve=1,
                              world="feedB", witness_mass="1/2", first_hit_three_ticks="0"),
                natural_balance={k: str(v) for k, v in natural.balance(24).items()},
                cut_balance={k: str(v) for k, v in cut.balance(9).items()},
                scope="Exact finite checks; existing Lean certificates carry all-time claims")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default=shutil.which("lake"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/published/cgb_unified_reference_20261026.json")
    args = parser.parse_args()
    if not args.lake:
        parser.error("provide --lake with the installed Lake executable")
    body = export(args.lake, "export_sensory_record_20261025.lean")
    world = export(args.lake, "unified_reference_20261025.lean")
    inventory_path = LEAN / ".lake" / "audit_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    sources = sorted((LEAN / "CGBAxioms").glob("*.lean")) + [LEAN / "CGBAxioms.lean"]
    current_hashes = {p.relative_to(LEAN).as_posix(): sha(p) for p in sources}
    assert inventory["module_coverage_passed"] and inventory["proof_trust_passed"]
    assert current_hashes == inventory["source_hashes"], "Build/audit current Lean sources first"
    guard = subprocess.run([sys.executable, str(ROOT / "scripts/check_lean_complexity_budget.py")],
                           cwd=ROOT, capture_output=True, encoding="utf-8", check=True)
    budget = json.loads(guard.stdout)
    contract_run = subprocess.run([sys.executable, str(ROOT / "scripts/validate_cgb_contract.py"),
                                   "--real-data", "--events", "64"], cwd=ROOT,
                                  capture_output=True, encoding="utf-8", check=True)
    contract = json.loads(contract_run.stdout)
    report = dict(status="parameterized_finite_reference_and_paid_byte_diagnostic_checked",
                  capability_claim=False, complete_life_definition_proved=False,
                  new_brain_architecture=False, production_lean_modules_added=0,
                  canonical_process="CGB.SensoryMaintenance.process",
                  components=validate_components(body, world), ghost=validate_ghost(world),
                  protocols=validate_protocols(), relearning=paid_relearning_path(),
                  parameterized=validate_families(), byte_interface=validate_bytes(world["ticket_vectors"]),
                  real_byte_world_interface=dict(
                      source_revision=contract["source_revision"], split=contract["source_split"],
                      alphabet=contract["alphabet"], source_hashes=contract["source_hashes"],
                      fresh_bytes_per_path=64, paths=["nominal", "resource_cut"],
                      actual_model_updates=0, capability_claim=False,
                      unit_checks=contract["unit_checks"],
                      scope="Existing zero-update interface oracle, not the new port or a trained organism"),
                  carrier_scope="Executable old-stock choice realizes the contract, not Lean choose identities",
                  port_separation_scope="Separate extension; only existing legacy_rules is imported",
                  build_and_trust=dict(passed=True, inventory_sha256=sha(inventory_path),
                                       compiled_declarations=inventory["summary"]["declarations"],
                                       production_modules=len(inventory["summary"]["modules"]),
                                       complexity_guard_passed=budget["passed"]),
                  source_hashes={name: sha(ROOT / name) for name in (
                      MODULE, "scripts/validate_cgb_unified_reference.py",
                      "information_boltzmann/core/cgb_low_resource_life.py",
                      "scripts/validate_cgb_contract.py",
                      "docs/information_boltzmann/CGB_UNIFIED_REFERENCE_20261026.md",
                      "lean/CGBAxioms/reviews/export_sensory_record_20261025.lean",
                      "lean/CGBAxioms/reviews/unified_reference_20261025.lean")},
                  lean_source_hashes=current_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "components", "ghost")}, indent=2))


if __name__ == "__main__":
    main()
