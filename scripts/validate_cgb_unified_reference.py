"""Consolidate and check the existing finite witness; no capability training."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "CGBAxioms"
MODULE = "information_boltzmann/core/cgb_unified_reference.py"
spec = importlib.util.spec_from_file_location("cgb_unified_reference", ROOT / MODULE)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(lake, name):
    result = subprocess.run([lake, "env", "lean", "reviews/" + name], cwd=LEAN,
                            capture_output=True, encoding="utf-8", timeout=240)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:] + result.stderr[-1000:])
    assert "sorryAx" not in result.stdout and "[sorry" not in result.stdout
    return next(json.loads(line) for line in result.stdout.splitlines() if line.startswith("{"))


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
    report = dict(status="existing_finite_process_consolidated_and_cross_checked",
                  capability_claim=False, complete_life_definition_proved=False,
                  new_brain_architecture=False, production_lean_modules_added=0,
                  canonical_process="CGB.SensoryMaintenance.process",
                  components=validate_components(body, world), ghost=validate_ghost(world),
                  protocols=validate_protocols(), relearning=paid_relearning_path(),
                  carrier_scope="Executable old-stock choice realizes the contract, not Lean choose identities",
                  port_separation_scope="Separate extension; only existing legacy_rules is imported",
                  build_and_trust=dict(passed=True, inventory_sha256=sha(inventory_path),
                                       compiled_declarations=inventory["summary"]["declarations"],
                                       production_modules=len(inventory["summary"]["modules"]),
                                       complexity_guard_passed=budget["passed"]),
                  source_hashes={name: sha(ROOT / name) for name in (
                      MODULE, "scripts/validate_cgb_unified_reference.py",
                      "docs/information_boltzmann/CGB_UNIFIED_REFERENCE_20261026.md",
                      "lean/CGBAxioms/reviews/export_sensory_record_20261025.lean",
                      "lean/CGBAxioms/reviews/unified_reference_20261025.lean")},
                  lean_source_hashes=current_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "components", "ghost")}, indent=2))


if __name__ == "__main__":
    main()
