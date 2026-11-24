"""Join saved geometry, actual one-round receipts and existing long-run logs."""
from __future__ import annotations

import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def external_rows(path):
    rows = [json.loads(x) for x in path.read_text(encoding="utf8").splitlines()]
    return [r for r in rows if r.get("autonomous_event", {}).get("kind") == "external"]


def main():
    project = Path(__file__).resolve().parents[1]
    published = project/"results/published"
    old = read(published/"cgb_coupling_credit_2845_head_geometry_20261123.json")["bodies"][0]
    current = read(published/"cgb_coupling_actual_gradient_replay_20261123.json")["bodies"][0]
    private = Path("E:/mengli-coupling-credit-audit-20261123/one_round_receipt")
    actual = external_rows(private/"events.jsonl")[0]
    strict = Path("E:/mengli-wallclock-10min-20261123/strict_10min")
    natural = external_rows(strict/"events.jsonl")[0]
    fields = ("cycle", "choice", "utterance_hex", "world", "response_nll", "metal_g", "wood_prediction", "bfss_dynamics")
    matches = {k: actual[k] == natural[k] for k in fields}
    assert all(matches.values())
    receipts = [json.loads(x) for x in (private/"coupling_update_receipts.jsonl").read_text(encoding="utf8").splitlines()]
    update_summary = []
    for r in receipts:
        if r["kind"] != "optimizer_execution" or r["gradient"] is None:
            continue
        update_summary.append(dict(call=r["optimizer_call"], gate=r["gate"],
            gradient_norm=sum(x*x for x in r["gradient"])**.5,
            nonzero_gradient_layers=sum(x != 0 for x in r["gradient"]),
            maximum_nominal_step=max(map(abs, r["ungated_raw_step"])),
            maximum_actual_step=max(map(abs, r["executed_raw_step"])),
            first_order_objective_change_without_gate=sum(g*d for g, d in zip(r["gradient"], r["ungated_raw_step"])),
            executed_step_reconstruction_error=r["reconstructed_parameter_max_error"]))
    long_receipts = {}
    for name in ("strict_10min", "phase_rhythm_10min"):
        rows = external_rows(strict.parent/name/"events.jsonl")
        gates = [r["adaptive_dynamics"]["budget_write_fraction"]["coupling"] for r in rows]
        long_receipts[name] = dict(external_rounds=len(rows), zero_gate_boundary_snapshots=gates.count(0),
            mean_boundary_gate=sum(gates)/len(gates),
            initial_mean_coupling=sum(rows[0]["adaptive_dynamics"]["coupling"])/32,
            final_mean_coupling=sum(rows[-1]["adaptive_dynamics"]["coupling"])/32)
    result = dict(schema=1, new_external_rounds=1, new_optimizer_updates=4,
        new_external_bytes=960, paid_calls=0, new_source_transition="private copy 2861 -> 2862",
        source_sha256=current["source_sha256"], source_unchanged=current["source_unchanged"],
        instrumentation_agrees_with_existing_natural_prefix=matches,
        geometries={"2845": old["physical"], "2861": current["physical"]},
        actual_updates=update_summary,
        exact_gradient_replay=current["actual_gradient_replay"],
        existing_longer_receipts=long_receipts,
        decision=dict(
            production_change="retain existing coupling, role groups, Wiener gate, LR and epsilon",
            corrected_measurement="report per-layer and per-head force ratios, frequencies, absolute forces and dt-specific displacement together",
            interpretation="weak layer aggregate coexists with appreciable slow-head coupling; zero optimizer gate holds learned coefficients while physical force continues",
            initialization_audit="separate birth coefficient, actual operating scale, task-credit return and executed optimizer update"))
    path = published/"cgb_coupling_credit_audit_summary_20261123.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(dict(report=str(path), updates=update_summary, historical=long_receipts), ensure_ascii=False))


if __name__ == "__main__":
    main()
