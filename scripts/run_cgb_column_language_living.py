"""Single-budget, continuously learning OWT protocol; no learning laws live here."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cgb_column_language_admission import CudaGraphValueCredit, load_split
from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody

REVISION = "433fe0f44ed7894fea29c08b3202aa348ccc6369"
BEST_CRITERION = "best_observed_first_pass_window"
LOG_NAMES = ("events.jsonl", "windows.jsonl")


def phase_plan(warmup: int = 100_000, phase_bytes: int = 64_000) -> list[dict]:
    """All coordinates are raw source tokens, not scored UTF-8 bytes."""
    if not 1 <= phase_bytes <= warmup:
        raise ValueError("require 1 <= phase-bytes <= warmup")
    specs = [
        ("warmup", "train", 0, warmup, False),
        ("A_firstpass", "train", warmup, 2 * warmup, False),
        ("B_firstpass", "validation", 0, phase_bytes, False),
        ("A_revisit", "train", warmup, warmup + phase_bytes, True),
        ("A_fresh", "train", 2 * warmup, 2 * warmup + phase_bytes, False),
    ]
    return [dict(name=name, split=split, start=start, stop=stop, replay=replay)
            for name, split, start, stop, replay in specs]


class SourceClock:
    """Monotonic within a process; one UTC offline gap on the first resumed packet."""

    def __init__(self, last_arrival_utc_ns: int | None = None):
        self.last_arrival_utc_ns = last_arrival_utc_ns
        self.last_monotonic_ns: int | None = None

    def next_interval(self) -> dict:
        mono, utc = time.perf_counter_ns(), time.time_ns()
        if self.last_arrival_utc_ns is not None and utc < self.last_arrival_utc_ns:
            raise ValueError("source clock error: UTC moved backwards; no elapsed time invented")
        if self.last_monotonic_ns is not None:
            gap = mono - self.last_monotonic_ns
            basis = "process_monotonic"
        elif self.last_arrival_utc_ns is not None:
            gap = utc - self.last_arrival_utc_ns
            basis = "resume_utc_gap_once"
        else:
            gap, basis = 0, "birth_zero"
        if gap < 0:
            raise ValueError("source clock error: monotonic clock moved backwards")
        self.last_monotonic_ns, self.last_arrival_utc_ns = mono, utc
        return {"elapsed_seconds": gap / 1e9, "arrival_utc_ns": utc,
                "arrival_monotonic_ns": mono, "clock_basis": basis}


def rng_state() -> dict:
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def source_identity() -> dict:
    paths = [Path(__file__).resolve(),
             ROOT / "scripts/run_cgb_column_language_admission.py",
             ROOT / "information_boltzmann/core/cgb_column_body.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_experiment(path: Path, experiment: dict, engines: dict, clock: SourceClock,
                    *, snapshots: dict | None = None) -> dict:
    """One transaction holds both arms and the already-consumed next raw cursor."""
    payload = {"format_version": 1, "experiment": copy.deepcopy(experiment),
               "engines": snapshots if snapshots is not None else {
                   name: engine.checkpoint_state() for name, engine in engines.items()},
               "last_arrival_utc_ns": clock.last_arrival_utc_ns, "torch_rng": rng_state(),
               "log_offsets": {name: (path.parent / name).stat().st_size
                               if (path.parent / name).exists() else 0 for name in LOG_NAMES}}
    atomic_save(path, payload)
    return payload


def verify_resume_logs(output: Path, payload: dict) -> None:
    """Uncheckpointed log tails may be real experience; never delete or replay them."""
    for name in LOG_NAMES:
        path = output / name
        actual = path.stat().st_size if path.exists() else 0
        if actual != payload["log_offsets"][name]:
            raise ValueError(f"cannot exactly resume: {name} size differs from checkpoint; "
                             "original logs preserved, no rollback or replay")


def restore_engines(payload: dict, device: str | torch.device) -> dict:
    engines = {name: CudaGraphValueCredit.from_checkpoint(state, device=device)
               for name, state in payload["engines"].items()}
    # Constructors/capture may use RNG; no such work becomes resumed experience.
    restore_rng(payload["torch_rng"])
    return engines


def fork_engines(warmup: CudaGraphValueCredit, device: str | torch.device) -> dict:
    state, rng = warmup.checkpoint_state(), rng_state()
    engines = {"s_on": CudaGraphValueCredit.from_checkpoint(state, device=device),
               "s_off": CudaGraphValueCredit.from_checkpoint(
                   state, device=device, s_contact_step=0.0)}
    restore_rng(rng)
    return engines


def deliver_packet(engines: dict, raw_tokens: np.ndarray, interval: dict) -> tuple[dict, int]:
    """Deliver the common raw stream without joining across elapsed markers."""
    raw = np.asarray(raw_tokens)
    if bool(((raw < 0) | (raw >= 260)).any()):
        raise ValueError("source contains tokens outside special[0:4]/UTF-8[4:260]")
    for engine in engines.values():
        engine.advance_empty_time(interval["elapsed_seconds"])
    results = {}
    for name, engine in engines.items():
        results[name] = engine.phase(raw, report_every=len(raw) + 1)
    # Timing the next arrival happens only after both arms have completed.
    devices = {engine.body.column_keys.device for engine in engines.values()}
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return results, int((raw < 4).sum())


def new_window(cursor: int, snapshot: dict) -> dict:
    return {"start": cursor, "raw_tokens": 0, "special_tokens": 0, "scored_bytes": 0,
            "total_nll_nats": 0.0, "elapsed_seconds": 0.0, "diagnostic_totals": [0.0] * 6,
            "s_contact_start": snapshot["body_state_dict"]["reciprocal_contact"][
                snapshot["model_config"]["d_model"] + 2:].clone(),
            "update_calls_start": copy.deepcopy(snapshot["lifetime"]["update_calls"])}


def accumulate(window: dict, metrics: dict, raw_count: int, special: int, dt: float) -> None:
    scored = int(metrics["scored_bytes"])
    window["raw_tokens"] += raw_count
    window["special_tokens"] += special
    window["scored_bytes"] += scored
    window["total_nll_nats"] += float(metrics["mean_nll_nats"]) * scored
    window["elapsed_seconds"] += dt
    window["diagnostic_totals"] = [a + float(b) for a, b in zip(
        window["diagnostic_totals"], metrics["diagnostic_totals"])]


def distribution(value: torch.Tensor) -> dict:
    values = value.detach().to(device="cpu", dtype=torch.float64).flatten()
    return {"Min": float(values.min()), "Median": float(values.quantile(0.5)),
            "Mean": float(values.mean()), "Max": float(values.max())}


@torch.no_grad()
def observe(engine: CudaGraphValueCredit) -> dict:
    """Read-only current state. No body step, VJP, update or elapsed advance."""
    state = engine.state()
    occupied = state.participation
    return {
        "wood_candidate": {"H": distribution(state.activity),
                           "H_saturated_fraction": float((state.activity.abs() >= 1).float().mean())},
        "earth_candidate": {"precision": distribution(state.precision),
                            "demand": distribution(engine.body.demand(state)),
                            "C_mean": distribution(state.natural / state.precision)},
        "metal_candidate": {
            "S_typed_rank": {source: [distribution(occupied[..., index, rank])
                                      for rank in range(occupied.shape[-1])]
                             for index, source in enumerate(("external", "self"))},
            "S_rank_occupancy": [distribution(occupied[..., rank].sum(-1))
                                 for rank in range(occupied.shape[-1])],
            "reciprocal_absolute_error": distribution(
                engine.body.reciprocal_participation_mismatch(occupied).abs())},
        "water_candidate": {"S_free_capacity": distribution(1 - occupied.sum((-1, -2)))},
        "interpretation": "contextual functional observations, not five entities or automatic diagnoses",
    }


def best_eligible(phase: dict, arm: str, scored_bytes: int, raw_count: int,
                  report_every: int) -> bool:
    return (arm == "s_on" and not phase["replay"] and phase["name"] != "warmup"
            and scored_bytes > 0 and raw_count == report_every)


def diagnostic_report(totals: list, profile: dict) -> dict:
    return {"S_actual_admitted_mass": totals[0], "S_actual_released_mass": totals[1],
            "mass_counting": "directed sum / 2; each reciprocal physical contact counted once",
            "H_live_response_coordinates": totals[2],
            "H_local_write_support_coordinates": totals[3],
            "H_live_response_ratio": totals[2] / totals[3] if totals[3] else None,
            "S_return_lambda_squared_norm_sum": totals[4] if profile["s_contact_step"] > 0 else None,
            "S_conserved_transfer_squared_norm_sum": totals[5] if profile["s_contact_step"] > 0 else None,
            "S_return_lambda_collection": "enabled" if profile["s_contact_step"] > 0 else "disabled_not_collected",
            "S_conserved_transfer_collection": "enabled" if profile["s_contact_step"] > 0 else "disabled_not_collected",
            "scope": "actual executed forward; masses not energy/reward; squared norms not language benefit"}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def append_json(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")


def finite_payload(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_payload(item) for item in value)
    return True


def window_report(phase: dict, arm: str, window: dict, cursor: int,
                  snapshot: dict, observations: dict, phase_end: bool) -> dict:
    scored = window["scored_bytes"]
    mean = window["total_nll_nats"] / scored if scored else None
    calls = snapshot["lifetime"]["update_calls"]
    slow_delta = snapshot["body_state_dict"]["reciprocal_contact"][
        snapshot["model_config"]["d_model"] + 2:] - window["s_contact_start"]
    observations["metal_candidate"]["S_slow_parameter_window_net_change"] = {
        "absolute_change": distribution(slow_delta.abs()),
        "nonzero_entries": int(torch.count_nonzero(slow_delta)),
        "scope": "window_net_change, not per-step changes or language benefit"}
    return {"phase": phase["name"], "split": phase["split"], "replay": phase["replay"],
            "arm": arm, "raw_start": window["start"], "raw_stop": cursor,
            **{key: value for key, value in window.items()
               if key not in ("start", "update_calls_start", "s_contact_start")},
            "mean_nll_nats": mean, "bits_per_byte": mean / math.log(2) if mean is not None else None,
            "phase_end": phase_end, "learning_profile": snapshot["profile"],
            "enabled_update_call_opportunities_window": {
                key: count - window["update_calls_start"][key] for key, count in calls.items()},
            "enabled_update_call_opportunities_lifetime": calls,
            "update_call_scope": "enabled calls, not certified nonzero parameter changes",
            "five_phase_observations": {**observations, "fire_candidate": {
                "prequential_mean_nll_nats": mean, "scored_bytes": scored}},
            "diagnostics": diagnostic_report(window["diagnostic_totals"], snapshot["profile"])}


def summarize_logs(output: Path, experiment: dict) -> None:
    groups = {}
    with (output / "windows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = record["phase"] + "/" + record["arm"]
            group = groups.setdefault(key, {"raw_tokens": 0, "special_tokens": 0,
                                           "scored_bytes": 0, "total_nll_nats": 0.0,
                                           "replay": record["replay"], "diagnostic_totals": [0.0] * 6,
                                           "learning_profile": record["learning_profile"],
                                           "window_means": [],
                                           "enabled_update_call_opportunities": {
                                               name: 0 for name in record["enabled_update_call_opportunities_window"]}})
            if record["mean_nll_nats"] is not None:
                group["window_means"].append(record["mean_nll_nats"])
            for field in ("raw_tokens", "special_tokens", "scored_bytes", "total_nll_nats"):
                group[field] += record[field]
            group["diagnostic_totals"] = [a + b for a, b in zip(
                group["diagnostic_totals"], record["diagnostic_totals"])]
            for name, count in record["enabled_update_call_opportunities_window"].items():
                group["enabled_update_call_opportunities"][name] += count
    for group in groups.values():
        means = group.pop("window_means")
        group["window_mean_nll_distribution"] = distribution(torch.tensor(means, dtype=torch.float64)) if means else None
        group["window_distribution_scope"] = "unweighted distribution of logged window means, not per-byte losses"
        group["diagnostics"] = diagnostic_report(group["diagnostic_totals"], group["learning_profile"])
        scored = group["scored_bytes"]
        group["mean_nll_nats"] = group["total_nll_nats"] / scored if scored else None
        group["bits_per_byte"] = group["mean_nll_nats"] / math.log(2) if scored else None
        group["phase_mean_scope"] = "scored-byte-weighted prequential NLL"
    comparisons = {}
    for phase in experiment["phases"][1:]:
        on, off = (groups.get(phase["name"] + "/" + arm) for arm in ("s_on", "s_off"))
        if on is not None and off is not None:
            comparable = on["scored_bytes"] > 0 and off["scored_bytes"] > 0
            comparisons[phase["name"]] = {
                "replay": phase["replay"], "scored_bytes_match": on["scored_bytes"] == off["scored_bytes"],
                "s_on_scored_bytes": on["scored_bytes"], "s_off_scored_bytes": off["scored_bytes"],
                "s_on_minus_s_off_mean_nll_nats": on["mean_nll_nats"] - off["mean_nll_nats"] if comparable else None,
                "s_on_minus_s_off_bits_per_byte": on["bits_per_byte"] - off["bits_per_byte"] if comparable else None}
    summary = {"status": experiment["status"], "scope": "budget-limited; not a convergence proof",
               "clock": "real common packet-arrival clock, not per-byte physical sampling",
               "language_only": "systems 1/5 conditions; 2/3 co-learning; system 4 deferred",
               "source_switch": "validation first-pass with online learning, not an asserted domain change",
               "common_delivered_raw": experiment["common_delivered_raw"],
               "cumulative_exposure_raw": experiment["cumulative_exposure_raw"],
               **{key: experiment[key] for key in ("common_delivered_scored_bytes", "common_delivered_special_tokens",
                                                   "cumulative_exposure_scored_bytes", "cumulative_exposure_special_tokens")},
               "best_criterion": BEST_CRITERION,
               "best_eligibility": "s_on non-warmup non-replay complete report_every-raw windows only",
               "best": experiment["best"], "phases": groups, "paired_differences": comparisons,
               "comparison_scope": "one common birth, budget-limited branch comparison; not significance or capability proof",
               "parameter_numbers": experiment["parameter_numbers"],
               "provenance": experiment["provenance"], "config": experiment["config"],
               "model_source": experiment["model_source"]}
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")


def run_protocol(experiment: dict, engines: dict, sources: dict, output: Path,
                 clock: SourceClock, device: str | torch.device) -> None:
    config, phases = experiment["config"], experiment["phases"]
    while experiment["phase_index"] < len(phases):
        phase = phases[experiment["phase_index"]]
        if phase["name"] != "warmup" and "warmup" in engines:
            engines = fork_engines(engines["warmup"], device)
            experiment["stage"] = "twoarm"
            experiment["windows"] = {name: new_window(experiment["next_raw_cursor"], engine.checkpoint_state())
                                     for name, engine in engines.items()}
            save_experiment(output / "latest.pt", experiment, engines, clock)
        start = experiment["next_raw_cursor"]
        window_raw = next(iter(experiment["windows"].values()))["raw_tokens"]
        stop = min(start + config["graph_block"], phase["stop"],
                   start + config["report_every"] - window_raw)
        raw = sources[phase["split"]][start:stop]
        interval = clock.next_interval()
        results, special = deliver_packet(engines, raw, interval)
        # Both arms completed this shared packet. Save the NEXT raw position, never its start.
        experiment["next_raw_cursor"] = stop
        experiment["common_delivered_raw"] += len(raw)
        experiment["cumulative_exposure_raw"] += len(raw) * len(engines)
        for unit, count in (("scored_bytes", len(raw) - special), ("special_tokens", special)):
            experiment["common_delivered_" + unit] += count
            experiment["cumulative_exposure_" + unit] += count * len(engines)
        for name, result in results.items():
            accumulate(experiment["windows"][name], result, len(raw), special,
                       interval["elapsed_seconds"])
        append_json(output / "events.jsonl", {
            "phase": phase["name"], "split": phase["split"], "replay": phase["replay"],
            "raw_start": start, "raw_stop": stop, "raw_tokens": len(raw),
            "special_tokens": special, "special_id_counts": [int((raw == token).sum()) for token in range(4)],
            "lexical_bytes": len(raw) - special, "arms": list(engines), **interval})
        failed = not finite_payload(results)
        phase_end = stop == phase["stop"]
        report_due = phase_end or next(iter(experiment["windows"].values()))["raw_tokens"] == config["report_every"]
        if not report_due and not failed:
            continue
        snapshots = {name: engine.checkpoint_state() for name, engine in engines.items()}
        failed = failed or not finite_payload(snapshots)
        best_changed = False
        reports = {name: window_report(phase, name, experiment["windows"][name], stop,
                                        snapshots[name], observe(engine), phase_end)
                   for name, engine in engines.items()}
        failed = failed or not finite_payload(reports)
        for name, report in reports.items():
            append_json(output / "windows.jsonl", report)
            print(json.dumps(json_safe(report), ensure_ascii=False, allow_nan=False), flush=True)
            mean = report["mean_nll_nats"]
            if (not failed and best_eligible(phase, name, report["scored_bytes"],
                                            report["raw_tokens"], config["report_every"])
                    and (experiment["best"] is None or mean < experiment["best"]["mean_nll_nats"])):
                experiment["best"] = {"criterion": BEST_CRITERION, "phase": phase["name"],
                                      "raw_start": report["raw_start"], "raw_stop": stop,
                                      "scored_bytes": report["scored_bytes"], "mean_nll_nats": mean}
                best_changed = True
        if failed:
            experiment["status"] = "failed_nonfinite"
            payload = save_experiment(output / "latest.pt", experiment, engines, clock, snapshots=snapshots)
            atomic_save(output / "failure.pt", payload)
            raise FloatingPointError("nonfinite numerical state; failure.pt retains consumed cursor, no rollback")
        if phase_end:
            experiment["phase_index"] += 1
            if experiment["phase_index"] < len(phases):
                experiment["next_raw_cursor"] = phases[experiment["phase_index"]]["start"]
            else:
                experiment["status"] = "completed_budget_limited"
        experiment["windows"] = {name: new_window(experiment["next_raw_cursor"], snapshot)
                                 for name, snapshot in snapshots.items()}
        payload = save_experiment(output / "latest.pt", experiment, engines, clock, snapshots=snapshots)
        if best_changed:
            atomic_save(output / "best.pt", payload)
    summarize_logs(output, experiment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/ib_owt_10000")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/cgb_column_language_living_20261101")
    parser.add_argument("--resume", type=Path, help="only this experiment's latest.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--phase-bytes", type=int, default=64_000)
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--graph-block", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.resume:
        if args.resume.name != "latest.pt":
            raise ValueError("resume requires latest.pt, not best or a failure rollback")
        output = args.resume.resolve().parent
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        verify_resume_logs(output, payload)
        experiment = payload["experiment"]
        if experiment["status"] == "failed_nonfinite":
            raise ValueError("this experiment failed nonfinite; do not continue fake training")
        if experiment["model_source"] != source_identity():
            raise ValueError("model/runner/driver source identity differs from checkpoint")
        config = experiment["config"]
    else:
        output = args.output_dir.resolve()
        if output.exists():
            raise FileExistsError("output directory already exists; choose a new directory or --resume latest.pt")
        phases = phase_plan(args.warmup, args.phase_bytes)
        if min(args.graph_block, args.report_every) < 1:
            raise ValueError("graph-block and report-every must be positive")
        config = {"warmup": args.warmup, "phase_bytes": args.phase_bytes,
                  "report_every": args.report_every, "graph_block": args.graph_block,
                  "d_model": 32, "grid": 8, "active_columns": 8, "contact_rank": 1,
                  "seed": 20261031, "head_step": 0.01, "value_step": 0.001,
                  "route_step": 0.001, "contact_step": 0.001, "s_contact_step": 0.001,
                  "prediction_contract": "score arriving x_t from state through x_(t-1); then assimilate x_t",
                  "special_token_contract": "advance physical time without concatenating language runs",
                  "record_s_return": True, "collect_diagnostics": True,
                  "protocol_sizing_override": (args.warmup, args.phase_bytes, args.report_every,
                                               args.graph_block) != (100_000, 64_000, 10_000, 128)}
    phases = experiment["phases"] if args.resume else phases
    sources, provenance = {}, {}
    for split in ("train", "validation"):
        required = max(phase["stop"] for phase in phases if phase["split"] == split)
        sources[split], provenance[split] = load_split(args.data_root.resolve(), split, required)
        if len(sources[split]) < required or provenance[split]["revision"] != REVISION:
            raise ValueError("cached registered OWT revision/split does not cover the protocol")
    if args.resume:
        if provenance != experiment["provenance"]:
            raise ValueError("data provenance differs from checkpoint")
        engines = restore_engines(payload, device)
        clock = SourceClock(payload["last_arrival_utc_ns"])
    else:
        output.mkdir(parents=True)
        torch.manual_seed(config["seed"])
        body = CGBColumnLanguageBody(32, (8, 8), active_columns=8, contact_rank=1).to(device).eval()
        engine = CudaGraphValueCredit(body, body.initial_state(1), 0.01, 0.001, 0.001, 0.001,
                                      config["graph_block"], s_contact_step=0.001,
                                      record_s_return=True, collect_diagnostics=True)
        engines, clock = {"warmup": engine}, SourceClock()
        experiment = {"status": "running_budget_limited", "stage": "warmup", "config": config,
                      "phases": phases, "phase_index": 0, "next_raw_cursor": 0,
                      "common_delivered_raw": 0, "cumulative_exposure_raw": 0,
                      "common_delivered_scored_bytes": 0, "common_delivered_special_tokens": 0,
                      "cumulative_exposure_scored_bytes": 0, "cumulative_exposure_special_tokens": 0,
                      "best": None,
                      "provenance": provenance, "model_source": source_identity(),
                      "parameter_numbers": sum(parameter.numel() for parameter in body.parameters()),
                      "windows": {"warmup": new_window(0, engine.checkpoint_state())}}
        save_experiment(output / "latest.pt", experiment, engines, clock)
    run_protocol(experiment, engines, sources, output, clock, device)


if __name__ == "__main__":
    main()
