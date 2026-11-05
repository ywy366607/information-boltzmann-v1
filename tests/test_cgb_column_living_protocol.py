"""Protocol/interface tests only: mock learners, no language capability experiment."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_cgb_column_language_living as living


class MockBody:
    def __init__(self):
        self.column_keys = torch.zeros(1)
        self.reciprocal_contact = torch.zeros(12, 3)

    def demand(self, state):
        return 1 / (1 + state.precision)

    def reciprocal_participation_mismatch(self, occupied):
        return torch.zeros_like(occupied)


class MockEngine:
    """Records interface calls; counters are not a toy learning task."""

    def __init__(self, *, s_contact_step=0.001):
        self.body = MockBody()
        self.profile = dict(head_step=0.01, value_step=0.001, route_step=0.001,
                            contact_step=0.001, s_contact_step=s_contact_step,
                            block_size=3, record_s_return=True, collect_diagnostics=True)
        self.buffers = {"activity": torch.zeros(1, 1, 1, 2),
                        "natural": torch.zeros(1, 1, 1, 2),
                        "precision": torch.ones(1, 1, 1, 1),
                        "causal_token": torch.tensor([-1]),
                        "participation": torch.zeros(1, 1, 1, 4, 2, 1)}
        self.lifetime = dict(scored_lifetime=0, special_lifetime=0, advanced_elapsed=0.0,
                             update_calls={key: 0 for key in ("head", "value", "route", "contact", "s_contact")})
        self.times, self.packets = [], []
        self.nonfinite = False

    def advance_empty_time(self, elapsed=1.0):
        self.times.append(elapsed)
        self.lifetime["advanced_elapsed"] += elapsed

    def phase(self, raw_tokens, report_every):
        self.packets.append(list(map(int, raw_tokens)))
        count = 0
        for raw in raw_tokens:
            if int(raw) >= 4:
                count += 1
        self.lifetime["scored_lifetime"] += count
        for name in self.lifetime["update_calls"]:
            if self.profile[name + "_step"] > 0:
                self.lifetime["update_calls"][name] += count
        return {"scored_bytes": count, "mean_nll_nats": float("nan") if self.nonfinite else 2.0,
                "diagnostic_totals": [count * 0.2, count * 0.1, count, count * 2, count * 0.3,
                                      count * 0.4 if self.profile["s_contact_step"] else 0.0]}

    def state(self):
        return SimpleNamespace(**{name: value.clone() for name, value in self.buffers.items()})

    def checkpoint_state(self):
        return {"model_config": {"d_model": 2}, "profile": copy.deepcopy(self.profile),
                "body_state_dict": {"column_keys": self.body.column_keys.clone(),
                                    "reciprocal_contact": self.body.reciprocal_contact.clone()},
                "buffers": {name: value.clone() for name, value in self.buffers.items()},
                "lifetime": copy.deepcopy(self.lifetime)}

    @classmethod
    def from_checkpoint(cls, payload, *, device="cpu", s_contact_step=None):
        torch.rand(1)  # driver must restore RNG after reconstruction
        engine = cls()
        engine.profile = copy.deepcopy(payload["profile"])
        if s_contact_step is not None:
            engine.profile["s_contact_step"] = s_contact_step
        engine.buffers = {name: value.clone() for name, value in payload["buffers"].items()}
        engine.body.column_keys = payload["body_state_dict"]["column_keys"].clone()
        engine.body.reciprocal_contact = payload["body_state_dict"]["reciprocal_contact"].clone()
        engine.lifetime = copy.deepcopy(payload["lifetime"])
        return engine


def experiment_for(engine, *, warmup=4, phase_bytes=2, report_every=2):
    return {"status": "running_budget_limited", "stage": "warmup",
            "config": {"graph_block": 3, "report_every": report_every},
            "phases": living.phase_plan(warmup, phase_bytes), "phase_index": 0,
            "next_raw_cursor": 0, "common_delivered_raw": 0, "cumulative_exposure_raw": 0,
            "common_delivered_scored_bytes": 0, "common_delivered_special_tokens": 0,
            "cumulative_exposure_scored_bytes": 0, "cumulative_exposure_special_tokens": 0,
            "best": None, "provenance": {}, "model_source": {}, "parameter_numbers": 1,
            "windows": {"warmup": living.new_window(0, engine.checkpoint_state())}}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_phase_plan_exact_raw_budget_and_replay():
    phases = living.phase_plan()
    assert [(p["split"], p["start"], p["stop"]) for p in phases] == [
        ("train", 0, 100_000), ("train", 100_000, 200_000),
        ("validation", 0, 64_000), ("train", 100_000, 164_000),
        ("train", 200_000, 264_000)]
    assert sum(p["stop"] - p["start"] for p in phases) == 392_000
    assert 100_000 + 2 * sum(p["stop"] - p["start"] for p in phases[1:]) == 684_000
    assert [p["name"] for p in phases if p["replay"]] == ["A_revisit"]
    with pytest.raises(ValueError):
        living.phase_plan(3, 4)


def test_clock_birth_monotonic_and_once_only_resume_gap(monkeypatch):
    mono = iter([100, 500_000_100, 900_000_100, 1_100_000_100])
    utc = iter([1_000_000_000, 1_500_000_000, 8_000_000_000, 8_200_000_000])
    monkeypatch.setattr(living.time, "perf_counter_ns", lambda: next(mono))
    monkeypatch.setattr(living.time, "time_ns", lambda: next(utc))
    clock = living.SourceClock()
    assert clock.next_interval()["elapsed_seconds"] == 0
    assert clock.next_interval()["elapsed_seconds"] == 0.5
    restored = living.SourceClock(clock.last_arrival_utc_ns)
    first = restored.next_interval()
    assert first["elapsed_seconds"] == 6.5
    assert first["clock_basis"] == "resume_utc_gap_once"
    assert restored.next_interval()["elapsed_seconds"] == 0.2


def test_clock_rejects_negative_offline_gap(monkeypatch):
    monkeypatch.setattr(living.time, "perf_counter_ns", lambda: 1)
    monkeypatch.setattr(living.time, "time_ns", lambda: 99)
    with pytest.raises(ValueError, match="clock error"):
        living.SourceClock(100).next_interval()


def test_packet_common_dt_and_specials_preserve_causal_language_boundaries():
    on, off = MockEngine(), MockEngine(s_contact_step=0)
    results, special = living.deliver_packet({"s_on": on, "s_off": off},
                                           np.array([0, 4, 1, 259, 2, 3, 17]),
                                           {"elapsed_seconds": 0.125})
    assert special == 4
    assert on.times == off.times == [0.125]
    assert on.packets == off.packets == [[0, 4, 1, 259, 2, 3, 17]]
    assert results["s_on"]["scored_bytes"] == 3
    empty, count = living.deliver_packet({"s_on": on, "s_off": off}, np.array([0, 3]),
                                        {"elapsed_seconds": 0.0})
    assert count == 2 and empty["s_on"]["scored_bytes"] == 0
    assert len(on.packets) == len(off.packets) == 2
    assert on.times == off.times == [0.125, 0.0]
    with pytest.raises(ValueError, match="outside"):
        living.deliver_packet({"s_on": on}, np.array([260]), {"elapsed_seconds": 1.0})
    assert on.times == [0.125, 0.0]


def test_atomic_checkpoint_restore_cursor_rng_and_independent_fork(tmp_path, monkeypatch):
    monkeypatch.setattr(living, "CudaGraphValueCredit", MockEngine)
    engine = MockEngine()
    experiment = experiment_for(engine)
    experiment["next_raw_cursor"] = 3
    experiment["windows"]["warmup"]["raw_tokens"] = 3
    engine.lifetime["scored_lifetime"] = 2
    clock = living.SourceClock(12345)
    torch.manual_seed(617)
    payload = living.save_experiment(tmp_path / "latest.pt", experiment, {"warmup": engine}, clock)
    expected_next_random = torch.rand(4)
    loaded = torch.load(tmp_path / "latest.pt", weights_only=True)
    restored = living.restore_engines(loaded, "cpu")
    assert torch.equal(torch.rand(4), expected_next_random)
    assert loaded["experiment"]["next_raw_cursor"] == 3
    assert loaded["experiment"]["windows"]["warmup"]["raw_tokens"] == 3
    assert loaded["last_arrival_utc_ns"] == 12345
    assert restored["warmup"].lifetime["scored_lifetime"] == 2
    assert not (tmp_path / "latest.pt.tmp").exists()
    assert payload["engines"]["warmup"]["buffers"]["activity"].data_ptr() != engine.buffers["activity"].data_ptr()
    rng = torch.get_rng_state()
    arms = living.fork_engines(restored["warmup"], "cpu")
    assert torch.equal(torch.get_rng_state(), rng)
    assert arms["s_on"].profile["s_contact_step"] == 0.001
    assert arms["s_off"].profile["s_contact_step"] == 0
    assert {k: v for k, v in arms["s_on"].profile.items() if k != "s_contact_step"} == {
        k: v for k, v in arms["s_off"].profile.items() if k != "s_contact_step"}
    assert arms["s_on"].body.column_keys.data_ptr() != arms["s_off"].body.column_keys.data_ptr()
    for key, value in arms["s_on"].buffers.items():
        assert torch.equal(value, arms["s_off"].buffers[key])
        assert value.data_ptr() != arms["s_off"].buffers[key].data_ptr()


@pytest.mark.parametrize("name", living.LOG_NAMES)
def test_resume_rejects_uncheckpointed_log_tail_without_changing_bytes(tmp_path, name):
    engine = MockEngine()
    for log in living.LOG_NAMES:
        (tmp_path / log).write_bytes(b'{"checkpointed":true}\n')
    payload = living.save_experiment(tmp_path / "latest.pt", experiment_for(engine),
                                    {"warmup": engine}, living.SourceClock())
    living.verify_resume_logs(tmp_path, payload)
    path = tmp_path / name
    with path.open("ab") as stream:
        stream.write(b'{"actual_uncheckpointed_experience":true}\n')
    original = {log: (tmp_path / log).read_bytes() for log in living.LOG_NAMES}
    with pytest.raises(ValueError, match="cannot exactly resume"):
        living.verify_resume_logs(tmp_path, payload)
    assert {log: (tmp_path / log).read_bytes() for log in living.LOG_NAMES} == original


def test_best_excludes_replay_warmup_off_and_partial_windows():
    phases = living.phase_plan(10, 6)
    for phase in (phases[1], phases[2], phases[4]):
        assert living.best_eligible(phase, "s_on", 8, 10, 10)
    assert not living.best_eligible(phases[3], "s_on", 10, 10, 10)
    assert not living.best_eligible(phases[0], "s_on", 10, 10, 10)
    assert not living.best_eligible(phases[1], "s_off", 10, 10, 10)
    assert not living.best_eligible(phases[1], "s_on", 0, 10, 10)
    assert not living.best_eligible(phases[1], "s_on", 6, 6, 10)


def test_observation_and_diagnostics_have_no_extra_events():
    engine = MockEngine(s_contact_step=0)
    before = engine.checkpoint_state()
    observed = living.observe(engine)
    assert set(observed["wood_candidate"]["H"]) == {"Min", "Median", "Mean", "Max"}
    assert len(engine.times) == len(engine.packets) == 0
    for key, tensor in before["buffers"].items():
        assert torch.equal(tensor, engine.buffers[key])
    diag = living.diagnostic_report([2, 3, 4, 8, 5, 0], engine.profile)
    assert diag["S_actual_admitted_mass"] == 2 and diag["S_actual_released_mass"] == 3
    assert diag["H_live_response_ratio"] == 0.5
    assert diag["S_return_lambda_squared_norm_sum"] is None
    assert diag["S_conserved_transfer_squared_norm_sum"] is None
    assert diag["S_conserved_transfer_collection"] == "disabled_not_collected"


def test_slow_parameter_observation_records_actual_net_change_not_call_count():
    engine = MockEngine()
    window = living.new_window(0, engine.checkpoint_state())
    engine.body.reciprocal_contact[4, 0] = 0.125
    report = living.window_report(living.phase_plan(4, 2)[0], "warmup", window, 0,
                                  engine.checkpoint_state(), living.observe(engine), False)
    observed = report["five_phase_observations"]["metal_candidate"]["S_slow_parameter_window_net_change"]
    assert observed["absolute_change"]["Max"] == 0.125
    assert observed["nonzero_entries"] == 1
    assert report["enabled_update_call_opportunities_window"]["s_contact"] == 0
    assert "s_contact_start" not in report
    json.dumps(report, allow_nan=False)


def test_protocol_common_cursor_reports_and_full_paired_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(living, "CudaGraphValueCredit", MockEngine)
    engine = MockEngine()
    experiment = experiment_for(engine)
    sources = {"train": np.array([4, 0, 5, 6, 7, 8, 1, 9, 10, 11]),
               "validation": np.array([2, 12])}
    living.run_protocol(experiment, {"warmup": engine}, sources, tmp_path, living.SourceClock(), "cpu")
    capsys.readouterr()
    events = read_jsonl(tmp_path / "events.jsonl")
    assert [(e["phase"], e["raw_start"], e["raw_stop"]) for e in events] == [
        ("warmup", 0, 2), ("warmup", 2, 4), ("A_firstpass", 4, 6),
        ("A_firstpass", 6, 8), ("B_firstpass", 0, 2), ("A_revisit", 4, 6), ("A_fresh", 8, 10)]
    assert all(e["raw_tokens"] <= 3 for e in events)
    assert experiment["common_delivered_raw"] == 14
    assert experiment["cumulative_exposure_raw"] == 24
    payload = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert set(payload["engines"]) == {"s_on", "s_off"}
    assert payload["experiment"]["phase_index"] == 5
    assert payload["experiment"]["next_raw_cursor"] == 10
    assert payload["engines"]["s_on"]["lifetime"]["advanced_elapsed"] == payload["engines"]["s_off"]["lifetime"]["advanced_elapsed"]
    assert payload["engines"]["s_off"]["profile"]["record_s_return"] is True
    windows = read_jsonl(tmp_path / "windows.jsonl")
    # Every lexical arrival is scored, then assimilated; markers advance time.
    assert all(w["raw_tokens"] == w["special_tokens"] + w["scored_bytes"] for w in windows)
    assert all("Min" in w["five_phase_observations"]["earth_candidate"]["precision"] for w in windows)
    assert sum(w["raw_tokens"] for w in windows) == 24
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert best["experiment"]["best"]["phase"] == "A_firstpass"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed_budget_limited"
    assert summary["phases"]["B_firstpass/s_on"]["scored_bytes"] == 1
    assert summary["phases"]["A_firstpass/s_on"]["window_mean_nll_distribution"] == {
        "Min": 2.0, "Median": 2.0, "Mean": 2.0, "Max": 2.0}
    assert summary["paired_differences"]["A_firstpass"]["scored_bytes_match"]
    assert summary["paired_differences"]["A_firstpass"]["s_on_minus_s_off_mean_nll_nats"] == 0
    assert summary["paired_differences"]["A_revisit"]["replay"] is True


def test_resume_next_packet_does_not_repeat_consumed_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(living, "CudaGraphValueCredit", MockEngine)
    engine = MockEngine()
    experiment = experiment_for(engine)
    first = np.array([4, 0])
    result, special = living.deliver_packet({"warmup": engine}, first, {"elapsed_seconds": 0.0})
    living.accumulate(experiment["windows"]["warmup"], result["warmup"], 2, special, 0)
    experiment.update(next_raw_cursor=2, common_delivered_raw=2, cumulative_exposure_raw=2,
                      common_delivered_scored_bytes=1, cumulative_exposure_scored_bytes=1,
                      common_delivered_special_tokens=1, cumulative_exposure_special_tokens=1)
    # This snapshot is in a partially filled report window, not at a phase boundary.
    experiment["config"]["report_every"] = 4
    payload = living.save_experiment(tmp_path / "latest.pt", experiment, {"warmup": engine}, living.SourceClock(100))
    restored = living.restore_engines(payload, "cpu")
    sources = {"train": np.arange(4, 14), "validation": np.array([20, 21])}
    living.run_protocol(payload["experiment"], restored, sources, tmp_path, living.SourceClock(), "cpu")
    capsys.readouterr()
    events = read_jsonl(tmp_path / "events.jsonl")
    assert events[0]["raw_start"] == 2 and events[0]["raw_stop"] == 4
    assert restored["warmup"].packets == [[6, 7]]
    windows = read_jsonl(tmp_path / "windows.jsonl")
    assert windows[0]["raw_start"] == 0 and windows[0]["raw_stop"] == 4
    assert windows[0]["raw_tokens"] == 4 and windows[0]["scored_bytes"] == 3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_runner_driver_interface_small_tensors(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(17)
    body = living.CGBColumnLanguageBody(4, (2, 2), active_columns=1, contact_rank=1).to(device).eval()
    engine = living.CudaGraphValueCredit(body, body.initial_state(1), 0.01, 0.001, 0.001,
                                         0.001, 2, s_contact_step=0.001,
                                         record_s_return=True, collect_diagnostics=True)
    results, special = living.deliver_packet({"warmup": engine}, np.array([4, 0, 5]),
                                             {"elapsed_seconds": 0.125})
    assert results["warmup"]["scored_bytes"] == 2 and special == 1
    assert len(results["warmup"]["diagnostic_totals"]) == 6
    before = engine.checkpoint_state()["lifetime"]
    living.observe(engine)
    assert engine.checkpoint_state()["lifetime"] == before
    experiment = experiment_for(engine)
    experiment["next_raw_cursor"] = 3
    payload = living.save_experiment(tmp_path / "latest.pt", experiment, {"warmup": engine},
                                    living.SourceClock(100))
    loaded = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=True)
    restored = living.restore_engines(loaded, device)["warmup"]
    assert restored.checkpoint_state()["lifetime"] == payload["engines"]["warmup"]["lifetime"]
    arms = living.fork_engines(restored, device)
    results, special = living.deliver_packet(arms, np.array([1, 6]), {"elapsed_seconds": 0.25})
    assert special == 1 and all(result["scored_bytes"] == 1 for result in results.values())
    assert all(arm.checkpoint_state()["lifetime"]["advanced_elapsed"] == 2.375 for arm in arms.values())
    assert all(arm.checkpoint_state()["profile"]["record_s_return"] for arm in arms.values())


def test_nonfinite_saves_actual_both_arm_failure_cursor_without_rollback(tmp_path, capsys):
    on, off = MockEngine(), MockEngine(s_contact_step=0)
    experiment = experiment_for(on)
    experiment["phase_index"], experiment["next_raw_cursor"] = 1, 4
    experiment["windows"] = {name: living.new_window(4, engine.checkpoint_state())
                             for name, engine in {"s_on": on, "s_off": off}.items()}
    on.nonfinite = True
    with pytest.raises(FloatingPointError, match="no rollback"):
        living.run_protocol(experiment, {"s_on": on, "s_off": off},
                            {"train": np.arange(4, 14)}, tmp_path, living.SourceClock(), "cpu")
    capsys.readouterr()
    failed = torch.load(tmp_path / "failure.pt", weights_only=True)
    assert failed["experiment"]["status"] == "failed_nonfinite"
    assert failed["experiment"]["next_raw_cursor"] == 6
    assert on.packets == off.packets == [[8, 9]]
    assert set(failed["engines"]) == {"s_on", "s_off"}
    assert not (tmp_path / "best.pt").exists()
