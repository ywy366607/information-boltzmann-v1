"""Offline telemetry and compatibility-launcher contracts; no API or training."""
import dataclasses
import json
from pathlib import Path

import pytest

from scripts.ib_local.mengli_harness.telemetry import MengLiVitalSigns
from scripts.ib_local.mengli_harness import server


HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "ib_local" / "mengli_harness"


def test_missing_measurements_are_null():
    assert all(value is None for value in MengLiVitalSigns().to_dict().values())
    assert MengLiVitalSigns.from_metrics_dict({}).to_dict() == MengLiVitalSigns().to_dict()


def test_flat_round_trip_preserves_zero_false_and_empty_text():
    data = dict(cycle=0, choice="silence", child_text="", spoke=False,
                metal_g=0, provider_hit_rate=0, latency_seconds=0,
                layer_gradient_norms=[0, None, 2, 0], metadata={"arm": "free"})
    vitals = MengLiVitalSigns.from_dict(data)
    assert {key: vitals.to_dict()[key] for key in data} == data
    assert MengLiVitalSigns.from_dict(vitals.to_dict()) == vitals


def test_flat_parser_does_not_project_nested_training_record():
    vitals = MengLiVitalSigns.from_dict({
        "world": {"child_text": "nested"},
        "causal_program": {"active_relations": 3},
        "child_text": "flat", "active_relations": 0,
    })
    assert vitals.child_text == "flat"
    assert vitals.active_relations == 0


def test_nested_parser_does_not_treat_flat_api_fields_as_measurements():
    vitals = MengLiVitalSigns.from_metrics_dict({
        "child_text": "flat", "provider_hit_rate": 0.4,
        "active_relations": 4, "layer_gradient_norms": [3],
    })
    assert vitals.child_text is None
    assert vitals.provider_hit_rate is None
    assert vitals.active_relations is None
    assert vitals.layer_gradient_norms is None


@pytest.mark.parametrize("usage", [None, {}])
def test_api_usage_none_or_missing_is_unmeasured(usage):
    vitals = MengLiVitalSigns.from_metrics_dict({"world": {"mother_harness": {"api_usage": usage}}})
    assert vitals.prompt_tokens is None
    assert vitals.cache_hit_tokens is None
    assert vitals.completion_tokens is None


def test_nested_zero_values_and_usage_are_not_replaced():
    vitals = MengLiVitalSigns.from_metrics_dict({
        "cycle": 0, "choice": "silence", "spoke": False, "metal_g": 0,
        "world": {"child_text": "", "mother_reply": "reply", "mother_harness": {
            "turn": 0, "rollover": False, "latency_seconds": 0, "provider_hit_rate": 0,
            "api_usage": {"prompt_tokens": 10, "cache_hit_tokens": 0,
                          "cache_miss_tokens": 10, "completion_tokens": 0},
            "reply_usage": {"input": 10, "output": 0},
            "summary_usage": {"input": 0, "output": 0},
        }},
        "causal_program": {"active_relations": 0, "evidence_count": 0,
                           "mother_prequential": {"mse": 0}, "question_norm": 0},
        "learned_head_coupling": {"per_layer_gradient_norm": [0, 1, 2, 0]},
    })
    assert vitals.cycle == 0
    assert vitals.spoke is False
    assert vitals.child_text == ""
    assert vitals.metal_g == 0
    assert vitals.turn_in_segment == 0
    assert vitals.rollover_occurred is False
    assert vitals.latency_seconds == 0
    assert vitals.provider_hit_rate == 0
    assert vitals.prompt_tokens == 10  # Cache hits are already part of total input.
    assert vitals.cache_hit_tokens == 0
    assert vitals.completion_tokens == 0
    assert vitals.active_relations == 0
    assert vitals.relation_capacity is None
    assert vitals.current_evidence_count == 0
    assert vitals.mother_prequential_mse == 0
    assert vitals.question_norm == 0
    assert vitals.layer_gradient_norms == [0, 1, 2, 0]
    assert vitals.reply_usage == {"input": 10, "output": 0}
    assert vitals.summary_usage == {"input": 0, "output": 0}


def test_gradients_and_question_update_are_not_physical_norms():
    vitals = MengLiVitalSigns.from_metrics_dict({
        "learned_head_coupling": {"per_layer_gradient_norm": [3, 4]},
        "causal_program": {"question_weight_step": 5},
    })
    assert vitals.layer_gradient_norms == [3, 4]
    assert vitals.s_state_norm is None
    assert vitals.p_momentum_norm is None
    assert vitals.q_invariant_norm is None
    assert vitals.question_norm is None
    assert "layer_momentums" not in vitals.to_dict()


def test_explicit_physical_norms_are_preserved():
    vitals = MengLiVitalSigns.from_metrics_dict({
        "s_state_norm": 0, "p_momentum_norm": 2, "q_invariant_norm": None,
    })
    assert vitals.s_state_norm == 0
    assert vitals.p_momentum_norm == 2
    assert vitals.q_invariant_norm is None


def test_schema_matches_dataclass_and_accepts_null():
    schema = json.loads((HARNESS / "schema.json").read_text(encoding="utf-8"))
    assert set(schema["properties"]) == {item.name for item in dataclasses.fields(MengLiVitalSigns)}
    for prop in schema["properties"].values():
        if "$ref" not in prop:
            assert "null" in prop["type"]
    assert "maximum" not in schema["properties"]["active_relations"]


def test_python_entry_point_only_forwards_to_node(monkeypatch):
    calls = []
    monkeypatch.setattr(server.subprocess, "call", lambda args: calls.append(args) or 7)
    assert server.main(["--port", "8900", "--arm", "free"]) == 7
    assert calls == [["node", str(HARNESS / "mengli_service.mjs"), "--port", "8900", "--arm", "free"]]
    calls.clear()
    assert server.run_server(8910) == 7
    assert calls == [["node", str(HARNESS / "mengli_service.mjs"), "--port", "8910"]]


def test_legacy_entries_have_no_external_writes_or_provider_calls():
    for name in ("launch_dsh_web.js", "run_dsh_mengli_web.ps1", "server.py",
                 "setup_mengli_dsh_home.js", "inspect_sessions.mjs", "read_full_session.mjs"):
        text = (HARNESS / name).read_text(encoding="utf-8")
        for removed in ("F:/", "F:\\", "copyFileSync", "writeFileSync", ".credentials", ".env",
                        "Access-Control-Allow-Origin", "api.deepseek.com", "ToddlerMother"):
            assert removed not in text, (name, removed)
    assert "process.argv.slice(2)" in (HARNESS / "launch_dsh_web.js").read_text(encoding="utf-8")
    assert "@args" in (HARNESS / "run_dsh_mengli_web.ps1").read_text(encoding="utf-8")
