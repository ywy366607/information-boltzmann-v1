"""Measured MengLi telemetry. None means unmeasured; zero is a real value.

Gradient norms are learning diagnostics, not physical S/P/Q measurements.
Flat API payloads and nested training records have separate constructors.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MengLiVitalSigns:
    cycle: Optional[int] = None
    choice: Optional[str] = None
    child_text: Optional[str] = None
    mother_reply: Optional[str] = None
    spoke: Optional[bool] = None
    metal_g: Optional[float] = None
    settlement_kind: Optional[str] = None

    q_invariant_norm: Optional[float] = None
    p_momentum_norm: Optional[float] = None
    s_state_norm: Optional[float] = None
    layer_gradient_norms: Optional[List[Optional[float]]] = None

    active_relations: Optional[int] = None
    relation_capacity: Optional[int] = None
    unknown_identity: Optional[str] = None
    current_evidence_count: Optional[int] = None
    mother_prequential_mse: Optional[float] = None
    question_norm: Optional[float] = None
    expected_gain: Optional[float] = None
    candidate_count: Optional[int] = None

    provider_hit_rate: Optional[float] = None
    latency_seconds: Optional[float] = None
    cache_hit_tokens: Optional[int] = None
    cache_miss_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    turn_in_segment: Optional[int] = None
    rollover_occurred: Optional[bool] = None
    reply_usage: Optional[Dict[str, Any]] = None
    summary_usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MengLiVitalSigns:
        """Read flat vitals without interpreting nested training structures."""
        names = {item.name for item in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})

    @classmethod
    def from_metrics_dict(cls, data: Dict[str, Any]) -> MengLiVitalSigns:
        """Project only explicitly reported measurements from a training record."""
        vitals = cls()
        for name in (
            "cycle", "choice", "spoke", "metal_g", "settlement_kind",
            "s_state_norm", "p_momentum_norm", "q_invariant_norm",
        ):
            setattr(vitals, name, data.get(name))

        world = data.get("world") or {}
        vitals.child_text = world.get("child_text")
        vitals.mother_reply = world.get("mother_reply")
        harness = world.get("mother_harness") or {}
        vitals.turn_in_segment = harness.get("turn")
        vitals.rollover_occurred = harness.get("rollover")
        vitals.latency_seconds = harness.get("latency_seconds")
        vitals.provider_hit_rate = harness.get("provider_hit_rate")
        usage = harness.get("api_usage") or {}
        for name in ("prompt_tokens", "cache_hit_tokens", "cache_miss_tokens", "completion_tokens"):
            setattr(vitals, name, usage.get(name))
        vitals.reply_usage = harness.get("reply_usage")
        vitals.summary_usage = harness.get("summary_usage")

        program = data.get("causal_program") or {}
        vitals.active_relations = program.get("active_relations")
        vitals.relation_capacity = program.get("relation_capacity")
        vitals.unknown_identity = (program.get("roles") or {}).get("unknown_identity")
        vitals.current_evidence_count = program.get("evidence_count")
        vitals.mother_prequential_mse = (program.get("mother_prequential") or {}).get("mse")
        vitals.question_norm = program.get("question_norm")
        vitals.expected_gain = program.get("expected_gain")
        vitals.candidate_count = program.get("candidate_count")

        coupling = data.get("learned_head_coupling") or {}
        vitals.layer_gradient_norms = coupling.get("per_layer_gradient_norm")
        return vitals
