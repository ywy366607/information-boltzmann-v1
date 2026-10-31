"""Unit tests for the CGB small lifeform experimental harness and 4 paired controls."""

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))

from cgb_experimental_harness import (
    EmpiricalBaselines,
    format_stats,
    run_four_paired_controls,
    run_trial,
)
from cgb_prequential_bridge import PackedTreePort
from cgb_low_resource_life import PrefixReplayWorld


def fixture_rule(code, hidden, bit, context, depth, prefix):
    auxiliary = context[-1] if len(context) > 1 else 0
    return ((code + 9 if hidden else 1) * (3 if hidden == bit else 1)
            * (2 if hidden == ((context[0] >> depth) & 1) else 1)
            * (3 if hidden == ((auxiliary >> depth) & 1) else 1))


def create_test_learner():
    return PackedTreePort(
        256,
        fixture_rule,
        context=(0,),
        seed=20261025,
        learning=True,
        rule_id="harness-unit-test-fixture",
    )


def test_format_stats():
    res = format_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert res["min"] == 1.0
    assert res["median"] == 3.0
    assert res["mean"] == 3.0
    assert res["max"] == 5.0
    assert "[1.0000 | 3.0000 | 3.0000 | 5.0000]" in res["display"]


def test_heldout_consumption_prepares_before_retrieving_each_event():
    from types import SimpleNamespace
    from scripts.measure_cgb_prequential_heldout import consume_prequential

    calls = []

    class Bridge:
        def prepare(self):
            calls.append("prepare")

        def step(self, obs, *, write_units, dissipation_units):
            calls.append(("observe", obs.tick, obs.payload))
            return SimpleNamespace(model_nll_nats=1.0)

    def events():
        for tick in range(2):
            calls.append(("publish", tick))
            yield SimpleNamespace(tick=tick, payload=65 + tick)

    total, last = consume_prequential(
        Bridge(), events(), lambda tick, payload: SimpleNamespace(tick=tick, payload=payload),
        2, 0, 0)
    assert calls == ["prepare", ("publish", 0), ("observe", 0, 65),
                     "prepare", ("publish", 1), ("observe", 1, 66)]
    assert total == 2.0 and last.tick == 1


def test_empirical_baselines():
    data = b"ABABABAB" * 10
    baselines = EmpiricalBaselines(data, vocab_size=256, alpha=0.5)
    u_bits_A = baselines.unigram_bits(ord("A"))
    u_bits_Z = baselines.unigram_bits(ord("Z"))
    assert u_bits_A < u_bits_Z  # 'A' is frequent, 'Z' is unseen

    b_bits_B_after_A = baselines.bigram_bits(ord("A"), ord("B"))
    b_bits_A_after_A = baselines.bigram_bits(ord("A"), ord("A"))
    assert b_bits_B_after_A < b_bits_A_after_A  # 'B' always follows 'A' in training


def test_run_four_paired_controls_contract():
    train_data = b"hello world this is a test stream for cgb lifeform " * 10
    val_data = b"hello world this is a test stream for cgb lifeform 123456" * 5
    events = 32

    results = run_four_paired_controls(
        learner_factory=create_test_learner,
        source_bytes=val_data,
        train_reference_bytes=train_data,
        events=events,
        reserve=16,
        capacity=32,
        nominal_supply=2,
    )

    assert results["events"] == events
    for key in ("nominal", "frozen_learning", "history_disrupted", "information_cut", "resource_cut"):
        assert key in results
        entry = results[key]
        assert "model_bits_per_byte" in entry
        assert "display" in entry["model_bits_per_byte"]
        assert "gain_over_unigram_bits" in entry
        assert "gain_over_bigram_bits" in entry

    # Resource cut must deplete and have lower survival than nominal
    assert results["resource_cut"]["active_ticks"] < results["nominal"]["active_ticks"]
    assert results["resource_cut"]["survival_rate"] < 1.0
    assert results["nominal"]["survival_rate"] == 1.0

    contrasts = results["pairwise_contrasts"]
    assert "plasticity_net_gain_bits_per_byte" in contrasts
    assert "resource_cut_starvation_ticks" in contrasts
    assert contrasts["resource_cut_starvation_ticks"] > 0
