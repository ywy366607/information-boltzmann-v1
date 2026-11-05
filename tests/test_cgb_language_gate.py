import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("cgb_language_gate", ROOT / "scripts" / "cgb_language_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
from cgb_language_candidates import CANDIDATES  # noqa: E402

THRESHOLDS = gate.load_obligations()["language_thresholds"]
REFERENCE = {"order0": 3.2, "order1": 2.5, "novel": np.array([True, False]),
             "table_nll": np.array([2.0, 1.0], dtype=np.float32)}


def test_every_obligation_has_a_source_and_a_known_check_kind():
    data = gate.load_obligations()
    ids = [row["id"] for row in data["obligations"]]
    assert len(ids) == len(set(ids))
    for row in data["obligations"]:
        assert row["theorems"] or row.get("source")
        assert row["check"]["kind"] in data["check_kinds"]


def test_missing_body_manifest_cannot_pass_a_lean_obligation():
    assert gate.lean_check(None, "column_refinement_instance", {"anything"})[0] == gate.FAIL


def test_declared_lean_name_must_be_in_the_inventory():
    manifest = {"lean": {"order_separation": "CGB.Body.order"}}
    assert gate.lean_check(manifest, "order_separation", set())[0] == gate.FAIL
    assert gate.lean_check(manifest, "order_separation", {"CGB.Body.order"})[0] == gate.PASS


def test_every_behaviour_check_refuses_missing_evidence():
    data = gate.load_obligations()
    ids = [row["check"]["id"] for row in data["obligations"] if row["check"]["kind"] == "language"]
    for check_id in ids + ["one_shot_gain", "state_repeatable", "finite_state"]:
        assert gate.judge(check_id, {}, THRESHOLDS, REFERENCE, 1024)[0] == gate.NOT_RUN


def test_revisit_threshold_is_enforced():
    evidence = lambda again: {"phases": {"A_firstpass": 2.0, "A_revisit": again}}
    assert gate.judge("revisit_drop", evidence(1.8), THRESHOLDS, None, None)[0] == gate.FAIL
    assert gate.judge("revisit_drop", evidence(1.6), THRESHOLDS, None, None)[0] == gate.PASS


def test_adaptation_needs_a_gain_on_the_same_held_out_bytes():
    evidence = lambda after: {"phases": {"A_firstpass": 2.0, "A_revisit": 2.0},
                              "switch": {"before": 2.5, "after": after}}
    assert gate.judge("drift_recovery", evidence(2.49), THRESHOLDS, None, None)[0] == gate.FAIL
    assert gate.judge("drift_recovery", evidence(2.45), THRESHOLDS, None, None)[0] == gate.PASS


def test_generalization_compares_only_novel_context_bytes():
    better = {"fresh_nll": np.array([1.9, 9.0], dtype=np.float32)}
    worse = {"fresh_nll": np.array([2.1, 0.1], dtype=np.float32)}
    assert gate.judge("generalizes", better, THRESHOLDS, REFERENCE, None)[0] == gate.PASS
    assert gate.judge("generalizes", worse, THRESHOLDS, REFERENCE, None)[0] == gate.FAIL


def test_last_byte_only_prediction_cannot_pass_by_repetition():
    row = {"repeatable": {"same_eight": 0.0, "last_byte_only": 0.0}}
    assert gate.judge("state_repeatable", row, THRESHOLDS, None, None)[0] == gate.FAIL


def test_dominance_uses_real_region_shares():
    uneven = torch.tensor([0.9] + [0.1 / 63] * 63)
    even = torch.full((64,), 1 / 64)
    assert gate.judge("no_dominance", {"shares": uneven}, THRESHOLDS, None, None)[0] == gate.FAIL
    assert gate.judge("no_dominance", {"shares": even}, THRESHOLDS, None, None)[0] == gate.PASS


def test_reference_only_candidate_is_never_admitted():
    candidate = CANDIDATES["context_table"](64)
    report = gate.evaluate("context_table", {}, candidate, 64, REFERENCE,
                           set(), {"fresh": True, "trusted": True})
    assert report["admitted"] is False and report["reference_only"] is True
    assert not any(row["status"] == gate.PASS and row["system"] == "4" for row in report["rows"])


def test_context_cells_event_touches_only_its_address():
    """The executable side of `GenerationContextBody`: writes stay in `region`."""
    candidate = CANDIDATES["context_cells"](64)
    text = gate.stream("train", 0, 2000)
    for byte in text[:600]:
        candidate.observe(byte)
    for byte in text[600:700]:
        region = {index ^ flip for index, _, _ in candidate.lookups() for flip in (0, 1)}
        before = list(candidate.slots)
        snapshot = [None if slot is None else (slot[0], dict(slot[1])) for slot in before]
        candidate.observe(byte)
        changed = {index for index, slot in enumerate(candidate.slots)
                   if (None if slot is None else (slot[0], dict(slot[1]))) != snapshot[index]}
        assert changed <= region
        assert len(region) <= 2 * len(candidate.orders)


def test_context_cells_never_exceed_their_worst_case():
    """A size read once mid-stream passed a body that kept growing."""
    candidate = CANDIDATES["context_cells"](64)
    assert candidate.capacity_bytes() <= 64 * 1024
    for index, byte in enumerate(gate.stream("train", 0, 60_000)):
        candidate.observe(byte)
        if index % 5000 == 0:
            assert candidate.state_bytes() <= candidate.capacity_bytes()
    live = [slot for slot in candidate.slots if slot is not None]
    assert candidate.held == sum(len(slot[1]) for slot in live)
    assert candidate.held >= candidate.limit - 8


def test_context_cells_tick_settles_lazily():
    candidate = CANDIDATES["context_cells"](64)
    for byte in gate.stream("train", 0, 800):
        candidate.observe(byte)
    before = [None if slot is None else dict(slot[1]) for slot in candidate.slots]
    prediction = candidate.predict()
    candidate.advance(500.0)
    assert before == [None if slot is None else dict(slot[1]) for slot in candidate.slots]
    assert not torch.equal(prediction, candidate.predict())


def test_candidates_are_causal_and_replay_exactly():
    text = gate.stream("train", 0, 400)
    for name, build in CANDIDATES.items():
        candidate = build(64)
        for byte in text[:32]:
            assert candidate.predict().shape == (256,)
            assert abs(candidate.log_prob(byte) - float(candidate.predict()[byte])) < 1e-3
            candidate.observe(byte)
        assert gate.check_causal(candidate)[0] == gate.PASS, name
        assert gate.check_resume(candidate, text[32:])[0] == gate.PASS, name
