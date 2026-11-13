"""Boundary/rollback tests for continuous inference, not capability studies."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.small_gdn_ruler import fingerprint, request_branch
from scripts.ib_local.small_gdn_stream_eval import CarriedStream


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(256)*.1)
        self.memory = nn.Module()
        self.memory.register_buffer('clock', torch.tensor(7))

    def initial_state(self, batch, device):
        return torch.zeros(batch, device=device)

    def predict_observed(self, ids, state, logp):
        logits = self.weight[ids]+logp
        logits[..., 0] += state[:, None]*.01
        return logits, state+ids.shape[1], {}, ids[0, :, None].float(), None

    def commit_observed(self, features, receipt):
        self.memory.clock.add_(len(features))


def engine():
    value = SimpleNamespace(model=Body(), recurrent=torch.tensor([11.]), pending=17,
        device=torch.device('cpu'), table=RetrievedCreditTable(orders=(0, 1), heads=1, slots_log2=3))
    value.table.history = np.uint64(0x616263)
    def inputs(raw, probability):
        return (torch.tensor(raw[:-1].astype(np.int64))[None],
                torch.tensor(raw[1:].astype(np.int64))[None],
                torch.tensor(np.log(np.maximum(probability, 1e-12)))[None])
    value._inputs = inputs
    return value


def test_continuation_carries_state_pending_history_and_restores_live_source():
    e = engine()
    original = fingerprint(e.model, e.table)
    with request_branch(e, reset_history=False) as table:
        assert int(table.history) == 0x616263
        reader = CarriedStream(e, table, window=2)
        assert torch.equal(reader.state, e.recurrent)
        result = reader.observe(np.array([65, 66, 67, 68], dtype=np.uint8))
        assert result['windows'][0]['first_input'] == 17
        assert result['windows'][1]['first_input'] == 66
        assert result['targets'].tolist() == [[65, 66, 67, 68]]
        assert result['final_pending'] == 68
        assert result['final_state'].item() == 15
    assert fingerprint(e.model, e.table) == original
    assert e.recurrent.item() == 11 and e.pending == 17


def test_native_reset_is_explicit_and_keeps_pending_and_lexical_context():
    e = engine()
    with request_branch(e, reset_history=False) as table:
        reader = CarriedStream(e, table, reset_native=True, window=2)
        assert reader.state.item() == 0 and reader.pending == 17
        assert int(table.history) == 0x616263


def test_legacy_request_still_resets_history_by_default():
    e = engine()
    with request_branch(e) as table:
        assert int(table.history) == 0
    assert int(e.table.history) == 0x616263


def test_two_continuations_equal_one_stream_with_same_window_boundaries():
    e = engine()
    with request_branch(e, reset_history=False) as table:
        reader = CarriedStream(e, table, window=2)
        whole = reader.observe(np.array([65, 66, 67, 68], dtype=np.uint8))
    with request_branch(e, reset_history=False) as table:
        reader = CarriedStream(e, table, window=2)
        first = reader.observe(np.array([65, 66], dtype=np.uint8))
        second = reader.observe(np.array([67, 68], dtype=np.uint8))
    assert torch.equal(whole['logits'], torch.cat((first['logits'], second['logits']), 1))
    assert torch.equal(whole['final_state'], second['final_state'])


def test_future_observations_do_not_change_earlier_window_predictions():
    e = engine()
    outputs = []
    for last in (68, 99):
        with request_branch(e, reset_history=False) as table:
            reader = CarriedStream(e, table, window=2)
            outputs.append(reader.observe(np.array([65, 66, 67, last], dtype=np.uint8)))
    assert torch.equal(outputs[0]['logits'][:, :2], outputs[1]['logits'][:, :2])


def test_answer_partial_window_does_not_commit_or_reset_once_per_byte(monkeypatch):
    from scripts.ib_local import evaluate_layer_reset_answers as module
    e = engine()
    calls = []
    def boundary(engine, state):
        calls.append(state.clone())
        return state*0
    monkeypatch.setattr(module, 'apply_native_boundary', boundary)
    with request_branch(e, reset_history=False) as table:
        stream = module.AnswerStream(e, table)
        stream.observe_prompt(b'a'*126)
        assert int(e.model.memory.clock) == 7
        assert stream.reader.state.item() == 11
        stream.predict_next()
        stream.accept(66)
        stream.predict_next()
        stream.accept(67)
        assert stream.completed_windows == 1
        assert int(e.model.memory.clock) == 135
        # Recomputed prefixes start from the SAME saved window boundary.
        assert [x.item() for x in calls] == [11, 11]
        assert stream.reader.state.item() == 128
        assert stream.raw == [67] and not stream.probabilities


def test_answer_stream_window_end_matches_bulk_observation():
    from scripts.ib_local.evaluate_layer_reset_answers import AnswerStream
    e = engine()
    sequence = np.array([65, 66, 67, 68]*32, dtype=np.uint8)
    with request_branch(e, reset_history=False) as table:
        bulk = CarriedStream(e, table).observe(sequence)
        bulk_hash = fingerprint(e.model, table)
    with request_branch(e, reset_history=False) as table:
        stream = AnswerStream(e, table)
        stream.observe_prompt(bytes(sequence[:-1]))
        logits = stream.predict_next()
        assert torch.equal(logits, bulk['logits'][0, -1])
        stream.accept(int(sequence[-1]))
        assert torch.equal(stream.reader.state.cpu(), bulk['final_state'])
        assert fingerprint(e.model, table) == bulk_hash


def test_continuing_answer_generation_cannot_read_gold_or_change_source():
    from scripts.ib_local.evaluate_layer_reset_answers import run_answer
    e = engine()
    before = fingerprint(e.model, e.table)
    row = dict(index=0, input='A number 123. Question:', outputs=['123'])
    a = run_answer(e, row, 8)
    b = run_answer(e, dict(row, outputs=['999']), 8)
    assert a['prediction'] == b['prediction']
    assert a['partial_window_arrivals'] == len(row['input'])+8
    assert fingerprint(e.model, e.table) == before
