"""Protocol tests with deterministic arithmetic; no capability claim."""
import copy

import pytest
import torch

from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream


class ProtocolBackend:
    device = torch.device('cpu')

    def __init__(self):
        self.state, self.weight = 0, 0
        self.arrivals, self.commits = [], []

    def boundary(self):
        return self.state

    def advance(self, tokens, *, clock_offset):
        rows = []
        for token in tokens:
            self.arrivals.append(token)
            self.state = (self.state * 7 + token + 1) % 101
            row = torch.zeros(11)
            row[(self.state + self.weight) % 11] = 2
            rows.append(row)
        return torch.stack(rows)

    def commit(self, inputs, targets, mask, boundary):
        self.commits.append(copy.deepcopy((inputs, targets, mask, boundary)))
        self.weight += int(any(mask))
        return {'ce': 1.} if any(mask) else {'observed_only': True}


def test_generated_tokens_and_eos_are_irreversibly_consumed():
    backend = ProtocolBackend()
    stream = IrreversibleCreditStream(backend)
    stream.receive([1, 2, 3], phase='world')
    stream.next_logits = torch.zeros(11)
    stream.next_logits[7] = 3
    generated = stream.generate(eos_token_id=7)
    assert generated == [7]
    assert backend.arrivals == [1, 2, 3, 7]
    assert stream.inputs[-1] == 7
    stream.receive([9, 8], phase='feedback', teach=True)
    assert backend.arrivals == [1, 2, 3, 7, 9, 8]
    assert stream.learn_mask == [False, False, False, True, True]
    assert stream.targets == [2, 3, 7, 9, 8]
    with pytest.raises(ValueError, match='independent teaching'):
        stream.receive([2], phase='self', teach=True)


def test_update_waits_for_real_next_token_and_keeps_cross_phase_boundary():
    backend = ProtocolBackend()
    stream = IrreversibleCreditStream(backend)
    stream.receive([1] * 128, phase='world')
    assert len(backend.commits) == 0
    assert len(stream.targets) == 127
    stream.receive([4], phase='feedback', teach=True)
    assert len(backend.commits) == 1
    inputs, targets, mask, _ = backend.commits[0]
    assert targets[-1] == 4 and sum(mask) == 1
    assert inputs == [1] * 128
    assert backend.arrivals == [1] * 128 + [4]
    assert stream.inputs == [4] and not stream.targets


def test_continuation_restores_pending_self_history_and_next_update():
    one = ProtocolBackend()
    first = IrreversibleCreditStream(one)
    first.receive([3] * 123, phase='world')
    first.receive([7, 2], phase='self')
    saved = copy.deepcopy(first.state_dict())
    two = copy.deepcopy(one)
    restored = IrreversibleCreditStream(two)
    restored.restore_latest(saved)
    tail = [5] * 140
    first.receive(tail, phase='feedback', teach=True)
    restored.receive(tail, phase='feedback', teach=True)
    assert first.chain == restored.chain
    assert one.arrivals == two.arrivals
    assert one.commits == two.commits
    assert first.inputs == restored.inputs
    assert first.targets == restored.targets
    assert one.state == two.state and one.weight == two.weight
    torch.testing.assert_close(first.next_logits, restored.next_logits, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='cannot rewind'):
        first.restore_latest(saved)


def test_fixed_pending_size_and_no_synthetic_targets_or_padding():
    backend = ProtocolBackend()
    stream = IrreversibleCreditStream(backend)
    tokens = [i % 11 for i in range(777)]
    stream.receive(tokens, phase='feedback', teach=True)
    assert backend.arrivals == tokens
    assert stream.events == len(tokens)
    assert stream.windows == 6
    assert len(stream.inputs) == 9 and len(stream.targets) == 8
    for i, (inputs, targets, mask, _) in enumerate(backend.commits):
        assert inputs == tokens[128*i:128*(i+1)]
        assert targets == tokens[128*i+1:128*(i+1)+1]
        assert all(mask)


def test_document_boundary_is_one_untrained_input_after_real_feedback():
    from scripts.ib_local.continue_irreversible_feedback import hear_feedback

    backend = ProtocolBackend()
    stream = IrreversibleCreditStream(backend)
    stream.receive([3] * 120, phase='world')
    original_state = backend.state
    hear_feedback(stream, [4, 5, 6], boundary_token=7)
    assert backend.arrivals == [3] * 120 + [4, 5, 6, 7]
    assert stream.events == 124 and not backend.commits
    assert stream.targets[-4:] == [4, 5, 6, 7]
    assert stream.learn_mask[-4:] == [True, True, True, False]
    expected_state = original_state
    for token in (4, 5, 6, 7):
        expected_state = (expected_state * 7 + token + 1) % 101
    assert backend.state == expected_state
    assert stream.phase_counts['world_marker'] == 1
