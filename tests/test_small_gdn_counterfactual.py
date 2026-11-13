"""Layout/segmentation/probe checks; these assert no learned task capability."""
import pytest
import torch
from types import SimpleNamespace

from scripts.ib_local.small_gdn_counterfactual import (
    answer_windows, counterpart, frozen_read, near_pair, top_one_value,
)
from scripts.ib_local.small_gdn_ruler import encode_case


def official_row(number='1234567', key='functional-component'):
    return dict(index=5, input=(
        'A special magic number is hidden within the following text.\n'
        f'One of the special magic numbers for {key} is: {number}.\n'
        'The grass is green. The sky is blue.\n'
        f'What is the special magic number for {key} mentioned in the provided text?'),
        answer_prefix=' Redundant official answer prefill', outputs=[number])


def test_pairs_change_only_clue_value_and_answer():
    left, right = near_pair(official_row())
    assert left['input'].replace('1234567', right['outputs'][0]) == right['input']
    assert left['answer_prefix'] == right['answer_prefix'] == ''
    assert left['input'].endswith('functional-component?')
    assert 'The grass is green.' in left['input']
    assert left['source_index'] == right['source_index'] == 5
    assert all(a != b for a, b in zip(left['outputs'][0], right['outputs'][0]))


@pytest.mark.parametrize('number', ['1234567', '9000000', '9999999'])
def test_changed_numbers_preserve_support(number):
    value = counterpart(number)
    assert len(value) == 7 and value.isdigit() and value[0] != '0'
    assert all(a != b for a, b in zip(number, value))


@pytest.mark.parametrize('key', ['x', 'functional-component', 'a'*36])
def test_answer_aligned_window_contains_all_digits_and_every_transition_once(key):
    row = near_pair(official_row(key=key))[0]
    prefix, gold = encode_case(row)
    raw, segments = answer_windows(prefix, gold, 10)
    joined = b''.join(bytes(raw[a:b]) for a, b, _ in segments)
    assert joined == bytes(raw[:-1])
    assert all(b-a <= 128 for a, b, _ in segments)
    assert sum(int(mask.sum()) for _, _, mask in segments) == len(gold)+1
    assert sum(bool(mask.any()) for _, _, mask in segments) == 1
    start, end, mask = segments[-1]
    assert b'1234567' in bytes(raw[start:end])
    assert bytes(raw[start+1:end+1][mask]) == gold+b'\n'
    # Changing the target cannot change the real prompt processing schedule.
    other = near_pair(official_row(key=key))[1]
    p, g = encode_case(other)
    _, other_segments = answer_windows(p, g, 10)
    assert [(a,b) for a,b,_ in segments] == [(a,b) for a,b,_ in other_segments]


def test_top1_read_includes_null_and_preserves_value_order():
    values = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    weights = torch.tensor([[.1, .6, .2], [.1, .1, .1]])
    result = top_one_value(weights, values, torch.tensor([.1, .7]))
    assert torch.equal(result, torch.tensor([[3., 4.], [0., 0.]]))


def test_unrecognized_official_input_is_not_silently_modified():
    row = official_row()
    row['input'] = row['input'].replace('What is', 'Tell me')
    with pytest.raises(ValueError):
        near_pair(row)


class ProbeMemory:
    def __init__(self):
        self.replayed_by_slot = {2: torch.tensor([[[1., 2.], [3., 4.]]])}
        self.capsules = [None, None, SimpleNamespace(version=7)]
        self.position_read = dict(weights=torch.tensor([[.2, .7]]), null_weight=torch.tensor([.1]),
            record_ids=torch.tensor([0, 0]), slots=torch.tensor([2]), value_positions=torch.tensor([0, 1]))

    def retrieve_context(self, query, block):
        return torch.tensor([[2.3, 3.2]])


@pytest.mark.parametrize('mode,expected', [('top1', [3., 4.]), ('first', [1., 2.]), ('wrong', [3., 4.])])
def test_frozen_route_selects_observed_values_and_restores_method(mode, expected):
    memory = ProbeMemory()
    bank = SimpleNamespace(memories=[memory])
    metadata = [{2: dict(start=100, end=102, version=7)}]
    with frozen_read(bank, mode, metadata, [100], 0) as trace:
        assert torch.equal(memory.retrieve_context(torch.zeros(1, 2), None), torch.tensor([expected]))
        assert trace[0]['layer'] == 1
    assert 'retrieve_context' not in memory.__dict__
    assert torch.equal(memory.retrieve_context(None, None), torch.tensor([[2.3, 3.2]]))


def test_probe_restores_method_when_observed_identity_is_wrong():
    memory = ProbeMemory()
    bank = SimpleNamespace(memories=[memory])
    metadata = [{2: dict(start=100, end=102, version=8)}]
    with pytest.raises(RuntimeError):
        with frozen_read(bank, 'first', metadata, [100], 0):
            memory.retrieve_context(torch.zeros(1, 2), None)
    assert 'retrieve_context' not in memory.__dict__
