"""Pure driver/data bookkeeping checks; no model, GPU or learning execution."""
from collections import Counter

import pytest

from scripts.ib_local.run_native_s_credit_retention import (
    MAX_BYTES, PHASE_TOKENS, arrival_positions, check_native, distribution,
    phase_for_window, receive_slice, summarize, traffic_segments, untimed,
)


class CountingStream:
    """Only next-target/input counters, not an alternative learner."""
    def __init__(self):
        self.inputs = list(range(69))
        self.targets = list(range(68))
        self.events = 47941
        self.arrivals = []
        self.commits = []

    def receive(self, tokens, *, phase, teach):
        assert teach is True
        for token in tokens:
            if self.inputs:
                self.targets.append(token)
                if len(self.targets) == 128:
                    assert len(self.inputs) == 128
                    self.commits.append((self.inputs, self.targets))
                    self.inputs, self.targets = [], []
            self.inputs.append(token)
            self.events += 1
            self.arrivals.append((phase, token))


def test_actual_bridge_and_96_windows_leave_real_one_token_partial():
    a, b = list(range(PHASE_TOKENS)), list(range(10000, 10000+PHASE_TOKENS))
    tokens = a[:59] + a + b + a + b[:1]
    segments = traffic_segments(59)
    stream = CountingStream()
    receive_slice(stream, tokens, segments, 0, 59)
    assert len(stream.inputs) == 128
    assert len(stream.commits) == 0  # No flush or padding.
    receive_slice(stream, tokens, segments, 59, 60)
    assert len(stream.commits) == 1
    cursor = 60
    for formal in range(1, 97):
        receive_slice(stream, tokens, segments, cursor, cursor+128)
        cursor += 128
        assert len(stream.commits) == formal+1
    assert cursor == len(tokens)
    assert stream.inputs == b[:1] and stream.targets == []
    assert stream.events-47941 == 59+3*4096+1
    assert [t for _, t in stream.arrivals] == tokens
    assert Counter(p for p, _ in stream.arrivals) == {
        'bridge': 59, 'A_first': 4096, 'B': 4096,
        'A_return': 4096, 'B_continuation': 1}
    assert stream.commits[32][0] == a[-128:]
    assert stream.commits[32][1][-1] == b[0]
    assert stream.commits[64][1][-1] == a[0]
    assert stream.commits[96][0] == a[-128:]
    assert stream.commits[96][1][-1] == b[0]
    assert [phase_for_window(i) for i in (0, 1, 32, 33, 64, 65, 96)] == [
        'bridge', 'A_first', 'A_first', 'B', 'B', 'A_return', 'A_return']


def test_source_positions_distinguish_reuse_and_boundary_target():
    segments = traffic_segments(59)
    assert arrival_positions(segments, 0, 59) == [dict(phase='bridge', block='a', raw_array_start=8000000,
        model_token_start=4096, model_token_stop=4155)]
    positions = arrival_positions(segments, 59+4096-127, 59+4096+1)
    assert positions == [
        dict(phase='A_first', block='a', raw_array_start=8000000,
             model_token_start=3969, model_token_stop=4096),
        dict(phase='B', block='b', raw_array_start=10000000,
             model_token_start=0, model_token_stop=1)]
    for index in (-1, 128):
        with pytest.raises(ValueError):
            traffic_segments(index)


def native(**changes):
    values = dict(teaching_available=True, has_factual_trace=True,
                  trace_empty_before=False, injected_gradient_norm=0.5,
                  persistent_bytes=1024, estimated_temporary_bytes=2048)
    values.update(changes)
    return dict(native_s=values)


def test_bridge_exception_is_not_a_silent_fresh_teacher_skip():
    empty = native(teaching_available=False, has_factual_trace=False,
                   trace_empty_before=True, injected_gradient_norm=0.)
    check_native(empty, bridge=True)
    with pytest.raises(RuntimeError):
        check_native(empty)
    with pytest.raises(RuntimeError):
        check_native({})
    check_native(native(trace_empty_before=True, injected_gradient_norm=0.), empty=True)
    with pytest.raises(AssertionError):
        check_native(native(trace_empty_before=True), empty=True)
    with pytest.raises(MemoryError):
        check_native(native(persistent_bytes=MAX_BYTES))


def test_primary_retention_and_later_relearning_are_separate():
    windows = [dict(phase='bridge', lived_preupdate_ce=999.)]
    for phase in ('A_first', 'B', 'A_return'):
        windows.extend(dict(phase=phase, lived_preupdate_ce=float(i)) for i in range(32))
    summary = summarize(windows)
    assert summary['return_first4']['mean'] == 1.5
    assert summary['return_later28']['mean'] == 17.5
    assert summary['return_last4']['mean'] == 29.5
    assert summary['phases']['A_first']['windows'] == 32
    assert summary['later_minus_return_first4'] == 16.
    assert distribution([3., 1., 2.]) == dict(min=1., median=2., mean=2., max=3., windows=3)


def test_restore_metric_filter_preserves_causal_cursors():
    assert untimed(dict(elapsed=1., timing_gpu=2., nested=dict(seconds=3., seen=7),
                        cuda_allocator_before_backward=dict(reserved=9),
                        events=4, old_credit_norm=0.1)) == dict(
        nested=dict(seen=7), events=4, old_credit_norm=0.1)
    assert untimed({1: dict(elapsed=2., seen=7), 12: dict(credit=.1)}) == {
        1: dict(seen=7), 12: dict(credit=.1)}


def test_resource_policies_are_explicit_and_checkpoint_values_match(tmp_path):
    from types import SimpleNamespace
    import torch
    from scripts.ib_local.run_native_s_credit_retention import configuration, load_checkpoint
    old = SimpleNamespace(model='model', vendor='vendor')
    selected = SimpleNamespace(model='model', vendor='vendor',
                               readonly_checkpoints=True, bounded_autotune=True)
    assert configuration(old)['readonly_checkpoints'] is False
    assert configuration(old)['bounded_autotune'] is False
    assert configuration(selected)['readonly_checkpoints'] is True
    assert configuration(selected)['bounded_autotune'] is True
    assert configuration(selected)['max_owned_archive_bytes'] == 6 * 2**30
    selected.archive_budget_gib = 10
    assert configuration(selected)['max_owned_archive_bytes'] == 10 * 2**30
    path = tmp_path/'candidate.pt'
    torch.save({'updates': 524, 'S': torch.arange(12).reshape(3, 4)}, path)
    a, b = load_checkpoint(path, old), load_checkpoint(path, selected)
    assert a['updates'] == b['updates'] == 524 and torch.equal(a['S'], b['S'])


def test_snapshot_budget_checks_existing_archives_before_writing(tmp_path):
    from types import SimpleNamespace
    from scripts.ib_local.run_native_s_credit_retention import snapshot, SOURCE_BYTES
    directory = tmp_path / 'raw'
    directory.mkdir()
    (tmp_path / 'curriculum.pt').write_bytes(b'fixed curriculum')
    learner = SimpleNamespace(state_dict=lambda: {})
    backend = SimpleNamespace(close=lambda: None)
    stream = SimpleNamespace(state_dict=lambda: {})
    path = directory / 'checkpoint_000032.pt'
    with pytest.raises(OSError, match='archive budget'):
        snapshot(path, learner, backend, stream, {'next_case': 14}, {},
                 archive_budget=SOURCE_BYTES)
    assert not path.exists() and not path.with_suffix('.tmp.pt').exists()
