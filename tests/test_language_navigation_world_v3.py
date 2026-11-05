import numpy as np
import pytest

from scripts.ib_local.language_navigation_world_v3 import BroadLanguageWorld, SequentialResponseWorld, StreamingLanguageWorld


def test_seed_restore_and_real_response():
    data = np.frombuffer(b'abcde FIRST 01234 abcde SECOND 56789 abcde THIRD abcde FOURTH', np.uint8)
    world = BroadLanguageWorld(data, buckets=128, per_bucket=8, seed=123)
    world.take(b'abcde', 3, 2)
    saved = world.state_dict()
    clone = BroadLanguageWorld(data, buckets=128, per_bucket=8, seed=123)
    clone.load_state_dict(saved)
    for _ in range(8):
        a, b, info = world.take(b'abcde', 3, 2)
        aa, bb, other = clone.take(b'abcde', 3, 2)
        assert bytes(a) == bytes(aa) and bytes(b) == bytes(bb) and info == other
        assert bytes(a) == bytes(data[info['start']:info['start']+3])


def test_silence_advances_and_does_not_wrap():
    data = np.frombuffer(b'first second third', np.uint8)
    world = BroadLanguageWorld(data, buckets=16, seed=1)
    a, b, info = world.take(b'', 5, 2)
    assert bytes(a) == b'first' and bytes(b) == b' s' and info['source'] == 'continuation'
    with pytest.raises(EOFError):
        world.take(b'', len(data), 1)


def test_exposure_does_not_choose_response():
    data = np.frombuffer(b'abcde first abcde second abcde third', np.uint8)
    a = BroadLanguageWorld(data, buckets=128, seed=7)
    b = BroadLanguageWorld(data, buckets=128, seed=7)
    b.seen[:] = True
    x, y, meta = a.take(b'abcde', 4, 2)
    xx, yy, other = b.take(b'abcde', 4, 2)
    assert bytes(x) == bytes(xx) and bytes(y) == bytes(yy)
    assert meta['start'] == other['start'] and meta['holdout_position_fresh']
    assert not other['holdout_position_fresh']


def test_identical_query_can_reach_distinct_matched_continuations():
    data = np.frombuffer(b'abcde first abcde second abcde third abcde fourth', np.uint8)
    world = BroadLanguageWorld(data, buckets=128, per_bucket=8, seed=3)
    starts = {world.take(b'abcde', 3, 2)[2]['start'] for _ in range(30)}
    assert len(starts) > 1


def test_speaking_control_ignores_utterance_for_response():
    data = np.frombuffer(b'abcde first abcde second abcde third', np.uint8)
    a = SequentialResponseWorld(data, buckets=128, seed=7)
    b = SequentialResponseWorld(data, buckets=128, seed=7)
    x, y, meta = a.take(b'abcde second', 4, 2)
    xx, yy, other = b.take(b'', 4, 2)
    assert bytes(x) == bytes(xx) and bytes(y) == bytes(yy)
    assert meta == other and meta['source'] == 'continuation'


def test_stream_only_commits_consumed_bytes_and_restores():
    data = np.frombuffer(b'abcde first abcde second abcde third', np.uint8)
    world = StreamingLanguageWorld(data, buckets=128, seed=7)
    event = world.act(b'abcde', required=4)
    assert not world.seen.any()
    before = world.state_dict()
    clone = StreamingLanguageWorld(data, buckets=128, seed=7)
    clone.load_state_dict(before)
    x, info = world.observe(2)
    y, other = clone.observe(2)
    assert bytes(x) == bytes(y) and info == other
    assert info['start'] == event['start'] and int(world.seen.sum()) == 2
    assert not world.seen[info['end']]
