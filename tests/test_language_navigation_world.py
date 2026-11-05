"""World interface and causal boundaries, not model capability tests."""
import numpy as np
import pytest
from scripts.ib_local.language_navigation_world import LanguageNavigationWorld


def test_utterance_selects_real_nonoverlapping_response_and_probe():
    data = np.frombuffer(b'abcdefgh FIRST_RESPONSE NEXT_PROBE\nxyzuvw12 SECOND_RESPONSE OTHER_PROBE\n', np.uint8)
    world = LanguageNavigationWorld(data, buckets=256, per_bucket=8)
    a, held, meta = world.take(b'abcdefgh', 8, 5)
    assert bytes(a) == b' FIRST_R' and bytes(held) == b'ESPON'
    assert meta['match'] == 8 and meta['source'] == 'matched'
    assert meta['end'] - meta['start'] == len(a)+len(held)
    b, _, other = world.take(b'xyzuvw12', 8, 5)
    assert bytes(b) == b' SECOND_' and other['start'] != meta['start']
    _, _, repeated = world.take(b'abcdefgh', 8, 5)
    assert meta['holdout_position_fresh'] and not repeated['holdout_position_fresh']
    assert repeated['holdout_previously_seen_bytes'] == 5


def test_world_exposure_provenance_restores_without_changing_index():
    data = np.frombuffer(b'abcde first passage and then the next', np.uint8)
    world = LanguageNavigationWorld(data,buckets=128)
    world.mark_seen(3,8); saved = world.state_dict()
    clone = LanguageNavigationWorld(data,buckets=128); clone.load_state_dict(saved)
    np.testing.assert_array_equal(world.seen,clone.seen)
    np.testing.assert_array_equal(world.positions,clone.positions)
    assert world.cursor == clone.cursor


def test_silence_and_no_match_continue_and_never_silently_wrap():
    data = np.frombuffer(b'the quick brown fox jumps over the dog', np.uint8)
    world = LanguageNavigationWorld(data, buckets=128, cursor=4)
    a, b, info = world.take(b'', 5, 4)
    assert bytes(a) == b'quick' and bytes(b) == b' bro' and info['source'] == 'continuation'
    assert world.cursor == 13
    a, b, info = world.take(b'ZZZZ', 5, 4)
    assert bytes(a) == b'wn fo' and info['source'] == 'continuation'
    with pytest.raises(EOFError):
        world.take(b'', len(data), 1)


def test_hash_collisions_do_not_count_as_matches():
    data = np.frombuffer(b'abcde first passage and then the next', np.uint8)
    world = LanguageNavigationWorld(data, buckets=1, per_bucket=2)
    a, _, info = world.take(b'zzzq', 3, 2)
    assert info['match'] == 0 and bytes(a) == b'abc'
