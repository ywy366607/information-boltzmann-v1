import numpy as np

from scripts.ib_local.symbol_navigation_world import SymbolNavigationWorld, validation_stream


def test_actual_utterance_routes_and_silence_stays():
    world = SymbolNavigationWorld(11)
    response, _, info = world.take(b'anything a', 8, 8)
    assert response.tobytes() == b'aabbaabb' and info['room'] == 'periodic'
    response, _, info = world.take(b'', 8, 8)
    assert response.tobytes() == b'aabbaabb' and info['match'] == 0
    _, _, info = world.take(b'anything b', 8, 8)
    assert info['room'] == 'random' and info['match'] == 1
    _, _, info = world.take(b'x', 8, 8)
    assert info['room'] == 'random' and info['match'] == 0


def test_snapshot_restores_reply_and_hidden_rng():
    world = SymbolNavigationWorld(11)
    state = world.state_dict()
    first = world.take(b'b', 160, 160)
    world.load_state_dict(state)
    second = world.take(b'b', 160, 160)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_two_rooms_share_alphabet_but_have_different_order():
    stream = validation_stream()
    assert set(stream) == {ord('a'), ord('b')}
    assert stream[:8].tobytes() == b'aabbaabb'
    assert not np.array_equal(stream[2048:2056], stream[:8])
