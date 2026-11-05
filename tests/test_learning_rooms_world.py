import numpy as np

from scripts.ib_local.learning_rooms_world import LearningRoomsWorld, validation_banks


def test_addresses_and_silence_have_observable_effects():
    w=LearningRoomsWorld(11)
    for room,word in enumerate((b'a',b'b',b'c',b'd',b'e')):
        response,_,info=w.take(word,257,257)
        assert info['anchor']==room
        assert set(response).issubset({97+2*room,98+2*room})
        assert w.destination(b'',room)==room
    assert w.destination(b'x',4)==4


def test_payload_induced_navigation_graph_is_strongly_connected():
    for initial in range(5):
        reached={initial}
        for _ in range(5):
            reached|={LearningRoomsWorld.destination(bytes([97+2*r+b]),r)
                       for r in list(reached) for b in (0,1)}
        assert reached==set(range(5))


def test_direct_keys_can_reach_every_room_without_a_noise_hallway():
    words=[bytes(97+((key>>bit)&1) for bit in (2,1,0)) for key in range(8)]
    for source in range(5):
        reached={LearningRoomsWorld.destination(w,source,'direct_bits') for w in words}
        assert reached==set(range(5))
    for phase in (b'aabb'*8,b'bbaa'*8):
        assert LearningRoomsWorld.destination(phase,0,'direct_bits') in (2,3)


def test_snapshot_and_rule_change_are_reproducible():
    w=LearningRoomsWorld(11,switch_turn=2)
    state=w.state_dict()
    a=w.take(b'e',257,257)
    w.load_state_dict(state)
    b=w.take(b'e',257,257)
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])
    assert a[2]==b[2] and not b[2]['rule_changed']
    w.take(b'e',257,257)
    assert w.take(b'e',257,257)[2]['rule_changed']


def test_periodic_phase_does_not_alias_the_32_byte_action_every_turn():
    w=LearningRoomsWorld()
    assert w.offset==0
    w.take(b'a',257,257)
    assert w.offset==2
    w.take(b'a',257,257)
    assert w.offset==0


def test_heldout_banks_are_separate_and_flip_only_changing_room():
    banks=validation_banks()
    for room in range(4):
        np.testing.assert_array_equal(banks[0][room],banks[1][room])
    assert not np.array_equal(banks[0][4],banks[1][4])
