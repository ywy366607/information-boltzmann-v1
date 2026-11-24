import copy
import numpy as np
import pytest

from scripts.ib_local.mother_reply_world import (MotherReplyWorld, StrictToddlerMother,
    RESPONSIVE_SYSTEM, RESPONSIVE_SYSTEM_V2)


class Ambient:
    def __init__(self):
        self.clock = 0
        self.words = None

    def take(self, words, n, h):
        self.words = words
        self.clock += 1
        return np.full(n, ord('x'), np.uint8), np.full(h, ord('y'), np.uint8), {'clock': self.clock}

    def state_dict(self):
        return {'clock': self.clock}

    def load_state_dict(self, state):
        self.clock = state['clock']


class Mother:
    def __init__(self, answer='hello there'):
        self.answer, self.seen = answer, []

    def reply(self, turns):
        self.seen.append(copy.deepcopy(turns))
        return self.answer


def test_contingent_response_and_complete_world_state_roundtrip():
    mother = Mother()
    world = MotherReplyWorld(Ambient(), mother, history=1)
    response, held, info = world.take(b'hello', 16, 5)
    assert bytes(response) == b'hello there\nxxxx'
    assert bytes(held) == b'yyyyy'
    assert info['match'] == 1 and info['mother_bytes'] == 12
    assert mother.seen == [[('child', 'hello')]]
    assert world.base.words == b'hello'
    state = world.state_dict()
    restored = MotherReplyWorld(Ambient(), Mother(), history=1)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    restored.take(b'', 16, 5)
    assert restored.mother.seen[0][-1] == ('child', '[The child does not speak this turn.]')
    assert world.state_dict() == state


def test_failed_reply_does_not_advance_world_or_create_fake_experience():
    world = MotherReplyWorld(Ambient(), Mother(''))
    before = world.state_dict()
    with pytest.raises(RuntimeError, match='not advanced'):
        world.take(b'hello', 16, 5)
    assert world.state_dict() == before


def test_ambient_read_preserves_conversation_but_actual_silence_calls_mother():
    mother = Mother()
    world = MotherReplyWorld(Ambient(), mother)
    response, held, info = world.take_ambient(16, 5)
    assert bytes(response) == b'x' * 16 and bytes(held) == b'y' * 5
    assert info['source'] == 'text' and info['mother_bytes'] == 0
    assert world.base.clock == 1 and world.transcript == [] and mother.seen == []
    world.take(b'', 16, 5)
    assert world.base.clock == 2 and len(mother.seen) == 1
    assert mother.seen[0][-1] == ('child', '[The child does not speak this turn.]')


def test_ambient_read_roundtrip_preserves_caregiver_history():
    world = MotherReplyWorld(Ambient(), Mother(), history=2)
    world.take(b'hello', 16, 5)
    history = copy.deepcopy(world.transcript)
    world.take_ambient(8, 8)
    state = world.state_dict()
    restored = MotherReplyWorld(Ambient(), Mother(), history=2)
    restored.load_state_dict(state)
    assert restored.state_dict() == state and restored.transcript == history
    restored.take_ambient(8, 8)
    assert restored.transcript == history and restored.mother.seen == []


def test_legacy_quiet_is_explicit_and_does_not_hide_migration():
    world = MotherReplyWorld(Ambient(), Mother(), ambient_policy='quiet')
    world.take(b'hello', 16, 5)
    assert world.base.words == b''
    old = world.state_dict()
    with pytest.raises(ValueError, match='incompatible'):
        MotherReplyWorld(Ambient(), Mother(), ambient_policy='reply').load_state_dict(old)
    restored = MotherReplyWorld(Ambient(), Mother(), ambient_policy='quiet')
    restored.load_state_dict(old)
    restored.ambient_policy = 'reply'
    restored.take(b'hello', 16, 5)
    assert restored.base.words == b'hello'


def test_identical_histories_share_a_real_reply_and_cache_is_bounded(tmp_path):
    import sqlite3
    from unittest.mock import patch

    class Delegate:
        def __init__(self):
            self.cache, self.calls = {}, 0
            self.base_system, self.model = 'system', 'model'
            self.temperature, self.max_tokens = 0., 20

        def reply(self, transcript):
            self.calls += 1
            return f'real reply {self.calls}'

    db = tmp_path/'world.sqlite'
    with patch('scratch.claude_mother_world_20261105.ToddlerMother', Delegate):
        a = StrictToddlerMother(shared_cache=db, shared_limit=2)
        b = StrictToddlerMother(shared_cache=db, shared_limit=2)
        assert a.reply([('child', 'hello')]) == 'real reply 1'
        assert b.reply([('child', 'hello')]) == 'real reply 1'
        assert b.delegate.calls == 0
        a.reply([('child', 'second')])
        a.reply([('child', 'third')])
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT COUNT(*) FROM replies').fetchone()[0] == 2


def test_caregiver_policy_migration_uses_new_prompt_preserves_history_and_isolates_cache(tmp_path):
    from unittest.mock import patch

    class Delegate:
        def __init__(self):
            self.cache, self.calls = {}, 0
            self.base_system, self.model = 'legacy instructions', 'model'
            self.temperature, self.max_tokens = 0., 20

        def get_prompt_parts(self, transcript=()):
            return 'legacy instructions', ''

        def reply(self, transcript):
            self.calls += 1
            return self.get_prompt_parts(transcript)[0]

    with patch('scratch.claude_mother_world_20261105.ToddlerMother', Delegate):
        m = StrictToddlerMother(shared_cache=tmp_path/'cache.sqlite')
        w = MotherReplyWorld(Ambient(), m, history=1)
        w.take(b'why', 32, 5)
        transcript = copy.deepcopy(w.transcript)
        m.delegate.cache['old transcript'] = 'legacy response'
        m.set_policy('responsive')
        assert w.transcript == transcript
        assert m.delegate.cache == {}
        assert m.policy_archives[-1]['cache']['old transcript'] == 'legacy response'
        assert m.delegate.get_prompt_parts([])[0] == RESPONSIVE_SYSTEM
        assert m.reply([('child','why')]) == RESPONSIVE_SYSTEM
        state = w.state_dict()
        restored = MotherReplyWorld(Ambient(),
            StrictToddlerMother(policy='responsive', shared_cache=tmp_path/'cache.sqlite'), history=1)
        restored.load_state_dict(state)
        assert restored.state_dict() == state
        restored.mother.set_policy('responsive_v2')
        assert restored.mother.delegate.get_prompt_parts()[0] == RESPONSIVE_SYSTEM_V2
        assert restored.transcript == transcript
        newer = restored.state_dict()
        resumed = MotherReplyWorld(Ambient(), StrictToddlerMother(policy='responsive_v2'), history=1)
        resumed.load_state_dict(newer)
        assert resumed.state_dict() == newer
        resumed.mother.set_policy('responsive_v3')
        stable, tail = resumed.mother.delegate.get_prompt_parts([])
        assert stable and 'present message' in tail
        assert resumed.mother.delegate.temperature == .7
        assert resumed.transcript == transcript
        v3_state = resumed.state_dict()
        final = MotherReplyWorld(Ambient(), StrictToddlerMother(policy='responsive_v3'), history=1)
        final.load_state_dict(v3_state)
        assert final.state_dict() == v3_state
