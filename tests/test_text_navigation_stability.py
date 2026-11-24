"""Addressing and continuation invariants, without capability training."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
from scripts.ib_local.natural_text_world import NaturalTextWorld
from scripts.ib_local.text_navigation import stable_suffix


def source(raw):
    return np.frombuffer(raw, np.uint8)


def test_full_suffix_wins_over_nearer_short_coincidence():
    data = source(b'cat----X123----cat----wholeX123----tail')
    query = source(b'wholeX123')
    start, score, count, _ = stable_suffix(query, data, 4, 8)
    assert start == bytes(data).index(b'wholeX123')+len(query)
    assert score == len(query) and count == 1


def test_ambiguous_short_suffix_uses_nearest_actual_context():
    data = source(b'abcd..abcd....abcd........')
    first = stable_suffix(source(b'xxabcd'), data, 4, 9)
    assert first[:3] == (10, 4, 3)
    assert stable_suffix(source(b'xxabcd'), data, 4, 9) == first


def test_tie_breaks_to_lower_address_and_excludes_insufficient_tail():
    data = source(b'abcd....abcd....abcd')
    assert stable_suffix(source(b'abcd'), data, 4, 8)[:3] == (4, 4, 2)


@pytest.mark.parametrize('query', [b'', b'a', b'abc', b'ZZZZ'])
def test_missing_address_is_explicit(query):
    assert stable_suffix(source(query), source(b'abcdefghijk'), 4, 0) == (-1,0,0,-1)


def world():
    raw = np.arange(256, dtype=np.uint8)
    return NaturalTextWorld(ReplyTopicWorld(raw,window=32,buckets=16,per_bucket=2,seed=11),
        {'mother_transcript':[['mother','historical evidence']]},
        navigation_policy='stable_context_v1')


def test_stable_world_uses_full_source_and_preserves_saved_state():
    original = world()
    original.base.cursor = 16
    before = original.state_dict()
    a,b,info = original.take(bytes(range(100,108)),8,8)
    assert info['start'] == 108 and info['match'] == 8
    assert info['address_policy'] == 'stable_context_v1'
    assert info['provider_calls'] == 0 and original.base.moves == 1
    assert bytes(np.concatenate((a,b))) == bytes(range(108,124))
    restored = world(); restored.load_state_dict(before)
    # A different random stream cannot change a tied address.
    restored.base.rng = np.random.default_rng(999)
    assert restored.take(bytes(range(100,108)),8,8)[2]['start'] == 108
    after = original.state_dict()
    continued = world(); continued.load_state_dict(after)
    assert continued.navigation_policy == 'stable_context_v1'
    assert continued.retained == original.retained
    assert bytes(continued.take(b'',8,8)[0]) == bytes(original.take(b'',8,8)[0])


def test_stable_source_accounting_and_eof_continuity():
    current = world(); current.source_manifest = dict(prefix_bytes=128)
    current.base.cursor = 120
    a,b,info = current.take(b'',12,4)
    assert info['text_source_bytes'] == dict(inherited_prefix=8,natural_append=8)
    assert bytes(np.concatenate((a,b))) == bytes(range(120,136))
    current.base.cursor = 250
    assert bytes(current.take(b'',8,8)[0]) == bytes(range(8))


def test_navigation_migration_is_explicit_and_preserves_history():
    from scripts.ib_local.run_mother_continuation import prepare_world
    original = world()
    base = original.base
    base.cursor = 99
    engine = SimpleNamespace(world=base)
    before = base.state_dict()
    state = dict(mother_transcript=[['mother','old actual reply']])
    config = dict(world_source='text')
    args = SimpleNamespace(world_source='text',natural_text_manifest=None,mode='real',
                           text_navigation='stable_context_v1')
    prepare_world(engine,config,args,state,True,7623,'a'*64,True)
    assert engine.world.navigation_policy == 'stable_context_v1'
    assert engine.world.cursor == before['cursor']
    assert engine.world.retained == state
    assert config['text_navigation_migrations'][0]['at_cycle'] == 7623
    assert np.array_equal(engine.world.seen,before['seen'])
    restored_config = copy.deepcopy(config)
    second = SimpleNamespace(world=world().base)
    args.text_navigation = None
    prepare_world(second,restored_config,args,state,True,7624,'b'*64,True)
    assert second.world.navigation_policy == 'stable_context_v1'
    assert len(restored_config['text_navigation_migrations']) == 1


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError,match='navigation policy'):
        NaturalTextWorld(world().base,{},navigation_policy='invalid')
