"""Address/accounting interface tests, without training or capability claims."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ib_local.text_source_availability import TextSourceAvailability


def fixture():
    base = SimpleNamespace(data=np.arange(100,dtype=np.uint8),seen=np.zeros(100,bool),
        window=16,cursor=60,anchor=60)
    inner = SimpleNamespace(base=base,navigation_policy='stable_context_v1',
        source_manifest=dict(prefix_bytes=50))
    def take(words,n,m):
        start = base.cursor
        response = base.data[start:start+n].copy()
        continuation = base.data[start+n:start+n+m].copy()
        base.seen[start:start+n+m] = True
        base.cursor += n+m; base.anchor = start
        return response,continuation,dict(start=start,anchor=start)
    inner.take = take
    return TextSourceAvailability(inner)


def test_change_available_source_keeps_absolute_addresses_and_seen_history():
    world = fixture(); data,seen = world.base.data,world.base.seen
    world.bounds = (50,100)
    response,continuation,info = world.take(b'actual words',4,4)
    assert np.array_equal(response,np.arange(60,64))
    assert np.array_equal(continuation,np.arange(64,68))
    assert world.base.data is data and world.base.seen is seen
    assert seen[60:68].all() and not seen[:60].any()
    assert world.base.cursor == 68 and world.base.anchor == 60
    assert info['start'] == 60
    assert info['text_source_bytes'] == dict(inherited_prefix=0,natural_append=8)


def test_failed_world_call_restores_original_storage():
    world = fixture(); data,seen = world.base.data,world.base.seen
    world.bounds = (50,100)
    def fail(*args): raise RuntimeError('external failure')
    world.world.take = fail
    with pytest.raises(RuntimeError): world.take(b'',4,4)
    assert world.base.data is data and world.base.seen is seen
