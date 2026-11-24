from types import SimpleNamespace

import numpy as np

from scripts.ib_local.natural_text_world import NaturalTextWorld
from scripts.run_interruptible_observation import configure


def test_silence_continues_from_only_the_bytes_already_acquired():
    data = np.arange(2048, dtype=np.uint8)
    base = SimpleNamespace(data=data, cursor=0, window=960, seen=np.zeros(len(data), dtype=bool))
    world = NaturalTextWorld(base, {}, navigation_policy='stable_context_v1')
    engine = SimpleNamespace(autonomous_events_schema=1, response_bytes=480, holdout_bytes=480)
    loaded = SimpleNamespace(engine=engine, config=dict(world_source='text', speech_only=True))
    configure(loaded)
    first, suffix, _ = world.take(b'', engine.response_bytes, engine.holdout_bytes)
    assert (len(first), len(suffix), base.cursor) == (240, 240, 480)
    assert base.seen[:480].all() and not base.seen[480:].any()
    configure(loaded)
    first, suffix, receipt = world.take(b'', engine.response_bytes, engine.holdout_bytes)
    assert receipt['start'] == 480 and base.cursor == 960
    np.testing.assert_array_equal(np.concatenate((first, suffix)), data[480:960])
