"""Fixed independent-validation cadence across real worker restarts."""
import json

import pytest

from scripts.ib_local.continuation_schedule import configure_validation_schedule, validation_due


def test_text_validation_preserves_stage_origin_across_restart():
    config=dict(world_source='text',world_source_migration=dict(at_cycle=5543))
    schedule=configure_validation_schedule(config,5791,128,8192)
    assert schedule['origin_cycle']==5543
    assert validation_due(5799,schedule)
    assert not validation_due(5792,schedule)
    restored=json.loads(json.dumps(config))
    assert configure_validation_schedule(restored,5801,128,8192)==schedule
    assert validation_due(5927,restored['validation_schedule'])
    assert validation_due(5810,schedule,final=True)
    assert not validation_due(5543,schedule)


def test_new_nontext_stage_begins_at_its_first_checkpoint():
    config={}
    schedule=configure_validation_schedule(config,500,64,8192)
    assert schedule['origin_cycle']==500 and validation_due(564,schedule)
    assert configure_validation_schedule(config,570,64,8192)==schedule


def test_invalid_or_silently_changed_validation_schedule_is_rejected():
    assert configure_validation_schedule({},100,0,8192) is None
    for every,size in [(-1,8192),(1,511)]:
        with pytest.raises(ValueError):configure_validation_schedule({},100,every,size)
    config={};configure_validation_schedule(config,100,64,8192)
    with pytest.raises(ValueError,match='explicit migration'):
        configure_validation_schedule(config,110,128,8192)
    with pytest.raises(ValueError,match='explicit migration'):
        configure_validation_schedule(config,110,64,4096)
