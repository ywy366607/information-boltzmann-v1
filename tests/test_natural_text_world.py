"""Interface/continuity checks; no synthetic capability training."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
from scripts.ib_local.natural_text_world import NaturalTextWorld, append_source, extended_source
from scripts.ib_local.navigation_mother_boundary import mother_boundary_feature
from scripts.ib_local.run_mother_continuation import prepare_world


def base():
    return ReplyTopicWorld(np.arange(256,dtype=np.uint8),window=32,buckets=16,per_bucket=2,seed=11)


def test_append_keeps_addresses_seen_cursor_and_rng(tmp_path):
    world=base()
    world.take(b'',8,8)
    before=world.state_dict()
    raw=b'natural source attachment'
    file=tmp_path/'text.bin'; file.write_bytes(raw)
    manifest=dict(schema=1,prefix_bytes=256,train_path=str(file),train_bytes=len(raw),train_sha256=hashlib.sha256(raw).hexdigest())
    changed=append_source(world,manifest)
    assert np.array_equal(changed.data[:256],world.data)
    assert np.array_equal(changed.seen[:256],before['seen'])
    assert not changed.seen[256:].any()
    assert changed.cursor==world.cursor and changed.anchor==world.anchor and changed.offset==world.offset
    assert changed.rng.bit_generator.state==before['rng']
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):extended_source(world.data,manifest)


def test_text_reads_forward_and_preserves_caregiver_history_on_resume():
    retained=dict(mother_transcript=[['child','old'],['mother','actual past reply']],mother_provider_state={'turns':9})
    world=NaturalTextWorld(base(),retained)
    a,_,info=world.take(b'',16,16)
    b,_,_=world.take(b'',16,16)
    assert bytes(a)==bytes(range(16)) and bytes(b)==bytes(range(32,48))
    assert info['provider_calls']==0 and info['source']=='text' and info['mother_bytes']==0
    assert 'mother_reply' not in info
    state=world.state_dict()
    restored=NaturalTextWorld(base(),{})
    restored.load_state_dict(state)
    assert restored.retained==retained
    assert np.array_equal(world.take(b'',16,16)[0],restored.take(b'',16,16)[0])


def test_text_keeps_real_suffix_navigation():
    world=NaturalTextWorld(base(),{})
    # Choose an indexed occurrence: the finite reservoir need not contain
    # every possible suffix, which is part of the production world's contract.
    p=next(int(p) for p in world.positions.flat if 0 <= p <= 220)
    _,_,info=world.take(bytes(world.data[p:p+4]),8,8)
    assert info['match']==4 and info['start']==p+4 and world.base.moves==1


def test_actual_text_boundary_uses_existing_relation_port():
    features=torch.arange(40,dtype=torch.float32).reshape(20,2)
    target=mother_boundary_feature(features,2,8,dict(source='text',navigation_response_bytes=8))
    assert torch.equal(target,features[10])
    assert mother_boundary_feature(features,2,8,dict(source='mother',mother_bytes=8)) is None


def test_text_prepare_never_constructs_provider_and_defaults_resume_to_text(monkeypatch):
    from scripts.ib_local.mengli_harness import mengli_world_client
    def forbidden(**kwargs):raise AssertionError('text world must not construct DSH provider')
    monkeypatch.setattr(mengli_world_client,'DshMother',forbidden)
    engine=SimpleNamespace(world=base())
    config=dict(mother_context_harness=dict(kind='dsh'),world_source='text')
    state=dict(mother_transcript=[['child','old'],['mother','reply']])
    args=SimpleNamespace(world_source=None,natural_text_manifest=None,mode='real')
    prepare_world(engine,config,args,state,True,5543,'a'*64,True)
    assert isinstance(engine.world,NaturalTextWorld)
    assert engine.world.retained==state
    assert config['world_source']=='text'



def test_source_accounting_tracks_delivered_bytes_without_changing_navigation():
    world=NaturalTextWorld(base(),{},dict(prefix_bytes=128))
    world.base.cursor=120
    _,_,info=world.take(b'',12,4)
    assert info['text_source_bytes']==dict(inherited_prefix=8,natural_append=8)
    assert info['text_response_source_bytes']==dict(inherited_prefix=8,natural_append=4)
    assert info['text_holdout_source_bytes']==dict(inherited_prefix=0,natural_append=4)
    before=world.state_dict()
    control=NaturalTextWorld(base(),{});control.load_state_dict(before)
    p=next(int(p) for p in world.positions.flat if 0<=p<=100)
    utterance=bytes(world.data[p:p+4])
    response,continuation,info=world.take(utterance,8,8)
    expected_response,expected_continuation,expected_info=control.take(utterance,8,8)
    assert np.array_equal(response,expected_response)
    assert np.array_equal(continuation,expected_continuation)
    assert info['start']==expected_info['start']==p+4
    assert info['text_source_bytes']==dict(inherited_prefix=16,natural_append=0)
    assert world.rng.bit_generator.state==control.rng.bit_generator.state
    assert np.array_equal(world.seen,control.seen)
