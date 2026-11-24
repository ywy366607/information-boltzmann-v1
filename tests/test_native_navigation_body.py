"""Tensor and checkpoint contracts for native birth; no capability benchmark."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local.native_navigation_body import (
    component_config, construct_engine, create, initialize_native, validate_spec)
from scripts.ib_local.run_causal_water_reply import configure_components
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable


def spec():
    path=Path('configs/information_boltzmann/mengli_native_l32_d256_20261115.json')
    result=json.loads(path.read_text())
    # Same complete graph with bounded tensor dimensions for interface tests.
    result['backbone'].update(d=32,layers=4,heads=2,mlp_hidden=48)
    result.update(latent_dim=16,experience_layer=3,memory_capacity=512)
    return result


def construct():
    s=spec();cfg=component_config(s,{})
    engine=construct_engine(cfg,SimpleNamespace(data=np.zeros(1,dtype=np.uint8)))
    configure_components(engine,cfg)
    receipt=initialize_native(engine,s)
    return cfg,engine,receipt


def test_native_every_layer_and_deep_port_are_live_from_birth():
    _,e,r=construct();m=e.model
    assert e.policy_replay_backend == 'batched_guarded'
    assert r['zero_residual_layers']==[] and len(e.optimizer.state)==0
    assert m.encoder.in_features==32 and m.encoder.out_features==16
    assert m.body.inject_layer==1 and m.experience_layer==3
    assert not hasattr(m,'feature_read')
    for layer in m.base.layers:
        assert layer.o_proj.weight.count_nonzero()==layer.o_proj.weight.numel()
        assert layer.mlp['down'].weight.count_nonzero()==layer.mlp['down'].weight.numel()
    ids=torch.tensor([[10,20,30,40,50,60,70,80]])
    lp=torch.full((1,8,256),-np.log(256),dtype=torch.float32)
    seen=[]
    handle=m.base.layers[2].register_forward_hook(lambda module,args,out:seen.append(out[0].clone()))
    logits,state,_,h,_=m.predict_observed(ids,e.recurrent,lp)
    handle.remove()
    torch.testing.assert_close(h,seen[0][0])
    # Readout/latent losses use deep contents; gradients reach all layers at birth.
    loss=logits.square().mean()+m.encoder(h).square().mean()
    loss.backward()
    for layer in m.base.layers:
        assert layer.o_proj.weight.grad.norm()>0
        assert layer.q_proj.weight.grad.norm()>0
    assert state.shape==e.recurrent.shape and not torch.equal(state,e.recurrent)


def test_native_full_graph_roundtrip_keeps_group_order_and_no_coordinate_bridge():
    cfg,e,r=construct()
    cfg2=copy.deepcopy(cfg)
    restored=construct_engine(cfg2,SimpleNamespace(data=np.zeros(1,dtype=np.uint8)),restoring=True)
    configure_components(restored,cfg2)
    assert restored.policy_replay_backend == 'batched_guarded'
    restored.model.load_state_dict(e.model.state_dict(),strict=True)
    restored.optimizer.load_state_dict(e.optimizer.state_dict())
    assert [len(g['params']) for g in e.optimizer.param_groups]==[
        len(g['params']) for g in restored.optimizer.param_groups]
    assert r['parameters']==sum(p.numel() for p in restored.model.parameters())
    assert not hasattr(restored.model,'feature_read')
    assert hasattr(restored.model,'relation_basis')
    assert hasattr(restored.model,'binding_choice') and hasattr(restored.model,'handover')


def test_native_chunk_boundary_is_not_a_reset():
    _,e,_=construct();m=e.model
    ids=torch.tensor([[9,7,3,2,8,6,5,4]])
    lp=torch.full((1,8,256),-np.log(256),dtype=torch.float32)
    with torch.no_grad():
        whole=m.predict(ids,e.recurrent,lp)
        left=m.predict(ids[:,:4],e.recurrent,lp[:,:4])
        right=m.predict(ids[:,4:],left[1],lp[:,4:])
    torch.testing.assert_close(torch.cat((left[0],right[0]),1),whole[0],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(right[1],whole[1],atol=2e-6,rtol=2e-5)


def test_native_birth_has_empty_evidence_and_all_existing_algorithm_connections(tmp_path):
    from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
    s=spec();cfg=component_config(s,{})
    # Only construct/storage interfaces are tested; this synthetic array is
    # never learned from and produces no ability or performance claim.
    world=ReplyTopicWorld(np.zeros(8192,dtype=np.uint8),window=4096,seed=s['seed'])
    e,r=create(cfg,world,table=RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1),
        causal_storage=tmp_path/'causal')
    assert len(e.optimizer.state)==0 and e.prev is None
    assert int(e.model.memory.clock)==0 and int(e.model.memory.episodes)==0
    assert e.gain_units_schema==1 and e.settlement_kind=='world_stream'
    assert e.model.memory.read_credit_enabled
    assert len(e.bfss_momenta)==4 and all(p.count_nonzero()==0 for p in e.bfss_momenta)
    assert e.model._causal_archive is not None
    assert getattr(e,'joint_relation_update_revision',None)==4
    assert cfg['birth']['parent_checkpoint'] is None
    e.model._causal_archive.db.close()


def test_native_rejects_a_shallow_shared_byte_experience_port():
    s=spec();s['experience_layer']=s['byte_table_layer']
    with pytest.raises(ValueError,match='deeper content'):
        validate_spec(s)


def test_native_lifetime_counts_survive_resume_and_do_not_accept_missing_history(tmp_path):
    from scripts.ib_local.native_navigation_body import restore_birth_counters, record_birth_event
    journal=tmp_path/'metrics.jsonl'
    journal.write_text(json.dumps(dict(cycle=1,updates=3,external_bytes=960,self_bytes=0))+'\n')
    e=SimpleNamespace()
    restore_birth_counters(e,1,journal)
    row=dict(updates=4,external_bytes=960,self_bytes=20)
    record_birth_event(e,row)
    assert row['since_birth_updates']==7 and row['since_birth_external_bytes']==1920
    restore_birth_counters(e,2,journal)  # Existing saved counters are authoritative.
    assert e.native_optimizer_updates==7 and e.native_self_bytes==20
    with pytest.raises(ValueError,match='incomplete journal'):
        restore_birth_counters(SimpleNamespace(),2,journal)
