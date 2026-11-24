import numpy as np
import pytest
import torch

from scripts.ib_local.arrived_language_actor import attach
from scripts.ib_local.native_navigation_body import component_config, create
from scripts.ib_local.natural_text_world import NaturalTextWorld
from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from test_native_navigation_body import spec


@pytest.mark.parametrize('mode',['observe','apply'])
def test_h2_reaches_only_next_original_update(tmp_path,mode):
    torch.manual_seed(31); torch.set_num_threads(1)
    s=spec(); manifest=dict(prefix_bytes=2048,train_bytes=8192)
    base=ReplyTopicWorld(np.arange(10240).astype(np.uint8),window=4096,seed=s['seed'])
    world=NaturalTextWorld(base,{},manifest,navigation_policy='stable_context_v1')
    e,_=create(component_config(s,manifest),world,table=RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1),causal_storage=tmp_path/'causal')
    try:
        attach(e,mode,23)
        row=e.step(torch.randn(s['latent_dim'],4)); state=e.arrived_actor_state
        assert 'arrived_language_credit' not in row and state['due'] is None
        head=e.model.kanli.choice
        index=next(i for i,p in enumerate(e.params) if p is head.bias)
        assert (e.prev['actor'][index] is None)==(mode=='apply')
        row=e.step(torch.randn(s['latent_dim'],4))
        assert row['arrived_language_credit']['action_round']==1
        assert row['arrived_language_credit']['endpoint_round']==2
        assert (state['due'] is not None)==(mode=='apply')
        before=head.weight.detach().clone()
        e.step(torch.randn(s['latent_dim'],4))
        assert state['applied']==int(mode=='apply')
        assert not torch.equal(before,head.weight)
    finally:
        e.model._causal_archive.close()
