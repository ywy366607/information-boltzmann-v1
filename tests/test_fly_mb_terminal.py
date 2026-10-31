"""Contact/chemistry, release conservation and actual learner replay contracts."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_fly_bptt_learning import make_model, physical
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState
from information_boltzmann.core.fly_mb_terminal import (
    attach_mb_terminal, step_mb_terminal, apply_mb_terminal_release,
    quarantine_mb_pooled_credit, MB_TERMINAL_PARAMETER_NAMES, MB_TERMINAL_ID_NAMES,
)
from information_boltzmann.core.fly_mb_consolidation import step_mb_consolidation, dpm_forgetting_factor
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights


def setup(tmp_path):
    prototype=make_model(tmp_path)
    path=tmp_path/'graph.npz'
    with np.load(path,allow_pickle=False) as packed:
        graph={k:packed[k] for k in packed.files}
    graph.update(edge_pre=np.arange(6),edge_post=np.array([3,4,4,5,5,0]),
        nt_sign=np.array([-1,-1,-1,1,1,1]),
        edge_pre_i=np.array([0,2,1]),edge_post_i=np.array([3,4,4]),
        edge_weight_i=np.array([.2,.3,.4],dtype=np.float32),edge_delay_i=np.array([1,1,2]),
        delay_splits_i=np.array([0,2,3,3,3]),delay_splits_e=np.array([0,3,3,3,3]),
        edge_pre_e=np.array([3,4,5]),edge_post_e=np.array([5,5,0]),
        edge_weight_e=np.array([.1,.1,.1],dtype=np.float32),edge_delay_e=np.ones(3,dtype=np.int32))
    np.savez(path,**graph)
    model=FlyReservoirLM(path,vocab_size=9,d_model=3,injection='sensory',read_surface='output',
                         synapse_model='coba',use_alif=True,use_stp=False)
    model.topographic_writer=prototype.topographic_writer
    model.injection_mode,model.input_proj='topographic',None
    ann=tmp_path/'ann.feather'
    pd.DataFrame({'bodyId':np.arange(6),'type':['DPM','APL','DPM','KC1','KC2','MBON'],
                  'class':['','','','Kenyon_Cell','Kenyon_Cell','MBON']}).to_feather(ann)
    share=tmp_path/'shares.feather'
    pd.DataFrame({'body_pre':[0,1,2],'body_post':[3,4,4],
                  'total':[10,10,10],'dendritic':[2,4,0]}).to_feather(share)
    before=torch.get_rng_state().clone()
    manifest=attach_mb_terminal(model,path,ann,share)
    assert torch.equal(before,torch.get_rng_state())
    return model,physical(model),manifest,ann,share


def test_source_target_identity_delay_and_known_axonal_share(tmp_path):
    model,current,manifest,_,_=setup(tmp_path)
    assert manifest['axonal_gaba_contacts']==3 and manifest['dpm_5ht_contacts']==2
    assert model.mb_terminal_source_body_ids.tolist()==[0,2,1]
    # First DPM reaches only KC3; the other hemisphere and APL are silent.
    current=replace(current,ring=(torch.tensor([[1.,0.,0.,0.,0.,0.]]),*current.ring[1:]))
    fraction,gaba,concentration,receptor=step_mb_terminal(model,current)
    assert gaba.shape==(3,2) and concentration.shape==(2,2)
    assert gaba[0,0]>0 and torch.count_nonzero(gaba)==1
    assert receptor[0,0]>0 and receptor[0,1]==0 and fraction[0,1]==1
    # Its .2 contact has .8 axonal share. Same normalized COBA kernel as soma.
    rho=model.get_decay_rates()[2][0,3]
    torch.testing.assert_close(gaba[0,0],(1.-rho)*.2*.8)
    quiet=tuple(torch.zeros_like(current.h) for _ in range(4))
    wrong=replace(current,ring=(torch.tensor([[0.,1.,0.,0.,0.,0.]]),*quiet[1:]))
    _,g,_,_=step_mb_terminal(model,wrong)
    assert g.sum()==0  # APL contact delay is2, not1
    correct=replace(current,ring=(quiet[0],wrong.ring[0],*quiet[2:]))
    _,g,_,_=step_mb_terminal(model,correct)
    assert g[2,1]>0 and torch.count_nonzero(g)==1


def test_release_is_gated_once_and_unreleased_vesicles_are_preserved(tmp_path):
    model,current,_,_,_=setup(tmp_path)
    model.use_stp=True
    packet=torch.tensor([[0.,0.,0.,.2,.4,0.]])
    ring=(packet,*current.ring[1:])
    pool=torch.tensor([[1.,1.,1.,.8,.6,1.]])
    new,x=apply_mb_terminal_release(model,ring,pool,torch.tensor([[.25,0.]]))
    torch.testing.assert_close(new[0][0,3:5],torch.tensor([.05,0.]))
    torch.testing.assert_close(x[0,3:5],torch.tensor([.95,1.]))
    assert all(a is b for a,b in zip(new[1:],ring[1:]))
    # No extra spike, threshold adjustment, or cross-target command is created.
    assert torch.equal(new[0][:,:3],packet[:,:3])


def test_pooled_rules_are_inert_and_native_score_uses_new_parameters(tmp_path):
    model,current,_,_,_=setup(tmp_path)
    model.mb_consolidation_enabled=True
    current=replace(current,dpm_trace=torch.tensor([[.8]]),apl_trace=torch.tensor([[.9]]))
    threshold=torch.ones_like(current.h)
    same,dpm,apl=step_mb_consolidation(model,current,threshold)
    assert same is threshold and dpm is current.dpm_trace and apl is current.apl_trace
    assert dpm_forgetting_factor(model,dpm)==1.
    names=native_parameter_names(model)
    assert set(MB_TERMINAL_PARAMETER_NAMES).issubset(names)
    assert 'mb_apl_feedback' not in names


def test_identity_checkpoint_and_disabled_terminal_are_exact(tmp_path):
    model,current,_,ann,shares=setup(tmp_path)
    saved=checkpoint_model_weights(model)
    assert set(MB_TERMINAL_ID_NAMES).issubset(saved)
    bad={**saved,'mb_terminal_source_body_ids':saved['mb_terminal_source_body_ids'].flip(0)}
    other=copy.deepcopy(model)
    # Rebuild, rather than copying cached anatomical indices.
    del other.mb_terminal_kc_index
    with pytest.raises(ValueError,match='identity mismatch'):
        attach_mb_terminal(other,tmp_path/'graph.npz',ann,shares,saved=bad)
    model.mb_terminal_enabled=False
    fraction,*state=step_mb_terminal(model,current)
    assert fraction is None and all(x.numel()==0 for x in state)
    assert apply_mb_terminal_release(model,current.ring,current.x,fraction)==(current.ring,current.x)


@pytest.mark.parametrize('field', ['mb_terminal_gaba','mb_dpm_concentration','mb_dpm_receptor'])
def test_incompatible_nonempty_memory_is_rejected_instead_of_reset(tmp_path,field):
    model,current,_,_,_=setup(tmp_path)
    changed=replace(current,**{field:torch.ones(1,3)})
    with pytest.raises(ValueError,match='Nonempty MB terminal'):
        step_mb_terminal(model,changed)
    assert getattr(changed,field).eq(1).all()


@pytest.mark.parametrize('device', ['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_eager_checkpointed_learning_and_full_chemical_state_match(tmp_path,device):
    torch.manual_seed(31)
    model,current,_,_,_=setup(tmp_path)
    current=replace(current,ring=tuple(torch.ones_like(current.h) for _ in current.ring))
    current.h[:,model.mb_terminal_kc_index]=2.*model.get_thresholds().detach().expand_as(current.h)[:,model.mb_terminal_kc_index]
    model=model.to(device)
    current=FlyPhysicalState(**{k:tuple(x.to(device) for x in v) if k=='ring' else v.to(device)
                                for k,v in current.state_dict().items()})
    a=FlyBPTTLearner(model,copy.deepcopy(current),use_checkpointing=False,settle_ticks=2)
    b=FlyBPTTLearner(copy.deepcopy(model),copy.deepcopy(current),use_checkpointing=True,settle_ticks=2)
    a.previous_token=b.previous_token=0
    for labels in ([1,2,3,4],[5,6,7,8]):
        p,_=a.observe(labels)
        q,_=b.observe(labels)
        torch.testing.assert_close(torch.tensor(p),torch.tensor(q))
        for name in MB_TERMINAL_PARAMETER_NAMES:
            torch.testing.assert_close(getattr(a.model,name),getattr(b.model,name))
        for name in ('mb_terminal_gaba','mb_dpm_concentration','mb_dpm_receptor'):
            torch.testing.assert_close(getattr(a.state,name),getattr(b.state,name))
    assert a.state.mb_dpm_receptor.sum()>0
    for name in MB_TERMINAL_PARAMETER_NAMES:
        assert name in a.adam_names and a.optimizer.state[getattr(a.model,name)]['exp_avg'].abs().sum()>0


def test_old_proxy_credit_is_archived_without_losing_other_experience():
    old=torch.tensor([.3])
    keep=torch.tensor([.7])
    credit={'E':{'mb_apl_feedback':old,'log_tau_m':keep},'response_score':None,
            'deferred_response_scores':{'turn1':{'mb_dpm_consolidation':old}},'clock_ticks':91}
    learner=SimpleNamespace(model=SimpleNamespace(mb_pooled_quarantined=True),
        consequence_state={'native_credit':credit,'reply_pending':{'word':123}})
    assert quarantine_mb_pooled_credit(learner)
    assert credit['E']=={'log_tau_m':keep} and credit['deferred_response_scores']['turn1']=={}
    assert learner.consequence_state['reply_pending']=={'word':123}
    archive=learner.consequence_state['quarantined_mb_pooled_credit'][0]
    assert torch.equal(archive['traces']['E']['mb_apl_feedback'],old)
    assert not quarantine_mb_pooled_credit(learner)
