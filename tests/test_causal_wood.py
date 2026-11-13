"""Declared action-address and typed-read invariants, no capability toy."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.ib_local.navigation_causal_wood import (
    CausalWoodEngine, common_suffix, pack_words, stratified_shuffle, configure_causal_wood)


def reader(gain=True):
    e=object.__new__(CausalWoodEngine)
    e.model=SimpleNamespace(encoder=nn.Identity(),memory=SimpleNamespace(topk=2,temperature=.2,long=512))
    e.device=torch.device('cpu');e.intero=lambda:torch.zeros(1,3)
    e.response_bytes=e.holdout_bytes=480;e.max_utterance=32;e.wood_decay=.95
    words,sizes=pack_words([b'abcd'],32,e.device)
    e._episodic_bank=dict(source=torch.tensor([[1.,0.]]),action=torch.tensor([[0.,1.]]),
        wood_target=torch.tensor([[.2,.3,1.,.4]]),wood_baseline=torch.zeros(1,4),
        wood_option=torch.tensor([1]),feeling=torch.zeros(1,3),slots=torch.tensor([7]),
        age=torch.tensor([1000]),words=words,sizes=sizes,
        valid=torch.tensor([[True,True,True,gain]]))
    e._episodic_deciding=True
    return e


def test_suffix_compares_real_ends_and_never_padding():
    q,n=pack_words([b'abcd',b'xxabcd',b'',b'cd'],32,'cpu')
    k,m=pack_words([b'abcd',b'yyabcd',b'zzcd'],32,'cpu')
    actual=common_suffix(q,n,k,m)
    expected=torch.tensor([[4,4,2],[4,4,2],[0,0,0],[2,2,2]])
    assert torch.equal(actual,expected)


def test_same_spoken_action_crosses_generate_reuse_categories():
    e=reader();q,n=pack_words([b'abcd'],32,'cpu')
    f,p=e._read_features(torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]),torch.tensor([2]),
                         torch.zeros(1,4),q,n)
    torch.testing.assert_close(f[0,:4],e._episodic_bank['wood_target'][0])
    assert p['support'].gt(0).all()


def test_old_reward_contract_does_not_erase_valid_near_outcome():
    e=reader(gain=False);q,n=pack_words([b'abcd'],32,'cpu')
    f,p=e._read_features(torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]),torch.tensor([2]),
                         torch.zeros(1,4),q,n)
    assert torch.equal(f[0,:3],torch.tensor([.2,.3,1.]))
    assert f[0,3]==0 and p['support'][0,3]==0
    assert p['support'][0,:3].gt(0).all()


def test_replay_self_exclusion_erases_all_payload_features():
    e=reader();e._episodic_deciding=False;e._episodic_exclude=torch.tensor([7])
    e._episodic_query_feeling=torch.zeros(1,3)
    q,n=pack_words([b'abcd'],32,'cpu')
    f,_=e._read_features(torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]),torch.tensor([1]),
                         torch.zeros(1,4),q,n)
    assert torch.equal(f,torch.zeros_like(f))


def test_unknown_action_address_has_no_fabricated_history():
    e=reader();q,n=pack_words([b'zzzz'],32,'cpu')
    f,_=e._read_features(torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]),torch.tensor([1]),
                         torch.zeros(1,4),q,n)
    assert torch.equal(f,torch.zeros_like(f))


def test_missing_merged_address_is_not_silence():
    e=reader();e._episodic_bank['sizes'].zero_();e._episodic_bank['wood_option'].zero_()
    q,n=pack_words([b''],32,'cpu');n.fill_(-1)
    f,_=e._read_features(torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]),torch.tensor([1]),
                         torch.zeros(1,4),q,n)
    assert torch.equal(f,torch.zeros_like(f))


def test_observed_silence_can_recall_observed_silence():
    e=reader();e._episodic_bank['sizes'].zero_();e._episodic_bank['wood_option'].zero_()
    q,n=pack_words([b''],32,'cpu')
    f,p=e._read_features(torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]),torch.tensor([0]),
                         torch.zeros(1,4),q,n)
    assert p['support'].gt(0).all() and f[0,0]>0


def test_shuffle_preserves_time_option_and_provenance_strata():
    values=torch.arange(32.).reshape(8,4)
    option=torch.tensor([0,0,0,0,1,1,1,1])
    age=torch.tensor([0,1,100000,100001,0,1,100000,100001])
    valid=torch.tensor([0,0,1,1,0,0,1,1],dtype=torch.bool)
    state=torch.get_rng_state().clone()
    actual=stratified_shuffle(values,option,age,valid,event_bytes=992,decay=.95)
    assert torch.equal(actual,values[torch.tensor([1,0,3,2,5,4,7,6])])
    assert torch.equal(state,torch.get_rng_state())


def test_new_f_input_is_neutral_preserves_old_parameters_and_can_learn():
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.kanli=nn.Module()
            self.kanli.outcome=nn.Sequential(nn.Linear(131,64),nn.SiLU(),nn.Linear(64,64))
            self.memory=nn.Module();self.memory.capacity=2
    e=object.__new__(CausalWoodEngine);e.model=Model();e.device=torch.device('cpu')
    old=list(e.model.parameters());e.optimizer=torch.optim.AdamW(old,lr=3e-4)
    e.groups=(old[:1],old[1:2],old[2:]);x=torch.randn(2,131)
    before=e.model.kanli.outcome(x).detach().clone();rng=torch.get_rng_state().clone()
    configure_causal_wood(e,'real')
    assert torch.equal(rng,torch.get_rng_state())
    h=e.model.kanli.outcome[0](x)+e.model.kanli.causal_input(torch.randn(2,14))
    after=e.model.kanli.outcome[2](e.model.kanli.outcome[1](h))
    torch.testing.assert_close(before,after.detach(),rtol=0,atol=0)
    assert [id(p) for p in e.model.parameters()][:len(old)]==[id(p) for p in old]
    after[:,:4].square().sum().backward()
    assert e.model.kanli.causal_input.weight.grad.norm()>0
    assert e.model.kanli.causal_input.weight.numel()==896
    groups=len(e.optimizer.param_groups)
    configure_causal_wood(e,'blocked')
    assert len(e.optimizer.param_groups)==groups
    assert not e.model.memory.causal_evidence_valid.any()
    assert e.model.memory.causal_evidence.shape==(2,14)


def test_replay_uses_the_read_observed_then_not_a_future_bank():
    e=object.__new__(CausalWoodEngine)
    e.model=SimpleNamespace(encoder=nn.Linear(2,64),kanli=SimpleNamespace(
        outcome=nn.Sequential(nn.Linear(131,64),nn.SiLU(),nn.Linear(64,64)),
        causal_input=nn.Linear(14,64,bias=False)))
    e._episodic_deciding=False;e._wood_replay=(True,)
    e._batch_baseline=torch.zeros(2,4)
    e._causal_query_words,e._causal_query_sizes=pack_words([b'abcd'],32,'cpu')
    e._causal_replay_evidence=torch.full((1,14),2.)
    calls=[]
    def first_read(sources,*args):
        calls.append(len(sources))
        return torch.ones(1,14),{}
    e._read_features=first_read
    source=torch.eye(2);action=source+.2;options=torch.tensor([0,1])
    before=e._residual(source,action,options)
    e._read_features=lambda sources,*args:(torch.full((1,14),100.),{})
    after=e._residual(source,action,options)
    assert calls==[1]
    torch.testing.assert_close(before[1],after[1],rtol=0,atol=0)
    assert not torch.equal(before[0],after[0])


def test_old_unknown_evidence_is_not_reconstructed_or_relabelled():
    e=object.__new__(CausalWoodEngine)
    m=SimpleNamespace(occupied=torch.ones(2,dtype=torch.bool),wood_target_count=torch.ones(2,dtype=torch.long),
        wood_option=torch.tensor([0,1]),count=torch.ones(2,dtype=torch.long),source=torch.eye(2),
        action=torch.eye(2),wood_target=torch.zeros(2,4),wood_baseline=torch.zeros(2,4),
        feeling=torch.zeros(2,3),causal_evidence=torch.full((2,14),99.),
        causal_evidence_valid=torch.tensor([False,True]))
    e.model=SimpleNamespace(memory=m);e.replay_rng=np.random.default_rng(4);e.replay_batch=32
    e.reward_label_valid=lambda slots,wood:torch.ones_like(slots,dtype=torch.bool)
    e._wood_replay_batch()
    known=m.causal_evidence_valid[e._episodic_replay_slots]
    assert (~known).any() and known.any()
    assert not e._causal_replay_evidence[~known].any()
    assert e._causal_replay_evidence[known].eq(99).all()


def test_new_tag_is_observed_before_result_and_merge_does_not_fabricate_a_tag():
    e=object.__new__(CausalWoodEngine)
    m=SimpleNamespace(wood_target_count=torch.zeros(1,dtype=torch.long),wood_option=torch.full((1,),-1),
        count=torch.ones(1,dtype=torch.long),wood_target=torch.zeros(1,4),wood_baseline=torch.zeros(1,4),
        causal_evidence=torch.zeros(1,14),causal_evidence_valid=torch.zeros(1,dtype=torch.bool))
    e.model=SimpleNamespace(memory=m);e._forecast=dict(base=torch.zeros(3,4))
    observed=torch.arange(42.).reshape(3,14)
    e._episodic_decision_read=dict(features=observed)
    e._store_label(dict(episode_slot=0,episode_merged=False),1,torch.ones(4))
    assert m.causal_evidence_valid[0] and torch.equal(m.causal_evidence[0],observed[1])
    m.count[0]=2
    e._store_label(dict(episode_slot=0,episode_merged=True),1,torch.ones(4)*2)
    assert not m.causal_evidence_valid[0] and not m.causal_evidence[0].any()


def test_complete_state_hash_covers_words_flags_and_array_dtype():
    from scripts.ib_local.audit_water_wood_resume import digest
    assert digest([b'ab',b'c'])!=digest([b'a',b'bc'])
    assert digest(torch.tensor([True]))!=digest(torch.tensor([False]))
    assert digest(np.array([1],dtype=np.int32))!=digest(np.array([1],dtype=np.int64))
    assert digest(dict(a=1,b=2))==digest(dict(b=2,a=1))
