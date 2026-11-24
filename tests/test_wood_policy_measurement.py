"""Numerical/continuation contracts on the existing learner, not a capability task."""
import random
import gzip

import numpy as np
import pytest
import torch

from test_persistent_rhythm import body
from scripts.ib_local.action_value_pair_audit import first_action
from scripts.ib_local.autonomous_cycle import dispatch
from scripts.ib_local.finite_choice_contract import trace_choices
from scripts.ib_local.wood_policy_measurement import action_receipt,contribution,score,serializable_engine


def _two(engine):
    rows=[]
    for _ in range(256):
        row=dispatch(engine)
        if row['autonomous_event']['kind']=='external':rows.append(row)
        if len(rows)==2:return rows
    raise AssertionError('two complete external transitions required')


def test_conditional_actual_choice_keeps_original_complete_learning(tmp_path):
    torch.set_num_threads(1)
    captured=[];seed=201120
    def reset():
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    reset()
    a,archive=body(tmp_path/'natural',root_mode='observe',scope_prior='off')
    original=a._utter
    def observe():
        if captured:return original()
        with trace_choices() as trace:words,record=original()
        captured.append(action_receipt(a,words,record,trace))
        return words,record
    a._utter=observe
    try:
        rows_a=_two(a)
        del a._utter
        params={k:v.detach().clone() for k,v in a.model.state_dict().items() if torch.is_tensor(v)}
        recurrent=a.recurrent.clone();rng=torch.get_rng_state()
        table={k:getattr(a.table,k).copy() for k in ('rows','count','tag','err','epoch','last_write')}
        # A readout must not assimilate the material or change any RNG/state.
        material=[np.frombuffer(b'Natural text read.',dtype=np.uint8).copy()]
        assert score(a,material)==score(a,material)
        fn=lambda *args,**kwargs:None
        a._advance=fn
        with serializable_engine(a):
            assert '_advance' not in vars(a)
            from scripts.ib_local.mother_checkpoint import save
            destination=tmp_path/'checkpoint'
            destination.mkdir()
            save(a,destination,2,{})
        with gzip.open(destination/'last.pt.gz','rb') as stream:
            checkpoint=torch.load(stream,map_location='cpu',weights_only=False)
        assert '_advance' not in checkpoint['engine'] and '_utter' not in checkpoint['engine']
        assert a._advance is fn
        del a._advance
    finally:archive.close()
    reset()
    b,archive=body(tmp_path/'conditional',root_mode='observe',scope_prior='off')
    try:
        receipts=[]
        with first_action(b,captured[0]['option'],receipts.append,keep_actor_learning=True):
            rows_b=_two(b)
        assert receipts[0]['behavior']['categorical_score_gradient']==1
        assert captured[0]['production']==pytest.approx(receipts[0]['nominal_probability'],abs=1e-7)
        for x,y in zip(rows_a,rows_b):
            assert x['choice']==y['choice'] and x['utterance_hex']==y['utterance_hex']
            assert x['world']==y['world']
            assert x['metal_g']==y['metal_g']
        assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in params.items())
        assert torch.equal(recurrent,b.recurrent) and torch.equal(rng,torch.get_rng_state())
        assert all(np.array_equal(v,getattr(b.table,k)) for k,v in table.items())
    finally:archive.close()


def test_probability_contribution_is_center_invariant_and_factors_moved_mass():
    p=[.45867449,.28138584,.25993967]
    h=[.5009,.2642,.2349]
    g=[-.0087282615,.0062743002,.0060920584]
    got=contribution(p,h,g)
    shifted=contribution(p,h,np.asarray(g)+7)
    assert got['C']==pytest.approx(shifted['C'],abs=1e-14)
    assert got['C']==pytest.approx(got['probability_moved']*got['receiver_minus_donor_G'])
    assert got['C']==pytest.approx(got['production_expected_G']-got['blind_expected_G'])
