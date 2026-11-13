"""Verify diagnostic accounting, not task capability."""
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from scripts.ib_local.audit_small_gdn_ruler_routes import margin_components, intervention
import scripts.ib_local.small_gdn_ruler as adapter


def test_margin_split_matches_full_softmax_log_odds():
    decoder=torch.tensor([2.,-1.,5.])
    table=torch.tensor([.2,.5,.3]).log()
    weight=torch.tensor(1.7)
    parts=margin_components(decoder,table,weight,0,1)
    logits=(decoder+weight*table).log_softmax(-1)
    assert parts['total_margin']==pytest.approx(float(logits[0]-logits[1]))


def test_output_table_cut_and_uniform_input_restore_after_exception():
    model=SimpleNamespace(body=SimpleNamespace(mix=nn.Linear(2,1)),to_hidden=nn.Linear(2,2))
    x=torch.ones(1,1,2)
    before=model.body.mix(x).detach().clone()
    original=adapter.probability_tensor
    with pytest.raises(RuntimeError):
        with intervention(model,'no_output_table'):
            assert model.body.mix(x).item()==-80
            raise RuntimeError('diagnostic interrupted')
    torch.testing.assert_close(model.body.mix(x),before)
    with intervention(model,'uniform_table'):
        assert adapter.probability_tensor is not original
    assert adapter.probability_tensor is original
    with intervention(model,'no_causal_injection'):
        assert model.to_hidden(x).abs().sum()==0
