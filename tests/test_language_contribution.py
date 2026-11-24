"""Read-only likelihood accounting on the existing graph, no capability task."""
import copy

import numpy as np
import torch

from scripts.audit_language_contribution import language_contribution
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_language_navigation_v2 import validate


def test_accounting_reuses_full_forward_and_restores_live_graph_and_memory():
    torch.manual_seed(49)
    base = GatedDeltaNetLM(256,16,1,1,'decay_then_delta',32,4)
    model = LatentNavigationV3(TableGDN(base,True,inject_layer=0),latent_dim=8,
        capacity=512,short=2,long=512,max_chunk=512)
    table = RetrievedCreditTable(orders=(0,),heads=1,slots_log2=2)
    valid = np.frombuffer(b'A complete observed text prefix and continuation.',np.uint8)
    expected = validate(model,table,valid)
    before = copy.deepcopy(model.state_dict())
    memory = model.memory
    rng = torch.get_rng_state().clone()
    count = [0]
    handle = model.base.decoder.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
    result = language_contribution(model,table,valid)
    handle.remove()
    assert count[0] == 1
    assert model.memory is memory and 'predict_observed' not in model.__dict__
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in before.items():
        if torch.is_tensor(value): assert torch.equal(value,model.state_dict()[key])
    assert result['bytes'] == len(valid)-1
    assert abs(result['full_nll']-expected)<1e-6
    assert abs(result['full_gain_over_table']-(result['table_prior_nll']-expected))<1e-6
