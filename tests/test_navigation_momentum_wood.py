"""Decision-time momentum receipt and real later boundary credit contracts."""
import copy

import torch

from scripts.ib_local.navigation_bfss_dynamics import configure_momentum
from scripts.ib_local.navigation_momentum_wood import configure_momentum_wood, decision_momentum_read
from scripts.ib_local.navigation_internal_rollout import InternalRolloutEngine
from test_navigation_internal_rollout import engine


def configured():
    e = engine()
    e.recurrent = e.model.body.base.initial_state(1)
    e.optimizer = torch.optim.AdamW(e.model.parameters(), lr=.001)
    e.params = list(e.model.parameters())
    e.groups = ([], [], e.params)
    configure_momentum(e)
    configure_momentum_wood(e)
    return e


def test_zero_initial_port_preserves_old_transition_and_does_not_change_rng():
    e = engine()
    e.recurrent = e.model.body.base.initial_state(1)
    e.optimizer = torch.optim.AdamW(e.model.parameters(), lr=.001)
    e.params = list(e.model.parameters()); e.groups = ([], [], e.params)
    configure_momentum(e)
    state = torch.get_rng_state().clone()
    old = copy.deepcopy(e.model.state_dict())
    configure_momentum_wood(e)
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(v, e.model.state_dict()[k]) for k, v in old.items() if torch.is_tensor(v))
    assert torch.count_nonzero(e.model.momentum_to_boundary.weight) == 0
    assert any(p is e.model.momentum_to_boundary.weight for g in e.optimizer.param_groups for p in g['params'])


def test_later_real_boundary_updates_actual_participating_momentum_read(monkeypatch):
    e = configured()
    for p in e.bfss_momenta:
        p.normal_()
    source, action, target = torch.randn(16), torch.randn(16), torch.randn(16)
    p_read = decision_momentum_read(e.model, e.bfss_momenta, source[None])[0].detach()
    e.internal_pending = {'momentum_read': p_read}
    e._internal_label = (source, action, target)
    monkeypatch.setattr(InternalRolloutEngine, '_outcome_loss',
        lambda self, *args: (self.model.encoder.weight.sum()*0, {}))
    loss, diag = e._outcome_loss(None, None, None)
    loss.backward()
    assert diag['momentum_boundary_samples'] == 1
    assert e.model.momentum_to_boundary.weight.grad.norm() > 0
    e.optimizer.step()
    assert e.model.momentum_to_boundary.weight.norm() > 0
    e._internal_label = None
    _, rejected = e._outcome_loss(None, None, None)
    assert rejected['momentum_boundary_samples'] == 0
