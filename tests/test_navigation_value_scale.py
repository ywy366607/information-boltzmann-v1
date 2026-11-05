"""Numerical invariants of a reparameterization, not task capability tests."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ib_local.navigation_value_scale import (
    configure_value_scale, critic_output, normalized_critic_loss, update_value_scale)


def body():
    torch.manual_seed(17)
    critic = nn.Sequential(nn.Linear(7, 9), nn.SiLU(), nn.Linear(9, 1)).double()
    return SimpleNamespace(model=SimpleNamespace(kanli=SimpleNamespace(critic=critic)))


def test_default_is_exactly_the_existing_critic_and_adds_no_state():
    engine = body()
    x = torch.randn(4, 7, dtype=torch.float64)
    configure_value_scale(engine, 'none')
    assert torch.equal(critic_output(engine, x), engine.model.kanli.critic(x))
    assert not hasattr(engine, 'value_scale')


def test_rebase_preserves_outputs_input_derivatives_and_choice_including_recall():
    engine = body()
    configure_value_scale(engine, 'popart')
    x = torch.randn(32, 7, dtype=torch.float64, requires_grad=True)
    y = critic_output(engine, x)
    grad = torch.autograd.grad(y.sum(), x)[0]
    choice = (y.squeeze(-1)+x[:, -2]).softmax(0)
    for target in (1600., -32., .001, 90.):
        update_value_scale(engine, target)
        after = critic_output(engine, x)
        torch.testing.assert_close(after, y, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(torch.autograd.grad(after.sum(), x)[0], grad,
                                   rtol=1e-11, atol=1e-11)
        torch.testing.assert_close((after.squeeze(-1)+x[:, -2]).softmax(0), choice,
                                   rtol=1e-11, atol=1e-11)


def test_loss_uses_normalized_residual_but_keeps_raw_recalled_value():
    engine = body()
    configure_value_scale(engine, 'popart')
    x = torch.randn(1, 7, dtype=torch.float64)
    x[:, -2] = 50.
    raw = float(critic_output(engine, x).squeeze().detach())+50.
    loss, diag = normalized_critic_loss(engine, x, 1500.)
    assert diag['value_raw_residual_error'] == pytest.approx((raw-1500.)**2)
    assert float(loss.detach()) == pytest.approx((raw-1500.)**2/engine.value_scale['scale']**2)
    assert diag['value_rebase_max_error'] < 1e-11
    assert loss.requires_grad
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in engine.model.kanli.critic.parameters())
