"""Motor policy regularization follows actual available decision intervals."""
import pytest
import torch

from information_boltzmann.core.fly_active_inference import habit_kl, motor_policy_kl


def test_unavailable_turn_has_no_motor_regularizer_gradient():
    logits = torch.tensor([-8., -5., -9.], requires_grad=True)
    value, ticks = motor_policy_kl(logits.sigmoid(), .2, [])
    value.backward()
    assert ticks == 0 and value.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_actual_decisions_exclude_protected_mother_ticks():
    logits = torch.tensor([-8., -2., 1., -7.], requires_grad=True)
    value, ticks = motor_policy_kl(logits.sigmoid(), .2, [(0, 1, 2, 0)])
    torch.testing.assert_close(value, habit_kl(logits[1:3].sigmoid(), .2).mean())
    value.backward()
    assert ticks == 2 and logits.grad[1:3].abs().min() > 0
    assert logits.grad[0] == logits.grad[3] == 0
    with pytest.raises(ValueError, match='physical clock'):
        motor_policy_kl(logits.sigmoid(), .2, [(0, 3, 2, 0)])


def test_unrestricted_motor_policy_keeps_existing_regularizer():
    h = torch.tensor([.1, .6])
    value, ticks = motor_policy_kl(h, .2)
    torch.testing.assert_close(value, habit_kl(h, .2).mean())
    assert ticks == 2
