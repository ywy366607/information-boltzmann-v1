"""Structural tests for the Q8 predictive-impedance write agent."""

import torch

from information_boltzmann.core.torus3d import (
    CBIMTorus3D,
    FullRankTorusWrite,
    PredictiveImpedanceWriteAgent,
)


def test_zero_innovation_is_exact_identity_port_event():
    torch.manual_seed(7)
    writer = FullRankTorusWrite(vocab_size=19, shape=(4, 4, 4), d=16,
                                 write_type="w2_impedance").double()
    agent = PredictiveImpedanceWriteAgent(16).double()
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64)
    precision = agent.initial_precision(2, device=field.device, dtype=field.dtype)
    token_ids = torch.tensor([3, 11])
    observed_token = writer.embedding(token_ids)

    next_field, _, reflected, diagnostics = agent(
        writer, field, token_ids, precision, predicted_token=observed_token)

    torch.testing.assert_close(next_field, field, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(reflected, torch.zeros_like(reflected),
                               atol=2e-12, rtol=2e-12)
    assert diagnostics["innovation_energy"].item() < 2e-24
    assert diagnostics["write_balance_residual"].item() < 2e-12


def test_belief_step_carries_precision_and_has_finite_gradients():
    torch.manual_seed(8)
    model = CBIMTorus3D(vocab_size=23, shape=(4, 4, 4), velocities=8,
                        content_dim=2, write_type="w4_predictive_agent",
                        micro_steps=1)
    belief = model.initial_belief(1)
    logits, next_belief, diagnostics = model.belief_step(
        belief, torch.tensor([5]))

    assert logits.shape == (1, 23)
    assert next_belief.field.shape == belief.field.shape
    assert next_belief.precision.shape == (1, 16)
    assert torch.isfinite(next_belief.precision).all()
    assert (next_belief.precision > 0).all()
    assert "write_free_energy" in diagnostics
    assert "_posterior_precision" not in diagnostics

    ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])
    loss, final_belief, final_diagnostics = model.forward_belief(ids, targets)
    assert torch.isfinite(loss)
    assert torch.isfinite(final_belief.field).all()
    assert torch.isfinite(final_belief.precision).all()
    assert "token_nll" in final_diagnostics
    assert "write_free_energy_mean" in final_diagnostics
    loss.backward()
    for parameter in model.write_agent.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
