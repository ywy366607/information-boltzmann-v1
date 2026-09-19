"""Structural tests for the pure periodic 3D kinetic model."""
import torch

from scripts.ib_local.cbim_torus3d import (
    CBIMTorus3D,
    LocalInvariantCollision3D,
    VelocityCayleyTransport3D,
)


def test_torus_transport_preserves_norm():
    torch.manual_seed(1)
    transport = VelocityCayleyTransport3D((4, 6, 8), 8, 2).double()
    field = torch.randn(2, 4, 6, 8, 16, dtype=torch.float64)
    output, _ = transport(field)
    torch.testing.assert_close(output.norm(), field.norm(), atol=3e-12, rtol=3e-12)


def test_collision_is_local_and_preserves_invariants():
    torch.manual_seed(2)
    collision = LocalInvariantCollision3D((4, 4, 4), 8, 2).double()
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64)
    output, _ = collision(field)
    flat_before, flat_after = field.reshape(2, -1, 16), output.reshape(2, -1, 16)
    constraints = collision.constraints.to(field)
    before = torch.einsum("cd,bnd->bnc", constraints, flat_before)
    after = torch.einsum("cd,bnd->bnc", constraints, flat_after)
    torch.testing.assert_close(after, before, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(output.square().sum(-1), field.square().sum(-1),
                               atol=2e-11, rtol=2e-11)


def test_full_model_gradients_and_interventions():
    torch.manual_seed(3)
    model = CBIMTorus3D(vocab_size=32, shape=(4, 4, 4),
                        velocities=8, content_dim=2)
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, state, diagnostics = model(ids, targets)
    assert state.shape == (1, 4, 4, 4, 16)
    assert torch.isfinite(loss)
    assert diagnostics["write_balance_residual"] < 1e-5
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)
    with torch.no_grad():
        full, _, _ = model.step(state, torch.tensor([6]))
        no_collision, _, _ = model.step(
            state, torch.tensor([6]), disable_collision=True)
    assert (full - no_collision).abs().amax() > 1e-7


def test_weights_transfer_across_torus_resolutions():
    coarse = CBIMTorus3D(vocab_size=32, shape=(4, 4, 4),
                         velocities=8, content_dim=2)
    fine = CBIMTorus3D(vocab_size=32, shape=(8, 8, 4),
                       velocities=8, content_dim=2)
    fine.load_state_dict(coarse.state_dict(), strict=True)
