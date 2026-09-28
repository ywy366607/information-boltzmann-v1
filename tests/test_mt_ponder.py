import torch

from information_boltzmann.core.mt_ponder import CBIMActivePonder3D
from information_boltzmann.core.torus3d import CBIMTorus3D


def _model() -> CBIMActivePonder3D:
    core = CBIMTorus3D(
        vocab_size=31, shape=(2, 2, 2), velocities=8, content_dim=4,
        collision_layers=2, relative_address=False, v2_coordinate_components=True,
        readout_type="kernel_r1", readout_probes=2, readout_rounds=1,
        write_type="w2_impedance", micro_steps=1, adaptive_clock=False,
        continuous_velocities=True, dissipation_type="quadratic",
    )
    return CBIMActivePonder3D(core, branches=4, integration_steps=3, latent_dim=8)


def test_mt_ponder_uses_complete_fields_and_joint_vfe_gradients():
    torch.manual_seed(0)
    model = _model()
    state = model.initial_state(2, warm_start=False)
    target = torch.tensor([7, 9])
    output = model.event(state, torch.tensor([3, 5]), target_ids=target)

    assert output.log_probs.shape == (2, 31)
    assert output.branch_fields.shape == (2, 4, 2, 2, 2, 32)
    assert output.next_field.shape == (2, 2, 2, 2, 32)
    assert torch.allclose(output.route_weights.sum(-1), torch.ones(2), atol=1e-6)
    assert float(output.diagnostics["branch_field_spread"]) > 0.0
    assert torch.allclose(
        output.diagnostics["integration_dt_sum"],
        output.diagnostics["internal_time_mean"],
        atol=1e-6,
    )
    marginal_nll = -output.log_probs.gather(-1, target[:, None]).mean()
    assert torch.allclose(output.diagnostics["vfe"], marginal_nll, atol=1e-5)

    loss = marginal_nll + output.diagnostics["critic_loss"] + 0.01 * (
        output.diagnostics["grpo_route_loss"] + output.diagnostics["grpo_path_loss"]
    )
    loss.backward()
    assert model.core.collision.angle[-1].weight.grad is not None
    assert model.prior.net[-1].weight.grad is not None


def test_mt_ponder_deployment_uses_prior_without_target():
    model = _model().eval()
    state = model.initial_state(1, warm_start=False)
    output = model.event(state, torch.tensor([4]), deterministic=True)
    assert output.log_probs.shape == (1, 31)
    assert not bool(output.diagnostics["posterior_active"])
    assert torch.isfinite(output.log_probs).all()
