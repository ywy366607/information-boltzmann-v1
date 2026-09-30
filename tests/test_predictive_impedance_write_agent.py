"""Structural tests for the Q8 predictive-impedance write agent."""

import torch
import math

from information_boltzmann.core.torus3d import (
    CBIMTorus3D,
    FullRankTorusWrite,
    PredictiveImpedanceWriteAgent,
    SelectiveOutflowBath,
    UnifiedKineticStateAgent,
    UnifiedTorusDissipation,
)


def test_zero_innovation_is_exact_identity_port_event():
    torch.manual_seed(7)
    writer = FullRankTorusWrite(vocab_size=19, shape=(4, 4, 4), d=16,
                                 write_type="w2_impedance").double()
    agent = PredictiveImpedanceWriteAgent(16, vocab_size=19).double()
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64)
    precision = agent.initial_precision(2, device=field.device, dtype=field.dtype)
    token_ids = torch.tensor([3, 11])
    observed_feature = torch.nn.functional.normalize(
        writer.embedding(token_ids), dim=-1)

    next_field, _, reflected, diagnostics = agent(
        writer, field, token_ids, precision, predicted_feature=observed_feature)

    torch.testing.assert_close(next_field, field, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(reflected, torch.zeros_like(reflected),
                               atol=2e-12, rtol=2e-12)
    assert diagnostics["innovation_energy"].item() < 2e-24
    assert diagnostics["write_balance_residual"].item() < 2e-12


def test_event_norm_opens_the_port_and_conserves_the_ledger():
    """An unpredictable token must exchange a finite fraction of one
    unit-norm incident mode instead of a grid-averaged pointwise sliver."""
    torch.manual_seed(13)
    writer = FullRankTorusWrite(vocab_size=19, shape=(4, 4, 4), d=16,
                                 write_type="w2_impedance").double()
    agent = PredictiveImpedanceWriteAgent(16, vocab_size=19).double()
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64)
    precision = agent.initial_precision(2, device=field.device, dtype=field.dtype)
    token_ids = torch.tensor([3, 11])
    observed_feature = torch.nn.functional.normalize(
        writer.embedding(token_ids), dim=-1)
    noise = torch.randn(observed_feature.shape, dtype=observed_feature.dtype)
    noise = noise - (noise * observed_feature).sum(-1, keepdim=True) * observed_feature
    predicted_feature = torch.nn.functional.normalize(noise, dim=-1)

    next_field, _, reflected, diagnostics = agent(
        writer, field, token_ids, precision, predicted_feature=predicted_feature)

    # The rotation acts on (field, unit incident mode), so the boundary
    # ledger is an exact orthogonal identity up to floating point.
    assert diagnostics["write_balance_residual"].item() < 1e-12
    # The event angle is driven by the event-total norm, not the per-site
    # magnitude: at unit admittance the angle must be an O(0.1) rad event
    # quantity rather than the ~1e-2 pointwise sliver of the old law.
    assert 0.05 < diagnostics["write_angle_abs_mean"].item() < 1.2
    assert 0.005 < diagnostics["accepted_fraction"].item() < 0.95
    assert diagnostics["innovation_norm"].item() > 0.05
    assert (next_field - field).norm() > 0.0
    assert reflected.abs().sum() > 0.0


def test_port_angle_at_registered_scale_is_finite_for_surprise():
    """At the registered d=128 packet calibration an unpredictable token
    has ||delta||_Pi ~ 1, hence a finite exchange at initial admittance."""
    torch.manual_seed(17)
    writer = FullRankTorusWrite(vocab_size=19, shape=(4, 4, 4), d=128,
                                 write_type="w2_impedance").double()
    agent = PredictiveImpedanceWriteAgent(128, vocab_size=19).double()
    field = torch.zeros(1, 4, 4, 4, 128, dtype=torch.float64)
    precision = agent.initial_precision(1, device=field.device, dtype=field.dtype)
    _, _, _, diagnostics = agent(
        writer, field, torch.tensor([3]), precision,
        predicted_feature=torch.zeros(1, 128, dtype=torch.float64))

    assert 0.5 < diagnostics["innovation_norm"].item() < 2.0
    assert 0.1 < diagnostics["write_angle_abs_mean"].item() < 1.3
    assert diagnostics["accepted_fraction"].item() > 0.03


def test_event_norm_and_port_angle_are_resolution_independent():
    """The site-averaged event norm must not change with the grid."""
    torch.manual_seed(5)
    agent = PredictiveImpedanceWriteAgent(16, vocab_size=19).double()
    writer_small = FullRankTorusWrite(vocab_size=19, shape=(4, 4, 4), d=16,
                                      write_type="w2_impedance").double()
    writer_large = FullRankTorusWrite(vocab_size=19, shape=(8, 4, 6), d=16,
                                      write_type="w2_impedance").double()
    with torch.no_grad():
        writer_large.embedding.weight.copy_(writer_small.embedding.weight)
        writer_large.channel_scale.copy_(writer_small.channel_scale)
    rows = []
    for writer in (writer_small, writer_large):
        field = torch.zeros(1, *writer.shape, 16, dtype=torch.float64)
        precision = agent.initial_precision(1, device=field.device,
                                            dtype=field.dtype)
        _, _, _, diagnostics = agent(
            writer, field, torch.tensor([3]), precision,
            predicted_feature=torch.zeros(1, 16, dtype=torch.float64))
        rows.append((diagnostics["innovation_norm"].item(),
                     diagnostics["write_angle_abs_mean"].item()))
    torch.testing.assert_close(rows[0][0], rows[1][0], rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(rows[0][1], rows[1][1], rtol=1e-8, atol=1e-12)


def test_factorized_packet_chart_commutes_with_categorical_expectation():
    torch.manual_seed(12)
    writer = FullRankTorusWrite(vocab_size=19, shape=(4, 4, 4), d=16,
                                 write_type="w2_impedance").double()
    agent = PredictiveImpedanceWriteAgent(16, vocab_size=19).double()
    field = torch.randn(1, 4, 4, 4, 16, dtype=torch.float64)
    precision = agent.initial_precision(1, device=field.device, dtype=field.dtype)
    field_summary = field.mean((1, 2, 3))
    prior_features = torch.cat((
        torch.nn.functional.rms_norm(field_summary, (16,)), precision.log()), -1)
    phi_a, phi_b = torch.randn(1, 16, dtype=torch.float64), torch.randn(1, 16, dtype=torch.float64)
    mixed = 0.3 * phi_a + 0.7 * phi_b

    packet_mixed, _, _, _ = agent._packet_chart(writer, field, prior_features, mixed)
    packet_a, _, _, _ = agent._packet_chart(writer, field, prior_features, phi_a)
    packet_b, _, _, _ = agent._packet_chart(writer, field, prior_features, phi_b)
    torch.testing.assert_close(packet_mixed, 0.3 * packet_a + 0.7 * packet_b,
                               atol=3e-12, rtol=3e-12)

    features = torch.nn.functional.normalize(writer.embedding.weight, dim=-1)
    probability = torch.softmax(torch.randn(19, dtype=torch.float64), dim=0)
    expected_feature = probability @ features
    expected_packet, _, _, _ = agent._packet_chart(
        writer, field, prior_features, expected_feature[None])
    categorical_packet = torch.zeros_like(expected_packet)
    for token, weight in enumerate(probability):
        packet, _, _, _ = agent._packet_chart(
            writer, field, prior_features, features[token][None])
        categorical_packet = categorical_packet + weight * packet
    torch.testing.assert_close(expected_packet, categorical_packet,
                               atol=4e-12, rtol=4e-12)


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
    assert "_write_free_energy" not in diagnostics
    assert "_read_action_complexity" not in diagnostics

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


def test_physical_read_agent_is_complete_and_field_only():
    """The read policy must see invariant and collision coordinates alike."""
    torch.manual_seed(23)
    model = CBIMTorus3D(
        vocab_size=23, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        micro_steps=1,
    ).double()
    agent = model.readout
    field = torch.randn(1, 4, 4, 4, 16, dtype=torch.float64)
    flat = field.reshape(1, -1, 16)
    coordinates = agent.physical_coordinates(flat)
    reconstructed = agent.reconstruct_physical_coordinates(coordinates)
    torch.testing.assert_close(reconstructed, flat, atol=2e-11, rtol=2e-11)

    precision = model.write_agent.initial_precision(
        1, device=field.device, dtype=field.dtype)
    zero = torch.zeros_like(field)
    baseline, _ = agent(zero, precision)
    invariant_shift = zero.clone()
    collision_shift = zero.clone()
    invariant_shift.reshape(1, -1, 16)[0, 0] = agent.invariant_basis[:, 0]
    collision_shift.reshape(1, -1, 16)[0, 0] = agent.nullspace[:, 0]
    invariant_feature, invariant_diag = agent(invariant_shift, precision, return_diag=True)
    collision_feature, collision_diag = agent(collision_shift, precision, return_diag=True)

    # Both tangent spaces alter the measurement; neither is hidden behind a
    # token residual or a value projection that can erase it at initialization.
    assert (invariant_feature - baseline).norm().item() > 1e-8
    assert (collision_feature - baseline).norm().item() > 1e-8
    assert invariant_diag["read_aperture_coverage"].item() > 0.0
    assert collision_diag["read_aperture_coverage"].item() > 0.0


def test_belief_read_agent_receives_next_token_gradients_without_token_input():
    torch.manual_seed(29)
    model = CBIMTorus3D(
        vocab_size=23, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        micro_steps=1,
    )
    ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])
    loss, _, diagnostics = model.forward_belief(ids, targets)
    assert torch.isfinite(loss)
    assert "read_action_kl" in diagnostics
    assert "read_action_entropy" in diagnostics
    loss.backward()
    for parameter in model.readout.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

def test_unified_dissipation_serves_the_w4_belief_path_field_only():
    """The three-layer unified bath must run token-free inside the port line
    with finite gradients and the same persistent precision semantics."""
    torch.manual_seed(31)
    model = CBIMTorus3D(
        vocab_size=23, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        micro_steps=2, dissipation_type="unified", dissipation_rank=2,
    )
    assert isinstance(model.bath, UnifiedTorusDissipation)

    ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])
    loss, belief, diagnostics = model.forward_belief(ids, targets)
    assert torch.isfinite(loss)
    assert torch.isfinite(belief.field).all()
    assert "dissipation_gamma0" in diagnostics
    assert "dissipation_nu" in diagnostics
    assert "dissipation_lambda_mean" in diagnostics

    loss.backward()
    for name in ("gamma0_param", "nu_param"):
        parameter = getattr(model.bath, name)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in (*model.bath.u_proj.parameters(),
                      *model.bath.lambda_net.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    # Structural invariant of the port line: the belief path never hands the
    # token embedding to the bath, so the three-layer dissipation stays
    # field-only and cannot become a second hidden write path.
    recorded = []
    original_forward = model.bath.forward

    def spy_forward(field, delta_tau, tok_embed=None, **kwargs):
        recorded.append(tok_embed)
        return original_forward(field, delta_tau, tok_embed=tok_embed, **kwargs)

    model.bath.forward = spy_forward
    loss, belief, diagnostics = model.forward_belief(ids, targets)
    assert recorded and all(token is None for token in recorded)


def test_unified_spectral_viscosity_annihilates_unit_torus_fundamental_at_t64():
    """Expose the physical time-scale of the optional spectral bath.

    This is an analytic operator test, not a language capability experiment.
    For a mode cos(2*pi*x), repeated ``dt=4`` evolution for 16 microsteps
    must yield exp(-2*T*nu*(2*pi)^2) in quadratic energy.  It prevents a
    raw physical viscosity from being mistaken for a duration-neutral bath.
    """
    shape = (8, 8, 4)
    bath = UnifiedTorusDissipation(
        shape=shape, d=1, rank=1, gamma0_init=1e-5, nu_init=0.02,
    ).double()
    x = torch.arange(shape[0], dtype=torch.float64) / shape[0]
    mode = torch.cos(2.0 * math.pi * x)[:, None, None, None]
    field = mode.expand(1, *shape, 1).clone()
    energy_initial = field.square().mean()
    nu = torch.nn.functional.softplus(bath.nu_param).item()
    field, _ = bath(
        field, delta_tau=4.0, gamma0_factor=0.0,
        disable_subspace=True,
    )
    measured_ratio = field.square().mean() / energy_initial
    fundamental_laplacian = bath.laplacian[1, 0, 0].item()
    assert math.isclose(fundamental_laplacian, (2.0 * math.pi) ** 2,
                        rel_tol=1e-6)
    expected_ratio = math.exp(-2.0 * 4.0 * nu * fundamental_laplacian)
    torch.testing.assert_close(
        measured_ratio, torch.tensor(expected_ratio, dtype=torch.float64),
        rtol=2e-8, atol=1e-14,
    )
    for _ in range(15):
        field, _ = bath(
            field, delta_tau=4.0, gamma0_factor=0.0,
            disable_subspace=True,
        )
    measured_ratio = field.square().mean() / energy_initial
    # The primary Q8 event lasts T=64; this is why the raw nu=0.02 bath
    # leaves DC as its sole long-lived spatial mode.  The analytic value is
    # about 1e-44; the looser bound accounts for accumulated FP64 FFT roundoff.
    assert measured_ratio.item() < 1e-30


def test_selective_outflow_bath_is_field_only_and_closes_energy_ledger():
    torch.manual_seed(37)
    bath = SelectiveOutflowBath(shape=(4, 4, 4), d=16).double()
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64)
    precision = torch.exp(torch.randn(2, 16, dtype=torch.float64))
    output, diag = bath(field, 0.3, precision=precision)
    assert torch.isfinite(output).all()
    assert diag["bath_out_energy"].item() > 0.0
    assert diag["bath_energy_residual"].item() < 2e-12
    assert diag["bath_selectivity"].item() >= 0.0
    energy_before = 0.5 * field.square().sum(-1).mean()
    energy_after = 0.5 * output.square().sum(-1).mean()
    torch.testing.assert_close(
        energy_before - energy_after, diag["bath_out_energy"],
        atol=2e-12, rtol=2e-12)


def test_state_agent_uses_palindromic_orders_over_a_microstep_pair():
    assert UnifiedKineticStateAgent.order_for_microstep(0) == (
        "transport", "collision", "bath")
    assert UnifiedKineticStateAgent.order_for_microstep(1) == (
        "bath", "collision", "transport")


def test_selective_bath_joint_state_path_has_finite_w4_gradients():
    torch.manual_seed(41)
    model = CBIMTorus3D(
        vocab_size=23, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        micro_steps=2, dissipation_type="selective")
    ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])
    loss, belief, diagnostics = model.forward_belief(ids, targets)
    assert torch.isfinite(loss)
    assert torch.isfinite(belief.field).all()
    assert diagnostics["state_agent_pairwise_symmetric"].item() == 1.0
    assert diagnostics["bath_out_energy"].item() >= 0.0
    loss.backward()
    for parameter in model.bath.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
