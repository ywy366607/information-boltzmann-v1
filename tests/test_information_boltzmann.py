"""Behavioral contracts for the isolated continuous-phase prototype."""
import math

import pytest
import torch

from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState
from fine_grain.information_boltzmann.collision import CollisionKernel, reflect, spatial_kernel
from fine_grain.information_boltzmann.density import InitialDensity
from fine_grain.information_boltzmann.force import DataForce
from fine_grain.information_boltzmann.streaming import StreamRunner


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(11)


def tiny(**kwargs):
    return InformationBoltzmann(vocab_size=12, phase_dim=2, particles=8,
                               hidden_dim=8, flow_layers=2, steps=2, **kwargs)


def test_joint_flow_inverse_jacobian_and_conditional_samples():
    op = InitialDensity(8, 2, 12, 4).double()
    distribution = op.condition(torch.tensor([1, 2]))
    z = torch.tensor([0.1, -0.3, 0.5, 0.2], dtype=torch.float64, requires_grad=True)
    transformed, ld = op.transform(z, distribution.context)
    recovered, ild = op.transform(transformed, distribution.context, inverse=True)
    assert torch.allclose(z, recovered, atol=1e-12)
    assert torch.allclose(ld, -ild, atol=1e-12)
    jac = torch.autograd.functional.jacobian(lambda a: op.transform(a, distribution.context)[0], z)
    assert torch.allclose(torch.linalg.slogdet(jac)[1], ld, atol=1e-10)
    assert jac[:2, 2:].abs().sum() > 1e-5  # joint x/v coupling, not independent embeddings
    a = distribution.sample(100, torch.Generator().manual_seed(2))
    b = op.condition(torch.tensor([3, 4])).sample(100, torch.Generator().manual_seed(2))
    assert not torch.allclose(a.x, b.x)
    assert a.x.std() > 0 and a.v.std() > 0
    assert torch.isfinite(distribution.log_prob(a.x, a.v)).all()


def test_density_normalization_by_two_dimensional_quadrature():
    op = InitialDensity(4, 1, 8, 2).double()
    density = op.condition(torch.tensor([1]))
    grid = torch.linspace(-5, 5, 251, dtype=torch.float64)
    x, v = torch.meshgrid(grid, grid, indexing="ij")
    with torch.no_grad():
        f = density.log_prob(x.reshape(-1, 1), v.reshape(-1, 1)).exp().reshape(251, 251)
    mass = torch.trapezoid(torch.trapezoid(f, grid, dim=1), grid)
    assert abs(float(mass) - 1) < 0.005


def test_free_transport_and_constant_force_analytic_solutions():
    force = DataForce(4, 2, 4, kappa=0, gamma=0, amplitude=0).double()
    state = PhaseState(torch.randn(6, 2, dtype=torch.float64), torch.randn(6, 2, dtype=torch.float64))
    out = force.transport(state, 1, 0.7)
    assert torch.allclose(out.x, state.x + 0.7 * state.v, atol=1e-12)
    assert torch.equal(out.v, state.v)
    acceleration = torch.tensor([0.3, -0.2], dtype=torch.float64)
    force.drive = lambda x, token, time: acceleration.expand_as(x)
    out = force.transport(state, 1, 0.7)
    assert torch.allclose(out.x, state.x + .7 * state.v + .5 * .7**2 * acceleration, atol=1e-12)
    assert torch.allclose(out.v, state.v + .7 * acceleration, atol=1e-12)


def test_damping_compression_and_force_bound():
    force = DataForce(4, 2, 8, kappa=1, gamma=.7, amplitude=.3).double()
    x = torch.randn(7, 2, dtype=torch.float64)
    v = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda q: force(x, q, 1, .2), v)
    flat = jac.reshape(14, 14)
    assert torch.allclose(flat, -.7 * torch.eye(14, dtype=torch.float64), atol=1e-12)
    assert force.drive(x, 1, 0).norm(dim=-1).max() <= .3
    kicked = force.kick(x, v, 1, .2, .5)
    kick_jac = torch.autograd.functional.jacobian(lambda q: force.kick(x, q, 1, .2, .5), v).reshape(14, 14)
    assert torch.allclose(torch.linalg.slogdet(kick_jac)[1], x.new_tensor(-.7*.5*14))
    assert torch.isfinite(kicked).all()


def test_transport_step_refinement_for_damped_oscillator():
    force = DataForce(4, 2, 4, kappa=1, gamma=1, amplitude=0).double()
    state = PhaseState(torch.tensor([[1., 0.]], dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64))
    omega = math.sqrt(3) / 2
    exact_x = math.exp(-.5) * (math.cos(omega) + .5/omega * math.sin(omega))
    exact_v = -math.exp(-.5) / omega * math.sin(omega)
    errors = []
    for steps in (4, 8, 16):
        current = state
        for _ in range(steps):
            current = force.transport(current, 1, 1 / steps)
        errors.append(abs(float(current.x[0, 0].detach())-exact_x) + abs(float(current.v[0, 0].detach())-exact_v))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .001


def test_collision_microreversibility_and_h_integrand():
    kernel = CollisionKernel(2, 8).double()
    background = torch.randn(10, 2, dtype=torch.float64)
    positions = torch.zeros_like(background)
    v, w = background[0], background[1]
    normal = torch.tensor([.6, .8], dtype=torch.float64)
    vp, wp = reflect(v, w, normal)
    rate = kernel.rate(positions[0], v, w, normal, positions, background)
    for a, b, n in ((w, v, normal), (vp, wp, normal), (v, w, -normal)):
        other = kernel.rate(positions[0], a, b, n, positions, background)
        assert torch.allclose(rate, other, atol=1e-12)
    logf = lambda z: (-torch.nn.functional.softplus(z) - torch.nn.functional.softplus(-z)).sum()
    log_a, log_b = logf(v)+logf(w), logf(vp)+logf(wp)
    # Direct symmetrized collision weak form for log(f), not entropy of Dirac samples.
    weak_h = .25 * rate * (log_b.exp()-log_a.exp()) * (log_a-log_b)
    assert weak_h <= 1e-14
    assert torch.allclose(v+w, vp+wp, atol=1e-12)
    assert torch.allclose(v.square().sum()+w.square().sum(), vp.square().sum()+wp.square().sum(), atol=1e-12)


def test_collision_keeps_positions_mass_momentum_energy_and_has_events():
    kernel = CollisionKernel(2, 8, max_rate=10).double()
    state = PhaseState(torch.zeros(8, 2, dtype=torch.float64), torch.randn(8, 2, dtype=torch.float64))
    out, lp, stats = kernel(state, 1, torch.Generator().manual_seed(3))
    assert stats["accepted"] > 0
    assert torch.equal(out.x, state.x)
    assert out.v.shape == state.v.shape
    assert torch.allclose(out.v.sum(0), state.v.sum(0), atol=1e-12)
    assert torch.allclose(out.v.square().sum(), state.v.square().sum(), atol=1e-10)
    assert abs(stats["cross_moment_change"]) < 1e-12
    assert torch.isfinite(lp)
    lp.backward()
    assert kernel.output.bias.grad.abs().sum() > 0


def test_spatial_kernel_has_density_scaling_and_compact_support():
    zero = torch.zeros(2)
    assert spatial_kernel(zero, 2) == .25
    assert spatial_kernel(torch.tensor([2., 0.]), 1) == 0


def test_collision_score_gradient_matches_expected_finite_difference(monkeypatch):
    # Condition on one dominating event (its count is parameter independent).
    monkeypatch.setattr(torch, "poisson", lambda value, generator=None: torch.ones_like(value))
    kernel = CollisionKernel(2, 4).double()
    with torch.no_grad():
        kernel.output.weight.zero_()
        kernel.output.bias.zero_()
    state = PhaseState(torch.zeros(2, 2, dtype=torch.float64),
                       torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64))
    generator = torch.Generator().manual_seed(19)
    gradients = []
    for _ in range(512):
        out, log_prob, _ = kernel(state, 1, generator)
        reward = out.v[0, 0].square()
        surrogate = (reward.detach() - .75) * log_prob
        gradients.append(float(torch.autograd.grad(surrogate, kernel.output.bias)[0]))
    # Uniform continuous 2D reflection has E[v'_x²]=1/2. Acceptance=sigmoid(b).
    expected = lambda bias: 1 - .5 / (1 + math.exp(-bias))
    finite_difference = (expected(1e-5)-expected(-1e-5)) / 2e-5
    samples = torch.tensor(gradients)
    standard_error = float(samples.std() / math.sqrt(len(samples)))
    assert abs(float(samples.mean())-finite_difference) < 4 * standard_error + .005


def test_decoder_permutation_invariance_and_belief_psd():
    model = tiny()
    state = model.initialize(torch.tensor([1]))
    order = torch.randperm(len(state.x))
    other = PhaseState(state.x[order], state.v[order])
    assert torch.allclose(model.decode(state), model.decode(other), atol=1e-6)
    mean, covariance = state.belief()
    assert torch.allclose(covariance, covariance.T)
    assert torch.linalg.eigvalsh(covariance).min() > -1e-7
    shifted = PhaseState(state.x+3, state.v)
    assert torch.allclose(shifted.belief()[0], mean+3)
    assert torch.allclose(shifted.belief()[1], covariance, atol=1e-6)


def test_stream_predict_before_observe_and_no_reset():
    model = tiny()
    runner = StreamRunner(model, 1, optimizer=torch.optim.Adam(model.parameters(), lr=.001), update_every=2)
    with pytest.raises(RuntimeError):
        runner.observe(3)
    before = runner.predict()
    with pytest.raises(RuntimeError):
        runner.predict()
    runner.observe(3)
    assert runner.current_token == 3 and runner.events == 1
    assert runner.state.time == pytest.approx(1)
    runner.predict()
    runner.observe(4)
    assert runner.updates == 1 and runner.state.time == pytest.approx(2)
    assert not runner.state.x.requires_grad
    assert before.shape == (12,)
    runner.predict()
    with pytest.raises(ValueError):
        runner.observe(5, external=False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_restores_online_rng_and_phase(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = tiny().to(device)
    runner = StreamRunner(model, 1, optimizer=torch.optim.Adam(model.parameters(), lr=.001), update_every=2)
    for token in [3, 4]:
        runner.predict()
        runner.observe(token)
    runner.save(tmp_path / "state.pt")
    reference = []
    for token in [5, 6]:
        reference.append(runner.predict())
        runner.observe(token)
    restored_model = tiny().to(device)
    restored = StreamRunner(restored_model, 1, optimizer=torch.optim.Adam(restored_model.parameters(), lr=.001), update_every=2)
    restored.load(tmp_path / "state.pt")
    for expected, token in zip(reference, [5, 6]):
        assert torch.equal(expected, restored.predict())
        restored.observe(token)
    assert torch.equal(runner.state.x, restored.state.x)
    assert torch.equal(runner.state.v, restored.state.v)
    for a, b in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(a, b)


def test_future_token_cannot_affect_pending_prediction():
    model = tiny()
    a, b = StreamRunner(model, 1, seed=3), StreamRunner(model, 1, seed=3)
    assert torch.equal(a.predict(), b.predict())
    a.observe(3)
    b.observe(4)
    assert not torch.allclose(a.predict(), b.predict())


def test_invalid_one_dimensional_elastic_model_is_rejected():
    with pytest.raises(ValueError):
        InformationBoltzmann(phase_dim=1)


def test_langevin_fluctuation_dissipation_prevents_variance_collapse():
    # Route B: Langevin fluctuation-dissipation thermal bath prevents collapse
    model_zero = tiny(temperature=0.0)
    model_langevin = tiny(temperature=0.1)
    gen_zero = torch.Generator().manual_seed(42)
    gen_lang = torch.Generator().manual_seed(42)
    state_zero = model_zero.initialize(torch.tensor([1]), generator=gen_zero)
    state_lang = model_langevin.initialize(torch.tensor([1]), generator=gen_lang)

    for _ in range(32):
        state_zero, _, _ = model_zero.advance(state_zero, 1, generator=gen_zero)
        state_lang, _, _ = model_langevin.advance(state_lang, 1, generator=gen_lang)

    var_zero = float(state_zero.belief()[1].trace().detach())
    var_lang = float(state_lang.belief()[1].trace().detach())
    assert var_zero < 1e-10  # Damped oscillator without noise collapses
    assert var_lang > 0.05   # Langevin thermal equilibrium maintains finite variance


def test_langevin_checkpoint_bit_for_bit_reproducibility(tmp_path):
    model = tiny(temperature=0.1)
    runner = StreamRunner(model, 1, optimizer=torch.optim.Adam(model.parameters(), lr=.001), update_every=2)
    for token in [3, 4]:
        runner.predict()
        runner.observe(token)
    runner.save(tmp_path / "langevin_state.pt")
    reference = []
    for token in [5, 6]:
        reference.append(runner.predict())
        runner.observe(token)
    restored_model = tiny(temperature=0.1)
    restored = StreamRunner(restored_model, 1, optimizer=torch.optim.Adam(restored_model.parameters(), lr=.001), update_every=2)
    restored.load(tmp_path / "langevin_state.pt")
    for expected, token in zip(reference, [5, 6]):
        assert torch.equal(expected, restored.predict())
        restored.observe(token)
    assert torch.equal(runner.state.x, restored.state.x)
    assert torch.equal(runner.state.v, restored.state.v)


def test_unvalidated_adaptive_controller_cannot_silently_run():
    model = tiny(temperature=.1, adaptive_gamma=True)
    state = model.initialize(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="controller disabled"):
        model.advance(state, 1, torch.Generator().manual_seed(1))


def test_zero_gamma_has_zero_thermal_diffusion():
    force = DataForce(4, 2, 4, kappa=0, gamma=0, amplitude=0, temperature=.1)
    x, v = torch.zeros(10, 2), torch.randn(10, 2)
    assert torch.equal(force.kick(x, v, 1, 0, 1), v)


def test_full_phase_response_preserves_velocity_perturbation():
    from fine_grain.information_boltzmann.diagnostics import renormalize
    base = PhaseState(torch.zeros(2, 2), torch.zeros(2, 2))
    shadow = PhaseState(torch.ones(2, 2), 2*torch.ones(2, 2))
    new, distance = renormalize(base, shadow, .01, 1)
    assert distance == pytest.approx(math.sqrt(20))
    assert new.v.norm() > 0
    assert torch.cat((new.x,new.v),-1).norm() == pytest.approx(.01)


def test_full_phase_response_undamped_oscillator():
    from fine_grain.information_boltzmann.diagnostics import conditional_response
    model = tiny(gamma=0, amplitude=0, collision_rate=0).double()
    model.steps=16
    state = model.initialize(torch.tensor([1]), torch.Generator().manual_seed(1))
    response = conditional_response(model, state, [1]*48, torch.Generator().manual_seed(2), burn_in=8)
    assert abs(response["finite_response_rate"]) < .002
