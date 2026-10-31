"""Behavioral contracts for the isolated continuous-phase prototype."""
import copy
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


def test_stream_runner_prepared_logits_score_before_observation():
    from information_boltzmann.core.cgb_information_ledger import PredictiveInformationLedger
    from information_boltzmann.core.cgb_low_resource_life import Observation
    from information_boltzmann.core.cgb_prequential_bridge import CGBPrequentialBridge

    runner = StreamRunner(tiny(), 1, seed=17)
    ledger = PredictiveInformationLedger()
    bridge = CGBPrequentialBridge(runner, ledger, lambda _: math.log(12), strict_scoring=True)
    for tick, target in enumerate((2, 3)):
        logits = bridge.prepare()
        expected = float(-torch.log_softmax(logits, dim=0)[target])
        result = bridge.step(Observation(tick, target), write_units=0, dissipation_units=0)
        assert math.isclose(result.model_nll_nats, expected, rel_tol=1e-6)
        assert torch.equal(result.prediction, logits.detach())
        assert runner.pending is None and runner.events == tick + 1


def test_gdn_byte_port_scores_before_actual_assimilation():
    from fine_grain.active_gdn2 import ActiveInferenceGDN2
    from information_boltzmann.core.cgb_information_ledger import PredictiveInformationLedger
    from information_boltzmann.core.cgb_low_resource_life import Observation
    from information_boltzmann.core.cgb_prequential_bridge import CGBPrequentialBridge, GDNByteReadoutPort

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        memory = ActiveInferenceGDN2(d_model=3, n_slots=2, res=2).to(device).double()
        embedding = torch.nn.Embedding(256, 3).to(device).double()
        head = torch.nn.Linear(3, 256).to(device).double()
        state = memory.initial_state(1, device, torch.float64)
        calls = []

        def query(current):
            calls.append("query")
            return torch.softmax(current.natural.sum(dim=-1).unsqueeze(1), dim=-1)

        def encode(byte):
            calls.append(("encode", byte))
            value = embedding(torch.tensor([[byte]], device=device))
            return value, memory.atlas[:1].unsqueeze(0), 1.0

        class PreScoredOnlyPort(GDNByteReadoutPort):
            def observe(self, token, *, external=True):
                raise AssertionError("bridge must reuse the prepared score")

        learner = PreScoredOnlyPort(memory, state, query, head, encode)
        ledger = PredictiveInformationLedger()
        bridge = CGBPrequentialBridge(learner, ledger, lambda _: math.log(256), strict_scoring=True)
        for tick, byte in enumerate((65, 66)):
            before = learner.state.natural.clone()
            logits = bridge.prepare()
            assert calls[-1] == "query"
            assert torch.equal(learner.state.natural, before)
            expected = float(-torch.log_softmax(logits, dim=0)[byte])
            result = bridge.step(Observation(tick, byte), write_units=0, dissipation_units=0)
            assert calls[-1] == ("encode", byte)
            assert math.isclose(result.model_nll_nats, expected, rel_tol=1e-12)
            assert not torch.equal(learner.state.natural, before)
            assert not learner.state.natural.requires_grad
        assert ledger.rows == 2
        assert embedding.weight.requires_grad and head.weight.requires_grad
        before = learner.state.natural.clone()
        bridge.prepare()
        with pytest.raises(ValueError, match="byte"):
            bridge.step(Observation(2, 256), write_units=0, dissipation_units=0)
        assert calls[-1] == "query" and torch.equal(learner.state.natural, before)
        bridge.step(Observation(2, 67), write_units=0, dissipation_units=0)
        assert calls[-1] == ("encode", 67) and ledger.rows == 3


def test_gdn_port_uses_elapsed_world_time_before_predicting():
    from fine_grain.active_gdn2 import ActiveInferenceGDN2
    from information_boltzmann.core.cgb_prequential_bridge import GDNByteReadoutPort

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        memory = ActiveInferenceGDN2(d_model=3, n_slots=2, res=2).to(device).double()
        embedding = torch.nn.Embedding(256, 3).to(device).double()
        head = torch.nn.Linear(3, 256).to(device).double()
        state = memory.initial_state(1, device, torch.float64)
        query = lambda s: torch.full((1, 1, 2), 0.5, device=device, dtype=torch.float64)
        encode = lambda b: (embedding(torch.tensor([[b]], device=device)),
                            memory.atlas[:1].unsqueeze(0), 1.0)
        port = GDNByteReadoutPort(memory, state, query, head, encode)
        port.predict()
        port.observe_pre_scored(65)
        before = port.state
        prior = memory.dynamic_prior(before, 2.0)
        port.advance_time(2.0)
        torch.testing.assert_close(port.state.natural, prior.natural)
        torch.testing.assert_close(port.state.precision, prior.precision)
        torch.testing.assert_close(port.state.natural / port.state.precision,
                                   before.natural / before.precision)
        assert torch.all(port.state.precision < before.precision)
        tokens, centers, _ = encode(66)
        incoming = (memory.address(tokens, centers).unsqueeze(-1) *
                    torch.nn.functional.softplus(memory.log_likelihood_precision(tokens)).unsqueeze(-2))[:, 0]
        q = 2.0 * torch.nn.functional.softplus(memory.process_variance_raw)
        wmin = incoming.min()
        actual_gain = incoming / (port.state.precision + incoming)
        floor = wmin / (1.0 / q + wmin)
        assert torch.all(actual_gain >= floor - 1e-12)
        assert not port.state.natural.requires_grad
        current = port.state
        port.advance_time(0)
        assert port.state is current
        for invalid in (-1, float("inf"), float("nan"), True):
            with pytest.raises(ValueError, match="elapsed time"):
                port.advance_time(invalid)
            assert port.state is current
        port.predict()
        with pytest.raises(ValueError, match="pending"):
            port.advance_time(1.0)
        assert port.state is current
        port.observe_pre_scored(66)


def test_gdn_local_affine_head_update_matches_current_byte_gradient():
    from fine_grain.active_gdn2 import ActiveInferenceGDN2
    from information_boltzmann.core.cgb_prequential_bridge import GDNByteReadoutPort

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        memory = ActiveInferenceGDN2(d_model=3, n_slots=2, res=2).to(device).double()
        embedding = torch.nn.Embedding(256, 3).to(device).double()
        head = torch.nn.Linear(3, 256).to(device).double()
        state = memory.initial_state(1, device, torch.float64)
        state.natural.fill_(0.2)
        query = lambda s: torch.full((1, 1, 2), 0.5, device=device, dtype=torch.float64)
        encode = lambda b: (embedding(torch.tensor([[b]], device=device)),
                            memory.atlas[:1].unsqueeze(0), 1.0)
        step = 0.03
        port = GDNByteReadoutPort(memory, state, query, head, encode, head_step_size=step)
        feature = memory.association_readout(state, query(state))[0, 0].detach()
        weight_before, bias_before = head.weight.detach().clone(), head.bias.detach().clone()
        loss = -torch.log_softmax(head(feature), dim=0)[65]
        grad_w, grad_b = torch.autograd.grad(loss, (head.weight, head.bias))
        logits = port.predict()
        torch.testing.assert_close(head.weight, weight_before)
        torch.testing.assert_close(head.bias, bias_before)
        port.observe_pre_scored(65)
        torch.testing.assert_close(head.weight, weight_before - step * grad_w)
        torch.testing.assert_close(head.bias, bias_before - step * grad_b)
        assert torch.log_softmax(head(feature), dim=0)[65] > torch.log_softmax(logits, dim=0)[65]
        assert port._pending is None and port._pending_features is None
        assert not port.state.natural.requires_grad


def test_gdn_external_future_signature_replays_sourced_bytes_without_live_mutation():
    """A branch audit uses the same GDN state and rejects self-copy evidence."""
    from information_boltzmann.core.cgb_observation_stream import StreamEvent
    from information_boltzmann.core.cgb_prequential_bridge import GDNByteReadoutPort
    from fine_grain.active_gdn2 import ActiveInferenceGDN2

    memory = ActiveInferenceGDN2(d_model=3, n_slots=2, res=2).double()
    embedding = torch.nn.Embedding(256, 3).double()
    head = torch.nn.Linear(3, 256).double()
    state = memory.initial_state(1, torch.device("cpu"), torch.float64)
    query = lambda s: torch.softmax(s.natural.sum(dim=-1).unsqueeze(1), dim=-1)
    encode = lambda byte: (embedding(torch.tensor([[byte]])), memory.atlas[:1].unsqueeze(0), 1.0)
    port = GDNByteReadoutPort(memory, state, query, head, encode, head_step_size=0.03)
    continuation = (
        (1.5, StreamEvent(10, 65, "owt_utf8_bytes", {"byte_offset": 10})),
        (0.0, StreamEvent(11, 66, "owt_utf8_bytes", {"byte_offset": 11})),
    )

    original_state = port.snapshot().state
    original_head = {name: value.detach().clone() for name, value in head.state_dict().items()}
    branch_head = torch.nn.Linear(3, 256).double()
    branch_head.load_state_dict(original_head)
    branch = GDNByteReadoutPort(memory, port.snapshot().state, query, branch_head, encode,
                                head_step_size=0.03)
    for elapsed, event in continuation:
        branch.advance_time(elapsed)
        branch.predict()
        branch.observe_pre_scored(event.payload)
    expected = branch.predict()

    actual = port.external_future_signature(continuation)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(port.state.natural, original_state.natural)
    torch.testing.assert_close(port.state.precision, original_state.precision)
    for name, value in original_head.items():
        torch.testing.assert_close(head.state_dict()[name], value)
    assert port._pending is None and port._pending_features is None

    with pytest.raises(ValueError, match="external event"):
        port.external_future_signature(((0.0, StreamEvent(12, 67, "self", {})),))
    torch.testing.assert_close(port.state.natural, original_state.natural)
    for name, value in original_head.items():
        torch.testing.assert_close(head.state_dict()[name], value)


def test_gdn_self_speech_persists_as_a_readable_difference_after_external_arrival():
    """Finite interface witness, not a claim about arbitrary-delay retention."""
    from fine_grain.active_gdn2 import ActiveInferenceGDN2
    from information_boltzmann.core.cgb_observation_stream import StreamEvent
    from information_boltzmann.core.cgb_prequential_bridge import LastByteGDNReadoutPort

    memory = ActiveInferenceGDN2(d_model=2, n_slots=2, res=2).double()
    encoder = torch.nn.Embedding(256, 2).double()
    head = torch.nn.Linear(2, 256).double()
    with torch.no_grad():
        encoder.weight.zero_()
        encoder.weight[65].copy_(torch.tensor([1.0, 0.0], dtype=torch.float64))
        encoder.weight[66].copy_(torch.tensor([-1.0, 0.0], dtype=torch.float64))
        memory.content_address.weight.zero_()
        head.weight.zero_()
        head.bias.zero_()
        head.weight[0, 0] = 1.0
    seed_logits = torch.zeros(2, dtype=torch.float64)
    anchor = memory.atlas[0].double()

    def make_port():
        return LastByteGDNReadoutPort(
            memory,
            memory.initial_state(1, torch.device("cpu"), torch.float64),
            encoder,
            head,
            seed_logits,
            anchor,
            observation_precision=1.0,
        )

    left, right = make_port(), make_port()
    for port, self_byte in ((left, 65), (right, 66)):
        port.predict()
        port.observe_pre_scored(self_byte, external=False)
    for port in (left, right):
        port.predict()
        port.observe_pre_scored(67, external=True)

    # The later external byte is identical in both bodies and has replaced the
    # last-byte query, yet the ordinary associative state still differs.
    assert torch.allclose(left.state.last_tokens, right.state.last_tokens)
    assert not torch.allclose(left.state.natural, right.state.natural)
    continuation = ((0.0, StreamEvent(1, 68, "mother", {})),)
    assert not torch.allclose(
        left.external_future_signature(continuation),
        right.external_future_signature(continuation),
    )


def test_last_byte_gdn_port_matches_seed_then_previous_byte_address():
    """Remove the arbitrary query callback from the concrete GDN byte path."""
    from fine_grain.active_gdn2 import ActiveInferenceGDN2
    from information_boltzmann.core.cgb_prequential_bridge import LastByteGDNReadoutPort

    memory = ActiveInferenceGDN2(d_model=4, n_slots=3, res=3).double()
    with torch.no_grad():
        memory.content_address.weight.copy_(torch.tensor([
            [0.2, -0.1, 0.3, -0.2],
            [-0.3, 0.4, 0.1, 0.2],
            [0.1, 0.2, -0.4, 0.3],
        ], dtype=torch.float64))
        memory.content_address_scale.fill_(1.0)
    encoder = torch.nn.Embedding(256, 4).double()
    head = torch.nn.Linear(4, 256).double()
    seed_logits = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64)
    state = memory.initial_state(1, torch.device("cpu"), torch.float64)
    port = LastByteGDNReadoutPort(memory, state, encoder, head, seed_logits, memory.atlas[0],
                                  observation_precision=0.7)

    torch.testing.assert_close(
        port._last_byte_query(state)[0, 0], torch.softmax(seed_logits, dim=0)
    )
    port.predict()
    port.observe_pre_scored(65)
    expected = memory.address(port.state.last_tokens, port.state.last_centers)
    torch.testing.assert_close(port._last_byte_query(port.state), expected)
    assert port.state.last_tokens.shape == (1, 1, 4)
    assert port.state.last_centers.shape == (1, 1, 2)

    with pytest.raises(ValueError, match="seed logits"):
        LastByteGDNReadoutPort(memory, state, encoder, head, torch.zeros(2), memory.atlas[0])


def test_prepared_logits_survive_buffer_reuse_before_observation():
    from information_boltzmann.core.cgb_information_ledger import PredictiveInformationLedger
    from information_boltzmann.core.cgb_low_resource_life import Observation
    from information_boltzmann.core.cgb_prequential_bridge import CGBPrequentialBridge

    class ReusedBufferLearner:
        def __init__(self):
            self.logits = torch.tensor([0.0, 2.0], requires_grad=True)

        def predict(self):
            return self.logits

        def observe(self, token, *, external=True):
            return 0.0

    learner = ReusedBufferLearner()
    bridge = CGBPrequentialBridge(learner, PredictiveInformationLedger(), lambda _: math.log(2))
    returned = bridge.prepare()
    expected = float(-torch.log_softmax(returned.detach(), dim=0)[1])
    with torch.no_grad():
        returned.copy_(torch.tensor([4.0, -4.0]))
    result = bridge.step(Observation(0, 1), write_units=0, dissipation_units=0)
    assert math.isclose(result.model_nll_nats, expected, rel_tol=1e-6)
    assert not result.prediction.requires_grad
    assert torch.equal(result.prediction, torch.tensor([0.0, 2.0]))


def test_strict_prequential_bridge_rejects_unprepared_or_unscoreable_prediction():
    from information_boltzmann.core.cgb_information_ledger import PredictiveInformationLedger
    from information_boltzmann.core.cgb_low_resource_life import Observation
    from information_boltzmann.core.cgb_prequential_bridge import (
        CGBPrequentialBridge, PrequentialBridgeViolation,
    )

    class UnsupportedLearner:
        def __init__(self):
            self.observations = 0

        def predict(self):
            return object()

        def observe(self, token, *, external=True):
            self.observations += 1
            return 0.0

    learner = UnsupportedLearner()
    ledger = PredictiveInformationLedger()
    bridge = CGBPrequentialBridge(learner, ledger, lambda _: math.log(2), strict_scoring=True)
    observation = Observation(0, 1)
    with pytest.raises(PrequentialBridgeViolation, match="prepared"):
        bridge.step(observation, write_units=0, dissipation_units=0)
    bridge.prepare()
    with pytest.raises(PrequentialBridgeViolation, match="scoreable"):
        bridge.step(observation, write_units=0, dissipation_units=0)
    assert learner.observations == 0 and ledger.rows == 0 and bridge.last_tick == -1


def test_lifetime_tensor_accounting_preserves_online_updates():
    import copy
    from scripts.train_information_boltzmann_lifetime import TensorBudget, energy
    a = tiny(gamma=.3, temperature=.1, gamma_mode='local').double()
    b = copy.deepcopy(a)
    left = StreamRunner(a, 1, seed=17, optimizer=torch.optim.Adam(a.parameters()), update_every=2)
    right = StreamRunner(b, 1, seed=17, optimizer=torch.optim.Adam(b.parameters()), update_every=2)
    for token in (2, 3, 4, 5):
        before = energy(a, left.state).detach()
        budget = TensorBudget()
        assert torch.equal(left.predict(budget), right.predict())
        total = (budget['drive_work']+budget['trap_work']-budget['deterministic_damping_loss']
                 +budget['ou_fluctuation_energy']+budget['drift_potential_change']
                 +budget['collision_energy_error'])
        assert abs(float((energy(a,left.state)-before-total).detach())) < 1e-12
        assert all(not value.requires_grad for value in budget.values())
        left.observe(token)
        right.observe(token)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)


def test_lifetime_validation_does_not_mutate_individual_or_rng():
    from scripts.train_information_boltzmann_lifetime import validate
    model = tiny(gamma=.3, temperature=.1, gamma_mode='local')
    runner = StreamRunner(model, 1, seed=17, optimizer=torch.optim.Adam(model.parameters()), update_every=2)
    for token in (2, 3):
        runner.predict()
        runner.observe(token)
    parameters = {k: v.clone() for k, v in model.state_dict().items()}
    x, v, rng = runner.state.x.clone(), runner.state.v.clone(), runner.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    result = validate(model, [2, 3, 4, 5], 1, length=3, burn_in=1)
    assert result['scored_tokens'] == 3
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(rng, runner.generator.get_state())
    assert torch.equal(x, runner.state.x) and torch.equal(v, runner.state.v)
    assert all(torch.equal(parameters[k], value) for k, value in model.state_dict().items())


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


def test_substep_energy_budget_and_instrumentation_equivalence():
    model = tiny(temperature=.1).double()
    state = model.initialize(torch.tensor([1]), torch.Generator().manual_seed(3))
    budget = {}
    with torch.no_grad():
        a, _, _ = model._advance_steps(state, 2, torch.Generator().manual_seed(4), budget=budget)
        b, _, _ = model._advance_steps(state, 2, torch.Generator().manual_seed(4))
    assert torch.equal(a.x,b.x) and torch.equal(a.v,b.v)
    energy = lambda s: float(((s.x.square()+s.v.square()).sum(-1).mean()/2).detach())
    predicted = (budget['drive_work']+budget['trap_work']-budget['deterministic_damping_loss']
                 +budget['ou_fluctuation_energy']+budget['drift_potential_change']
                 +budget['collision_energy_error'])
    assert abs(energy(a)-energy(state)-predicted) < 1e-12


def test_two_streams_isolate_phase_parameters_and_rng():
    import copy
    source = tiny(temperature=.1)
    deployed = copy.deepcopy(source)
    reference = copy.deepcopy(source)
    train = StreamRunner(source,1,seed=9,optimizer=torch.optim.Adam(source.parameters()),update_every=2)
    inference = StreamRunner(deployed,1,seed=7)
    control = StreamRunner(reference,1,seed=7)
    for token in [2,3,4,5]:
        train.predict()
        train.observe(token)
        assert torch.equal(inference.predict(),control.predict())
        inference.observe(token)
        control.observe(token)
    assert any(not torch.equal(a,b) for a,b in zip(source.parameters(),deployed.parameters()))
    assert torch.equal(inference.state.x,control.state.x)


@pytest.mark.parametrize('mode',['global','local'])
def test_learned_gamma_bounds_gradients_and_checkpoint(mode,tmp_path):
    model=tiny(gamma=.3,temperature=.1,gamma_mode=mode).double()
    state=model.initialize(torch.tensor([1]))
    g=model.force.damping(state.x)
    assert torch.allclose(g,torch.full_like(g,.3))
    out=model.force.transport(state,2,.2,torch.Generator().manual_seed(1))
    out.v.square().sum().backward()
    assert model.force.gamma_field.bias.grad.abs().sum()>0
    with torch.no_grad():model.force.gamma_field.weight.fill_(3)
    gamma=model.force.damping(state.x)
    assert (gamma>.01).all() and (gamma<2).all()
    order=torch.randperm(len(state.x))
    assert torch.allclose(model.force.damping(state.x[order]),gamma[order])
    if mode=='local':assert gamma.std()>0
    torch.save(model.state_dict(),tmp_path/'model.pt')
    restored=tiny(gamma=.3,temperature=.1,gamma_mode=mode).double()
    restored.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True))
    assert torch.equal(restored.force.damping(state.x),gamma)

@pytest.mark.parametrize('mode', ['global', 'local'])
def test_learned_gamma_energy_accounting(mode):
    model = tiny(gamma=.3, temperature=.1, gamma_mode=mode).double()
    with torch.no_grad():
        model.force.gamma_field.weight.fill_(.2)
    state = model.initialize(torch.tensor([1]))
    budget = {}
    with torch.no_grad():
        after, _, _ = model._advance_steps(state, 2, torch.Generator().manual_seed(4), budget=budget)
    energy = lambda s: float((s.x.square()+s.v.square()).sum(-1).mean().detach()/2)
    accounted = (budget['drive_work']+budget['trap_work']-budget['deterministic_damping_loss']
                 +budget['ou_fluctuation_energy']+budget['drift_potential_change']
                 +budget['collision_energy_error'])
    assert abs(energy(after)-energy(state)-accounted) < 1e-12


def test_continuation_probe_identical_states_and_no_mutation():
    from fine_grain.information_boltzmann.behavior import continuation_distance
    model = tiny(temperature=.1)
    state = model.initialize(torch.tensor([1]))
    before = state.x.clone()
    rng = torch.get_rng_state().clone()
    result = continuation_distance(model, state, state, [[1, 2], [3]], seed=7)
    assert result['total_variation_by_suffix'] == [[0., 0., 0.], [0., 0.]]
    assert torch.equal(state.x, before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_stream_boot_context_conditions_f0_once_without_reset():
    model = tiny(collision_rate=0, temperature=0)
    context = torch.tensor([1, 2, 3])
    expected = model.initialize(context, torch.Generator().manual_seed(7))
    runner = StreamRunner(model, 1, seed=7, boot_tokens=context)
    assert torch.equal(runner.state.x, expected.x)
    assert torch.equal(runner.state.v, expected.v)
    assert runner.current_token == 3
    initial_time = runner.state.time
    runner.predict()
    runner.observe(4)
    assert runner.state.time > initial_time
    assert runner.current_token == 4


def test_initial_joint_density_uses_ordered_input_context():
    model = tiny()
    left = model.initial.condition(torch.tensor([1, 2]))
    right = model.initial.condition(torch.tensor([2, 1]))
    a = left.sample(4, torch.Generator().manual_seed(9))
    b = right.sample(4, torch.Generator().manual_seed(9))
    assert not torch.allclose(a.x, b.x) or not torch.allclose(a.v, b.v)


def test_last_byte_value_credit_matches_one_arrival_forward_derivative():
    """The value update is the Lean-proved causal carrier, not BPTT."""
    from fine_grain.active_gdn2 import ActiveInferenceGDN2
    from information_boltzmann.core.cgb_prequential_bridge import LastByteGDNReadoutPort

    torch.manual_seed(71)
    memory = ActiveInferenceGDN2(d_model=2, n_slots=2, res=2).double()
    value_encoder = torch.nn.Embedding(256, 2).double()
    address_encoder = torch.nn.Embedding(256, 2).double()
    likelihood_encoder = torch.nn.Embedding(256, 2).double()
    head = torch.nn.Linear(2, 256).double()
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    for parameter in address_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in likelihood_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    anchor = memory.atlas[0].double()
    source, target, step = 65, 66, 0.03

    # Exact differentiable reference for the sole admitted source-value path:
    # write source -> one process prior -> query source address -> score target.
    reference = memory.initial_state(1, torch.device("cpu"), torch.float64)
    source_byte = torch.tensor([[source]])
    value = value_encoder(source_byte)
    address = address_encoder(source_byte)
    likelihood = likelihood_encoder(source_byte)
    reference = memory.assimilate(
        reference, value, anchor.reshape(1, 1, 2), 1.0, update_motion=False,
        address_tokens=address, likelihood_tokens=likelihood,
    )
    reference = memory.dynamic_prior(reference, 1.0)
    query = memory.address(reference.last_tokens, reference.last_centers)
    features = memory.association_readout(reference, query)[0, 0]
    score = torch.log_softmax(head(features), dim=0)[target]
    expected_gradient = torch.autograd.grad(score, value)[0][0, 0].detach()

    state = memory.initial_state(1, torch.device("cpu"), torch.float64)
    port = LastByteGDNReadoutPort(
        memory, state, value_encoder, head, torch.zeros(2, dtype=torch.float64), anchor,
        observation_precision=1.0, value_step_size=step,
        address_encoder=address_encoder, likelihood_encoder=likelihood_encoder,
    )
    before = value_encoder.weight[source].detach().clone()
    port.predict()
    port.observe_pre_scored(source)
    port.advance_time(1.0)
    port.predict()
    port.observe_pre_scored(target)
    torch.testing.assert_close(value_encoder.weight[source], before + step * expected_gradient,
                               rtol=1e-10, atol=1e-10)
    assert port._last_byte_token == target
    assert port._previous_value_eligibility is not None

    with pytest.raises(ValueError, match="distinct fixed"):
        LastByteGDNReadoutPort(
            memory, memory.initial_state(1, torch.device("cpu"), torch.float64),
            value_encoder, head, torch.zeros(2, dtype=torch.float64), anchor,
            value_step_size=step,
        )


def test_last_byte_snapshot_preserves_causal_credit_carrier():
    from fine_grain.active_gdn2 import ActiveInferenceGDN2
    from information_boltzmann.core.cgb_prequential_bridge import LastByteGDNReadoutPort

    memory = ActiveInferenceGDN2(d_model=2, n_slots=2, res=2).double()
    value_encoder = torch.nn.Embedding(256, 2).double()
    address_encoder = torch.nn.Embedding(256, 2).double()
    likelihood_encoder = torch.nn.Embedding(256, 2).double()
    port = LastByteGDNReadoutPort(
        memory, memory.initial_state(1, torch.device("cpu"), torch.float64), value_encoder,
        torch.nn.Linear(2, 256).double(), torch.zeros(2, dtype=torch.float64),
        memory.atlas[0].double(), value_step_size=0.01,
        address_encoder=address_encoder, likelihood_encoder=likelihood_encoder,
    )
    port.predict()
    port.observe_pre_scored(65)
    snapshot = port.snapshot()
    token, eligibility = port._last_byte_token, port._previous_value_eligibility
    port.predict()
    port.observe_pre_scored(66)
    port.restore(snapshot)
    assert port._last_byte_token == token
    assert port._previous_value_eligibility == eligibility
