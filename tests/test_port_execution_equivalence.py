"""Execution changes must preserve the causal belief and training gradient."""
import copy

import torch
import torch.nn.functional as F

from information_boltzmann.core.torus3d import (
    CBIMTorus3D, FullRankTorusWrite, KineticBeliefState,
    PredictiveImpedanceWriteAgent,
)


def test_innovation_chart_fusion_preserves_values_and_gradients():
    torch.manual_seed(53)
    writer = FullRankTorusWrite(vocab_size=19, shape=(4, 4, 4), d=16,
                                write_type="w2_impedance").double()
    agent = PredictiveImpedanceWriteAgent(16, vocab_size=19).double()
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64, requires_grad=True)
    prior = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
    observed = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    predicted = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    observed_packet, *_ = agent._packet_chart(writer, field, prior, observed)
    predicted_packet, *_ = agent._packet_chart(writer, field, prior, predicted)
    reference = observed_packet - predicted_packet
    fused, *_ = agent._packet_chart(writer, field, prior, observed - predicted)
    torch.testing.assert_close(fused, reference, rtol=2e-12, atol=2e-12)
    inputs = (field, prior, observed, predicted, *agent.chart_gate.parameters())
    old_grad = torch.autograd.grad(reference.square().sum(), inputs, retain_graph=True)
    new_grad = torch.autograd.grad(fused.square().sum(), inputs)
    for actual, expected in zip(new_grad, old_grad):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_batched_decoder_preserves_joint_loss_belief_and_all_gradients():
    torch.manual_seed(59)
    model = CBIMTorus3D(
        vocab_size=23, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        micro_steps=2, dissipation_type="selective").double()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 7], [5, 6, 8]])
    initial = KineticBeliefState(
        torch.randn(2, 4, 4, 4, 16, dtype=torch.float64) * 0.1,
        torch.ones(2, 16, dtype=torch.float64))
    loss, belief, diag = model.forward_belief(ids, targets, initial,
                                             port_free_energy_weight=0.7)
    reference_belief = initial
    reference_nll, reference_fe = 0.0, 0.0
    features = F.normalize(reference.source.embedding.weight, dim=-1)
    for index in range(ids.shape[1]):
        logits, reference_belief, row = reference.belief_step(
            reference_belief, ids[:, index], include_private=True,
            token_features=features)
        reference_nll = reference_nll + F.cross_entropy(logits, targets[:, index])
        reference_fe = reference_fe + row["_write_free_energy"]
    reference_loss = (reference_nll + 0.7 * reference_fe) / ids.shape[1]
    torch.testing.assert_close(loss, reference_loss, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(diag["token_nll"], reference_nll / ids.shape[1])
    torch.testing.assert_close(belief.field, reference_belief.field, rtol=0, atol=0)
    torch.testing.assert_close(belief.precision, reference_belief.precision, rtol=0, atol=0)
    loss.backward()
    reference_loss.backward()
    for (name, actual), (expected_name, expected) in zip(
            model.named_parameters(), reference.named_parameters()):
        assert name == expected_name
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad,
                                       rtol=2e-9, atol=2e-10, msg=name)


def test_decoder_runs_once_per_training_chunk_and_step_still_returns_logits():
    model = CBIMTorus3D(
        vocab_size=23, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        micro_steps=1)
    calls = []
    handle = model.decoder.register_forward_hook(
        lambda module, args, output: calls.append(tuple(args[0].shape)))
    ids = torch.tensor([[1, 2, 3]])
    _, belief, _ = model.forward_belief(ids, ids)
    assert calls == [(1, 3, 16)]
    logits, _, _ = model.belief_step(belief, torch.tensor([4]))
    assert logits.shape == (1, 23)
    assert calls[-1] == (1, 16)
    handle.remove()


def test_adjacent_transport_fusion_preserves_split_flow_and_gradients():
    torch.manual_seed(61)
    model = CBIMTorus3D(
        vocab_size=23, shape=(4, 4, 4), velocities=8, content_dim=2,
        write_type="w4_predictive_agent", micro_steps=4).double()
    with torch.no_grad():
        model.transport.residual_scale.fill_(0.4)
    reference = copy.deepcopy(model)
    field = torch.randn(1, 4, 4, 4, 16, dtype=torch.float64,
                        requires_grad=True)
    reference_field = field.detach().clone().requires_grad_()
    output, _ = model.state_agent.evolve(
        field, None, micro_steps=4, tau_0=model.tau_0_tensor)
    for index in range(4):
        for term in reference.state_agent.order_for_microstep(index):
            if term == "transport":
                multiplier, _ = reference.transport.multiplier(reference.tau_0_tensor)
                reference_field = reference.transport.apply_multiplier(reference_field, multiplier)
            elif term == "collision":
                reference_field, _ = reference.collision(reference_field, reference.tau_0_tensor)
            else:
                reference_field, _ = reference.bath(reference_field, reference.tau_0_tensor)
    torch.testing.assert_close(output, reference_field, rtol=2e-12, atol=2e-12)
    direction = torch.randn_like(output)
    (output * direction).sum().backward()
    (reference_field * direction).sum().backward()
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad,
                                       rtol=2e-9, atol=2e-10, msg=name)
