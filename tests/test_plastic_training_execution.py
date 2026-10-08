"""Numerical training-path equivalence; no training capability claims."""
import os

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import (
    CapturedPlasticChunk, belief_tensors, quiet_training_chunk)


def test_quiet_chunk_matches_public_likelihood_and_all_gradients():
    torch.manual_seed(143)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                hidden=8, bath_type='conductance').double()
    ids, targets = torch.tensor([[1, 2]]), torch.tensor([[2, 3]])
    initial = model.initial_belief()
    loss, expected, _ = model(ids, targets, initial, event_duration=0.005)
    loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual_loss, actual, _ = quiet_training_chunk(model, ids, targets, initial, event_duration=0.005)
    actual_loss.backward()
    torch.testing.assert_close(actual_loss, loss, atol=2e-12, rtol=2e-12)
    for x, y in zip(belief_tensors(actual), belief_tensors(expected)):
        torch.testing.assert_close(x, y, atol=2e-12, rtol=2e-12)
    actual_gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert actual_gradients.keys() == gradients.keys()
    for name, gradient in actual_gradients.items():
        torch.testing.assert_close(gradient, gradients[name], atol=2e-11, rtol=2e-10)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA allocation')
@pytest.mark.parametrize('activity_adaptation,short_term_plasticity,medium_execution',
                         [(False, False, 'native'), (True, False, 'native'),
                          (True, True, 'native'), (True, True, 'fused')])
def test_training_graph_accumulates_chunks_and_follows_updated_parameters(activity_adaptation, short_term_plasticity, medium_execution):
    torch.manual_seed(143)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                hidden=8, bath_type='conductance',
                                activity_adaptation=activity_adaptation,
                                short_term_plasticity=short_term_plasticity,
                                medium_execution=medium_execution,
                                port_execution=medium_execution).cuda()
    ids = torch.tensor([[1, 2, 3, 4]], device='cuda')
    targets = torch.tensor([[2, 3, 4, 5]], device='cuda')
    initial = model.initial_belief()
    capture = CapturedPlasticChunk(model, ids[:, :2], targets[:, :2], initial,
                                   event_duration=0.005, loss_scale=0.5)
    for version in range(2):
        if version:
            with torch.no_grad():
                model.medium.material.coefficients.add_(0.001)
                model.write_agent.chart_gate[-1].weight.add_(0.001)
                model.readout.k_proj.weight.mul_(1.01)
                model.read_norm.weight.mul_(1.03)
        capture.zero_grad()
        actual = initial
        for offset in (0, 2):
            _, actual, _ = capture.backward(ids[:, offset:offset + 2], targets[:, offset:offset + 2], actual)
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        saved = actual.medium.field.clone()
        capture.zero_grad()
        expected = initial
        for offset in (0, 2):
            loss, expected, _ = quiet_training_chunk(
                model, ids[:, offset:offset + 2], targets[:, offset:offset + 2],
                expected, event_duration=0.005)
            (loss * 0.5).backward()
            expected = expected.detach()
        for x, y in zip(belief_tensors(actual), belief_tensors(expected)):
            torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-4)
        capture.zero_grad()
        capture.backward(ids[:, :2], targets[:, :2], initial)
        torch.testing.assert_close(actual.medium.field, saved, atol=0, rtol=0)
