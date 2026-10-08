"""Opt-in CUDA execution checks; no GPU use in the default CPU-oriented suite."""
import os
from dataclasses import replace

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D, PlasticBelief
from information_boltzmann.core.plastic_medium import MediumState
from information_boltzmann.runtime import ContinuousStream

pytestmark = pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                                reason='Explicit GPU window required')


@pytest.mark.parametrize('bath_type,short_term_plasticity,medium_execution',
                         [('quadratic', False, 'native'), ('conductance', False, 'native'),
                          ('conductance', True, 'native'), ('conductance', True, 'fused')])
def test_captured_runtime_matches_variable_time_and_preserves_snapshots(bath_type, short_term_plasticity, medium_execution):
    net = PlasticMediumPorts3D(vocab_size=17, shape=(8, 8, 4), channels=128,
                              bath_type=bath_type, short_term_plasticity=short_term_plasticity,
                              activity_adaptation=(medium_execution == 'fused'),
                              medium_execution=medium_execution,
                              port_execution=medium_execution).eval().cuda()
    with torch.no_grad():
        net.medium.material.coefficients.normal_(std=0.01)
        initial = net.initial_belief()
        state = replace(initial.medium, field=torch.randn_like(initial.medium.field),
                        flux=tuple(torch.randn_like(x) for x in initial.medium.flux))
        initial = PlasticBelief(state, initial.precision)
        captured = ContinuousStream(net, max_step=0.01, belief=initial, compile_backend='cuda_graph')
        reference = ContinuousStream(net, max_step=0.01, belief=initial)
        for stream in (captured, reference):
            stream.advance_to(0.005)
        first = captured.belief.medium
        saved_first = first.field.clone()
        for stream in (captured, reference):
            stream.observe(0.005, torch.tensor([2]), training_terms=False)
            stream.advance_to(0.014)
        torch.cuda.synchronize()
        torch.testing.assert_close(first.field, saved_first, atol=0, rtol=0)
        torch.testing.assert_close(captured.belief.medium.field, reference.belief.medium.field,
                                   atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(captured.belief.medium.conduction, reference.belief.medium.conduction,
                                   atol=2e-6, rtol=2e-5)
        if initial.medium.receptors is not None:
            torch.testing.assert_close(captured.belief.medium.receptors, reference.belief.medium.receptors,
                                       atol=2e-6, rtol=2e-5)
        if initial.medium.transmission is not None:
            torch.testing.assert_close(captured.belief.medium.transmission, reference.belief.medium.transmission,
                                       atol=2e-6, rtol=2e-5)
        old_graph = captured._kernel.graph
        net.medium.material.coefficients.add_(0.001)
        for stream in (captured, reference):
            stream.advance_to(0.02)
        assert captured._kernel.graph is not old_graph
        torch.testing.assert_close(captured.belief.medium.field, reference.belief.medium.field,
                                   atol=2e-6, rtol=2e-5)


def test_deployment_graph_rejects_differentiable_training():
    net = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8).eval().cuda()
    stream = ContinuousStream(net, max_step=0.01, compile_backend='cuda_graph')
    with pytest.raises(ValueError, match='inference'):
        stream.advance_to(0.01)
