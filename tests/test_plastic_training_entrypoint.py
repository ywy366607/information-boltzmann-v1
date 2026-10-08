"""Continuation and evaluation invariants of the real-corpus trainer."""
import os

import pytest
import torch

from scripts.ib.train_plastic_conductance import (
    EvaluationGraph, evaluation_chunk, learning_rate, pack_belief, unpack_belief)
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import belief_tensors


def test_full_belief_roundtrip_preserves_values_clock_and_ownership():
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2,2,2), channels=4,
                                hidden=8, bath_type='conductance').double()
    old = model.initial_belief()
    with torch.no_grad():
        old, _ = model.assimilate(old, torch.tensor([2]), diagnostics=False)
        old, _ = model.advance(old, 0.005, diagnostics=False)
    restored = unpack_belief(pack_belief(old),'cpu')
    for original, new in zip(belief_tensors(old), belief_tensors(restored)):
        torch.testing.assert_close(original,new,atol=0,rtol=0)
        assert original.data_ptr() != new.data_ptr()
    assert restored.medium.elapsed.dtype == torch.float64


def test_evaluation_carries_mature_state_without_mutating_training_belief():
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2,2,2), channels=4,
                                hidden=8, bath_type='conductance').double()
    initial = model.initial_belief()
    mature, _ = model.advance(initial, 0.07, diagnostics=False)
    frozen = [x.clone() for x in belief_tensors(mature)]
    ids, targets = torch.tensor([[2,3]]), torch.tensor([[3,4]])
    nll, evolved = evaluation_chunk(model,ids,targets,mature,event_duration=0.005,substeps=1)
    assert torch.isfinite(nll)
    torch.testing.assert_close(evolved.medium.elapsed,mature.medium.elapsed+0.01)
    for value, expected in zip(belief_tensors(mature),frozen):
        torch.testing.assert_close(value,expected,atol=0,rtol=0)
    assert all(x.grad_fn is None for x in belief_tensors(evolved))


def test_wsd_budget_endpoints_and_stable_phase():
    def rate(step):
        return learning_rate(step,steps=3000,warmup=100,decay=300,peak=1e-4,floor=1e-6)
    assert rate(0) == 1e-6
    assert rate(100) == 1e-4
    assert rate(2700) == 1e-4
    assert rate(3000) == pytest.approx(1e-6)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',reason='Opt-in CUDA allocation')
@pytest.mark.parametrize('activity_adaptation,short_term_plasticity,medium_execution',
                         [(False, False, 'native'), (True, False, 'native'),
                          (True, True, 'native'), (True, True, 'fused')])
def test_evaluation_graph_matches_sequential_eager_and_reads_updated_weights(activity_adaptation, short_term_plasticity, medium_execution):
    model = PlasticMediumPorts3D(vocab_size=17,shape=(2,2,2),channels=4,
                                hidden=8,bath_type='conductance',
                                activity_adaptation=activity_adaptation,
                                short_term_plasticity=short_term_plasticity,
                                medium_execution=medium_execution,
                                port_execution=medium_execution).cuda().eval()
    ids, targets = torch.tensor([[2,3]],device='cuda'), torch.tensor([[3,4]],device='cuda')
    mature = model.initial_belief()
    graph = EvaluationGraph(model,ids,targets,mature,0.005,1)
    for version in range(2):
        if version:
            with torch.no_grad():
                model.medium.material.coefficients.add_(0.001)
                model.write_agent.chart_gate[-1].weight.add_(0.001)
                model.readout.k_proj.weight.mul_(1.01)
                model.read_norm.weight.mul_(1.03)
        reference, actual = mature, mature
        for shifted in (ids, targets):
            expected_nll, reference = evaluation_chunk(model,shifted,targets,reference,
                                                       event_duration=0.005,substeps=1)
            actual_nll, actual = graph(shifted,targets,actual)
            torch.testing.assert_close(expected_nll,actual_nll,atol=3e-6,rtol=3e-5)
        for expected,value in zip(belief_tensors(reference),belief_tensors(actual)):
            torch.testing.assert_close(expected,value,atol=3e-6,rtol=3e-5)
