"""Execution identity and owned output buffers, not capability experiments."""
import os
import pytest
import torch

from information_boltzmann.core.segment_graph import NoGradSegmentGraph


def test_segment_cpu_fallback_preserves_gradients_and_nested_values():
    operation = NoGradSegmentGraph(lambda state, dt: {'state': state * dt, 'clock': dt})
    state = torch.tensor([2., 3.], requires_grad=True)
    dt = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    output = operation(state, dt)
    output['state'].sum().backward()
    torch.testing.assert_close(state.grad, torch.full_like(state, .1))
    torch.testing.assert_close(dt.grad, torch.tensor(5., dtype=torch.float64))
    assert not operation.records
    with torch.no_grad():
        torch.testing.assert_close(operation(state, dt)['state'], state * dt)
    assert not operation.records


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA capture')
def test_segment_outputs_survive_replay_and_parameters_remain_current():
    parameter = torch.nn.Parameter(torch.tensor([2.], device='cuda'))
    def execute(state, dt):
        return {'state': parameter * state, 'clock': dt + 1.}
    operation = NoGradSegmentGraph(execute, max_variants=1)
    with torch.no_grad():
        first = operation(torch.tensor([3.], device='cuda'),
                          torch.tensor([.125], device='cuda', dtype=torch.float64))
        parameter.fill_(4.)
        second = operation(torch.tensor([5.], device='cuda'),
                           torch.tensor([.25], device='cuda', dtype=torch.float64))
        torch.testing.assert_close(first['state'], torch.tensor([6.], device='cuda'))
        torch.testing.assert_close(second['state'], torch.tensor([20.], device='cuda'))
        assert first['clock'].dtype == torch.float64
        torch.testing.assert_close(first['clock'], torch.tensor([1.125], device='cuda', dtype=torch.float64))
        # Extra shape falls back instead of allocating another graph pool.
        operation(torch.ones(2, device='cuda'), torch.ones(2, device='cuda'))
    assert len(operation.records) == 1
    leaf = torch.tensor([3.], device='cuda', requires_grad=True)
    operation(leaf, torch.zeros(1, device='cuda'))['state'].sum().backward()
    torch.testing.assert_close(leaf.grad, torch.tensor([4.], device='cuda'))
    torch.testing.assert_close(parameter.grad, torch.tensor([3.], device='cuda'))
