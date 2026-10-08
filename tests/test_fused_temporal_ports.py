"""Compiled temporal ports retain the native physical state and first-order credit."""
import copy
import os

import pytest
import torch

from information_boltzmann.runtime.training import quiet_training_chunk, belief_tensors
from test_medium_structural_integration import candidate


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA allocation')
def test_fused_temporal_ports_with_diagnostics_match_native_all_credit():
    torch.set_num_threads(1)
    from information_boltzmann.runtime.execution_cache import configure_execution_cache
    configure_execution_cache('scratch/compiler_cache')
    native = candidate().float().cuda()
    fused = copy.deepcopy(native)
    fused.port_execution = 'fused'
    noise = torch.randn_like(native.medium.structural_posterior.mean)
    ids = torch.tensor([[1, 2]], device='cuda')
    targets = torch.tensor([[2, 3]], device='cuda')

    class Health:
        def record(self, *args):
            self.parts = args[4:6]

        def record_decode(self, *args):
            pass

    result = []
    for net in (native, fused):
        net.medium.structural_posterior.begin_window(noise)
        health = Health()
        initial = net.initial_belief()
        loss, final, _ = quiet_training_chunk(net, ids, targets, initial,
            event_duration=.023, health_capture=health, activation_checkpointing=True,
            checkpoint_granularity='event')
        loss.backward()
        result.append((loss.detach(), [x.detach() for x in belief_tensors(final)],
            {n: p.grad.detach().clone() for n, p in net.named_parameters()
             if p.grad is not None}))
        # Unprepared monitoring still uses the Python coefficient transaction.
        with torch.no_grad():
            net.port_snapshot(final.detach())
    assert hasattr(fused, '_compiled_assimilate_diagnostic')
    assert hasattr(fused, '_compiled_assimilate')  # writer-only auxiliary replay
    assert hasattr(fused, '_compiled_read')
    expected, actual = result
    torch.testing.assert_close(actual[0], expected[0], rtol=5e-5, atol=5e-6)
    for a, b in zip(actual[1], expected[1]):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-6)
    assert actual[2].keys() == expected[2].keys()
    for name, gradient in actual[2].items():
        assert gradient.isfinite().all(), name
        torch.testing.assert_close(gradient, expected[2][name], rtol=8e-4, atol=1e-5, msg=name)
