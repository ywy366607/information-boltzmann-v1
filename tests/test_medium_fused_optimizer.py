"""Optimizer execution parity only, with saved moment and None semantics."""
import copy
import os

import pytest
import torch

from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.optimizer_storage import optimizer_state_on_host


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA optimizer execution')
def test_fused_adam_preserves_saved_groups_moments_none_and_host_staging():
    torch.manual_seed(87)
    original = torch.nn.Sequential(torch.nn.Linear(16, 12), torch.nn.Linear(12, 4)).cuda()
    reference = make_medium_optimizer(original, lr=2e-4)
    original(torch.ones(2, 16, device='cuda')).square().sum().backward()
    reference.step()
    reference.zero_grad(set_to_none=True)
    actual = copy.deepcopy(original)
    fused = make_medium_optimizer(actual, lr=2e-4,
        saved_state=copy.deepcopy(reference.state_dict()), fused=True)
    for index in range(3):
        for i, (a, b) in enumerate(zip(original.parameters(), actual.parameters())):
            if i == index:
                a.grad = b.grad = None
            else:
                gradient = torch.linspace(-.7, .9, a.numel(), device='cuda').reshape_as(a)
                a.grad, b.grad = gradient.clone(), gradient.clone()
        with optimizer_state_on_host(fused, enabled=True):
            pass
        versions = [p._version for p in actual.parameters()]
        reference.step()
        fused.step()
        for i, (a, b) in enumerate(zip(original.parameters(), actual.parameters())):
            torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-6)
            if i != index:
                assert b._version > versions[i]
            for name in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(reference.state[a][name].cpu(), fused.state[b][name].cpu(),
                                           atol=3e-8, rtol=3e-6)
