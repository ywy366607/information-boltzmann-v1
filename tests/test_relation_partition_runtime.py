"""Equivalent batched evidence solves; these are numerical, not ability tests."""
import pytest
import torch

from scripts.ib_local.generative_relations import marginal, partition_evidence


@pytest.mark.parametrize('n', [4, 8, 10])
@pytest.mark.parametrize('noise_prior', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batched_partition_matches_individual_evidence(n, noise_prior, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA numerical comparison')
    torch.manual_seed(15)
    x = torch.randn(n, 5, dtype=torch.float64, device=device)
    y = torch.randn(n, 11, dtype=x.dtype, device=device)
    order = torch.randperm(n, device=device)
    noise = torch.rand(11, dtype=x.dtype, device=device)+.1 if noise_prior else None
    result = partition_evidence(x, y, order, .2, noise_prior=noise)
    expected = torch.stack([marginal(x[order[:i]], y[order[:i]], .2, noise_prior=noise)+
        marginal(x[order[i:]], y[order[i:]], .2, noise_prior=noise) for i in range(2, n-1)])
    torch.testing.assert_close(result, expected, atol=2e-10, rtol=2e-10)
