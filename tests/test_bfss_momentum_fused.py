import pytest
import torch

from scripts.ib_local.bfss_gdn_capture import matrix_view, observed_momentum
from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA observation replay')
@pytest.mark.parametrize('budget', [0., 3., 32.])
def test_fused_actual_innovation_and_capacity_correction(monkeypatch, budget):
    torch.manual_seed(714)
    layer = GatedDeltaNetLayer(128, 4, 'decay_then_delta', 256, 4,
                              budget, decay_init='log_timescale').cuda()
    state = torch.randn(1, layer.matrix_size + 3*128*3, device='cuda')*.1
    p = torch.randn_like(matrix_view(layer, state))*.1
    x = torch.randn(1, 137, 128, device='cuda')
    monkeypatch.setenv('CGB_MOMENTUM_BACKEND', 'reference')
    expected = observed_momentum(layer, x, state, p)
    monkeypatch.setenv('CGB_MOMENTUM_BACKEND', 'fused')
    actual = observed_momentum(layer, x, state, p)
    for old, new in zip(expected, actual):
        torch.testing.assert_close(new, old, atol=2e-5, rtol=2e-4)
    first = observed_momentum(layer, x[:, :65], state, p)
    mid = state.clone()
    mid[:, :layer.matrix_size] = first[0].flatten(1)
    # Carry the actual convolution cache across the split, too.
    _, packed, _ = layer(x[:, :65], state)
    mid[:, layer.matrix_size:] = packed[:, layer.matrix_size:]
    last = observed_momentum(layer, x[:, 65:], mid, first[1])
    torch.testing.assert_close(last[1], actual[1], atol=2e-5, rtol=2e-4)
