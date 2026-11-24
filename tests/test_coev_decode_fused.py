"""Production cell parity at actual and reduced tensor dimensions."""
import pytest
import torch

pytest.importorskip('triton')

from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer
from scripts.ib_local.past_present_coevolution import PastPresentField
from scripts.ib_local.coev_decode_fused import exchange


@pytest.mark.skipif(not torch.cuda.is_available(), reason='fused decode kernel')
@pytest.mark.parametrize('width,rank', [(32, 8), (256, 32)])
@pytest.mark.parametrize('enabled', [0., 1.])
def test_fused_cell_keeps_shared_parameters_gates_and_workspace(width, rank, enabled):
    torch.manual_seed(112)
    layer = GatedDeltaNetLayer(width, 1, 'decay_then_delta', 3*width, 4).cuda()
    field = PastPresentField(width, rank, slots=8, rank=rank).cuda()
    h = torch.randn(1, 1, width, device='cuda')
    incoming = torch.randn(1, 1, 8, rank, device='cuda')
    previous = torch.randn(1, 8*rank, device='cuda')
    keys = torch.randn(8, rank, device='cuda')
    bias = torch.randn(8, device='cuda')
    field.valid[::2] = True
    field.enabled.fill_(enabled)
    with torch.no_grad():
        expected = field.exchange(layer, h, incoming, previous, keys, bias)
        actual = exchange(field, layer, h, incoming, previous, keys, bias)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)
