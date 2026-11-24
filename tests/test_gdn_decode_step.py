"""Original one-event law, recurrent trajectories and live graph refresh."""
import pytest
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer
from scripts.ib_local.gdn_decode_step import make_bundle, refresh, step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA decode kernel')


def initial_state(layer):
    shape = (1, layer.state_size) if layer.conv_kernel else (1, layer.heads, layer.dk, layer.dv)
    return torch.zeros(shape, device='cuda')


@pytest.mark.parametrize('d,heads,mlp,conv,budget,decay', [
    (32, 2, 0, 0, 0., 'constant'),
    (64, 2, 96, 4, .6, 'constant'),
    (256, 8, 768, 4, 32., 'log_timescale'),
])
@torch.no_grad()
def test_original_state_law_across_successive_events(d, heads, mlp, conv, budget, decay):
    torch.manual_seed(522)
    layer = GatedDeltaNetLayer(d, heads, 'decay_then_delta', mlp, conv, budget,
                               decay_init=decay).cuda()
    layer._decode_no_diagnostics = True
    state = torch.randn_like(initial_state(layer))*.05
    other = state.clone()
    bundle = make_bundle(layer)
    for _ in range(32):
        x = torch.randn(1, 1, d, device='cuda')*.2
        expected, state, _ = gdn_fused.fused_forward(layer, x, state)
        actual, other, _ = step(layer, x, other, bundle)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
        torch.testing.assert_close(other, state, atol=3e-5, rtol=5e-5)
    before = set(layer.state_dict())
    layer.q_proj.weight.add_(.003)
    layer.mlp_norm.weight.add_(.01) if mlp else None
    refresh(layer, bundle)
    expected, nxt, _ = gdn_fused.fused_forward(layer, x, state)
    actual, actual_nxt, _ = step(layer, x, state, bundle)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
    torch.testing.assert_close(actual_nxt, nxt, atol=3e-5, rtol=5e-5)
    assert set(layer.state_dict()) == before


def test_single_event_optimization_leaves_training_derivatives_live():
    layer = GatedDeltaNetLayer(32, 2, 'decay_then_delta', 64, 4).cuda()
    layer._decode_step_bundle = make_bundle(layer)
    x = torch.randn(1, 1, 32, device='cuda', requires_grad=True)
    y, state, _ = gdn_fused.fused_forward(layer, x, initial_state(layer))
    (y.square().mean()+state.square().mean()).backward()
    assert x.grad.norm() > 0
    assert layer.q_proj.weight.grad.norm() > 0
    assert layer.mlp['gate'].weight.grad.norm() > 0
