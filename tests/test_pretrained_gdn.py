"""Numerical boundaries of the optional pretrained-GDN compatibility port."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn import RecallInlet, streaming_short_convolution


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('split', [1, 3, 7])
def test_pretrained_convolution_carries_prefix_without_mutating_branch(width, split):
    torch.manual_seed(12)
    layer = nn.Conv1d(5, 5, width, groups=5, bias=False).double()
    layer.activation = 'silu'
    x = torch.randn(2, 12, 5, dtype=torch.float64)
    first, state = streaming_short_convolution(layer, x[:, :split], output_final_state=True)
    saved = state.clone()
    second, final = streaming_short_convolution(layer, x[:, split:], cache=state,
                                                output_final_state=True)
    expected = F.silu(F.conv1d(F.pad(x.transpose(1, 2), (width-1, 0)),
                               layer.weight, groups=5)).transpose(1, 2)
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected,
                               atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state, saved, atol=0, rtol=0)
    torch.testing.assert_close(final, x.transpose(1, 2)[..., -width:], atol=0, rtol=0)


def test_pretrained_convolution_chunk_boundary_has_correct_input_gradient():
    torch.manual_seed(17)
    layer = nn.Conv1d(3, 3, 4, groups=3, bias=False).double()
    layer.activation = 'silu'
    x = torch.randn(1, 10, 3, dtype=torch.float64, requires_grad=True)
    _, cache = streaming_short_convolution(layer, x[:, :3], output_final_state=True)
    result, _ = streaming_short_convolution(layer, x[:, 3:], cache=cache,
                                           output_final_state=True)
    actual = torch.autograd.grad(result.square().sum(), (x, layer.weight))
    expected = F.silu(F.conv1d(F.pad(x.transpose(1, 2), (3, 0)),
                               layer.weight, groups=3)).transpose(1, 2)[:, 3:]
    reference = torch.autograd.grad(expected.square().sum(), (x, layer.weight))
    for left, right in zip(actual, reference):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    assert actual[0][:, :3].abs().sum() > 0


def test_recall_inlet_is_neutral_trainable_and_removes_temporary_hook():
    torch.manual_seed(20)
    block = nn.Linear(8, 8, bias=False)
    block.requires_grad_(False)
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    inlet = RecallInlet(8, latent_dim=4)
    x = torch.randn(1, 5, 8)
    recalled = torch.randn(1, 1, 4)
    baseline = block(x)
    with inlet.attach(model, recalled, layer=0):
        result = block(x)
        torch.testing.assert_close(result, baseline, atol=0, rtol=0)
        result.square().sum().backward()
    assert torch.isfinite(inlet.projection.weight.grad).all()
    assert inlet.projection.weight.grad.norm() > 0
    assert not block._forward_pre_hooks
    assert block.weight.grad is None
