"""Numerical reduction and VJP contracts; no capability experiment."""
import pytest
import torch

from information_boltzmann.core.triton_synapse import (
    build_incoming_layout, IncomingDelayedTransmission,
    PyTorchDelayedSynapticTransmission,
)


@pytest.mark.parametrize('device,dtype,batch', [
    ('cpu', torch.float64, 2), ('cuda', torch.float32, 1),
    ('cuda', torch.float32, 2),
])
def test_fixed_incoming_order_matches_full_vjp_and_uses_current_weights(device, dtype, batch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pre = torch.tensor([2, 0, 2, 1, 3, 2, 2], device=device)
    post = torch.tensor([1, 1, 1, 3, 2, 1, 1], device=device)
    splits = (0, 3, 3, 5, 7)
    layouts = tuple(tuple(t.to(device) for t in row)
                    for row in build_incoming_layout(pre, post, splits, 5))
    pulse = tuple(torch.tensor([[0., .1, 0., .3, .4]], device=device,
                              dtype=dtype).repeat(batch, 1).requires_grad_() for _ in range(4))
    weight = torch.arange(1, 8, dtype=dtype, device=device).requires_grad_()
    reference = PyTorchDelayedSynapticTransmission.apply(*pulse, pre, post, weight, splits)
    expected = torch.autograd.grad(reference.sum(), (*pulse, weight))
    out = IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts)
    actual = torch.autograd.grad(out.sum(), (*pulse, weight))
    torch.testing.assert_close(out, reference)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    # Silent cells can become active: their nonzero transmission VJP is kept.
    assert pulse[0][0, 2] == 0 and actual[0][0, 2] > 0
    assert torch.count_nonzero(actual[1]) == 0  # Empty delay tier.
    assert out[:, 0].eq(0).all() and out[:, 4].eq(0).all()  # Empty rows.
    for _ in range(4):
        assert torch.equal(out, IncomingDelayedTransmission.apply(
            *pulse, pre, post, weight, splits, layouts))
    updated_weight = (weight.detach()*2).requires_grad_()
    changed = IncomingDelayedTransmission.apply(*pulse, pre, post, updated_weight, splits, layouts)
    torch.testing.assert_close(changed, out*2)


def test_empty_incoming_graph_has_zero_output_and_full_zero_pulse_vjp():
    index = torch.empty(0, dtype=torch.int64)
    weight = torch.empty(0, requires_grad=True)
    splits = (0,)
    layout = build_incoming_layout(index, index, splits, 3)
    pulse = tuple(torch.zeros(1, 3, requires_grad=True) for _ in range(4))
    out = IncomingDelayedTransmission.apply(*pulse, index, index, weight, splits, layout)
    out.sum().backward()
    assert out.eq(0).all() and weight.grad is not None
    assert all(p.grad is not None and p.grad.eq(0).all() for p in pulse)
