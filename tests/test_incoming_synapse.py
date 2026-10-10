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


def test_direct_edge_grad_accumulates_the_same_weight_gradient_as_autograd():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import information_boltzmann.core.triton_synapse as ts
    gen = torch.Generator().manual_seed(3)
    n, e = 40, 400
    pre = torch.randint(0, n, (e,), generator=gen).cuda()
    post = torch.randint(0, n, (e,), generator=gen).cuda()
    splits = (0, 100, 220, 300, 400)
    layouts = tuple(tuple(t.cuda() for t in row) for row in build_incoming_layout(pre, post, splits, n))
    pulse = tuple((torch.rand(1, n, generator=gen) < .3).float().cuda().requires_grad_() for _ in range(4))
    weight = torch.rand(e, generator=gen).cuda().requires_grad_()
    probe = torch.randn(1, n, generator=gen).cuda()
    ref = PyTorchDelayedSynapticTransmission.apply(*pulse, pre, post, weight, splits)
    expected = torch.autograd.grad((ref * probe).sum(), (*pulse, weight))
    out = IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts)
    plain = torch.autograd.grad((out * probe).sum(), (*pulse, weight))      # ordinary contract keeps working
    for a, b in zip(plain, expected):
        torch.testing.assert_close(a, b)
    with ts.direct_edge_grad():
        for _ in range(2):                                                   # two calls accumulate into .grad
            IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts).mul(probe).sum().backward()
    torch.testing.assert_close(weight.grad, 2 * expected[4])
    assert not ts.DIRECT_EDGE_GRAD


def test_run_sum_pulse_gradient_matches_autograd_and_is_bitwise_reproducible():
    """Edges grouped by (tier, presynaptic cell), with rows longer than a 256-edge block, an empty tier and silent
    cells: the run-sum pulse gradient equals the reference VJP and two backward passes agree bit for bit."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import information_boltzmann.core.triton_synapse as ts
    gen = torch.Generator().manual_seed(5)
    n = 300
    pres, posts, splits = [], [], [0]
    for tier in range(4):
        if tier == 2:                                                        # empty tier
            splits.append(splits[-1])
            continue
        degree = torch.randint(0, 6, (n,), generator=gen)
        degree[torch.randint(0, n, (4,), generator=gen)] = torch.tensor([700, 300, 257, 1000])   # straddling rows
        for cell in range(n):
            d = int(degree[cell])
            pres.append(torch.full((d,), cell))
            posts.append(torch.sort(torch.randint(0, n, (d,), generator=gen)).values)
        splits.append(splits[-1] + int(degree.sum()))
    pre, post = torch.cat(pres).int().cuda(), torch.cat(posts).int().cuda()
    e = pre.numel()
    layouts = tuple(tuple(t.cuda() for t in row) for row in build_incoming_layout(pre, post, splits, n))
    assert ts.pulse_grad_layout(pre, splits, n) is not None
    pulse = tuple((torch.rand(1, n, generator=gen) < .3).float().cuda().requires_grad_() for _ in range(4))
    weight = torch.randn(e, generator=gen).cuda().requires_grad_()
    probe = torch.randn(1, n, generator=gen).cuda()
    ref = PyTorchDelayedSynapticTransmission.apply(*pulse, pre, post, weight, splits)
    expected = torch.autograd.grad((ref * probe).sum(), (*pulse, weight))
    runs = [torch.autograd.grad((IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts) * probe)
                                .sum(), (*pulse, weight)) for _ in range(2)]
    for a, b in zip(runs[0], expected):
        torch.testing.assert_close(a, b)
    for a, b in zip(runs[0][:4], runs[1][:4]):
        assert torch.equal(a, b)
    old = ts.PULSE_RUNS
    ts.PULSE_RUNS = False
    try:
        atomic = torch.autograd.grad((IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts)
                                      * probe).sum(), pulse)
    finally:
        ts.PULSE_RUNS = old
    for a, b in zip(runs[0][:4], atomic):
        torch.testing.assert_close(a, b)
    weight.grad = None
    with ts.direct_edge_grad():
        for _ in range(2):                                                   # fused run kernel accumulates into .grad
            IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts).mul(probe).sum().backward()
    torch.testing.assert_close(weight.grad, 2 * expected[4])


def test_compacted_event_forward_is_bitwise_identical_to_the_row_scan():
    """The forward over a device-compacted list of spiking (tier, cell) slots gives exactly the row-scan event result,
    call after call (exact integer sums; the list order may differ)."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import information_boltzmann.core.triton_synapse as ts
    gen = torch.Generator().manual_seed(7)
    n, pres, posts, splits = 500, [], [], [0]
    for tier in range(4):
        degree = torch.randint(0, 40, (n,), generator=gen)
        for cell in range(n):
            pres.append(torch.full((int(degree[cell]),), cell))
            posts.append(torch.sort(torch.randint(0, n, (int(degree[cell]),), generator=gen)).values)
        splits.append(splits[-1] + int(degree.sum()))
    pre, post = torch.cat(pres).int().cuda(), torch.cat(posts).int().cuda()
    layouts = tuple(tuple(t.cuda() for t in row) for row in build_incoming_layout(pre, post, splits, n))
    weight = torch.randn(pre.numel(), generator=gen).cuda()
    pulse = tuple(((torch.rand(1, n, generator=gen) < .05).float() * torch.rand(1, n, generator=gen)).cuda()
                  for _ in range(4))
    old = ts.EVENT_COMPACT
    try:
        ts.EVENT_COMPACT = False
        scan = IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts)
        ts.EVENT_COMPACT = True
        for _ in range(50):
            assert torch.equal(IncomingDelayedTransmission.apply(*pulse, pre, post, weight, splits, layouts), scan)
    finally:
        ts.EVENT_COMPACT = old
