"""Unit tests for biological multi-delay synaptic transmission."""
import pytest
import torch
from information_boltzmann.core.triton_synapse import (
    execute_delayed_synaptic_transmission,
    PyTorchDelayedSynapticTransmission,
    HAS_TRITON,
)


def test_delayed_synaptic_transmission_cpu():
    torch.manual_seed(42)
    N = 100
    E = 400
    pre = torch.randint(0, N, (E,))
    post = torch.randint(0, N, (E,))
    w = torch.randn(E)
    splits = [0, 150, 280, 360, 400]

    s1 = torch.randn(1, N, requires_grad=True)
    s2 = torch.randn(1, N, requires_grad=True)
    s3 = torch.randn(1, N, requires_grad=True)
    s4 = torch.randn(1, N, requires_grad=True)

    out = execute_delayed_synaptic_transmission((s1, s2, s3, s4), pre, post, w, splits)
    assert out.shape == (1, N)

    loss = out.sum()
    loss.backward()

    assert s1.grad is not None and s1.grad.shape == (1, N)
    assert s2.grad is not None and s2.grad.shape == (1, N)
    assert s3.grad is not None and s3.grad.shape == (1, N)
    assert s4.grad is not None and s4.grad.shape == (1, N)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA & Triton required")
def test_delayed_synaptic_transmission_cuda_equivalence():
    torch.manual_seed(42)
    N = 500
    E = 2000
    pre = torch.randint(0, N, (E,), device="cuda")
    post = torch.randint(0, N, (E,), device="cuda")
    w = torch.randn(E, device="cuda", requires_grad=True)
    w_pt = w.detach().clone().requires_grad_(True)
    splits = [0, 800, 1400, 1800, 2000]

    s1_a = torch.randn(1, N, device="cuda", requires_grad=True)
    s2_a = torch.randn(1, N, device="cuda", requires_grad=True)
    s3_a = torch.randn(1, N, device="cuda", requires_grad=True)
    s4_a = torch.randn(1, N, device="cuda", requires_grad=True)

    s1_b = s1_a.detach().clone().requires_grad_(True)
    s2_b = s2_a.detach().clone().requires_grad_(True)
    s3_b = s3_a.detach().clone().requires_grad_(True)
    s4_b = s4_a.detach().clone().requires_grad_(True)

    out_triton = execute_delayed_synaptic_transmission((s1_a, s2_a, s3_a, s4_a), pre, post, w, splits)
    out_pt = PyTorchDelayedSynapticTransmission.apply(s1_b, s2_b, s3_b, s4_b, pre, post, w_pt, splits)

    assert torch.allclose(out_triton, out_pt, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(out_triton)
    out_triton.backward(grad)
    out_pt.backward(grad)

    for g_a, g_b in zip([s1_a.grad, s2_a.grad, s3_a.grad, s4_a.grad],
                        [s1_b.grad, s2_b.grad, s3_b.grad, s4_b.grad]):
        assert torch.allclose(g_a, g_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w.grad, w_pt.grad, atol=1e-5, rtol=1e-5)


def test_delayed_connection_gradient_matches_native_autograd():
    """All delays, duplicate edges and batch reduction receive exact VJPs."""
    torch.manual_seed(7)
    pre = torch.tensor([0, 0, 1, 3, 2, 4, 1, 1])
    post = torch.tensor([1, 1, 4, 2, 0, 3, 0, 0])
    splits = (0, 2, 4, 6, 8)
    pulses = [torch.randn(2, 5, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    weight = torch.randn(8, dtype=torch.float64, requires_grad=True)
    expected = torch.zeros_like(pulses[0])
    for tier in range(4):
        left, right = splits[tier:tier+2]
        expected = expected.index_add(1, post[left:right],
            pulses[tier][:, pre[left:right]] * weight[left:right])
    actual = execute_delayed_synaptic_transmission(pulses, pre, post, weight, splits)
    error = torch.randn_like(actual)
    target_gradients = torch.autograd.grad((expected*error).sum(), (*pulses, weight))
    actual_gradients = torch.autograd.grad((actual*error).sum(), (*pulses, weight))
    torch.testing.assert_close(actual, expected)
    for actual_gradient, target_gradient in zip(actual_gradients, target_gradients):
        torch.testing.assert_close(actual_gradient, target_gradient, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA & Triton required")
def test_sparse_pulses_preserve_silent_neuron_gradient_and_graph_replay():
    """Zero forward traffic must not remove credit to silent presynaptic cells."""
    torch.manual_seed(12)
    n, edges = 127, 4099
    pre = torch.randint(n, (edges,), device='cuda')
    post = torch.randint(n, (edges,), device='cuda')
    splits = (0, 2201, 2201, 3700, edges)  # Includes an empty delay tier.
    weight = torch.randn(edges, device='cuda', requires_grad=True)
    pulses = [((torch.rand(1, n, device='cuda') < .04).float()
               * torch.randn(1, n, device='cuda')).requires_grad_(True)
              for _ in range(4)]
    pulses[3].data.zero_()  # Entirely silent tier must still have pulse credit.
    expected_pulses = [p.detach().clone().requires_grad_(True) for p in pulses]
    expected_weight = weight.detach().clone().requires_grad_(True)
    output = execute_delayed_synaptic_transmission(pulses, pre, post, weight, splits)
    reference = PyTorchDelayedSynapticTransmission.apply(
        *expected_pulses, pre, post, expected_weight, splits)
    cotangent = torch.randn_like(output)
    gradients = torch.autograd.grad(output, (*pulses, weight), cotangent)
    expected_gradients = torch.autograd.grad(
        reference, (*expected_pulses, expected_weight), cotangent)
    torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert gradients[3].abs().sum() > 0
    assert torch.count_nonzero(gradients[-1][splits[3]:]) == 0
    # The active mask is evaluated on device every replay, not baked in at
    # capture: a later pulse in an originally silent tier must propagate.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute_delayed_synaptic_transmission(pulses, pre, post, weight, splits)
    with torch.no_grad():
        pulses[3].fill_(.5)
    graph.replay()
    torch.cuda.synchronize()
    updated = PyTorchDelayedSynapticTransmission.apply(*pulses, pre, post, weight, splits)
    torch.testing.assert_close(captured, updated, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA & Triton required")
def test_long_presynaptic_runs_cross_reduction_blocks():
    """A cell with more than one Triton block of edges keeps every contribution."""
    torch.manual_seed(13)
    pre = torch.repeat_interleave(torch.arange(5, device='cuda'),
                                 torch.tensor([1, 2057, 27, 1024, 5099], device='cuda'))
    edges = pre.numel()
    post = torch.randint(11, (edges,), device='cuda')
    weight = (torch.randn(edges, device='cuda') * .1).requires_grad_(True)
    # Repeated same cell across tier and block boundaries tests both reductions.
    splits = (0, 1700, 2800, 6100, edges)
    pulses = [torch.randn(1, 11, device='cuda', requires_grad=True) for _ in range(4)]
    output = execute_delayed_synaptic_transmission(pulses, pre, post, weight, splits)
    reference = PyTorchDelayedSynapticTransmission.apply(*pulses, pre, post, weight, splits)
    cotangent = torch.randn_like(output)
    actual = torch.autograd.grad(output, (*pulses, weight), cotangent)
    expected = torch.autograd.grad(reference, (*pulses, weight), cotangent)
    torch.testing.assert_close(output, reference, atol=1e-4, rtol=2e-5)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=5e-5, rtol=2e-5)
