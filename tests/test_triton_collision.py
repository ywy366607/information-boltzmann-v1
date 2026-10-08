"""Numerical scattering, generator gradients, batch layout and graph replay."""
import copy
import os

import pytest
import torch

pytest.importorskip("triton")
from information_boltzmann.core.triton_givens import (
    native_givens, triton_givens, triton_givens_adjoint,
)
from information_boltzmann.core.torus3d import LocalInvariantCollision3D
from information_boltzmann.core.triton_projection import (
    NullspaceProjection, NullspaceReconstruction,
)


def reference(values, angles):
    d = values.shape[-1]
    for layer in range(angles.shape[-2]):
        pair = torch.roll(torch.arange(d, device=values.device), layer).reshape(-1, 2)
        left, right = values[..., pair[:, 0]], values[..., pair[:, 1]]
        sine, cosine = angles[..., layer, :].sin(), angles[..., layer, :].cos()
        updated = values.clone()
        updated[..., pair[:, 0]] = cosine * left - sine * right
        updated[..., pair[:, 1]] = sine * left + cosine * right
        values = updated
    return values


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_native_fallback_and_reverse_order_adjoint(layers):
    torch.manual_seed(67)
    value = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    angle = torch.randn(2, 7, layers, 6, dtype=torch.float64, requires_grad=True)
    actual = triton_givens(value, angle)
    expected = reference(value, angle)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    direction = torch.randn_like(value)
    expected_grad = torch.autograd.grad(expected, value, direction)[0]
    torch.testing.assert_close(triton_givens_adjoint(direction, angle), expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical kernel test")
@pytest.mark.parametrize("batch,nodes,d,layers", [
    (1, 256, 124, 2), (2, 17, 124, 2), (2, 9, 252, 2),
    (2, 9, 252, 3), (1, 5, 428, 2),
    (2, 11, 12, 2), (1, 13, 32, 2),
])
def test_cuda_values_state_and_angle_derivatives(batch, nodes, d, layers):
    torch.manual_seed(71)
    value = torch.randn(batch, nodes, d, device="cuda", requires_grad=True)
    angle = (torch.randn(batch, nodes, layers, d // 2, device="cuda") * 4).requires_grad_()
    actual = triton_givens(value, angle)
    if layers == 2:
        assert "TritonGivensFunctionBackward" in type(actual.grad_fn).__name__
    expected = reference(value, angle)
    direction = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, (value, angle), direction)
    expected_grad = torch.autograd.grad(expected, (value, angle), direction)
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)
    for got, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, wanted, rtol=6e-5, atol=8e-6)
    torch.testing.assert_close(actual.square().sum(-1), value.square().sum(-1),
                               rtol=3e-6, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiled training test")
@pytest.mark.parametrize("transposed", [False, True])
def test_compiled_collision_graph_has_full_gradients_and_invariants(transposed, monkeypatch):
    torch.manual_seed(73)
    torch.set_num_threads(2)
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    # Earlier suite tests may already have imported Inductor, whose Windows
    # launcher option is cached at import. Patch the live option as well.
    import torch._inductor.config as inductor_config
    monkeypatch.setattr(inductor_config, "use_static_cuda_launcher", False)
    collision = LocalInvariantCollision3D((4, 4, 4), 8, 16, layers=2).cuda()
    expected_collision = copy.deepcopy(collision)
    expected_collision._triton_givens = native_givens
    expected_collision._structured_projection = False
    value = (torch.randn(2, 128, 4, 4, 4, device="cuda").permute(0, 2, 3, 4, 1)
             if transposed else torch.randn(2, 4, 4, 4, 128, device="cuda")).requires_grad_()
    expected_value = value.detach().clone().requires_grad_()
    direction = torch.randn_like(value)
    expected, _ = expected_collision(expected_value, 4.0)
    (expected * direction).sum().backward()
    compiled = torch.compile(collision.forward, fullgraph=True, dynamic=False)

    def run():
        collision.zero_grad(set_to_none=True)
        value.grad = None
        output, _ = compiled(value, 4.0)
        (output * direction).sum().backward()
        return output

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(value.grad, expected_value.grad, rtol=8e-5, atol=1e-5)
    for (name, parameter), (_, expected_parameter) in zip(
            collision.named_parameters(), expected_collision.named_parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad,
                                   rtol=1e-4, atol=2e-4, msg=name)
    assert collision.angle[-1].weight.grad.norm() > 0
    before = value.detach().reshape(2, -1, 128)
    after = output.detach().reshape_as(before)
    constraints = collision.constraints.to(value)
    torch.testing.assert_close(before @ constraints.T, after @ constraints.T,
                               rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(before.square().sum(-1), after.square().sum(-1),
                               rtol=3e-6, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA projection derivatives")
@pytest.mark.parametrize("d,nodes", [(64, 32), (128, 256), (256, 32), (432, 7)])
@pytest.mark.parametrize("transposed", [False, True])
def test_structured_basis_preserves_exact_coordinates_and_adjoint(d, nodes, transposed):
    torch.manual_seed(79)
    collision = LocalInvariantCollision3D((nodes, 1, 1), 8, d // 8).cuda()
    assert collision._structured_projection
    basis64 = collision.nullspace
    identity = torch.eye(d, device="cuda", dtype=torch.float64)[:, 4:]
    torch.testing.assert_close(identity + collision.projection_left @ collision.projection_right.T,
                               basis64, rtol=1e-12, atol=1e-12)
    left, right, basis = collision.projection_left.float(), collision.projection_right.float(), basis64.float()
    x = (torch.randn(2, d, nodes, device="cuda").transpose(1, 2) if transposed
         else torch.randn(2, nodes, d, device="cuda")).requires_grad_()
    c = torch.randn(2, nodes, d - 4, device="cuda", requires_grad=True)
    p = NullspaceProjection.apply(x, left, right)
    p_ref = x @ basis
    out = NullspaceReconstruction.apply(x, c, left, right)
    out_ref = x + c @ basis.T
    torch.testing.assert_close(p, p_ref, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(out, out_ref, rtol=2e-5, atol=3e-6)
    g_p, g_out = torch.randn_like(p), torch.randn_like(out)
    actual = torch.autograd.grad((p, out), (x, c), (g_p, g_out))
    expected = torch.autograd.grad((p_ref, out_ref), (x, c), (g_p, g_out))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=3e-5, atol=4e-6)
