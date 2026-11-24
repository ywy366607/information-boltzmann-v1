"""Sequence length must not create a new binary or change any derivative."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Triton runtime-length contract')
def test_workspace_reuses_one_binary_for_all_lengths_and_keeps_gradients(monkeypatch):
    import triton
    from scripts.ib_local import causal_workspace_scan as scan
    kernels = [triton.jit(kernel.fn, do_not_specialize=['T'])
               for kernel in (scan._forward, scan._backward)]
    monkeypatch.setattr(scan, '_forward', kernels[0])
    monkeypatch.setattr(scan, '_backward', kernels[1])
    torch.manual_seed(1619)
    for t in (1, 17, 65, 129):
        a = (torch.rand(1, t, 3, 8, device='cuda')*.7).requires_grad_()
        u = torch.randn_like(a, requires_grad=True)
        s = torch.randn(1, 3, 8, device='cuda', requires_grad=True)
        derivative = torch.randn_like(a)
        actual = scan.causal_scan(a, u, s)
        expected = scan.reference_scan(a, u, s)
        ga = torch.autograd.grad((actual*derivative).sum(), (a, u, s))
        ge = torch.autograd.grad((expected*derivative).sum(), (a, u, s))
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
        for x, y in zip(ga, ge):
            torch.testing.assert_close(x, y, atol=3e-6, rtol=2e-5)
    for kernel in kernels:
        parameter = next(p for p in kernel.params if p.name == 'T')
        assert parameter.do_not_specialize and not parameter.is_constexpr
        assert sum(len(cache) for cache in kernel.cache.values()) == 1
