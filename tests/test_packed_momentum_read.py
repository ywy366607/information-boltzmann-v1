"""Numerical contracts for the production P-query inference packing."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from scripts.ib_local.gated_deltanet_2 import GatedDeltaNet2Layer
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_packed_reads_preserve_layers_batches_updates_and_gradient_path(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(201120)
    model = nn.Module()
    layers = nn.ModuleList([GatedDeltaNet2Layer(256, 8) for _ in range(32)]).to(device)
    model.body = SimpleNamespace(base=SimpleNamespace(layers=layers))
    model.feature_write = nn.Linear(256, 256, bias=False).to(device)
    p = torch.randn(32, 1, 8, 32, 32, device=device)
    source = torch.randn(17, 256, device=device)
    before_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        model._packed_momentum_enabled = False
        reference = decision_momentum_read(model, p, source)
        model._packed_momentum_enabled = True
        packed = decision_momentum_read(model, p, source)
        torch.testing.assert_close(packed, reference, rtol=2e-5, atol=3e-6)
        # Splitting and stacking the caller's candidate batch is equivalent.
        single = torch.cat([decision_momentum_read(model, p, s[None]) for s in source])
        torch.testing.assert_close(packed, single, rtol=2e-5, atol=3e-6)
        old_cache = model._packed_momentum_query
        layers[7].q_proj.weight.add_(.03)
        layers[11].norm.weight.mul_(1.04)
        updated = decision_momentum_read(model, p, source)
        assert model._packed_momentum_query is not old_cache
        model._packed_momentum_enabled = False
        torch.testing.assert_close(updated, decision_momentum_read(model, p, source), rtol=2e-5, atol=3e-6)
        model._packed_momentum_enabled = True
        # Momenta are live inputs, not values frozen in the weight cache.
        p.mul_(.7)
        torch.testing.assert_close(decision_momentum_read(model, p, source), .7*updated, rtol=2e-5, atol=3e-6)
        captured, _ = decision_momentum_read(model, p, source, capture=True)
        torch.testing.assert_close(captured, .7*updated, rtol=2e-5, atol=3e-6)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all('packed' not in k for k in model.state_dict())
    # Learning and eligibility reads retain the original differentiable path.
    source = source.detach().requires_grad_()
    got = decision_momentum_read(model, p, source)
    got.square().mean().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert all(layer.q_proj.weight.grad is not None for layer in layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@torch.no_grad()
def test_fused_step_refreshes_live_projection_like_cold_restore():
    from information_boltzmann.budgeted_adamw import BudgetedAdamW
    layer = GatedDeltaNet2Layer(64, 2).cuda()
    model = SimpleNamespace(body=SimpleNamespace(base=SimpleNamespace(layers=[layer])))
    source = torch.randn(2, 64, device='cuda')
    momentum = torch.randn(1, 1, layer.heads, layer.dk, layer.dv, device='cuda')
    weight = layer.q_proj.weight
    norm = layer.norm.weight
    opt = BudgetedAdamW([weight, norm], allocation=True, cooling=True, fused=True)
    before = decision_momentum_read(model, momentum, source)
    version_w, version_n = weight._version, norm._version
    weight.grad = torch.randn_like(weight)
    norm.grad = torch.randn_like(norm)
    opt.step()
    assert weight._version > version_w and norm._version > version_n
    live = decision_momentum_read(model, momentum, source)
    del model._packed_momentum_query
    torch.testing.assert_close(live, decision_momentum_read(model, momentum, source), rtol=0, atol=0)
    assert not torch.equal(live, before)
    model._packed_momentum_enabled = False
    torch.testing.assert_close(live, decision_momentum_read(model, momentum, source), rtol=2e-5, atol=3e-6)
