"""Invariants and resolution transfer for the continuous CBIM truncation."""
import pytest
import torch
from scripts.ib_local.cbim_operator3d import (
    CBIMOperator3D, ConservativeScattering3D,
    ContinuousCayleyTransport3D, LocalSourceOutflow3D)


def test_3d_transport_preserves_norm():
    torch.manual_seed(4)
    model = ContinuousCayleyTransport3D((4, 6, 8), d=16).double()
    h = torch.randn(2, 4, 6, 8, 16, dtype=torch.float64)
    out, _ = model(h)
    torch.testing.assert_close(out.norm(), h.norm(), atol=2e-12, rtol=2e-12)


def test_3d_scattering_preserves_sum_and_energy():
    torch.manual_seed(5)
    model = ConservativeScattering3D((4, 4, 6), d=16).double()
    h = torch.randn(2, 4, 4, 6, 16, dtype=torch.float64)
    out, _ = model(h)
    torch.testing.assert_close(out.sum((1, 2, 3)), h.sum((1, 2, 3)),
                               atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(out.square().sum((1, 2, 3, 4)),
                               h.square().sum((1, 2, 3, 4)),
                               atol=2e-11, rtol=2e-12)


def test_local_source_bound_is_resolution_uniform():
    for shape in ((4, 4, 4), (8, 8, 4)):
        model = LocalSourceOutflow3D(32, shape, d=16, cell_radius=1.25)
        h = torch.randn(2, *shape, 16) * 100
        out, _ = model(h, torch.tensor([1, 2]))
        assert out.norm(dim=-1).amax() <= 1.250001
        assert (.5 * out.square().sum(-1).mean()) <= .5 * 1.25 ** 2 + 1e-6


def test_weights_load_across_spatial_resolutions():
    coarse = CBIMOperator3D(vocab_size=32, shape=(4, 4, 4), d=16)
    fine = CBIMOperator3D(vocab_size=32, shape=(8, 8, 4), d=16)
    fine.load_state_dict(coarse.state_dict(), strict=True)
    assert not any('position_features' in key or 'symbol_input' in key
                   for key in coarse.state_dict())


def test_scattering_receives_direct_ce_gradient_and_affects_logits():
    torch.manual_seed(6)
    model = CBIMOperator3D(vocab_size=32, shape=(4, 4, 4), d=16)
    h = torch.randn(2, 4, 4, 4, 16) * .1
    tokens, targets = torch.tensor([1, 2]), torch.tensor([3, 4])
    logits, _, _ = model.step(h, tokens)
    off, _, _ = model.step(h, tokens, disable_scattering=True)
    assert (logits - off).abs().max() > 1e-7
    torch.nn.functional.cross_entropy(logits, targets).backward()
    assert model.scattering.angle[-1].weight.grad.norm() > 0


def test_3d_causality_and_all_gradients():
    torch.manual_seed(7)
    model = CBIMOperator3D(vocab_size=32, shape=(4, 4, 4), d=16)
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, state, diag = model(ids, targets)
    assert state.shape == (1, 4, 4, 4, 16)
    assert diag['final_energy'] <= .5 * 1.25 ** 2 + 1e-6
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_3d_graph_training_step():
    from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer
    model = CBIMOperator3D(vocab_size=32, shape=(4, 4, 4), d=16).cuda()
    runner = CBIMGraphTrainer(model, tokens=4)
    loss, state, _ = runner.step(torch.randint(32, (1, 4), device='cuda'),
                                 torch.randint(32, (1, 4), device='cuda'))
    assert torch.isfinite(loss)
    assert state.shape == (1, 4, 4, 4, 16)
