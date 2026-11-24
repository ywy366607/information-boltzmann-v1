"""Actual optimizer arithmetic/state checks, never capability or convergence tests."""
import copy

import pytest
import torch

from information_boltzmann.budgeted_adamw import (
    BudgetedAdamW, attach_budgeted_adamw, balanced_cells, coherence_gate, tiled, untiled,
)


@pytest.mark.parametrize('shape', [(128, 256), (7, 11), (17,)])
def test_neutral_real_parameter_updates_match_adamw(shape):
    torch.manual_seed(13)
    p = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    a = torch.optim.AdamW([p], lr=.003, weight_decay=.13, foreach=False)
    b = BudgetedAdamW([q], lr=.003, weight_decay=.13, allocation=False, cooling=False)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            torch.testing.assert_close(a.state[p][key], b.state[q][key])


def test_actual_tile_broadcast_has_both_budgets():
    x = torch.arange(128 * 256).reshape(128, 256)
    assert torch.equal(untiled(tiled(x, 64)), x)
    phase = torch.tensor([[-1., -.3, 0., 1.], [.2, .5, -.1, -.9]])
    a = balanced_cells(phase)
    torch.testing.assert_close(a.sum(-1), torch.full((2, 4, 2), 2.))
    torch.testing.assert_close(a.sum(-2), torch.full((2, 4, 2), 2.))
    assert bool(((a >= 0) & (a <= 2)).all())


def test_same_observer_can_close_and_reopen_without_an_age_schedule():
    energy = torch.ones(3)
    torch.testing.assert_close(coherence_gate(torch.tensor([0., .1, 1.]), energy, .1),
                               torch.tensor([0., 0., 1.]))
    # A large amount of incoherent energy does not itself imply more writing.
    assert coherence_gate(torch.tensor(1.), torch.tensor(1000.), .1) == 0


def test_closed_gate_including_decay_holds_real_weights_but_observes():
    p = torch.nn.Parameter(torch.ones(64, 64, dtype=torch.float64))
    opt = BudgetedAdamW([p], betas=(.5, .9), weight_decay=.2, cooling=True)
    p.grad = torch.ones_like(p)
    opt.step()
    before = p.detach().clone()
    # m_next = .5*(.5) + .5*(-.5) = 0: noisy/reversing evidence, not a task.
    p.grad = torch.full_like(p, -.5)
    opt.step()
    assert torch.equal(p, before)
    assert opt.state[p]['step'] == 2
    assert bool((opt.state[p]['budget_gate'] == 0).all())
    for _ in range(12):
        p.grad = torch.ones_like(p)
        opt.step()
    assert bool((opt.state[p]['budget_gate'] > .9).all())
    assert not torch.equal(p, before)


def test_exact_resume_including_protection_observer_and_none_gradient():
    torch.manual_seed(17)
    p = torch.nn.Parameter(torch.randn(64, 128))
    a = BudgetedAdamW([p], cooling=True)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        a.step()
    q = torch.nn.Parameter(p.detach().clone())
    b = BudgetedAdamW([q], cooling=True)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    for _ in range(4):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        a.step()
        b.step()
        assert torch.equal(p, q)
    before = q.detach().clone()
    count = b.state[q]['step'].clone()
    q.grad = None
    b.step()
    assert torch.equal(q, before) and torch.equal(count, b.state[q]['step'])


def test_adopt_old_adam_without_inventing_observer_history_or_resetting_moments():
    p = torch.nn.Parameter(torch.ones(4, 4))
    a = torch.optim.AdamW([p])
    p.grad = torch.ones_like(p)
    a.step()
    b = BudgetedAdamW([p], cooling=True)
    saved = a.state_dict()
    saved['param_groups'][0].update(allocation=True, cooling=True, tile_size=64)
    b.load_state_dict(saved)
    old_m = b.state[p]['exp_avg'].clone()
    old_step = int(b.state[p]['step'])
    b.step()
    torch.testing.assert_close(b.state[p]['budget_prior_rms'], old_m.square().mean().sqrt())
    assert b.state[p]['budget_count'] == 1
    assert int(b.state[p]['step']) == old_step + 1
    torch.testing.assert_close(b.state[p]['exp_avg'], .9 * old_m + .1 * p.grad)


def test_unknown_inherited_direction_does_not_masquerade_as_no_new_signal():
    p = torch.nn.Parameter(torch.ones(64,64, dtype=torch.float64))
    opt = torch.optim.AdamW([p], betas=(.9,.999))
    for _ in range(12):
        p.grad = torch.ones_like(p); opt.step()
    attach_budgeted_adamw(opt, allocation=False)
    # Fresh evidence is perfectly coherent but opposes inherited momentum.
    # A lower norm bound can be zero; zero is not evidence of a noisy stream.
    for _ in range(8):
        p.grad = -torch.ones_like(p); opt.step()
        torch.testing.assert_close(opt.state[p]['budget_gate'],
            torch.ones_like(opt.state[p]['budget_gate']), atol=1e-12, rtol=1e-12)
    assert opt.state[p]['budget_count'] == 8


def test_attachment_preserves_real_adam_identity_moments_groups_and_new_parameters():
    p = torch.nn.Parameter(torch.ones(64, 64))
    opt = torch.optim.AdamW([dict(params=[p], lr=.002, weight_decay=.13)], foreach=True)
    p.grad = torch.ones_like(p)
    opt.step()
    identity, moment = id(opt), opt.state[p]['exp_avg']
    before = p.detach().clone()
    attach_budgeted_adamw(opt)
    assert id(opt) == identity and opt.state[p]['exp_avg'] is moment
    assert torch.equal(p, before) and opt.param_groups[0]['lr'] == .002
    q = torch.nn.Parameter(torch.ones(8))
    opt.add_param_group(dict(params=[q]))
    assert opt.param_groups[1]['cooling'] and opt.param_groups[1]['allocation']
    p.grad = -torch.ones_like(p)
    q.grad = torch.ones_like(q)
    opt.step()
    assert opt.state[p]['budget_count'] == 1 and int(opt.state[p]['step']) == 2


def test_navigation_restore_recovers_optimizer_execution_not_only_its_tensors():
    from types import SimpleNamespace
    import numpy as np
    from scripts.ib_local.run_navigation_residual_wood import snapshot
    from scripts.ib_local.navigation_checkpoint import restore
    class World:
        def state_dict(self): return dict(cursor=13)
        def load_state_dict(self, value): self.cursor = value['cursor']
    class Table:
        def state_dict(self): return dict(history=5)
        @classmethod
        def from_state_dict(cls, value, **kw): return cls()
    def engine():
        model = torch.nn.Linear(64,64)
        return SimpleNamespace(model=model, optimizer=torch.optim.AdamW(model.parameters()),
                               world=World(), table=Table(), recurrent=torch.ones(2), prev=None)
    a = engine()
    attach_budgeted_adamw(a.optimizer)
    for _ in range(3):
        for p in a.model.parameters(): p.grad = torch.randn_like(p)
        a.optimizer.step()
    saved = snapshot(a)
    b = engine(); restore(b, saved)
    assert isinstance(b.optimizer, BudgetedAdamW)
    for p,q in zip(a.model.parameters(),b.model.parameters()):
        p.grad = torch.randn_like(p); q.grad = p.grad.clone()
    a.optimizer.step(); b.optimizer.step()
    for p,q in zip(a.model.parameters(),b.model.parameters()):
        assert torch.equal(p,q)
        assert torch.equal(a.optimizer.state[p]['budget_gate'],b.optimizer.state[q]['budget_gate'])
    assert np.isfinite(next(b.model.parameters()).detach().numpy()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('shape', [(128, 256), (8, 256), (768, 1, 4), (8,)])
def test_fused_matches_executed_reference_on_actual_layer_shapes(shape):
    torch.manual_seed(23)
    p = torch.nn.Parameter(torch.randn(shape, device='cuda'))
    q = torch.nn.Parameter(p.detach().clone())
    a = BudgetedAdamW([p], cooling=True, fused=False)
    b = BudgetedAdamW([q], cooling=True, fused=True)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q, rtol=2e-6, atol=2e-7)
        for key in ('exp_avg', 'exp_avg_sq', 'budget_energy', 'budget_phase', 'budget_gate'):
            torch.testing.assert_close(a.state[p][key], b.state[q][key], rtol=2e-5, atol=2e-6)


def test_wiener_gate_formula_and_noise_rejection():
    energy = torch.ones(4, dtype=torch.float64)
    signal = torch.tensor([0.0, 0.05, 0.10, 1.0], dtype=torch.float64)
    w = 0.05
    got = coherence_gate(signal, energy, w, wiener=True)
    # At S<=wE -> 0; at S=0.10 -> (0.10-0.05)/(0.95*0.10) = 0.526315...; at S=1 -> 1.
    expected = torch.tensor([0.0, 0.0, (0.10 - w) / ((1.0 - w) * 0.10), 1.0], dtype=torch.float64)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('shape', [(128, 256), (32, 8), (32, 2), (32, 1), ()])
def test_wiener_fused_matches_cpu_reference_on_actual_and_scalar_shapes(shape):
    torch.manual_seed(31)
    p = torch.nn.Parameter(torch.randn(shape, device='cuda'))
    q = torch.nn.Parameter(p.detach().clone())
    a = BudgetedAdamW([p], cooling=True, wiener_gate=True, fused=False)
    b = BudgetedAdamW([q], cooling=True, wiener_gate=True, fused=True)
    drift = torch.full_like(p, 0.25)
    for _ in range(8):
        g = drift + torch.randn_like(p)
        p.grad = g
        q.grad = g.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q, rtol=2e-6, atol=2e-7)
        for key in ('exp_avg', 'exp_avg_sq', 'budget_energy', 'budget_phase', 'budget_gate'):
            torch.testing.assert_close(a.state[p][key], b.state[q][key], rtol=2e-5, atol=2e-6)
