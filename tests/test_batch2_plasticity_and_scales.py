"""Tests for Batch 2 opt-in fixes: split Hamiltonian theta, learnable ridge/temperature, Wiener gate."""
import copy
import math

import torch

from information_boltzmann.budgeted_adamw import attach_budgeted_adamw
from scripts.ib_local.adaptive_hamiltonian import (
    AdaptiveHamiltonian, configure as configure_adaptive, migrate_split_theta,
)
from scripts.ib_local.autonomous_cycle import configure as configure_autonomous, dispatch
from scripts.ib_local.learnable_precision_scales import (
    configure as configure_scales, diagnostics as scale_diagnostics,
    effective_ridge, effective_temperature,
)
from scripts.ib_local.navigation_checkpoint import restore, snapshot
from scripts.ib_local.optimizer_policy import configure_wiener_gate
from scripts.quiet_force_jet import commit_engine, finish_choice, prepare_engine
from test_autonomous_cycle import force_choices
from test_generative_cycle import configured


def test_split_hamiltonian_theta_migration_preserves_state_jet_and_restore(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    e = configured()
    for _ in range(3):
        e.step(torch.randn(8, 4))
    assert int(e.model.memory.occupied.sum()) >= 2
    configure_adaptive(e)
    configure_autonomous(e)
    attach_budgeted_adamw(e.optimizer, allocation=True, cooling=True, fused=False)
    config = {'optimizer_plasticity': {'kind': 'budgeted'}}
    configure_wiener_gate(e, config, True)

    # Run one real event before migration so theta has populated Adam m/v.
    force_choices(monkeypatch, [1, 0])
    dispatch(e); dispatch(e); dispatch(e)

    mod = e.model.head_coupling
    before_theta = mod.theta.detach().clone()
    old_m = e.optimizer.state[mod.theta]['exp_avg'].clone()
    old_v = e.optimizer.state[mod.theta]['exp_avg_sq'].clone()
    old_param_count = len(e.params)

    migrate_split_theta(e)
    assert mod._split_theta is True
    assert len(e.params) == old_param_count + 3
    torch.testing.assert_close(mod.theta.detach(), before_theta, rtol=0, atol=0)

    # Check exact m/v slice preservation across the 4 role tensors.
    h = mod.heads
    slices = ((0, h), (h, h + 1), (h + 1, h + 2), (h + 2, h + 4))
    for part, (lo, hi) in zip(mod.theta_parts(), slices):
        st = e.optimizer.state[part]
        torch.testing.assert_close(st['exp_avg'], old_m[:, lo:hi], rtol=0, atol=0)
        torch.testing.assert_close(st['exp_avg_sq'], old_v[:, lo:hi], rtol=0, atol=0)
        assert 'budget_gate' not in st  # fresh per-role observer on next step

    # PreparedQuiet (quiet_force_jet) works with split theta parts.
    mod._prepared_quiet_enabled = True
    prop = prepare_engine(e)
    assert prop is not None and prop.valid(e.recurrent, e.bfss_momenta)
    finish_choice(e, prop, True)
    committed = commit_engine(e, 1)
    assert committed is not None

    # Save snapshot, run an external event, verify per-role gates and exact restore.
    force_choices(monkeypatch, [1, 0])
    dispatch(e); dispatch(e)
    saved = snapshot(e)
    row = dispatch(e)
    gates = row['adaptive_dynamics']['budget_write_fraction']
    assert isinstance(gates, dict) and set(gates.keys()) == {'freq', 'context', 'coupling', 'pair'}
    assert all(v is not None for v in gates.values())
    assert not torch.equal(mod.theta.detach(), before_theta)
    expected_sd = copy.deepcopy(e.model.state_dict())

    fresh = configured()
    configure_adaptive(fresh, initialize=False, split_theta=True)
    configure_autonomous(fresh)
    attach_budgeted_adamw(fresh.optimizer, allocation=True, cooling=True, fused=False)
    restore(fresh, saved)
    repeated = dispatch(fresh)
    assert row['adaptive_dynamics'] == repeated['adaptive_dynamics']
    assert row['utterance_hex'] == repeated['utterance_hex']
    for k, v in expected_sd.items():
        if torch.is_tensor(v):
            assert torch.equal(v, fresh.model.state_dict()[k]), k


def test_learnable_ridge_and_temperature_learn_and_restore(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    e = configured()
    for _ in range(3):
        e.step(torch.randn(8, 4))
    assert int(e.model.memory.occupied.sum()) >= 2
    configure_adaptive(e)
    configure_autonomous(e)
    attach_budgeted_adamw(e.optimizer, allocation=True, cooling=True, fused=False)
    configure_wiener_gate(e, {'optimizer_plasticity': {'kind': 'budgeted'}}, True)

    init_ridge = float(e.relation_ridge)
    init_temp = float(e.model.memory.temperature)
    configure_scales(e, dict(schema=1, ridge=True, temperature=True), initialize=True)
    assert abs(float(effective_ridge(e.model).detach()) - init_ridge) < 1e-6
    assert abs(float(effective_temperature(e.model.memory).detach()) - init_temp) < 1e-6

    # Capture snapshot before event, run an external event, verify both parameters moved.
    force_choices(monkeypatch, [0])
    dispatch(e)
    saved = snapshot(e)
    before_ridge = float(e.model.log_relation_ridge.detach())
    before_temp = float(e.model.memory.log_temperature.detach())
    row = dispatch(e)
    diag = scale_diagnostics(e)
    assert diag['relation_ridge'] > 0 and diag['memory_temperature'] > 0
    assert diag['relation_ridge_gate'] is not None and diag['memory_temperature_gate'] is not None
    assert float(e.model.log_relation_ridge.detach()) != before_ridge
    assert float(e.model.memory.log_temperature.detach()) != before_temp
    expected_sd = copy.deepcopy(e.model.state_dict())

    fresh = configured()
    configure_adaptive(fresh, initialize=False)
    configure_autonomous(fresh)
    configure_scales(fresh, dict(schema=1, ridge=True, temperature=True), initialize=False)
    attach_budgeted_adamw(fresh.optimizer, allocation=True, cooling=True, fused=False)
    restore(fresh, saved)
    repeated = dispatch(fresh)
    assert row['utterance_hex'] == repeated['utterance_hex']
    assert row['metal_g'] == repeated['metal_g']
    for k, v in expected_sd.items():
        if torch.is_tensor(v):
            assert torch.equal(v, fresh.model.state_dict()[k]), k


def test_ridge_lr_migration_preserves_states_forward_and_exact_restore(monkeypatch):
    from scripts.ib_local.learnable_precision_scales import set_scale_learning_rates
    torch.set_num_threads(1)
    torch.manual_seed(47)
    e = configured()
    for _ in range(3):
        e.step(torch.randn(8, 4))
    configure_adaptive(e)
    configure_autonomous(e)
    attach_budgeted_adamw(e.optimizer, allocation=True, cooling=True, fused=False)
    config = {'optimizer_plasticity': {'kind': 'budgeted'}}
    configure_wiener_gate(e, config, True)
    configure_scales(e, dict(schema=1, ridge=True, temperature=True), initialize=True)
    config['learnable_precision_scales'] = dict(schema=1, ridge=True, temperature=True)

    # Run 1 real event so both scalar parameters have populated Adam + budget observer states.
    force_choices(monkeypatch, [0])
    dispatch(e); dispatch(e)

    p_r = e.model.log_relation_ridge
    p_t = e.model.memory.log_temperature
    before_r = p_r.detach().clone()
    before_t = p_t.detach().clone()
    st_r_before = dict(e.optimizer.state[p_r])
    st_t_before = dict(e.optimizer.state[p_t])
    old_group_count = len(e.optimizer.param_groups)
    base_lr = e.optimizer.param_groups[0]['lr']

    # Check forward output before vs after migration is bit-for-bit identical.
    probe_x = torch.randn(2, e.model.memory.width)
    with torch.no_grad():
        read_before = e.model.memory.read(probe_x, e.model.encoder).value.clone()

    set_scale_learning_rates(e, config, ridge_lr=1e-2)
    assert len(e.optimizer.param_groups) == old_group_count + 1
    assert torch.equal(p_r.detach(), before_r) and torch.equal(p_t.detach(), before_t)
    # Exact tensor identity preserved for ALL Adam + budget observer states.
    for k, v in st_r_before.items():
        assert e.optimizer.state[p_r][k] is v if torch.is_tensor(v) else e.optimizer.state[p_r][k] == v
    for k, v in st_t_before.items():
        assert e.optimizer.state[p_t][k] is v if torch.is_tensor(v) else e.optimizer.state[p_t][k] == v
    with torch.no_grad():
        read_after = e.model.memory.read(probe_x, e.model.encoder).value
    assert torch.equal(read_before, read_after)

    diag = scale_diagnostics(e)
    assert diag['relation_ridge_lr'] == 1e-2
    assert diag['memory_temperature_lr'] == base_lr
    assert config['learnable_precision_scales'] == dict(schema=1, ridge=True, temperature=True, ridge_lr=1e-2)

    # Full save/restore must preserve param groups, LRs, and bit-for-bit next transition.
    force_choices(monkeypatch, [0])
    dispatch(e)
    saved = snapshot(e)
    row = dispatch(e)
    expected_sd = copy.deepcopy(e.model.state_dict())

    fresh = configured()
    configure_adaptive(fresh, initialize=False)
    configure_autonomous(fresh)
    configure_scales(fresh, config['learnable_precision_scales'], initialize=False)
    attach_budgeted_adamw(fresh.optimizer, allocation=True, cooling=True, fused=False)
    restore(fresh, saved)
    fresh_diag = scale_diagnostics(fresh)
    assert fresh_diag['relation_ridge_lr'] == 1e-2
    assert fresh_diag['memory_temperature_lr'] == base_lr
    repeated = dispatch(fresh)
    assert row['utterance_hex'] == repeated['utterance_hex']
    assert row['metal_g'] == repeated['metal_g']
    for k, v in expected_sd.items():
        if torch.is_tensor(v):
            assert torch.equal(v, fresh.model.state_dict()[k]), k
