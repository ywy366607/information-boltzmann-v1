"""Numerical/data protocol checks; no synthetic capability task."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("material_owt_runner", ROOT / "scripts/run_material_optimizer_owt.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_cached_document_bounds_do_not_include_previous_separator():
    array = np.array([69, 70, 2, 1, 71, 2, 1])
    assert runner.document_bounds(array) == [(0, 2), (4, 5)]
    with pytest.raises(ValueError):
        runner.document_bounds(array[:-1])


def test_registration_uses_disjoint_document_domains_and_byte_units():
    import json
    registration = json.loads(runner.DEFAULT_REGISTRATION.read_text())
    assert registration["data"]["thresholds"][0] < registration["data"]["thresholds"][1]
    assert registration["experiment"]["phases"] == [["A", 2000], ["B", 1000], ["A", 2000]]
    assert sum(n for _, n in registration["experiment"]["phases"]) == registration["experiment"]["updates"]
    assert registration["experiment"]["targets_per_arm"] == 5000 * 8 * 128
    for domain, needed in (("A", 4000 * 1024), ("B", 1000 * 1024)):
        stream = registration["data"]["streams"]["train_" + domain]
        assert stream["tokens"] > needed + 1
    assert registration["model"]["vocabulary"] == 260
    assert registration["measurements"]["replay"].startswith("none:")


def test_continuous_optimizer_constructor_keeps_weights_and_ignores_metadata():
    import json
    import torch
    record = json.loads(runner.DEFAULT_REGISTRATION.read_text())
    model = torch.nn.Linear(3, 2)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = runner.make_optimizer(model, record, "cycle")
    assert optimizer.state == {}
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def test_resource_optimizer_constructor_and_source_gate_contract():
    import json
    import torch
    record = json.loads(runner.DEFAULT_REGISTRATION.read_text())
    record["experiment"]["resource"] = dict(lr=.001, observer_rate=.125)
    model = torch.nn.Linear(3, 2)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = runner.make_optimizer(model, record, "resource")
    assert optimizer.state == {}
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def test_feedback_arms_have_identical_configuration_except_coupling():
    import torch
    record = {"experiment": {"resource": dict(lr=.001, observer_rate=.125, chunk_compartments=2)}}
    models = [torch.nn.Linear(3, 2) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    a = runner.make_optimizer(models[0], record, "feedback_on")
    b = runner.make_optimizer(models[1], record, "feedback_off")
    first = {k:v for k,v in a.param_groups[0].items() if k != "params"}
    second = {k:v for k,v in b.param_groups[0].items() if k != "params"}
    assert first.pop("feedback") is True and second.pop("feedback") is False
    assert first == second and not a.state and not b.state


def test_auto_step_arms_use_identical_state_and_initial_weights():
    import torch
    record = {'experiment': {'resource': dict(lr=.001, auto_step=True)}}
    first, second = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    second.load_state_dict(first.state_dict())
    a = runner.make_optimizer(first, record, 'step_fixed')
    b = runner.make_optimizer(second, record, 'step_adaptive')
    for model, opt in ((first, a), (second, b)):
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
        assert a.state[p].keys() == b.state[q].keys()
        for key in a.state[p]:
            torch.testing.assert_close(a.state[p][key], b.state[q][key], atol=0, rtol=0)


def test_adam_cycle_arms_only_change_competition_strength():
    import torch
    record = {'experiment': {'resource': dict(lr=.0003, adam_base=True)}}
    models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
    opts = [runner.make_optimizer(models[0], record, 'adam_cycle_control'),
            runner.make_optimizer(models[1], record, 'adam_cycle')]
    a = {k:v for k,v in opts[0].param_groups[0].items() if k != 'params'}
    b = {k:v for k,v in opts[1].param_groups[0].items() if k != 'params'}
    assert a.pop('competition_strength') == 0 and b.pop('competition_strength') == 1
    assert a == b


def test_update_diagnostics_measure_applied_parameter_difference():
    import math
    import torch
    model = torch.nn.Linear(2, 2, bias=False)
    before = model.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    x = torch.tensor([[[1., 2.], [2., -1.]]])
    y = torch.tensor([[0, 1]])
    stats = runner.step(model, opt, x, y, record_update=True)
    change = (model.weight.detach() - before).double().norm().item()
    assert stats['update_rms'] == pytest.approx(change / math.sqrt(before.numel()))
    assert stats['relative_update_l2'] == pytest.approx(change / before.double().norm().item())


def test_latch_demo_uses_matched_state_control():
    import torch
    record = {'experiment': {'resource': dict(lr=.001, adam_base=True, latch_state=True)}}
    models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
    opts = [runner.make_optimizer(models[0], record, 'latch_control'),
            runner.make_optimizer(models[1], record, 'latch_demo')]
    a = {k:v for k,v in opts[0].param_groups[0].items() if k != 'params'}
    b = {k:v for k,v in opts[1].param_groups[0].items() if k != 'params'}
    assert a.pop('latch_strength') == 0 and b.pop('latch_strength') == 1
    assert a == b


def test_causal_controls_change_only_declared_mechanism():
    import torch
    record = {'experiment': {'resource': dict(lr=.001, adam_base=True,
                 latch_state=True, latch_strength=0)}}
    modes = ['latch_control', 'causal_allocation_off', 'causal_phase_off']
    groups = []
    for mode in modes:
        opt = runner.make_optimizer(torch.nn.Linear(3, 2), record, mode)
        groups.append({k:v for k,v in opt.param_groups[0].items() if k != 'params'})
    assert groups[0]['competition_strength'] == 1 and groups[1]['competition_strength'] == 0
    assert groups[0]['feedback'] is True and groups[2]['feedback'] is False
    groups[1]['competition_strength'] = 1
    groups[2]['feedback'] = True
    assert groups[0] == groups[1] == groups[2]


def test_causal_update_distribution_uses_applied_rows():
    import torch
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ff1 = torch.nn.Linear(2, 2, bias=False)
        def forward(self, x):
            return self.ff1(x)
    model = TestModel()
    before = model.ff1.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    opt.record_causal_diagnostics = True
    stats = runner.step(model, opt, torch.tensor([[[1., 2.], [2., -1.]]]),
                        torch.tensor([[0, 1]]), record_update=True)
    e = (model.ff1.weight.detach() - before).square().sum(dim=1)
    expected = e / e.sum()
    assert stats['ff_update_distribution']['ff1.weight']['max_row_share'] == pytest.approx(float(expected.max()))


def test_fatigue_control_changes_only_resource_consumption():
    import torch
    record = {'experiment': {'resource': dict(lr=.001, adam_base=True, feedback=False)}}
    a = runner.make_optimizer(torch.nn.Linear(3, 2), record, 'fatigue_on')
    b = runner.make_optimizer(torch.nn.Linear(3, 2), record, 'fatigue_off')
    ga = {k:v for k,v in a.param_groups[0].items() if k != 'params'}
    gb = {k:v for k,v in b.param_groups[0].items() if k != 'params'}
    assert ga.pop('resource_fatigue') is True and gb.pop('resource_fatigue') is False
    assert ga == gb


def test_outcome_arms_change_only_third_factor_gate():
    import torch
    record = {'experiment': {'resource': dict(lr=.001, adam_base=True,
                 feedback=False, outcome_deadband=1e-5)}}
    on = runner.make_optimizer(torch.nn.Linear(3, 2), record, 'outcome_on')
    off = runner.make_optimizer(torch.nn.Linear(3, 2), record, 'outcome_off')
    a = {k: v for k, v in on.param_groups[0].items() if k != 'params'}
    b = {k: v for k, v in off.param_groups[0].items() if k != 'params'}
    assert a.pop('outcome_modulation') is True
    assert b.pop('outcome_modulation') is False
    assert a == b


@pytest.mark.parametrize('arm,changed', [
    ('factor_anchor', 'latch_strength'),
    ('factor_phase_latch', 'phase_latch_enabled'),
    ('factor_rho', 'outcome_rho_enabled'),
])
def test_factor_controls_change_one_setting_and_keep_state_layout(arm, changed):
    import torch
    record = {'experiment': {'resource': dict(lr=.001, adam_base=True, feedback=False)}}
    models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
    opts = [runner.make_optimizer(models[0], record, 'factor_base'),
            runner.make_optimizer(models[1], record, arm)]
    groups = [{k: v for k, v in o.param_groups[0].items() if k != 'params'} for o in opts]
    assert groups[0].pop(changed) != groups[1].pop(changed)
    assert groups[0] == groups[1]
    for model, opt in zip(models, opts):
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
    assert [set(opts[0].state[p]) for p in models[0].parameters()] == [
        set(opts[1].state[p]) for p in models[1].parameters()]
