"""CPU-only operator identities; these fixtures make no capability claim."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'fly_repair_instrument', ROOT / 'scripts/ib/diagnose_fly_pipeline_repair.py')
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class ResetInterface:
    """Small interface fixture isolates reset from transmitted event outputs."""
    read_indices = torch.tensor([0, 1])
    read_centering = False
    output_read = torch.nn.Identity()
    read_norm = torch.nn.Identity()
    decoder = torch.nn.Identity()

    def prepare_coba_tick(self, h, *args, **kwargs):
        return {'alpha': torch.ones_like(h), 'beta_int': torch.zeros_like(h),
                'base_current': torch.zeros_like(h), 'eff_threshold': .1}

    def finish_coba_tick(self, context, drive, *, return_biophysics=False):
        v = context['v']
        spike = repair.reservoir.SpikeFn.apply(v - .1)
        outputs = (v * (1 - spike), spike, .2 * spike)
        bio = {'v_pre': v, 'eff_threshold': .1}
        return (outputs, bio) if return_biophysics else outputs


def state(h):
    zero = torch.zeros_like(h)
    return repair.FlyPhysicalState(h, (zero,) * 4, zero, zero, zero,
                                   zero, zero, zero, zero)


@pytest.mark.parametrize('entry', ['begin', 'finish'])
def test_reset_only_preserves_forward_and_fixed_event_membrane_partial(entry):
    model = ResetInterface()
    voltage = torch.tensor([[-.2, .2]], requires_grad=True)
    if entry == 'begin':
        original, _ = repair.pipeline.begin_fly_prediction(model, state(voltage))
    else:
        original = model.finish_coba_tick({'v': voltage}, None)[0]
    original_gradient, = torch.autograd.grad(original.sum(), voltage)
    expected_spike = (voltage >= .1).float()
    phi = 1 / (1 + (torch.pi * (voltage - .1)).square())
    torch.testing.assert_close(original_gradient, 1 - expected_spike - voltage * phi)
    with repair.instrument(model, reset_only=True) as events:
        if entry == 'begin':
            modified, _ = repair.pipeline.begin_fly_prediction(model, state(voltage))
        else:
            modified = model.finish_coba_tick({'v': voltage}, None)[0]
        gradient, = torch.autograd.grad(modified.sum(), voltage)
    torch.testing.assert_close(modified, original, rtol=0, atol=0)
    torch.testing.assert_close(gradient, 1 - expected_spike, rtol=0, atol=0)
    assert events['calls'] == 1


def test_reset_exclusion_retains_transmitted_event_gradient_and_restores_globals():
    model = ResetInterface()
    original_spike = repair.reservoir.SpikeFn
    original_begin = repair.pipeline.begin_fly_prediction
    voltage = torch.tensor([[-.2, .2]], requires_grad=True)
    with repair.instrument(model, reset_only=True):
        result = model.finish_coba_tick({'v': voltage}, None)
        pulse_gradient, = torch.autograd.grad(result[1].sum(), voltage, retain_graph=True)
        other_event_gradient, = torch.autograd.grad(result[2].sum(), voltage)
    phi = 1 / (1 + (torch.pi * (voltage - .1)).square())
    torch.testing.assert_close(pulse_gradient, phi)
    torch.testing.assert_close(other_event_gradient, .2 * phi)
    assert repair.reservoir.SpikeFn is original_spike
    assert repair.pipeline.begin_fly_prediction is original_begin
    with pytest.raises(RuntimeError, match='intentional'):
        with repair.instrument(model, reset_only=True):
            raise RuntimeError('intentional')
    assert repair.reservoir.SpikeFn is original_spike
    assert repair.pipeline.begin_fly_prediction is original_begin


def test_updated_parameter_replay_gate_rejects_changed_events_at_equal_scores():
    physical = state(torch.zeros(1, 2))
    reference = {'scores': [1.], 'state': physical,
                 'masks': [np.zeros((1, 2), dtype=bool)] * 2}
    actual = {**reference, 'masks': [np.ones((1, 2), dtype=bool)] * 2}
    replay = repair.selection_replay(reference, actual)
    assert replay['max_score_error'] == 0
    assert not replay['hard_masks_stable']
    assert replay['branches']['total_changes'] == 4


def test_state_floor_accumulates_all_repeats_and_budget_is_fixed():
    previous = {'h': {'max_absolute_difference': .1}}
    observed = {'h': {'max_absolute_difference': .01},
                'ring': {'max_absolute_difference': .03}}
    result = repair.merge_state_floor(previous, observed)
    assert result['h']['max_absolute_difference'] == .1
    assert result['ring']['max_absolute_difference'] == .03
    assert sum(repair.PLAN_COUNTS.values()) == 44
    assert repair.PLAN_COUNTS['selection_control_repeats'] == 12


def test_same_spikes_and_scores_with_drifting_state_are_not_stable():
    reference = {'scores':[1.], 'state':state(torch.zeros(1,2)),
                 'masks':[np.zeros((1,2),dtype=bool)]*2}
    actual = {**reference, 'state':state(torch.ones(1,2)*.01)}
    result = repair.selection_replay(reference,actual)
    assert result['hard_masks_stable']
    assert not result['continuous_state_stable']


def test_unresolved_matched_radius_and_relative_mismatch_cannot_support_direction():
    tiny = repair.matched_norm_gate(1e-8,1e-8,1e-8)
    assert tiny['norms_match']
    assert not tiny['radius_resolved']
    good = repair.matched_norm_gate(.01,.01,.01)
    assert good['norms_match'] and good['radius_resolved']
    mismatch = repair.matched_norm_gate(.01,.010001,.01)
    assert mismatch['radius_resolved']
    assert not mismatch['norms_match']


def test_report_preserves_numpy_numbers_booleans_and_error_report(tmp_path):
    value = {'floor':np.float32(1e-5),'nested':({'passed':np.bool_(True)},
              np.float64(1.25)),'complete':False,'error':'example'}
    path = tmp_path/'partial.json'
    repair.write_json(path,value)
    restored = json.loads(path.read_text(encoding='utf-8'))
    assert restored['floor'] == float(value['floor'])
    assert restored['nested'] == [{'passed':True},1.25]
    assert restored['complete'] is False
    assert restored['error'] == 'example'
