from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.ruler_readout_adaptation import (
    ReadoutAdaptationLearner, scheduled_gap, validate_splits,
)


def fake_learner():
    native = torch.nn.Parameter(torch.tensor([2.]))
    master = torch.nn.Parameter(torch.tensor([2.]))
    extra = torch.nn.Parameter(torch.tensor([1.]))
    null = torch.nn.Parameter(torch.tensor([0.]))
    optimizer = torch.optim.AdamW([
        dict(params=[master], lr=.01), dict(params=[extra, null], lr=.01)], weight_decay=0)
    master.grad = torch.ones_like(master)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return SimpleNamespace(native=[('native', native)], masters=[master],
        extras=[extra], null_parameters=[null], optimizer=optimizer)


def test_adaptation_preserves_native_weights_and_adam_history():
    learner = fake_learner()
    master = learner.masters[0]
    before = {k: v.clone() if torch.is_tensor(v) else v
              for k, v in learner.optimizer.state[master].items()}
    value = master.detach().clone()
    for parameter in [master, learner.native[0][1], *learner.extras, *learner.null_parameters]:
        parameter.grad = torch.ones_like(parameter)
    ReadoutAdaptationLearner.apply_gradients(learner)
    assert torch.equal(master, value)
    assert float(learner.native[0][1].detach()) == 2.
    for key, item in before.items():
        assert torch.equal(item, learner.optimizer.state[master][key])
    assert float(learner.extras[0].detach()) < 1.
    assert float(learner.null_parameters[0].detach()) < 0.
    assert learner.last_adaptation_gradient_norm > 0


def test_nonfinite_readout_update_is_rejected():
    learner = fake_learner()
    learner.extras[0].grad = torch.tensor([float('nan')])
    with pytest.raises(FloatingPointError):
        ReadoutAdaptationLearner.apply_gradients(learner)


def test_schedule_and_target_separation():
    assert [scheduled_gap(i) for i in range(4)] == [0, 512, 0, 512]
    with pytest.raises(ValueError):
        scheduled_gap(-1)
    validate_splits([dict(outputs=['123'])], [dict(outputs=['456'])])
    with pytest.raises(ValueError):
        validate_splits([dict(outputs=['123'])], [dict(outputs=['123'])])


def test_user_stop_checkpoints_current_complete_example_before_reporting(tmp_path):
    from scripts.ib_local.run_ruler_readout_adaptation import checkpoint_on_stop
    stop = tmp_path/'STOP'
    report, calls = {}, []
    checkpoint = lambda count: calls.append(('checkpoint', count))
    save = lambda: calls.append(('save', report.copy()))
    assert not checkpoint_on_stop(stop, 150, checkpoint, report, save)
    assert not calls
    stop.write_text('inspect delivery before training', encoding='utf-8')
    assert checkpoint_on_stop(stop, 150, checkpoint, report, save)
    assert calls[0] == ('checkpoint', 150)
    assert calls[1][1]['status'] == 'stopped'
    assert report['completed'] == 150 and report['checkpointed_stop']


def test_stop_save_failure_never_claims_latest_state_was_saved(tmp_path):
    from scripts.ib_local.run_ruler_readout_adaptation import checkpoint_on_stop
    stop = tmp_path/'STOP'
    stop.write_text('stop', encoding='utf-8')
    report = {'status': 'running'}
    def failed(count):
        raise OSError('disk full')
    with pytest.raises(OSError, match='disk full'):
        checkpoint_on_stop(stop, 150, failed, report, lambda: None)
    assert report == {'status': 'running'}


def test_training_is_blocked_without_completed_version_bound_repair(tmp_path):
    import hashlib
    import json
    from scripts.ib_local.run_ruler_readout_adaptation import require_repair_report
    source, code = tmp_path/'source.pt', tmp_path/'program.py'
    source.write_bytes(b'existing source')
    code.write_bytes(b'current program')
    with pytest.raises(RuntimeError, match='remains stopped'):
        require_repair_report(None, source, [code])
    report = tmp_path/'repair.json'
    report.write_text(json.dumps(dict(training_ready=False)), encoding='utf-8')
    with pytest.raises(RuntimeError):
        require_repair_report(report, source, [code])
    checks = ('storage_replay','query_conditioned_read','tail_native_derivative',
              'causal_and_cached_read','source_age_controls','frozen_delivery',
              'disk_next_event_restore','resource_budget')
    report.write_text(json.dumps(dict(training_ready=True,checks={n:'passed' for n in checks},
        source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        execution_sources={str(code.resolve()):hashlib.sha256(code.read_bytes()).hexdigest()})),encoding='utf-8')
    assert require_repair_report(report,source,[code])['training_ready']
    code.write_bytes(b'a different program')
    with pytest.raises(ValueError, match='execution version'):
        require_repair_report(report,source,[code])
