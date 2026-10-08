"""Causality, real continuation and measurement interfaces; no capability study."""
import copy
import json
import os
import sys

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.local_credit import LocalPlasticTrainer
from information_boltzmann.runtime.lifelong_evaluation import (
    LiveOnlineEvaluation, recovery_summary, savings_summary)
from information_boltzmann.runtime.training import belief_tensors


def make_runner(cadence=3):
    torch.manual_seed(211)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                hidden=8, bath_type='conductance', activity_adaptation=True,
                                short_term_plasticity=True).double()
    learner = LocalPlasticTrainer(model, event_duration=.005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return LiveOnlineEvaluation(learner, optimizer, tokens_per_update=cadence, window_tokens=4)


def test_scores_precede_updates_and_target_only_enters_likelihood():
    runner = make_runner(cadence=1)
    model = runner.learner.model
    belief = runner.learner.belief
    with torch.no_grad():
        conditioned, _ = model.assimilate(belief, torch.tensor([1]), diagnostics=False)
        evolved, _ = model.advance(conditioned, .005, diagnostics=False)
        logits, _ = model.read(evolved)
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))
    before = model.decoder.weight.clone()
    score = runner.step(torch.tensor([1]), torch.tensor([2]))
    torch.testing.assert_close(score['token_nll'], expected)
    assert not torch.equal(before, model.decoder.weight)
    assert runner.events == 1 and runner.optimizer_updates == 1
    alternative = make_runner(cadence=1)
    alternative.step(torch.tensor([1]), torch.tensor([3]))
    # Changing the future target changes learning, not this event's physics.
    for a, b in zip(belief_tensors(runner.learner.belief), belief_tensors(alternative.learner.belief)):
        torch.testing.assert_close(a, b)


def test_aba_runs_the_exact_same_continuous_learner_as_unsegmented_stream():
    runner, reference = make_runner(), make_runner()
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([5, 6, 7, 8, 9])
    scores = []
    for left, right in zip(a[:-1], a[1:]):
        scores.append(float(runner.step(left.reshape(1), right.reshape(1))['token_nll']))
        reference.step(left.reshape(1), right.reshape(1))
    trace_object = runner.learner.eligibility.trace
    report = runner.revisit_after_change(a, scores, b, block_tokens=1, hold_blocks=2)
    for token in b:
        reference.observe(token)
    reference.observe(a[0], novel=False)
    for token in a[1:]:
        reference.observe(token, novel=False)
    assert report['savings']['actual_intervening_events'] == 6
    assert runner.events == 12 and runner.optimizer_updates == 4
    assert runner.novel_events == 8  # A1 three + B five; bridge/A2 tracked separately
    assert runner.learner.eligibility.trace is trace_object
    assert float(runner.learner.belief.medium.elapsed[0]) == pytest.approx(12 * .005)
    for p, q in zip(runner.learner.model.parameters(), reference.learner.model.parameters()):
        torch.testing.assert_close(p, q)
    for p, q in zip(belief_tensors(runner.learner.belief), belief_tensors(reference.learner.belief)):
        torch.testing.assert_close(p, q)
    torch.testing.assert_close(runner.learner.eligibility.trace, reference.learner.eligibility.trace)


def test_a_revisit_requires_real_prior_scores_not_a_declared_experience_label():
    runner = make_runner()
    a = torch.tensor([1, 2, 3])
    scores = [float(runner.step(left.reshape(1), right.reshape(1))['token_nll'])
              for left, right in zip(a[:-1], a[1:])]
    with pytest.raises(ValueError, match='ledger'):
        runner.revisit_after_change(a, [s + 1 for s in scores], torch.tensor([4]),
                                     block_tokens=1, hold_blocks=1)
    assert runner.events == 2  # rejection precedes any extra experience


def test_checkpoint_resumes_pending_gradients_optimizer_scores_and_physics():
    runner = make_runner(cadence=3)
    runner.step(torch.tensor([1]), torch.tensor([2]))
    assert runner.pending == 1
    model_state = copy.deepcopy(runner.learner.model.state_dict())
    learner_state = runner.learner.state_dict()
    optimizer_state = copy.deepcopy(runner.optimizer.state_dict())
    metric_state = runner.state_dict()
    resumed = make_runner(cadence=3)
    resumed.learner.model.load_state_dict(model_state)
    resumed.learner.load_state_dict(learner_state)
    resumed.optimizer.load_state_dict(optimizer_state)
    resumed.load_state_dict(metric_state)
    for token in (3, 4, 5):
        left, right = runner.observe(torch.tensor([token])), resumed.observe(torch.tensor([token]))
        torch.testing.assert_close(left['token_nll'], right['token_nll'])
    assert runner.summary() == resumed.summary()
    for p, q in zip(runner.learner.model.parameters(), resumed.learner.model.parameters()):
        torch.testing.assert_close(p, q)


def test_stream_boundaries_cannot_skip_a_prediction_and_no_phase_flush():
    runner = make_runner(cadence=4)
    runner.step(torch.tensor([1]), torch.tensor([2]))
    with pytest.raises(ValueError, match='bridge'):
        runner.step(torch.tensor([3]), torch.tensor([4]))
    assert runner.pending == 1 and runner.optimizer_updates == 0
    runner.observe(torch.tensor([3]))
    assert runner.pending == 2 and runner.optimizer_updates == 0


def test_recovery_requires_sustained_blocks_and_reports_no_recovery_honestly():
    result = recovery_summary([10., 2., 9., 8., 3., 3.], block_tokens=1, hold_blocks=2)
    assert result['half_recovery_tokens'] == 6
    bad = recovery_summary([2., 3., 4., 5.], block_tokens=1, hold_blocks=1)
    assert bad['status'] == 'no_observed_improvement'
    assert bad['half_recovery_tokens'] is None
    short = recovery_summary([5.], block_tokens=8, hold_blocks=2)
    assert short['status'] == 'insufficient_blocks'


def test_savings_uses_one_shared_absolute_threshold_for_both_encounters():
    report = savings_summary([10., 8., 6., 4., 2.], [8., 4., 3., 2., 2.],
                             intervening_events=73, block_tokens=1, hold_blocks=1)
    assert report['actual_intervening_events'] == 73
    assert report['shared_threshold_nll'] == 6
    assert report['initial_threshold_tokens'] == 3
    assert report['revisit_threshold_tokens'] == 2
    assert report['relearning_acceleration'] == 1.5


def test_real_owt_entrypoint_runs_live_eval_and_resumes_pending_credit(tmp_path, monkeypatch):
    """Real corpus, small numerical model; only test CLI/checkpoint plumbing."""
    from pathlib import Path
    import numpy as np
    import scripts.ib.train_online_plastic as entry
    from scripts.ib.train_plastic_conductance import pack_belief
    source = Path('data/ib_owt_gpt2')
    if not (source/'train.npy').exists():
        pytest.skip('Optional local real OWT fixture')
    torch.set_num_threads(1)
    constructor = dict(vocab_size=50257, shape=(2, 2, 2), channels=8, hidden=8,
                       bath_type='conductance', activity_adaptation=True,
                       short_term_plasticity=True, medium_execution='native', port_execution='native')
    model = PlasticMediumPorts3D(**constructor)
    learner = LocalPlasticTrainer(model, event_duration=.005)
    source_ids = np.load(source/'train.npy', mmap_mode='r')
    # Establish one actual observed event before exporting the continuation.
    learner.backward_event(torch.tensor([int(source_ids[0])]), torch.tensor([int(source_ids[1])]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    optimizer.step()
    online_state = learner.state_dict()
    del online_state['belief']
    birth = tmp_path/'initial.pt'
    torch.save({'config': {'constructor': constructor, 'event_duration': .005, 'substeps': 1},
                'model': model.state_dict(), 'belief': pack_belief(learner.belief),
                'data_offset': 1}, birth)
    output = tmp_path/'live'
    common = ['entry', '--output', str(output), '--credit', 'local-receptors', '--device', 'cpu',
              '--execution', 'eager', '--tokens', '2', '--event-duration', '.005',
              '--validate-every', '1', '--change-tokens', '2', '--measurement-block', '1',
              '--recovery-hold-blocks', '1', '--warmup', '0', '--decay', '0']
    monkeypatch.setattr(sys, 'argv', common + ['--steps', '1', '--initialize-from', str(birth)])
    entry.main()
    saved = torch.load(output/'last.pt', weights_only=False)
    assert saved['adaptation_offset'] == 2
    assert saved['online']['events'] == 7  # A1 2 + B2 + bridge1 + A2 2
    assert saved['live_evaluation']['pending'] == 1
    assert saved['live_evaluation']['optimizer_updates'] == 3
    assert saved['live_evaluation']['novel_events'] == 4
    assert float(saved['belief']['elapsed'][0]) > .005  # mature time retained
    monkeypatch.setattr(sys, 'argv', common + ['--steps', '2', '--resume', str(output/'last.pt')])
    entry.main()
    resumed = torch.load(output/'last.pt', weights_only=False)
    assert resumed['adaptation_offset'] == 4
    assert resumed['online']['events'] == 14
    assert resumed['live_evaluation']['pending'] == 0
    assert resumed['live_evaluation']['optimizer_updates'] == 7
    reports = [json.loads(line) for line in (output/'lifelong_evaluation.jsonl').read_text().splitlines()]
    assert [r['adaptation_stream_start'] for r in reports] == [0, 2]
    assert all(r['savings']['actual_intervening_events'] == 3 for r in reports)
    assert all(r['learning_active'] and not r['reset'] for r in reports)
    # Numerical CLI fixtures leave no model assets behind.
    for path in (birth, output/'best_prequential.pt', output/'last.pt'):
        if path.exists():
            path.unlink()


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='Optional GPU allocation')
def test_live_optimizer_and_pending_resume_preserve_cuda_capture_storage():
    from information_boltzmann.runtime.local_credit import CapturedLocalEvent
    original = make_runner(cadence=3)
    original.learner.model.float().cuda()
    # Birth is created once on the correct device, before this numeric history.
    original.learner = LocalPlasticTrainer(original.learner.model, event_duration=.005)
    original.optimizer = torch.optim.AdamW(original.learner.model.parameters(), lr=1e-4)
    original.captured = CapturedLocalEvent(original.learner, loss_scale=1 / 3)
    original.step(torch.tensor([1], device='cuda'), torch.tensor([2], device='cuda'))
    states = (copy.deepcopy(original.learner.model.state_dict()), original.learner.state_dict(),
              copy.deepcopy(original.optimizer.state_dict()), original.state_dict())
    other_model = copy.deepcopy(original.learner.model)
    other_model.load_state_dict(states[0])
    learner = LocalPlasticTrainer(other_model, event_duration=.005)
    learner.load_state_dict(states[1])
    optimizer = torch.optim.AdamW(other_model.parameters(), lr=1e-4)
    optimizer.load_state_dict(states[2])
    captured = CapturedLocalEvent(learner, loss_scale=1 / 3)
    resumed = LiveOnlineEvaluation(learner, optimizer, tokens_per_update=3, window_tokens=4,
                                   captured=captured)
    resumed.load_state_dict(states[3])
    for target in (3, 4, 5, 6):
        token = torch.tensor([target], device='cuda')
        a, b = original.observe(token), resumed.observe(token)
        torch.testing.assert_close(a['token_nll'], b['token_nll'])
    assert original.summary() == resumed.summary()
    for a, b in zip(original.learner.model.parameters(), resumed.learner.model.parameters()):
        torch.testing.assert_close(a, b)
