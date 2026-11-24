"""Paired root-only readout of a q-calibration lift into persistent values.

At an actual final decision, Vp is the persistent scalar predictor and F0/F1
are action-plan expectations before/after option calibration, on the SAME
successor states and original branch masses. The experimental alternative is
Vp+(F1-F0). Shadow execution returns the original Vp, including to teachers
and subsequent learning. Its final alternative probability is evaluated on
the original concrete menu. No additional native teacher is executed.
"""
from contextlib import contextmanager
import types

import torch


@contextmanager
def observe(engine):
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local import persistent_rhythm as rhythm
    from scripts.ib_local.competing_action_teachers import action_key
    from scripts.option_basis_calibration import calibrated_predictions

    original_value, original_read = planner.value, rhythm.read_plans
    original_relation = engine._relation_value
    own = vars(engine).get('_relation_value')
    had_own = '_relation_value' in vars(engine)
    records, active = [], []

    @torch.no_grad()
    def read(e, states, sources, *, carried=None):
        baseline, cases, count = original_read(e, states, sources, carried=carried)
        if e is not engine or not active:
            return baseline, cases, count
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if sources[0].is_cuda else None
        revision = (e.native_value_bank['revision'], e.latent_action_bank['revision'])
        old, _, _ = planner.plan(e, states, sources, carried=carried)
        with calibrated_predictions(e):
            new, _, _ = planner.plan(e, states, sources, carried=carried)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
        assert revision == (e.native_value_bank['revision'], e.latent_action_bank['revision'])
        active[-1]['reads'].append(dict(persistent=baseline.detach(),
                                        old=old.detach(), new=new.detach()))
        return baseline, cases, count

    @torch.no_grad()
    def value(e, state, source, action, mother, distribution, momentum):
        roots = getattr(e, '_competing_action_roots', None)
        key = action_key(source, action) if e is engine and roots else None
        if not roots or key not in roots or records:
            return original_value(e, state, source, action, mother, distribution, momentum)
        row = dict(key=key, reads=[])
        active.append(row)
        try:
            answer = original_value(e, state, source, action, mother, distribution, momentum)
            assert len(row['reads']) == 1
            got = row['reads'][0]
            weights = answer.new_tensor(e.successor_distribution_last['weights'])
            torch.testing.assert_close(answer[0], (weights*got['persistent']).sum(), atol=2e-5, rtol=2e-5)
            delta = (weights*(got['new']-got['old'])).sum()
            root_calls.append(dict(key=key, original=float(answer[0]), delta=float(delta),
                weights=weights.tolist(), persistent=got['persistent'].tolist(),
                action_plan_old=got['old'].tolist(), action_plan_calibrated=got['new'].tolist()))
            return answer
        finally:
            active.pop()

    root_calls = []

    @torch.no_grad()
    def relation(e, *args, **kwargs):
        if records:
            return original_relation(*args, **kwargs)
        root_calls.clear()
        planner.value, rhythm.read_plans = value, read
        try:
            result = original_relation(*args, **kwargs)
        finally:
            planner.value, rhythm.read_plans = original_value, original_read
        critic, _ = result
        roots = e._competing_action_roots
        future = critic.new_tensor(e.epistemic_value_last['contextual_future'])
        delta = torch.zeros_like(critic)
        for key, options in roots.items():
            calls = [r for r in root_calls if r['key'] == key]
            assert calls, 'each legal root must have actual successor reads'
            original = critic.new_tensor([r['original'] for r in calls]).mean()
            correction = critic.new_tensor([r['delta'] for r in calls]).mean()
            for option in options:
                torch.testing.assert_close(future[option], original, atol=2e-5, rtol=2e-5)
                delta[option] = correction
        alternative = critic+10*e.gamma*delta
        records.append(dict(critic_original=critic.tolist(), critic_alternative=alternative.tolist(),
            future_delta=delta.tolist(), calls=list(root_calls),
            base='persistent scalar value', correction='paired action-plan difference at fixed successors',
            extra_teacher_calls=0, extra_world_calls=0, return_mode='original prediction',
            direct_q_patch_root_effect=0))
        return result

    engine._relation_value = types.MethodType(relation, engine)
    try:
        yield records
    finally:
        planner.value, rhythm.read_plans = original_value, original_read
        if had_own:
            engine._relation_value = own
        else:
            del engine._relation_value


def choice_receipt(engine, receipt, shadow):
    """Reconstruct both final probabilities using the actual habit and units."""
    critic = torch.tensor(receipt['critic'], dtype=torch.double)
    torch.testing.assert_close(critic, critic.new_tensor(shadow['critic_original']), atol=2e-5, rtol=2e-5)
    habit = critic.new_tensor(receipt['habit'])
    legal = habit > -1e3
    precision = float(engine.precision())
    def probability(values):
        offset = precision*(values-values[legal].mean())
        return (habit+offset.masked_fill(~legal, 0)).softmax(0)
    original = probability(critic)
    torch.testing.assert_close(original, critic.new_tensor(receipt['nominal_probability']), atol=2e-7, rtol=2e-6)
    alternative = probability(critic.new_tensor(shadow['critic_alternative']))
    return dict(**shadow, original_probability=original.tolist(),
        alternative_probability=alternative.tolist(), habit_probability=habit.softmax(0).tolist(),
        precision=precision, legal=torch.nonzero(legal).flatten().tolist(),
        source_sha256=receipt['source_sha256'], endpoint_sha256=receipt['endpoint_sha256'],
        unknown=receipt['unknown'], proposals=receipt['proposals'],
        selected_proposal=receipt['selected_proposal'], target='first-choice probability effect; original subsequent learning')
