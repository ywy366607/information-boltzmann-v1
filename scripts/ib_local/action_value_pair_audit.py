"""Private first-action intervention; normal sampling and learning resume after it."""
from contextlib import contextmanager
import copy
import hashlib

import torch

from scripts.ib_local.finite_choice_contract import trace_choices


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def first_action(engine, option, publish, *, keep_actor_learning=False):
    """Condition one legal actor leaf while consuming the original random draw.

    Question/proposal scores remain endogenous. The forced categorical has
    behavior probability one and contributes zero derivative, while its original
    logits remain available for the pre-outcome ranking/replay receipt. Passing
    keep_actor_learning=True preserves the original nominal-policy derivative;
    conditioning on the naturally sampled leaf then reproduces normal learning.
    """
    if option not in (0, 1, 2):
        raise ValueError('first-action audit requires a legal silence/generate/reuse leaf')
    original_utter, original_term = engine._utter, engine._choice_policy_term
    prior = {k:vars(engine).get(k) for k in ('_utter','_choice_policy_term')}
    present = {k:k in vars(engine) for k in prior}
    used = []

    def utter():
        if used:
            return original_utter()
        with trace_choices({'actor': option}) as trace:
            words, record = original_utter()
        actor = next(v for v in trace.events if v['stage'] == 'actor')
        if actor['selected'] != option:
            raise AssertionError('intervention did not reach the actual actor')
        behavior = dict(option=option, behavior_probability=1.,
            nominal_probability=float(actor['probability'][option]),
            categorical_score_gradient=1. if keep_actor_learning else 0.,
            learning='original nominal policy' if keep_actor_learning else 'forced-category stopped')
        if not keep_actor_learning:
            record['action_pair_intervention'] = behavior
        e = engine.epistemic_value_last
        receipt = dict(option=option, utterance_hex=words.hex(),
            nominal_probability=actor['probability'].cpu().tolist(),
            source_sha256=tensor_digest(engine._internal_source),
            endpoint_sha256=[tensor_digest(v) for v in engine._internal_action],
            recurrent_sha256=tensor_digest(engine.recurrent),
            unknown=engine._program_question_state.binding.identity,
            questions=copy.deepcopy(engine._generative_query['alternatives']),
            proposals=[p['words'].hex() for p in engine._joint_proposals],
            selected_proposal=engine._joint_selection['selected'],
            local=e['local_expected'], future=e['contextual_future'], total=e['contextual_total'],
            gamma=e['gamma'], critic=record['values'], habit=record['habit_logits'],
            offset=record['offset'], units='wood=10*U; critic=100*U',
            sampled_option=actor['sampled'], reused_slot=record['reused_slot'],
            wood_argmax=max(range(len(e['contextual_total'])),
                            key=lambda k:e['contextual_total'][k]
                            if actor['probability'][k]>0 else -float('inf')),
            habit_argmax=max(range(len(record['habit_logits'])),
                             key=lambda k:record['habit_logits'][k]),
            behavior=behavior)
        used.append(receipt)
        publish(receipt)
        return words, record

    def term(targets, own, record, **kwargs):
        value, replay = original_term(targets, own, record, **kwargs)
        if record.get('action_pair_intervention'):
            value = value*0
            replay['action_pair_intervention'] = record['action_pair_intervention']
        return value, replay

    engine._utter, engine._choice_policy_term = utter, term
    try:
        yield used
    finally:
        for name in prior:
            if present[name]:
                setattr(engine, name, prior[name])
            else:
                delattr(engine, name)
