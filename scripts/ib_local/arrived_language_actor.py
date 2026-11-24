"""Two-external-round language credit for the existing final categorical head."""
import copy
import numpy as np
import torch

from scripts.audit_arrived_language_credit import train_probe
from scripts.ib_local.wood_policy_measurement import score


class ArrivedLanguageActor:
    def _policy_grad(self, targets, own, record):
        grads, replay = super()._policy_grad(targets, own, record)
        head = self.model.kanli.choice
        indices = {id(p): i for i,p in enumerate(self.params)}
        iw, ib = indices[id(head.weight)], indices[id(head.bias)]
        probability = head.bias.new_tensor(record['choice_logits']).softmax(0)
        expected = -probability
        expected[record['choice']] += 1
        torch.testing.assert_close(grads[ib], expected, atol=2e-5, rtol=2e-5)
        s = self.arrived_actor_state
        s['current'] = dict(gradient=(grads[iw].detach().clone(), grads[ib].detach().clone()),
            action_round=s['round']+1, choice=record['choice'], baseline=s['baseline'])
        if s['mode'] == 'apply':
            # Replace local TD ONLY for the two new final-head eligibilities.
            # The inherited pending actor and all other participants are untouched.
            grads = list(grads)
            grads[iw] = grads[ib] = None
        return tuple(grads), replay

    def _advance(self, loss, *args, **kwargs):
        s = self.arrived_actor_state
        due = s['due']
        if due is not None:
            head = self.model.kanli.choice
            for param, trace in zip((head.weight, head.bias), due['gradient']):
                loss = loss - due['advantage']*((param-param.detach())*trace.to(param)).sum()
        result = super()._advance(loss, *args, **kwargs)
        if due is not None:
            s['due'] = None
            s['applied'] += 1
        return result

    def step(self, *args, **kwargs):
        row = super().step(*args, **kwargs)
        s = self.arrived_actor_state
        previous, current = s['pending'], s.pop('current')
        s['round'] += 1
        s['pending'] = current
        if previous is not None:
            # Dedicated probe RNG: unaffected by navigation and restored with the receipt.
            rng = np.random.default_rng(0)
            rng.bit_generator.state = s['rng']
            raw, start = train_probe(self.world, 2048, lambda n: int(rng.integers(n)))
            y = -score(self, [raw])['mean_nll']
            advantage = y-previous['baseline']
            if s['mode']=='apply':
                s['due'] = dict(gradient=previous['gradient'], advantage=advantage)
            s['baseline'] = y  # only decisions made *after* this endpoint may use it
            s['rng'] = rng.bit_generator.state
            row['arrived_language_credit'] = dict(action_round=previous['action_round'],
                endpoint_round=s['round'], choice=previous['choice'], start_in_train=start,
                endpoint_log_score=y, baseline=previous['baseline'], advantage=advantage,
                mode=s['mode'], applied_to_date=s['applied'])
        return row


def attach(engine, mode, seed):
    from scripts.ib_local.natural_text_world import NaturalTextWorld
    if mode not in ('observe','apply') or not isinstance(engine.world, NaturalTextWorld) or engine.mode=='silence':
        raise ValueError('requires a naturally sampled actor and ordinary training text')
    if not hasattr(engine, 'arrived_actor_state'):
        engine.arrived_actor_state = dict(mode=mode, seed=seed, round=0, applied=0,
            baseline=0., pending=None, due=None,
            rng=copy.deepcopy(np.random.default_rng(seed).bit_generator.state))
    elif (engine.arrived_actor_state['mode'],engine.arrived_actor_state['seed']) != (mode,seed):
        raise ValueError('cannot switch arrived credit on resume')
    if not isinstance(engine, ArrivedLanguageActor):
        engine.__class__ = type('ArrivedActorEngine',(ArrivedLanguageActor,type(engine)),{})
