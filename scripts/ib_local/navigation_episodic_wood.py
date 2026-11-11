"""Candidate-conditioned reactivation in the existing complete reply organism.

The bank is the existing finite experience memory. Retained observations have
no historical graph; their present re-encoding and read weights are live.
Read a correction to the existing predictor, never a historical absolute gain
as today's gain. Evidence is scored before the observed turn is stored.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.ib_local.run_causal_water_reply import WaterReplyEngine
from scripts.ib_local.navigation_settlement import complete_action_labels
from scripts.ib_local.navigation_value_scale import critic_output


def candidate_read(query_source, query_action, options, bank_source, bank_action,
                   bank_options, values, bank_slots, *, exclude=None, topk=8,
                   temperature=.2, context_penalty=None, log_reliability=None):
    """Convex, same-option read with explicit replay-self exclusion.

    All vectors are current encoded observations; values may carry gradients.
    No world position, response, target or future label is in a query.
    """
    batch = len(options)
    if not len(bank_slots):
        return values.new_zeros(batch, values.shape[-1]), {
            'slots': bank_slots.new_empty(batch, 0),
            'weights': values.new_zeros(batch, 0)}
    score = (F.normalize(query_source, dim=-1) @ F.normalize(bank_source, dim=-1).T
             + F.normalize(query_action-query_source, dim=-1)
             @ F.normalize(bank_action-bank_source, dim=-1).T)/2
    if context_penalty is not None:
        score = score-context_penalty
    score = score/temperature
    if log_reliability is not None:
        score = score+log_reliability[None]
    valid = options[:, None] == bank_options[None]
    if exclude is not None:
        valid = valid & (exclude[:, None] != bank_slots[None])
    k = min(topk, len(bank_slots))
    logits, ids = score.masked_fill(~valid, -1e4).topk(k, dim=-1)
    selected_valid = valid.gather(1, ids)
    weights = logits.softmax(-1)*selected_valid
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    recalled = (weights[..., None]*values[ids]).sum(1)
    return recalled, dict(slots=bank_slots[ids], weights=weights)


def within_option_shuffle(values, options):
    """Destroy key/value association, retain option-specific value marginals.

    Fixed cyclic permutation uses no actor/world/replay RNG. A singleton cannot
    be shuffled and is reported as such by the caller's counts.
    """
    indices = torch.arange(len(values), device=values.device)
    for option in range(3):
        rows = torch.nonzero(options == option).flatten()
        if len(rows) > 1:
            indices[rows] = rows.roll(max(1, len(rows)//2))
    return values[indices]


class EpisodicWoodEngine(WaterReplyEngine):
    """Water -> candidate forecast -> existing fire -> same world -> settlement.

    Existing GDN, exact-byte table, language/SIGReg losses, actor, critic and
    settlement continue. The new read has no neural parameters or extra slots.
    ``blocked`` trains the same forecaster but shields its contribution to the
    current decision, retaining all other memory paths.
    """
    def _freeze_wood_bank(self):
        m = self.model.memory
        valid = complete_action_labels(m) & self.reward_label_valid(wood=True)
        slots = torch.nonzero(valid).flatten()
        # Fixed-size data snapshot predates the action and all its consequences.
        self._episodic_bank = {name: getattr(m, name)[slots].detach().clone()
            for name in ('source', 'action', 'wood_target', 'wood_baseline', 'wood_option', 'feeling')}
        self._episodic_bank['slots'] = slots
        self._episodic_bank['age'] = (m.clock-m.last_write[slots]).detach().clone()

    def _wood_replay_batch(self):
        m = self.model.memory
        valid = torch.nonzero(m.occupied & (m.wood_target_count > 0)
                              & (m.wood_option >= 0)).flatten()
        if not len(valid):
            self._episodic_replay_slots = valid
            return None
        pick = valid[torch.as_tensor(self.replay_rng.integers(0, len(valid), self.replay_batch),
                                     device=valid.device)]
        self._episodic_replay_slots = pick.detach().clone()
        self._episodic_replay_feeling = m.feeling[pick].detach().clone()
        self._reward_replay_valid = self.reward_label_valid(pick, wood=True).clone()
        return tuple(x[pick].detach().clone() for x in
                     (m.source, m.action, m.wood_target, m.wood_option, m.wood_baseline))

    def _training_batch(self, target):
        result = super()._training_batch(target)
        # The new episode was not in the decision-time bank. Replay examples
        # must not read their own target from the retained bank.
        self._episodic_exclude = torch.cat((result[2].new_tensor([-1]),
            self._episodic_replay_slots if self._wood_replay is not None
            else result[2].new_empty(0)))
        # Settlement has already advanced interoception by the final gain
        # update. Always reuse the feeling recorded BEFORE this action.
        self._episodic_query_feeling = self._pending_record['feeling'].reshape(1, -1)
        if self._wood_replay is not None:
            self._episodic_query_feeling = torch.cat((self._episodic_query_feeling,
                                                     self._episodic_replay_feeling))
        return result

    def _residual(self, sources, actions, options):
        parametric = super()._residual(sources, actions, options)
        bank = self._episodic_bank
        old_parametric = super()._residual(bank['source'], bank['action'], bank['wood_option'])
        # Historical labels and baselines have the same objective and units.
        # Read the present predictor's error on old experiences as a local
        # correction, not an old level that may drift with ongoing learning.
        innovations = bank['wood_target']-bank['wood_baseline']-old_parametric
        channel_mask = innovations.new_tensor([1., 1., 0., 1.])
        innovations = innovations*channel_mask  # Bernoulli channel unchanged.
        enc = self.model.encoder
        qs, qa = enc(sources), enc(actions)
        ks, ka = enc(bank['source']), enc(bank['action'])
        kwargs = dict(exclude=None if self._episodic_deciding else self._episodic_exclude,
                      topk=self.model.memory.topk, temperature=self.model.memory.temperature)
        feeling = self.intero().expand(len(options), -1) if self._episodic_deciding else self._episodic_query_feeling
        kwargs['context_penalty'] = (feeling[:, None]-bank['feeling'][None]).norm(dim=-1)/(feeling.shape[-1]**.5)
        # Use the existing innovation/evidence clock. A learning-progress
        # correction is not assumed stationary for hundreds of old turns.
        step_bytes = self.response_bytes+self.holdout_bytes+self.max_utterance
        kwargs['log_reliability'] = bank['age'].to(parametric.dtype)/step_bytes * parametric.new_tensor(self.wood_decay).log()
        correction, read = candidate_read(qs, qa, options, ks, ka, bank['wood_option'],
                                         innovations, bank['slots'], **kwargs)
        if self._episodic_deciding:
            shuffled, _ = candidate_read(qs, qa, options, ks, ka, bank['wood_option'],
                within_option_shuffle(innovations, bank['wood_option']), bank['slots'], **kwargs)
            self._episodic_decision_read = dict(parametric=parametric.detach().clone(),
                shuffled=(parametric+shuffled).detach().clone(),
                correction=correction.detach().clone(),
                slots=read['slots'].detach().clone(), weights=read['weights'].detach().clone())
        return parametric+correction

    @torch.no_grad()
    def _value(self, h, films):
        self._freeze_wood_bank()
        self._episodic_deciding = True
        try:
            full_values, inputs = super()._value(h, films)
        finally:
            self._episodic_deciding = False
        forecast, read = self._forecast, self._episodic_decision_read
        base, confidence = forecast['base'], forecast['confidence']
        plain_used = self._compose_used_prediction(base, read['parametric'], confidence)
        shuffled_used = self._compose_used_prediction(base, read['shuffled'], confidence)
        width = self.model.encoder.out_features
        plain_inputs = inputs.clone()
        plain_inputs[:, width:width+4] = plain_used
        plain_values = critic_output(self, plain_inputs).squeeze(-1)+plain_inputs[:, -2]
        # Gain channel is in 10*nats, existing choice values in 100*nats.
        # Its calibrated contextual correction goes directly to the choice;
        # the post-action settlement remains the source of teaching evidence.
        direct_gain = 10*(forecast['used'][:, 3]-plain_used[:, 3])
        full_values = full_values+direct_gain
        available = len(self._draft_actions) if self._draft_actions else 1
        delta = torch.zeros_like(full_values)
        delta[:available] = self.precision()*((full_values-plain_values)[:available]
                                      -(full_values-plain_values)[:available].mean())
        forecast['episodic_offset'] = delta
        forecast['episodic_plain_used'] = plain_used
        forecast['episodic_shuffled_used'] = shuffled_used
        forecast['episodic_raw_shuffled'] = self._compose_prediction(base, read['shuffled'])
        forecast['episodic_raw_plain'] = self._compose_prediction(base, read['parametric'])
        if self.episodic_wood_mode == 'blocked':
            return plain_values, plain_inputs
        return full_values, inputs

    def step(self, directions, age_ablate=False):
        row = super().step(directions, age_ablate)
        choice, forecast = self._pending_record['choice'], self._forecast
        target = forecast['raw'].new_tensor(row['wood_target'])
        read = self._episodic_decision_read
        logits = target.new_tensor(self._pending_record['choice_logits'])
        delta = forecast['episodic_offset']
        # Actual control/real choice plus both same-decision read forecasts.
        row.update(episodic_wood_mode=self.episodic_wood_mode,
            episodic_real_error=(forecast['raw'][choice]-target).square().tolist(),
            episodic_shuffled_error=(forecast['episodic_raw_shuffled'][choice]-target).square().tolist(),
            episodic_plain_error=(forecast['episodic_raw_plain'][choice]-target).square().tolist(),
            episodic_used_real_error=(forecast['used'][choice]-target).square().tolist(),
            episodic_used_shuffled_error=(forecast['episodic_shuffled_used'][choice]-target).square().tolist(),
            episodic_correction=read['correction'][choice].tolist(),
            episodic_bank_count=len(self._episodic_bank['slots']),
            episodic_policy_offset=delta.tolist(),
            episodic_policy_tv=float((logits.softmax(-1)-(logits-delta).softmax(-1)).abs().sum()/2)
                if self.episodic_wood_mode == 'real' else
                float((logits.softmax(-1)-(logits+delta).softmax(-1)).abs().sum()/2),
            episodic_read_slots=read['slots'][choice].tolist(),
            episodic_read_weights=read['weights'][choice].tolist(),
            episodic_max_bank_age=int(self._episodic_bank['age'].max())
                if len(self._episodic_bank['age']) else 0)
        return row


def configure_episodic_wood(engine, mode):
    if mode not in ('real', 'blocked'):
        raise ValueError(mode)
    engine.__class__ = EpisodicWoodEngine
    engine.episodic_wood_mode = mode
    engine._episodic_deciding = False
    engine._episodic_exclude = torch.empty(0, dtype=torch.long, device=engine.device)
    engine._episodic_replay_slots = engine._episodic_exclude
    return engine
