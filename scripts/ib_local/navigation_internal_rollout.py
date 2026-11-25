"""Bounded internal continuation of the existing action/experience transition.

Schema 1 explicitly migrates the existing long predictor from a byte horizon
to the next *actual decision boundary*. No extra neural weights, reward or
goal state. Observed successor tags share the finite episode slots. Predicted
states can read that bank but never write observations. The two-turn read
increment is calibrated only after both actual settlements have arrived.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from scripts.ib_local.language_action import peek
from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
from scripts.ib_local.navigation_direct_read import (
    DirectReadGainEngine, gain_forecast, gain_state, observe_gain)


class InternalRolloutBody(WoodObservationBody):
    def long_prediction_loss(self, h):
        # The very same head is now trained in _outcome_loss on actual
        # decision-boundary pairs. Do not mix it with byte-horizon labels.
        return h.sum() * 0


def next_boundary_prediction(model, source, action):
    return next_boundary_encoded(model, model.encoder(source), model.encoder(action))


def next_boundary_encoded(model, source, action):
    """Shared transition also accepts a predicted state and a latent action.

    The caller owns its finite planning depth. It advances no observed clock.
    """
    return model.long_predictor(torch.cat((source, action), -1))


def continuation(model, latent, *, reward_valid, shuffle=False):
    """Predict the next choice's gain from action-conditioned retained episodes.

    Historical action displacements are transported to the predicted source.
    This is a learned-model approximation, not a restored full GDN state.
    Only compatible, unmerged observed transitions can supply candidates.
    No new randomness, table write, event clock or external call is used.
    """
    m, enc = model.memory, model.encoder
    from scripts.ib_local.preview_runtime import _scope, encoder_read, read, tensor_key
    active = None if torch.is_grad_enabled() else _scope.get()
    in_preview = active is not None and active[0] is model
    if in_preview:
        elig_key = (id(m), tensor_key(reward_valid), m.occupied._version, m.internal_valid._version,
                    m.count._version, m.internal_stamp._version, m.last_write._version,
                    m.wood_option._version, m.wood_target_count._version)
        eligible = read(model, 'continuation_eligible', elig_key, lambda: (
            m.occupied & reward_valid & m.internal_valid & (m.count == 1) &
            (m.internal_stamp == m.last_write) & (m.wood_option >= 0) &
            (m.wood_target_count == m.count)), anchors=(m, reward_valid))
    else:
        eligible = (m.occupied & reward_valid & m.internal_valid & (m.count == 1) &
                    (m.internal_stamp == m.last_write) & (m.wood_option >= 0) &
                    (m.wood_target_count == m.count))
    read_out = m.read_encoded(latent, enc, eligible=eligible)
    slots = read_out.slots
    valid = eligible[slots]
    options = m.wood_option[slots].clamp(0, 2)
    if in_preview:
        proj = encoder_read(enc, 'preview_memory_projection',
            (id(m), *(tensor_key(t) for t in (m.source, m.action, m.long_successor))),
            lambda: (F.normalize(enc(m.source), dim=-1), enc(m.action), enc(m.long_successor), enc(m.source)),
            anchors=(m,))
        if len(proj) >= 4:
            src, act = proj[3][slots], proj[1][slots]
        else:
            src, act = enc(m.source[slots]), proj[1][slots]
    else:
        src, act = enc(m.source[slots]), enc(m.action[slots])
    delta = act-src
    if shuffle:
        # Diagnostic only: preserve each query's candidates, change pairing.
        delta = delta.roll(1, dims=1)
    z = latent[:, None].expand_as(src)
    x = torch.cat((z, delta, F.one_hot(options, 3).to(z.dtype)), -1)
    head = model.kanli.outcome
    # The parametric branch of the SAME existing F. No fabricated historical
    # evidence for a future event, no new word/address side channel.
    evidence = getattr(model, '_causal_use_value_evidence', model.wood_evidence)
    if in_preview:
        bases, conf = read(model, 'continuation_evidence_stats', id(evidence),
                           lambda: (evidence.bases(), evidence.confidence()), anchors=(evidence,))
    else:
        bases, conf = evidence.bases(), evidence.confidence()
    if hasattr(model, '_causal_use_value_evidence'):
        from scripts.ib_local.causal_value_learning import prediction as gain_prediction
        raw = gain_prediction(model, z, delta, options)
        base = bases[options]
        prediction = base + conf[options]*(raw-base)
    else:
        residual = head(x)[..., :4]
        prediction = bases[options] + conf[options]*residual
    gain = prediction[..., 3]
    # Recalled action prior, reweighted by the currently predicted gain;
    # a bounded soft choice, not an oracle selecting a known future reward.
    weight = read_out.weights*valid
    logits = gain + weight.clamp_min(1e-30).log()
    weights = logits.masked_fill(~valid, -1e4).softmax(-1)*valid
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    # Center before summing: equal candidate values must remain exactly equal
    # across different attention weights, not create roundoff "gain evidence".
    reference = gain[:, :1]
    expected = reference[:, 0] + (weights*(gain-reference)).sum(-1)
    expected = torch.where(valid.any(-1), expected, 0.)
    return expected, dict(
        slots=slots, weights=weights, support=valid.any(-1), prediction=prediction)


class InternalRolloutEngine(DirectReadGainEngine):
    @torch.no_grad()
    def _capture_boundary(self):
        self.decision.start(self.recurrent)
        _, h = self.decision.step(self.pending, peek(self.table, self.pending))
        target = h.reshape(-1).detach().clone()
        m, pending = self.model.memory, self.internal_pending
        self._internal_label = None
        self._internal_pre_error = None
        self._internal_stay_error = None
        self._internal_other_action_error = None
        if pending is not None:
            slot = pending['slot']
            # Recycled/merged addresses do not inherit a previous episode's tag.
            valid = (int(m.count[slot]) == 1 and
                     int(m.last_write[slot]) == pending['stamp'])
            from scripts.ib_local.identified_event_credit import complete_boundary
            event_label = complete_boundary(self, pending, target)
            if valid or event_label is not None:
                if valid:
                    m.internal_next[slot].copy_(target)
                    m.internal_stamp[slot] = pending['stamp']
                    m.internal_valid[slot] = True
                self._internal_label = (pending['source'], pending['action'], target) if event_label is None else event_label
                # Forecast was made before the action, in the then-current
                # coordinate system; score the target using that encoder too.
                old_target = F.linear(target, pending['encoder_weight'])
                self._internal_pre_error = float((pending['next_latent']-old_target).square().mean())
                # Same target and decision-time encoder, no post-update leakage.
                if 'stay_latent' in pending:
                    self._internal_stay_error = float((pending['stay_latent']-old_target).square().mean())
                    self._internal_other_action_error = float(
                        (pending['other_action_latents']-old_target).square().mean(-1).mean())
        self._internal_boundary = target

    def _utter(self):
        self._capture_boundary()
        return super()._utter()

    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        actions = list(self._draft_actions)
        while len(actions) < 3:
            actions.append(h.reshape(-1))
        actions = torch.stack(actions)
        source = h.expand_as(actions)
        latent = next_boundary_prediction(self.model, source, actions)
        current = self.model.encoder(source)
        reward_valid = self.reward_label_valid(wood=True)
        future, packet = continuation(self.model, latent, reward_valid=reward_valid)
        blind, blind_packet = continuation(self.model, current, reward_valid=reward_valid)
        wrong, _ = continuation(self.model, latent, reward_valid=reward_valid, shuffle=True)
        # Baseline and increment share the same two-turn units (10*g).
        # We calibrate the additional prediction against observed returns,
        # rather than add an uncalibrated second copy of critic value.
        immediate = self._gain_forecast['mean']
        baseline = immediate + self.gamma*blind.cpu().numpy()
        delta = self.gamma*(future-blind).cpu().numpy()
        eligible = (packet['support'] & blind_packet['support']).cpu().numpy() & (delta != 0)
        forecast = gain_forecast(self.internal_gain_state, baseline, delta, eligible)
        self._internal_forecast = forecast
        self._internal_next_latent = latent.detach().clone()
        self._internal_action = actions.detach().clone()
        self._internal_source = h.reshape(-1).detach().clone()
        self._internal_encoder_weight = self.model.encoder.weight.detach().clone()
        self._internal_stay_latent = current[0].detach().clone()
        self._internal_packet = dict(
            next_latent=latent.detach().clone(), future=future.cpu().tolist(),
            blind=blind.cpu().tolist(), shuffled=wrong.cpu().tolist(),
            support=packet['support'].cpu().tolist(),
            slots=packet['slots'].cpu().tolist(), weights=packet['weights'].cpu().tolist())
        self._internal_offset = values.new_tensor(forecast['offset'])
        return values, inputs

    def _choice_offset(self, values, available):
        offset = super()._choice_offset(values, available)
        self._internal_delivered_offset = torch.zeros_like(offset)
        if self.internal_mode == 'real':
            extra = self._internal_offset.clone()
            extra -= extra[available].mean()
            self._internal_delivered_offset = extra.masked_fill(~available, 0.)
            offset = offset + self._internal_delivered_offset
        return offset

    def _outcome_loss(self, source, action, long):
        loss, diag = super()._outcome_loss(source, action, long)
        m = self.model.memory
        valid = (m.occupied & m.internal_valid & (m.count == 1) & (m.internal_stamp == m.last_write))
        # The newest pair is appended below using its exact decision-time
        # source/action. Do not count it again among the retained replay rows.
        if self._internal_label is not None and self.internal_pending is not None:
            valid[self.internal_pending['slot']] = False
        excl = getattr(self, '_exclude_internal_slots', None)
        if excl is not None and len(excl) > 0:
            valid = valid.clone()
            valid[excl.to(device=valid.device, dtype=torch.long)] = False
        slots = torch.nonzero(valid).flatten()
        # Fixed budget, use the existing replay RNG, all retained in snapshots.
        if len(slots) > self.replay_batch:
            picks = self.replay_rng.choice(len(slots), self.replay_batch, replace=False)
            slots = slots[torch.as_tensor(picks, device=self.device)]
        src, act, nxt = m.source[slots], m.action[slots], m.internal_next[slots]
        if self._internal_label is not None:
            s, a, n = self._internal_label
            src, act, nxt = (torch.cat((v[None], old)) for v, old in ((s,src),(a,act),(n,nxt)))
        if len(src):
            predicted = next_boundary_prediction(self.model, src, act)
            transition_loss = F.mse_loss(predicted, self.model.encoder(nxt))
        else:
            transition_loss = loss*0
        # The outer inherited update already multiplies outcome_loss by .1.
        return loss+transition_loss, dict(diag,
            internal_transition_loss=float(transition_loss.detach()),
            internal_transition_samples=len(src))

    def _store_label(self, result, choice, target):
        super()._store_label(result, choice, target)
        # A write/merge invalidates the previous successor of this exact slot.
        self.model.memory.internal_valid[result['episode_slot']] = False

    def step(self, directions, age_ablate=False):
        previous = self.internal_pending
        row = super().step(directions, age_ablate)
        choice = self._pending_record['choice']
        actual = float(row['wood_target'][3])
        settled = None
        if previous is not None and previous['settlement'] == self.settlement_kind:
            target = previous['gain'] + self.gamma*actual
            option, forecast = previous['choice'], previous['forecast']
            settled = dict(option=option, target=target,
                baseline=float(forecast['baseline'][option]),
                prediction=float(forecast['mean'][option]),
                uncalibrated=float(forecast['baseline'][option]+forecast['delta'][option]),
                eligible=bool(forecast['eligible'][option]))
            observe_gain(self.internal_gain_state, previous['choice'], target, previous['forecast'])
        slot = row['episode_slot']
        self.internal_pending = dict(
            slot=slot, stamp=int(self.model.memory.last_write[slot]),
            source=self._internal_source, action=self._internal_action[choice],
            next_latent=self._internal_next_latent[choice],
            stay_latent=self._internal_stay_latent,
            other_action_latents=self._internal_next_latent[
                torch.arange(3, device=self.device) != choice].detach().clone(),
            encoder_weight=self._internal_encoder_weight,
            choice=choice, gain=actual, forecast=self._internal_forecast,
            settlement=self.settlement_kind)
        row['internal_rollout'] = dict(
            mode=self.internal_mode, depth=2, predicted_future=self._internal_packet['future'],
            baseline_future=self._internal_packet['blind'],
            shuffled_future=self._internal_packet['shuffled'],
            support=self._internal_packet['support'],
            slots=self._internal_packet['slots'], weights=self._internal_packet['weights'],
            next_boundary_prequential_mse=self._internal_pre_error,
            unchanged_state_prequential_mse=self._internal_stay_error,
            other_action_prequential_mse=self._internal_other_action_error,
            settled_two_turn=settled,
            proposed_offset=self._internal_offset.tolist(),
            enabled=(self.internal_mode == 'real'),
            delivered_offset=self._internal_delivered_offset.tolist(),
            calibration_counts=self.internal_gain_state['count'].tolist(),
            valid_boundary_pairs=int(self.model.memory.internal_valid.sum()))
        logits = self._internal_delivered_offset.new_tensor(self._pending_record['choice_logits'])
        without = logits-self._internal_delivered_offset
        row['internal_rollout']['choice_probability_tv'] = float(
            (logits.softmax(-1)-without.softmax(-1)).abs().sum()/2)
        return row


def configure_internal_rollout(engine, mode):
    if mode not in ('observe', 'real'):
        raise ValueError('unknown internal rollout mode')
    if type(engine) not in (DirectReadGainEngine, InternalRolloutEngine):
        raise TypeError('requires the existing complete direct-read body')
    if type(engine.model) not in (WoodObservationBody, InternalRolloutBody):
        raise TypeError('unsupported body, cannot migrate long target silently')
    if not hasattr(engine, 'internal_gain_state'):
        m = engine.model.memory
        m.register_buffer('internal_next', torch.zeros_like(m.source))
        m.register_buffer('internal_stamp', torch.zeros_like(m.last_write))
        m.register_buffer('internal_valid', torch.zeros_like(m.occupied))
        engine.internal_pending = None
        engine.internal_gain_state = gain_state(engine.wood_decay)
    engine.__class__ = InternalRolloutEngine
    engine.model.__class__ = InternalRolloutBody
    engine.internal_mode = mode
    engine.internal_schema = 1
