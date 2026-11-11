"""Bounded, observational imagined evidence on the existing wood/water loop.

Historical action-conditioned residuals are candidate outcome samples, not a
calibrated mother model. This first stage logs the subsequent relation and
continuation changes without adding an uncalibrated policy reward.
"""
from __future__ import annotations

import copy
import torch
from torch.nn import functional as F

from scripts.ib_local.navigation_question_context import QuestionContextEngine
from scripts.ib_local.navigation_relation_wood import local_relation
from scripts.ib_local.navigation_internal_rollout import continuation
from scripts.ib_local.navigation_direct_read import gain_forecast, gain_state, observe_gain


@torch.no_grad()
def preview(model, source, actions, current_latent, reward_valid, *, ridge, scale, samples=2,
            candidate_kind='raw', relation_receipt=None, relation_actual=None,
            outcome_kind='decision'):
    """Use earlier matching episodes to propose residuals, then shadow U.

    Candidate weights are similarity weights, not calibrated probabilities.
    The same relation solve handles actual/imagined evidence; shared Adam and
    reservoir eviction are not simulated. No persistent state is modified.
    """
    if samples < 1 or samples > model.memory.topk:
        raise ValueError('invalid fixed preview budget')
    m, enc = model.memory, model.encoder
    if outcome_kind == 'mother':
        eligible = m.occupied & m.mother_valid & (m.count == 1) & (m.mother_stamp == m.last_write)
        targets = m.mother_next
    elif outcome_kind == 'decision':
        eligible = m.occupied & m.internal_valid & (m.count == 1) & (m.internal_stamp == m.last_write)
        targets = m.internal_next
    else:
        raise ValueError('unknown shadow outcome kind')
    old_s, old_a = enc(m.source), enc(m.action)
    query_s, query_a = enc(source), enc(actions)
    scores = (F.normalize(query_s,dim=-1) @ F.normalize(old_s,dim=-1).T +
              F.normalize(query_a-query_s,dim=-1) @ F.normalize(old_a-old_s,dim=-1).T)/2
    scores = scores.masked_fill(~eligible[None], -1e4)
    values, slots = scores.topk(samples,-1)
    valid = eligible[slots] & (values > -1e3)
    weights = values.masked_fill(~valid,-1e4).softmax(-1)*valid
    weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    historical_actual = enc(targets[slots])
    historical_pred = model.long_predictor(torch.cat((old_s[slots],old_a[slots]),-1))
    if outcome_kind == 'mother':
        historical_pred = historical_pred + model.mother_from_boundary(historical_pred)
    residuals = historical_actual-historical_pred
    if candidate_kind == 'centered':
        residuals = residuals-(weights[...,None]*residuals).sum(1,keepdim=True)
    elif candidate_kind != 'raw':
        raise ValueError('unknown shadow candidate kind')
    current_relation, _ = local_relation(model, source, actions, receipt=relation_receipt,
                                         prior_extra=relation_actual,
                                         ridge=ridge, scale=scale, outcome_kind=outcome_kind)
    base_future, base_packet = continuation(model, current_latent, reward_valid=reward_valid)
    futures, shifts, support = [], [], []
    for j in range(samples):
        # Only a candidate imagined latent successor; never write it to memory.
        imagined = current_latent + residuals[:,j]
        updated, _ = local_relation(model, source, actions,
            receipt=relation_receipt,
            prior_extra=relation_actual,
            extra=(source,actions,imagined,valid[:,j]), extra_encoded=True,
            ridge=ridge, scale=scale, outcome_kind=outcome_kind)
        later_latent = current_latent + updated-current_relation
        future, packet = continuation(model, later_latent, reward_valid=reward_valid)
        futures.append(future)
        shifts.append((later_latent-current_latent).norm(dim=-1))
        support.append(packet['support'])
    futures, shifts = torch.stack(futures,-1), torch.stack(shifts,-1)
    support = torch.stack(support,-1) & valid & base_packet['support'][:,None]
    usable = weights*support
    usable = usable/usable.sum(-1,keepdim=True).clamp_min(1e-12)
    return dict(slots=slots, stamps=m.last_write[slots].clone(), valid=valid,
        weights=weights, shadow_state_shift=shifts,
        future_delta=(usable*(futures-base_future[:,None])).sum(-1),
        usable=support.any(-1), candidate_latents=current_latent[:,None]+residuals,
        base_latent=current_latent)


@torch.no_grad()
def score_candidate_receipt(receipt, target, encoder_weight):
    """Score the previous decision's frozen candidates in its coordinates."""
    actual = F.linear(target, encoder_weight)
    candidates, weights = receipt['candidates'], receipt['weights']
    baseline = receipt['baseline']
    if not bool(receipt['valid'].any()):
        return None
    mean = (weights[:, None]*candidates).sum(0)
    errors = (candidates-actual).square().mean(-1)
    return dict(baseline_mse=float((baseline-actual).square().mean()),
        candidate_mean_mse=float((mean-actual).square().mean()),
        candidate_weighted_mse=float((weights*errors).sum()),
        candidate_best_mse=float(errors.masked_fill(~receipt['valid'], float('inf')).min()),
        candidate_spread=float((weights[:,None]*(candidates-mean).square()).sum()/actual.numel()))


class ShadowPreviewEngine(QuestionContextEngine):
    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        actions = self._internal_action
        receipt = None
        if getattr(self, 'bound_question_shadow', False):
            receipt = self._shadow_relation_receipt(len(actions))
        actual = self._shadow_relation_extra(len(actions)) if receipt is not None else None
        mother_mode = getattr(self, 'mother_shadow_enabled', False)
        latent = self._internal_next_latent
        if mother_mode:
            from scripts.ib_local.navigation_persistent_question import mother_response_evidence
            actual = mother_response_evidence(self, getattr(self, '_retained_question_receipt', None),
                                              len(actions)) if receipt is not None else None
            latent = latent + self.model.mother_from_boundary(latent)
        self._shadow_packet = preview(self.model, h.expand_as(actions), actions,
            latent, self.reward_label_valid(wood=True),
            ridge=self.relation_ridge, scale=self.relation_scale, samples=self.shadow_samples,
            candidate_kind=self.shadow_candidate_kind, relation_receipt=receipt, relation_actual=actual,
            outcome_kind='mother' if mother_mode else 'decision')
        self._shadow_packet['bound_receipt_used'] = torch.tensor(receipt is not None, device=h.device)
        self._shadow_packet['actual_evidence_used'] = torch.tensor(actual is not None, device=h.device)
        delta = self.gamma*self._shadow_packet['future_delta'].cpu().numpy()
        eligible = self._shadow_packet['usable'].cpu().numpy() & (delta != 0)
        self._shadow_forecast = gain_forecast(self.shadow_gain_state,
            self._internal_forecast['mean'], delta, eligible)
        self._shadow_offset = values.new_tensor(self._shadow_forecast['offset'])
        return values, inputs

    def _choice_offset(self, values, available):
        offset = super()._choice_offset(values, available)
        self._shadow_delivered = torch.zeros_like(offset)
        if self.shadow_mode == 'real':
            extra = self._shadow_offset.clone()
            extra -= extra[available].mean()
            self._shadow_delivered = extra.masked_fill(~available, 0.)
            offset = offset + self._shadow_delivered
        return offset

    def step(self, directions, age_ablate=False):
        previous = self.internal_pending
        row = super().step(directions, age_ablate)
        outcome_score = None
        if (previous is not None and 'shadow_candidates' in previous and self._internal_label is not None
                and previous.get('shadow_outcome_kind', 'decision') == 'decision'):
            outcome_score = score_candidate_receipt(previous['shadow_candidates'],
                self._internal_label[2], previous['encoder_weight'])
        settled = None
        if (previous is not None and 'shadow_forecast' in previous and
                previous['settlement'] == self.settlement_kind and
                previous.get('shadow_candidate_kind', 'raw') == self.shadow_candidate_kind):
            forecast, option = previous['shadow_forecast'], previous['choice']
            target = previous['gain'] + self.gamma*float(row['wood_target'][3])
            observe_gain(self.shadow_gain_state, option, target, forecast)
            settled = dict(option=option, target=target, baseline=float(forecast['baseline'][option]),
                prediction=float(forecast['mean'][option]), eligible=bool(forecast['eligible'][option]))
        self.internal_pending['shadow_forecast'] = self._shadow_forecast
        self.internal_pending['shadow_candidate_kind'] = self.shadow_candidate_kind
        self.internal_pending['shadow_outcome_kind'] = ('mother' if getattr(self, 'mother_shadow_enabled', False)
                                                       else 'decision')
        packet = self._shadow_packet
        option = self._pending_record['choice']
        self.internal_pending['shadow_candidates'] = dict(
            candidates=packet['candidate_latents'][option].detach().clone(),
            baseline=packet['base_latent'][option].detach().clone(),
            weights=packet['weights'][option].detach().clone(),
            valid=packet['valid'][option].detach().clone())
        mother_score = None
        if getattr(self, 'mother_shadow_enabled', False) and self._mother_boundary_target is not None:
            mother_score = score_candidate_receipt(self.internal_pending['shadow_candidates'],
                self._mother_boundary_target, self._mother_pre_encoder)
        row['shadow_preview'] = {k:v.detach().cpu().tolist() for k,v in packet.items()
                                 if k not in ('candidate_latents', 'base_latent')}
        row['shadow_preview'].update(mode=self.shadow_mode, candidate_kind=self.shadow_candidate_kind,
            delivered_offset=self._shadow_delivered.tolist(),
            calibration_counts=self.shadow_gain_state['count'].tolist(),
            alpha=self._shadow_forecast['alpha'].tolist(), settled_two_turn=settled,
            settled_candidate_prediction=outcome_score,
            mother_prequential=mother_score,
            outcome_kind=self.internal_pending['shadow_outcome_kind'],
            settled_candidate_kind=(previous.get('shadow_candidate_kind', 'raw')
                                    if outcome_score is not None else None),
            meaning='historical candidate residual; correction calibrated on actual two-turn gain')
        return row


def configure_shadow_preview(engine, *, samples=2, mode='observe', candidate_kind='raw'):
    if type(engine) not in (QuestionContextEngine, ShadowPreviewEngine):
        raise TypeError('requires the current question-conditioned body')
    if type(samples) is not int or not 1 <= samples <= engine.model.memory.topk:
        raise ValueError('invalid fixed preview budget')
    if mode not in ('observe', 'real'):
        raise ValueError('invalid shadow preview mode')
    if candidate_kind not in ('raw', 'centered'):
        raise ValueError('unknown shadow candidate kind')
    if not hasattr(engine, 'shadow_gain_state'):
        engine.shadow_gain_state = gain_state(engine.wood_decay)
    engine.__class__ = ShadowPreviewEngine
    engine.shadow_samples = samples
    engine.shadow_mode = mode
    engine.shadow_candidate_kind = candidate_kind
    engine.shadow_preview_schema = 1
    return engine


def migrate_shadow_candidates(engine, candidate_kind):
    """Keep inherited forecasts but do not mix calibration across semantics."""
    if candidate_kind not in ('raw', 'centered'):
        raise ValueError('unknown shadow candidate kind')
    if engine.shadow_candidate_kind != candidate_kind:
        if not hasattr(engine, 'shadow_gain_archive'):
            engine.shadow_gain_archive = []
        engine.shadow_gain_archive.append(dict(kind=engine.shadow_candidate_kind,
                                               state=copy.deepcopy(engine.shadow_gain_state)))
        engine.shadow_gain_state = gain_state(engine.wood_decay)
        engine.shadow_candidate_kind = candidate_kind


def enable_mother_shadow(engine):
    """Explicitly migrate calibration; pending real observations remain intact."""
    if not getattr(engine, 'joint_question_update', False):
        raise ValueError('mother shadow requires the shared versioned relation update')
    if getattr(engine, 'mother_shadow_enabled', False):
        return
    if not hasattr(engine, 'shadow_gain_archive'):
        engine.shadow_gain_archive = []
    engine.shadow_gain_archive.append(dict(kind='before_mother_shadow',
                                           state=copy.deepcopy(engine.shadow_gain_state)))
    engine.shadow_gain_state = gain_state(engine.wood_decay)
    if engine.internal_pending is not None and 'shadow_forecast' in engine.internal_pending:
        engine.internal_pending['pre_mother_shadow_forecast'] = engine.internal_pending.pop('shadow_forecast')
    engine.mother_shadow_enabled = True
