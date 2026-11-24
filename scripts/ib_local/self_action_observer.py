"""Passive, pre-response scoring of the continuing body's own action model.

Compare the SAME observed outcome under predictions for alternative actions.
This is factual likelihood discrimination, not a claim to have observed the
world's unexecuted counterfactual responses. Four generated drafts provide a
within-action-type check, separate from silence/generate/reuse discrimination.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from scripts.ib_local.causal_relation_program import log_score


@dataclass
class FrozenActionForecast:
    mean: torch.Tensor
    base: torch.Tensor
    scale2: torch.Tensor
    encoder: torch.Tensor
    actions: torch.Tensor
    df: float
    legacy: dict | None = None
    geometry: dict | None = None

    @classmethod
    def capture(cls, model, distribution, actions, *, legacy=None):
        freeze = lambda value: value.detach().to('cpu').clone()
        geometry=None
        if distribution.get('basis_schema')==1:
            x=distribution['x'].double()
            c,a=x[:,:2],x[:,2:4]
            residual=a-c@torch.linalg.pinv(c)@a
            rank=lambda v:int(torch.linalg.matrix_rank(v,tol=1e-6))
            geometry=dict(input_dimension=x.shape[1],support=len(x),
                condition_rank=rank(c),action_rank_given_condition=rank(residual),
                full_rank=rank(x),noise_mean=float(distribution['noise'].mean()))
        old=None if legacy is None else {k:freeze(legacy[k]) for k in ('mean','scale2')}
        if old is not None:
            old['df']=legacy['df']
        return cls(freeze(distribution['mean']), freeze(distribution['base']),
                   freeze(distribution['scale2']), freeze(model.encoder.weight),
                   freeze(actions), float(distribution['df']),old,geometry)

    @torch.no_grad()
    def score(self, observed, chosen):
        if not 0 <= chosen < len(self.mean):
            raise ValueError('executed action outside the frozen candidate set')
        target = F.linear(observed.detach().cpu().reshape(1, -1), self.encoder)
        mse = (self.mean-target).square().mean(-1)
        baseline = (self.base-target).square().mean(-1)
        scores = log_score(dict(mean=self.mean, scale2=self.scale2, df=self.df), target)
        different = (self.actions-self.actions[chosen]).norm(dim=-1) > 1e-6
        different[chosen] = False
        alternatives = torch.nonzero(different).flatten()
        result = dict(chosen=chosen, candidates=len(self.mean),
                      different_alternatives=alternatives.tolist(),
                      relation_mse=mse.tolist(), original_mse=baseline.tolist(),
                      relation_log_score=scores.tolist(),
                      action_distances=(self.actions-self.actions[chosen]).norm(dim=-1).tolist())
        result['predicted_variance']=(self.scale2.mean(-1)*self.df/(self.df-2)).tolist()
        if self.geometry is not None:
            result['shared_basis_geometry']=self.geometry
        if self.legacy is not None:
            result['legacy_relation_mse']=(self.legacy['mean']-target).square().mean(-1).tolist()
            result['legacy_relation_log_score']=log_score(self.legacy,target).tolist()
        if len(alternatives):
            result.update(
                original_margin=float(baseline[alternatives].mean()-baseline[chosen]),
                relation_margin=float(mse[alternatives].mean()-mse[chosen]),
                relation_log_margin=float(scores[chosen]-scores[alternatives].mean()),
                original_wins=int((baseline[chosen] < baseline[alternatives]).sum()),
                relation_wins=int((mse[chosen] < mse[alternatives]).sum()))
        return result


def attach_self_action_observer(engine):
    """Keep every original forward, sampling and update; only clone predictions.

    No optimizer, RNG, world call or teacher is added. Install in a separate
    continuation process, never into an already running worker.
    """
    from scripts.ib_local import navigation_joint_cycle as joint
    from scripts.ib_local import run_navigation_residual_wood as checkpoint_runtime
    if getattr(engine, '_self_action_observer_attached', False):
        raise ValueError('self-action observer already attached')
    # These are runtime callbacks, like the existing decision/world handles.
    # They must never be included among persistent engine fields.
    checkpoint_runtime._DERIVED = checkpoint_runtime._DERIVED | {
        '_value', 'step', '_self_action_observer_attached'}
    capture = {}
    original_imagine, original_value, original_step = joint.imagine, engine._value, engine.step

    def imagine(*args, **kwargs):
        result = original_imagine(*args, **kwargs)
        model, state, source, actions = args[:4]
        if model is engine.model:
            legacy=None
            if hasattr(model,'relation_basis'):
                from scripts.ib_local.causal_relation_program import mechanism
                legacy=mechanism(model,state,source,actions,ridge=kwargs['ridge'],
                    momentum=kwargs.get('momentum'),legacy=True)
            capture['drafts'] = FrozenActionForecast.capture(model, result['path']['mother'], actions,legacy=legacy)
        return result

    def value(*args, **kwargs):
        result = original_value(*args, **kwargs)
        distribution = dict(engine._program_mother_forecast,
                            base=engine._mother_pre_base)
        if hasattr(engine.model,'relation_basis'):
            actual=engine._program_forecasts['path']['mother']
            distribution.update({k:actual[k] for k in ('basis_schema','x','noise')})
        legacy=None
        if hasattr(engine.model,'relation_basis'):
            from scripts.ib_local.causal_relation_program import mechanism
            actions=engine._internal_action
            legacy=mechanism(engine.model,engine._program_question_state,args[0].expand_as(actions),
                actions,ridge=engine.relation_ridge,
                momentum=engine._momentum_decision_read[None].expand(len(actions),-1),legacy=True)
        capture['options'] = FrozenActionForecast.capture(
            engine.model, distribution, engine._internal_action,legacy=legacy)
        return result

    def annotate(row):
        target = engine._mother_boundary_target
        if target is None:
            row['self_action_observer'] = dict(available=False, reason='no observed mother boundary')
            return row
        chosen = int(engine._pending_record['choice'])
        options = capture['options'].score(target, chosen)
        reported = row['causal_program']['mother_prequential']['mse']
        error = abs(options['relation_mse'][chosen]-reported)
        if error > 1e-5:
            raise AssertionError(('frozen actual score failed to reproduce production', error))
        within = None
        if row['choice'] == 'generate':
            selected = int(engine._joint_selection['selected'])
            words = engine._joint_proposals[selected]['words']
            if words.hex() != row['utterance_hex']:
                raise AssertionError('executed speech differs from the frozen draft candidate')
            within = capture['drafts'].score(target, selected)
            within['utterances_hex'] = [p['words'].hex() for p in engine._joint_proposals]
        row['self_action_observer'] = dict(
            available=True, options=options, generated_drafts=within,
            actual_score_reproduction_error=error,
            coordinates='encoder and action predictions frozen before actual world response',
            scope='same factual outcome; alternative conditional forecasts; no counterfactual world calls')
        return row

    def step(*args, **kwargs):
        capture.clear()
        basis=getattr(engine.model,'relation_basis',None)
        before=None if basis is None else {k:p.detach().clone() for k,p in basis.named_parameters()}
        row = original_step(*args, **kwargs)
        if before is not None:
            row['shared_relation_basis']=dict(schema=1,
                parameter_update_norms={k:float((p.detach()-before[k]).norm())
                                        for k,p in basis.named_parameters()},
                noise_prior_mean=basis.noise(mother=True).mean().item(),
                same_mechanism_in_real_replay_imagination_and_action=True)
        try:
            return annotate(row)
        except Exception as error:
            # An observer failure must not discard an already real world event
            # or prevent its complete body checkpoint from being committed.
            row['self_action_observer'] = dict(available=False,
                reason='passive score failed; actual learning event retained',
                error=f'{type(error).__name__}: {error}')
            return row

    joint.imagine, engine._value, engine.step = imagine, value, step
    engine._self_action_observer_attached = True
