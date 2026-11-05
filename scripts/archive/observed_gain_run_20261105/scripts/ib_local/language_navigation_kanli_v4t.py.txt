"""Typed Bernoulli outcome on the existing wood predictor, no new parameters.

Column 2 answers whether the reply-world suffix matcher accepted the utterance.
Its probability is learned from real outcomes. The historical error weight is
a separate mixture weight, not the probability and not an uncertainty proof.
"""
import torch

from scripts.ib_local.language_navigation_kanli_v4q import SharedActionEvidence
from scripts.ib_local.language_navigation_kanli_v4s import RepresentativeWitnessEngine
from scripts.ib_local.navigation_value_scale import critic_output


def match_probability(base, log_odds_correction):
    # Only protect the machine logit at exact endpoints. The forecast mean
    # remains the observed action baseline; the learned correction is in odds.
    eps = torch.finfo(base.dtype).eps
    return torch.sigmoid(torch.logit(base.clamp(eps, 1-eps)) + log_odds_correction)


class ProbabilityActionEvidence(SharedActionEvidence):
    def bases(self):
        result = super().bases().clone()
        result[:,2] = result[:,2].clamp(0,1)
        return result


class TypedWoodEngine(RepresentativeWitnessEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        old = self.model.wood_evidence
        new = ProbabilityActionEvidence(self.wood_decay).to(self.device)
        new.load_state_dict(old.state_dict())
        self.model.wood_evidence = new

    def _compose_prediction(self, base, residual):
        result = super()._compose_prediction(base,residual)
        result = result.clone()
        result[:,2] = match_probability(base[:,2],residual[:,2])
        return result

    def _compose_used_prediction(self, base, residual, confidence):
        result = super()._compose_used_prediction(base,residual,confidence)
        probability = match_probability(base[:,2],residual[:,2])
        result = result.clone()
        result[:,2] = base[:,2] + confidence[:,2]*(probability-base[:,2])
        return result

    @torch.no_grad()
    def _value(self,h,films):
        values,inputs = super()._value(h,films)
        zwidth = self.model.encoder.out_features
        # Measure only this channel's delivery, holding every other critic
        # input and the memory's recalled value fixed.
        without = inputs.clone()
        without[:,zwidth+2] = self._forecast['base'][:,2]
        delta = critic_output(self, inputs).squeeze(-1)-critic_output(self, without).squeeze(-1)
        available = len(self._draft_actions) if self._draft_actions else 1
        offset = torch.zeros_like(delta)
        offset[:available] = self.precision()*(delta[:available]-delta[:available].mean())
        self._forecast['typed_match_offset'] = offset
        return values,inputs

    def step(self,directions,age_ablate=False):
        out = super().step(directions,age_ablate)
        choice = self._pending_record['choice']
        p = self._forecast['used'][choice,2]
        raw = self._forecast['raw'][choice,2]
        y = out['wood_target'][2]
        logits = p.new_tensor(self._pending_record['choice_logits'])
        delta = self._forecast['typed_match_offset']
        out.update(match_probability=float(p),match_model_probability=float(raw),
            match_distribution=[float(1-p),float(p)],
            match_brier=float((p-y).square()),match_model_brier=float((raw-y).square()),
            match_prediction_error=float(y-p),match_score=float(-(p-y).square()),
            match_residual_mixture_weight=float(self._forecast['confidence'][choice,2]),
            match_policy_tv=float((logits.softmax(-1)-(logits-delta).softmax(-1)).abs().sum()/2))
        return out
