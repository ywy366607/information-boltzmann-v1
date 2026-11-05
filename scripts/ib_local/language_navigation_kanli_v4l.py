"""Candidate: action-outcome and SIGReg gradients share the current encoder.

Stored features are finite, detached observations; re-encoding them now is
differentiable. No graph from a previous turn is retained.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
from scripts.ib_local.language_navigation_kanli_v4k import LatentDraftEngine


def live_sigreg(current, historical, valid, directions):
    z = torch.cat((current, historical), 0)
    weight = torch.cat((torch.ones(len(current), device=z.device, dtype=z.dtype), valid.to(z.dtype)))
    count = weight.sum().clamp_min(1)
    t = torch.linspace(0, 3, 17, device=z.device, dtype=z.dtype)
    phi = torch.exp(-t.square()/2)
    quad = torch.ones_like(t) * (6/16)
    quad[0] /= 2
    quad[-1] /= 2
    phase = (z @ F.normalize(directions, dim=0)).unsqueeze(-1) * t
    cosine = (phase.cos()*weight[:, None, None]).sum(0)/count
    sine = (phase.sin()*weight[:, None, None]).sum(0)/count
    return (((cosine-phi).square()+sine.square()) @ (quad*phi)).mean()*count


class LiveSigregBody(LatentNavigationV3):
    def objective(self, ids, targets, state, table_logp, directions,
                  prediction_weight=.1, regularization_weight=.01, score_mask=None):
        logits, next_state, diagnostics, h, read = self.predict_observed(ids, state, table_logp)
        if score_mask is not None:
            # The scored response window begins after the self-utterance.
            # Cache its real near outcome as fixed features: the encoder is
            # evaluated again during the later outcome update, without BPTT.
            own = int((score_mask[0] == 0).sum())
            self._near_outcome = h[min(own + self.memory.short, len(h)-1)].detach().clone()
        token_ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none')
        if score_mask is None:
            ce = token_ce.mean()
        else:
            mask = score_mask.reshape(-1).to(token_ce.dtype)
            ce = (token_ce*mask).sum()/mask.sum().clamp_min(1)
        short = self.memory.short
        if len(h) > short:
            source = h[:-short]
            recall = self.memory.read(source, self.encoder)
            pred = self.predictor(torch.cat((self.encoder(source), recall.value), -1))
            short_loss = F.mse_loss(pred, self.encoder(h[short:]))
        else:
            short_loss = h.sum()*0
        prior = self.memory.pending[:len(h)]
        prior_view = self.memory.pending_view[:len(h)]
        valid = self.memory.pending_valid[:len(h)].to(h.dtype)
        pred = self.long_predictor(torch.cat((self.encoder(prior), self.encoder(prior_view)), -1))
        error = (pred-self.encoder(h)).square().mean(-1)
        long_loss = (error*valid).sum()/valid.sum().clamp_min(1)
        # history is stored bounded data; only its current re-encoding is live.
        regularizer = live_sigreg(self.encoder(h), self.encoder(self.memory.history),
                                 self.memory.history_valid, directions)
        self._last_sigreg = float(regularizer.detach())
        loss = ce + prediction_weight*(short_loss+long_loss) + regularization_weight*regularizer
        return loss, next_state, h, dict(ce=ce.detach(), latent=short_loss.detach(),
            latent_long=long_loss.detach(), sigreg=regularizer.detach(),
            read_slots=read.slots.detach(), read_weights=read.weights.detach(),
            logits=logits, **diagnostics)


class LiveTargetEngine(LatentDraftEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcome_horizon = 'near'
        self.detach_outcome_target = False

    def _outcome_loss(self, source, action, long):
        memory, enc, outcome = self.model.memory, self.model.encoder, self.model.kanli.outcome
        # Current and stored examples must use the same target horizon.
        if self.outcome_horizon == 'near':
            current_target, stored_target = self.model._near_outcome, memory.successor
        elif self.outcome_horizon == 'long':
            current_target, stored_target = long, memory.long_successor
        else:
            raise ValueError('outcome_horizon must be near or long')
        occupied = torch.nonzero(memory.occupied).flatten()
        if len(occupied):
            pick = occupied[torch.as_tensor(self.replay_rng.integers(0, len(occupied), self.replay_batch),
                                            device=occupied.device)]
            src = torch.cat((source[None], memory.source[pick]))
            act = torch.cat((action[None], memory.action[pick]))
            lng = torch.cat((current_target[None], stored_target[pick]))
        else:
            src, act, lng = source[None], action[None], current_target[None]
        target = enc(lng)
        if self.detach_outcome_target:
            target = target.detach()
        zs, za = enc(src), enc(act)
        error = (outcome(torch.cat((zs, za), -1))-target).square().mean(-1)
        with torch.no_grad():
            shuffled = (outcome(torch.cat((zs, za.roll(1, 0)), -1))-target).square().mean(-1)
            blind = (outcome(torch.cat((zs, zs), -1))-target).square().mean(-1)
        return error.mean(), dict(outcome_error=float(error.detach().mean()),
            outcome_error_current=float(error.detach()[0]),
            outcome_error_shuffled=float(shuffled.mean()), outcome_error_as_silent=float(blind.mean()))
