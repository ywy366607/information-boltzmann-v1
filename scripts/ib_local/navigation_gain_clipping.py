"""Bound the one-step gain-forecast innovation, never the observed gain."""
from __future__ import annotations

import torch

from scripts.ib_local.language_navigation_kanli_v4t import ProbabilityActionEvidence


class ClippedSharedActionEvidence(ProbabilityActionEvidence):
    @torch.no_grad()
    def observe(self, choice, target, base, raw):
        n = int(self.seen[choice])
        if n:
            rate = 1. if n == 1 else 1-self.decay
            self.base_error[choice].lerp_((base-target).square(), rate)
            self.raw_error[choice].lerp_((raw-target).square(), rate)
        any_seen = bool(self.seen.any())
        self.unconditional.lerp_(target, 1-self.decay if any_seen else 1.)
        bounded = target.clone()
        error = target[3]-base[3]
        limit = (3*self.gain_innovation_scale).clamp_min(1e-6)
        clipped = error.clamp(-limit, limit)
        self.last_gain_clip = dict(innovation=float(error), limit=float(limit),
                                   used=float(clipped), scale_before=float(self.gain_innovation_scale))
        bounded[3] = base[3]+clipped
        if not any_seen:
            self.blind.copy_(target)
        elif not n:
            self.mean[choice].copy_(bounded-self.blind)
            shift = self.mean.mean(0)
            self.mean.sub_(shift)
            self.blind.add_(shift)
        else:
            innovation = (1-self.decay)*(bounded-base)/(5/3)
            self.blind.add_(innovation)
            self.mean.sub_(innovation/3)
            self.mean[choice].add_(innovation)
        self.gain_innovation_scale.lerp_(clipped.abs(), 1-self.decay)
        self.last_gain_clip['scale_after'] = float(self.gain_innovation_scale)
        self.seen[choice] += 1


def prepare_gain_clipping(engine):
    """Create the checkpoint schema before loading its saved buffer."""
    e = engine.model.wood_evidence
    if isinstance(e, ClippedSharedActionEvidence):
        return
    if type(e) is not ProbabilityActionEvidence:
        raise TypeError('requires the existing typed shared action evidence')
    e.register_buffer('gain_innovation_scale', e.blind.new_zeros(()))
    e.__class__ = ClippedSharedActionEvidence


def configure_gain_clipping(engine):
    """Migrate from existing versioned real observations; no body reset."""
    e, m = engine.model.wood_evidence, engine.model.memory
    if isinstance(e, ClippedSharedActionEvidence):
        return
    if type(e) is not ProbabilityActionEvidence:
        raise TypeError('requires the existing typed shared action evidence')
    valid = (m.occupied & (m.count == 1) & (m.wood_target_count == 1) &
             (m.wood_option >= 0) & engine.reward_label_valid(wood=True))
    slots = torch.nonzero(valid).flatten()
    if len(slots) == 0:
        raise ValueError('need real unmerged gain observations to initialize scale')
    slots = slots[m.last_write[slots].argsort()[-32:]]
    residual = (m.wood_target[slots, 3] - m.wood_baseline[slots, 3]).abs()
    if not bool(torch.isfinite(residual).all()):
        raise ValueError('non-finite existing gain observations')
    scale = residual.median().clamp_min(1e-6).detach().clone()
    prepare_gain_clipping(engine)
    e.gain_innovation_scale.copy_(scale)
