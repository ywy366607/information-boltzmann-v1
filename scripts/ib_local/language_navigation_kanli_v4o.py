"""V4o: wood sees what its own action changed; per-channel wood metal (Claude, 2026-11-04).

The v4n prereg failed (handover doc section 11). With a fast lr, F's match
channel only just beat the action-blind running mean (0.197 vs 0.212). The
by-action mean reached 0.016. Confidence stayed 0 for all 64 second-half turns.

Cause. F sees [z_source, z_action]. For silence z_action == z_source, so
"was this silence?" is an equality test between two drifting 64-d vectors.
The generate-silence gap is 3.6-9 L2, while z itself moves about 10 between
states. That test is non-linear and is not learnable from tens of samples.
The action-type law of the reply world was therefore invisible to wood.

Changes (two, separable in the analysis):
1. Input: F reads [z, z_action - z], i.e. what my own act changed in me. For
   silence this is exactly 0. Same parameters; the wrapper only re-expresses
   the input.
2. Wood's metal per channel: each of the 4 outcome channels has its own
   prequential confidence, 1 - err/err_baseline, against the action-blind
   running mean. A noisy channel no longer zeroes a reliable one.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scripts.ib_local.language_navigation_kanli_v4n import FastWoodEngine


class ActionDelta(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        z, a = x.chunk(2, -1)
        return self.inner(torch.cat((z, a - z), -1))


class DeltaWoodEngine(FastWoodEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.kanli.outcome = ActionDelta(self.model.kanli.outcome)  # same parameter objects
        self.channel_mean = None
        self.channel_err = np.zeros(4)
        self.channel_base = np.zeros(4)
        self.channel_count = 0
        self.wood_confidence = torch.zeros(4, device=self.device)

    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        self.draft_diagnostics['wood_confidence'] = self.wood_confidence.tolist()
        return values, inputs

    def step(self, directions, age_ablate=False):
        used = self.wood_confidence.tolist()
        result = super().step(directions, age_ablate)
        p, t = np.asarray(result['wood_prediction']), np.asarray(result['wood_target'])
        if self.channel_count:
            e, b = (p - t)**2, (self.channel_mean - t)**2
            if self.channel_count == 1:
                self.channel_err, self.channel_base = e, b
            else:
                d = self.wood_decay
                self.channel_err = d*self.channel_err + (1-d)*e
                self.channel_base = d*self.channel_base + (1-d)*b
            conf = np.clip(1 - self.channel_err/(self.channel_base+1e-8), 0, 1)
            self.channel_mean = self.wood_decay*self.channel_mean + (1-self.wood_decay)*t
        else:
            conf = np.zeros(4)
            self.channel_mean = t.copy()
        self.channel_count += 1
        self.wood_confidence = torch.tensor(conf, dtype=torch.float32, device=self.device)
        result.update(wood_confidence_used=used, wood_confidence_next=conf.tolist())
        return result
