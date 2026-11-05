"""V4n: wood's outcome channels learn at the pace their data arrives (Claude, 2026-11-04).

The v4m 64-turn run (scratch/wood_calibration_replay_20261104.json) shows the
world's consequences are strongly action-dependent and non-semantic. Second
half, mean +- sd by action:
* match: silence 0, generate 0.96, reuse 1.0.
* NLL4/3: reuse 0.13, generate 0.75, silence 0.50.
* 10g: silence 0.042, generate 0.020, reuse 0.010.
So wood has something to learn. A by-action running mean beats F on every
channel (match MSE 0.031 vs F 0.43).

Cause, computable in advance: AdamW moves each parameter by about lr per step.
At lr 3e-4 and about 2 F steps per turn, 64 turns allow about 0.04 per
parameter. F's mean outputs stayed at 0.05-0.18 while the targets sit at
0.5-0.75. F was too slow to reach even the mean, so its confidence stayed 0.

Only change: F's parameters (kanli.outcome) get their own param group at
WOOD_LR. The encoder, body, critic and all loss weights are unchanged from v4m.
"""
from __future__ import annotations

from scripts.ib_local.language_navigation_kanli_v4m import WoodCalibratedEngine

WOOD_LR = 1e-2


class FastWoodEngine(WoodCalibratedEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wood = {id(p) for p in self.model.kanli.outcome.parameters()}
        for group in self.optimizer.param_groups:
            group['params'] = [p for p in group['params'] if id(p) not in wood]
        self.optimizer.add_param_group(dict(params=list(self.model.kanli.outcome.parameters()), lr=WOOD_LR))
