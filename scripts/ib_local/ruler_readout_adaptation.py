"""Shared SFT adaptation of existing experience readouts, before credit arms.

Native parameters and their Adam moments are held. Existing encoder/inlet/null
parameters learn from ordinary answer CE and SIGReg; no new architecture.
"""
import math

import torch

from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner


class ReadoutAdaptationLearner(PretrainedExperienceLearner):
    def apply_gradients(self):
        # Preserve both native weights and their old Adam state. Native local
        # derivatives remain available for propagating loss to the readout.
        for master, (_, parameter) in zip(self.masters, self.native):
            master.grad = None
            parameter.grad = None
        parameters = self.extras + self.null_parameters
        for parameter in parameters:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError('nonfinite adaptation gradient')
        norm = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                             for p in parameters if p.grad is not None))
        self.optimizer.step()
        self.last_adaptation_gradient_norm = norm


def scheduled_gap(index):
    """Same task, alternating no-extra-gap /512 real text tokens."""
    if type(index) is not int or index < 0:
        raise ValueError('nonnegative integer index required')
    return 512 if index % 2 else 0


def validate_splits(train, validation):
    if not train or not validation:
        raise ValueError('nonempty adaptation and validation streams required')
    a = {answer for row in train for answer in row['outputs']}
    b = {answer for row in validation for answer in row['outputs']}
    if a & b:
        raise ValueError('train/validation target overlap; do not filter by scores')
