"""Experimental response student with a lossless motor-observation channel.

Regional latent dynamics remain approximate. The complete raw motor membrane
and raw running mean retain their semantics across read-weight revisions.
There is no future-teacher injection during rollout. This prototype is not
automatically installed in the production learner/runtime.
"""
from __future__ import annotations

import copy
import torch
from torch import nn


class MotorSurfaceForecaster(nn.Module):
    def __init__(self, latent_student, motor_count: int, *, use_motor_state=True,
                 mean_decay=.99):
        super().__init__()
        if motor_count < 1 or not 0 <= mean_decay < 1:
            raise ValueError('Positive motor surface and valid existing mean decay required')
        self.latent_student = copy.deepcopy(latent_student)
        # This arm decodes predicted raw membrane through the original readout;
        # its former learned latent-to-read adapter/attention are not used.
        for name in ('query', 'key', 'horizon_bias', 'motor_adapter'):
            delattr(self.latent_student, name)
        self.use_motor_state = bool(use_motor_state)
        self.mean_decay = float(mean_decay)
        motor_latent = len(latent_student.codec.graph.motor_regions) * latent_student.codec.latent_dim
        self.motor_delta = nn.Linear(motor_latent + motor_count, motor_count)
        nn.init.zeros_(self.motor_delta.weight)
        nn.init.zeros_(self.motor_delta.bias)

    def forward(self, history, motor, mean):
        """Only origin motor is observed; all subsequent motor inputs are drafts."""
        if motor.shape != mean.shape:
            raise ValueError('Raw motor and mean must share their full surface shape')
        z_values, m_values, means = [history[0]], [motor], [mean]
        for _ in range(self.latent_student.horizon):
            following, history = self.latent_student.transition_step(history)
            regional = following[:, self.latent_student.codec.graph.motor_regions].flatten(1)
            supplied_motor = motor if self.use_motor_state else torch.zeros_like(motor)
            motor = motor + self.motor_delta(torch.cat((regional, supplied_motor), dim=-1))
            mean = self.mean_decay * mean + (1 - self.mean_decay) * motor
            z_values.append(following)
            m_values.append(motor)
            means.append(mean)
        return torch.stack(z_values), torch.stack(m_values), torch.stack(means)

    @staticmethod
    def decode(model, motor, mean):
        """Read current model weights from fixed raw-state coordinates."""
        observable = motor - mean if model.read_centering else motor
        return model.output_read(observable)
