"""Local, causal conduction adaptation with an analytic invariant interval.

This is an engineered activity-dependent law, not a fitted biological model.
Optimizer parameters learn the metric/gain/time scale; runtime state adapts
without an optimizer or targets. Each stored positive-axis edge has one value.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LocalConductionPlasticity(nn.Module):
    """tau * dp/dt = gain * evidence - p, where |evidence| <= 1.

    p is a dimensionless log-speed offset. For fixed trained parameters, finite
    bounded spatial gain gives an invariant interval for p and hence finite,
    strictly positive propagation speeds. Time constants use physical units.
    """

    def __init__(self, channels: int, material_width: int,
                 time_reference: float = 1.0, *, spatial_metric: bool = False):
        super().__init__()
        if not (math.isfinite(time_reference) and time_reference > 0):
            raise ValueError("Positive finite plasticity time reference required")
        self.time_reference = float(time_reference)
        # Historical shared metric remains a reproducible option. The local
        # variant uses the same material field as electrical/propagation laws.
        self.metric_logits = nn.Parameter(torch.zeros(3, channels))
        self.metric_material = (nn.Linear(material_width, 3 * channels, bias=False)
                                if spatial_metric else None)
        self.channels = channels
        self.gain = nn.Linear(material_width, 3)
        self.log_rate = nn.Linear(material_width, 3)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, math.log(math.expm1(1.0)))
        nn.init.zeros_(self.log_rate.weight)
        nn.init.zeros_(self.log_rate.bias)

    def coefficients(self, material: torch.Tensor):
        gain = F.softplus(self.gain(material))
        rate = self.log_rate(material).exp() / self.time_reference
        logits = self.metric_logits
        if self.metric_material is not None:
            logits = logits + self.metric_material(material).reshape(
                *material.shape[:-1], 3, self.channels)
        metric = logits.softmax(-1)
        return gain, rate, metric

    @staticmethod
    def evidence(field: torch.Tensor, flux: tuple[torch.Tensor, ...],
                 metric: torch.Tensor) -> torch.Tensor:
        """[B,X,Y,Z,3] symmetric endpoint coherence plus edge occupancy.

        s=(2<f_i,f_j>_M + ||j_e||_M^2)/
          (||f_i||_M^2 + ||f_j||_M^2 + ||j_e||_M^2 + eps).
        Cauchy-Schwarz gives -1 <= s <= 1. Zero activity gives s=0.
        Positive activity is a local routing signal, not a semantic reward.
        """
        signals = []
        for axis in range(3):
            neighbor = torch.roll(field, -1, axis + 1)
            # One edge metric is used at BOTH endpoints. It may be [3,D]
            # or a material-conditioned [X,Y,Z,3,D] positive diagonal metric.
            weight = metric[..., axis, :]
            cross = (field * neighbor * weight).sum(-1)
            endpoints = ((field.square() + neighbor.square()) * weight).sum(-1)
            current = (flux[axis].square() * weight).sum(-1)
            denominator = endpoints + current + torch.finfo(field.dtype).eps
            signals.append((2 * cross + current) / denominator)
        return torch.stack(signals, -1)

    def forward(self, offset: torch.Tensor, field: torch.Tensor,
                flux: tuple[torch.Tensor, ...], duration: torch.Tensor,
                coefficients: tuple[torch.Tensor, ...], inhibition=None) -> torch.Tensor:
        """Exact frozen-evidence relaxation; convexity holds at every step size.

        duration broadcasts as [B,1,1,1,1]. Whole coupled dynamics still needs
        solver refinement because field/evidence change over physical time.
        """
        gain, rate, metric = coefficients
        evidence = self.adapted_evidence(field, flux, metric, gain, inhibition)
        target = gain[None] * evidence
        fraction = -torch.expm1(-duration * rate[None])
        return offset + fraction * (target - offset)

    def adapted_evidence(self, field, flux, metric, gain, inhibition=None):
        evidence = self.evidence(field, flux, metric)
        if inhibition is None:
            return evidence
        # Symmetric edge contrast promotes a path joining differently occupied
        # regions. It changes reciprocal conductance, not an imposed direction
        # or a guaranteed downhill energy current. Convex mixing keeps |s|<=1.
        contrast = torch.stack([(inhibition - torch.roll(inhibition, -1, axis + 1)).abs()
                                for axis in range(3)], -1)
        fraction = contrast * gain[None] / (1 + gain[None])
        return (1 - fraction) * evidence + fraction
