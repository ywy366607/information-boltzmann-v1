"""Mean-rate Tsodyks--Markram kinetics for reciprocal continuum pathways.

This uses continuous local activity instead of presynaptic spikes. An edge has
one shared resource/utilization pair across feature channels, not one bouton
per feature. It modulates conservative coupling; it never deletes stored flux.
References: Tsodyks, Pawelzik & Markram (1998), Neural Computation 10:821--835.
"""
from __future__ import annotations

import math
import os

import torch
from torch import nn


class LocalShortTermPlasticity(nn.Module):
    """dx/dt = recovery*(1-x)-u*r*x; du/dt=closing*(U-u)+U*r*(1-u).

    For nonnegative activity r, [0,1]^2 is invariant. Spatial material learns
    positive rates and U in (0,1). Unit references are initialization scales,
    not fixed biological constants. The frozen-rate split is positivity
    preserving and converges to these equations with timestep refinement.
    """

    def __init__(self, material_width: int, time_reference: float = 1.0):
        super().__init__()
        if not math.isfinite(time_reference) or time_reference <= 0:
            raise ValueError('Positive finite STP time reference required')
        self.time_reference = float(time_reference)
        # Three axes, each recovery, facilitation closing, activity rate, U.
        self.parameters_map = nn.Linear(material_width, 12)
        nn.init.zeros_(self.parameters_map.bias)

    def coefficients(self, material):
        raw = self.parameters_map(material).reshape(*material.shape[:-1], 3, 4)
        rates = raw[..., :3].exp() / self.time_reference
        baseline = raw[..., 3].sigmoid()
        torch._assert_async((torch.isfinite(rates) & (rates > 0)).all(),
                            'STP rates must remain finite and positive')
        torch._assert_async(((baseline > 0) & (baseline < 1)).all(),
                            'STP utilization must remain strictly inside (0,1)')
        return rates, baseline

    @staticmethod
    def initial_state(field, coefficients):
        _, baseline = coefficients
        u = baseline[None].expand(*field.shape[:-1], 3).to(field).detach().clone()
        return torch.stack((torch.ones_like(u), u), -1)

    @staticmethod
    def activity(field, flux):
        """Symmetric edge in-flight fraction in [0,1], zero for static storage.

        The edge coordinate is shared by both endpoints. This engineered
        continuum rate proxy is amplitude invariant, with no firing threshold.
        """
        signals = []
        # Roll the scalar energy, not a full D-channel field. The mean square
        # commutes with a spatial permutation; its graph can be shared by all
        # three edges instead of recomputing it at both endpoints per axis.
        stored = field.square().mean(-1)
        for axis, current in enumerate(flux):
            other = torch.roll(stored, -1, axis + 1)
            moving = current.square().mean(-1)
            total = moving + 0.5 * (stored + other)
            signals.append(moving / total.clamp_min(torch.finfo(field.dtype).eps))
        return torch.stack(signals, -1)

    @staticmethod
    def _relax(value, target, rate, duration):
        return value + (-torch.expm1(-duration * rate)) * (target - value)

    def native_step(self, state, field, flux, duration, coefficients):
        rates, baseline = coefficients
        recovery, closing, activity_rate = rates.unbind(-1)
        r = self.activity(field, flux) * activity_rate[None]
        u0 = baseline[None]
        x, u = state.unbind(-1)
        facilitation_rate = closing[None] + u0 * r
        facilitation_target = u0 * (closing[None] + r) / facilitation_rate
        # Symmetric u-half / x-full / u-half frozen-activity integration.
        half_fraction = -torch.expm1(-0.5 * duration * facilitation_rate)
        half_u = u + half_fraction * (facilitation_target - u)
        depletion_rate = recovery[None] + half_u * r
        next_x = self._relax(x, recovery[None] / depletion_rate, depletion_rate, duration)
        next_u = half_u + half_fraction * (facilitation_target - half_u)
        return torch.stack((next_x, next_u), -1)

    def forward(self, state, field, flux, duration, coefficients):
        # AOT autograd fuses the rate/relaxation arithmetic AND backward. CPU
        # and FP64 retain the native expression, including higher derivatives.
        if (getattr(self, 'fuse_execution', True) and field.is_cuda
                and field.dtype == torch.float32 and not torch.compiler.is_compiling()):
            from torch.autograd.forward_ad import unpack_dual
            values = (state, field, *flux, duration, *coefficients)
            if any(isinstance(value, torch.Tensor) and unpack_dual(value).tangent is not None
                   for value in values):
                # AOT backward compilation has no forward-mode contract. The
                # exact native expression retains every state/rate/time dual.
                return self.native_step(state, field, flux, duration, coefficients)
            if not hasattr(self, '_compiled_step'):
                # This Windows build lacks the optional static CUDA launcher.
                if os.name == 'nt':
                    import torch._inductor.config as config
                    config.use_static_cuda_launcher = False
                self._compiled_step = torch.compile(self.native_step, fullgraph=True, dynamic=False)
            return self._compiled_step(state, field, flux, duration, coefficients)
        return self.native_step(state, field, flux, duration, coefficients)

    @staticmethod
    def transmission_gain(state, coefficients):
        """ux/U: unity at resting x=1,u=U; can facilitate OR depress.

        0<=gain<=1/U for each fixed finite trained U>0. Neither activity nor
        recovery imposes an external sparse-rate target or an energy clamp.
        """
        _, baseline = coefficients
        return state[..., 0] * state[..., 1] / baseline[None]

    def rhs(self, state, field, flux, coefficients):
        rates, baseline = coefficients
        recovery, closing, activity_rate = rates.unbind(-1)
        r = self.activity(field, flux) * activity_rate[None]
        x, u = state.unbind(-1)
        dx = recovery[None] * (1 - x) - u * r * x
        du = closing[None] * (baseline[None] - u) + baseline[None] * r * (1 - u)
        return torch.stack((dx, du), -1)
