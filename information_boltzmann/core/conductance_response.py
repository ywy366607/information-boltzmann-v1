"""Conductance-based local response with persistent first-order receptor gates.

Equations: Hodgkin-Huxley membrane current balance; Destexhe et al. (1994)
two-state receptor kinetics. Neural opening-rate functions are learned model
choices, not measured fly receptor chemistry. No degree-to-time mapping.
Energy coordinates z=sqrt(C)*V and q=sqrt(L)*I keep the existing storage metric.
Positive reversal sources supply work; resistive branches release Joule heat.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ConductanceCoefficients:
    capacitance: torch.Tensor
    leak: torch.Tensor
    maximum: torch.Tensor       # [...,2,D], E and I positive conductances
    reversal: torch.Tensor      # [...,2,D], relative to the zero resting gauge
    closing: torch.Tensor       # [...,2,D], independent receptor closing rates
    inductance: torch.Tensor    # [...,3,D]
    resistance: torch.Tensor    # [...,3,D]
    material: torch.Tensor
    adaptation_gain: torch.Tensor | None = None  # [...,2,D], positive E/I feedback


class LocalConductanceResponse(nn.Module):
    """Learned positive C,L,g,R, kinetics and reversal potentials on the torus.

    Reference scales only specify nondimensionalization/initialization; every
    coefficient is trainable. For finite positive coefficients and gates in[0,1],
    frozen receptor/membrane flows have exact bounded updates for all dt>=0.
    Coupled propagation/scattering/kinetics still requires timestep refinement.
    """

    def __init__(self, channels: int, material_width: int, hidden: int,
                 time_reference: float = 1.0, voltage_reference: float = 1.0,
                 capacitance_reference: float = 1.0, activity_adaptation: bool = False):
        super().__init__()
        references = (time_reference, voltage_reference, capacitance_reference)
        if any(not math.isfinite(x) or x <= 0 for x in references):
            raise ValueError('Positive finite unit references required')
        self.channels = channels
        self.activity_adaptation = bool(activity_adaptation)
        self.time_reference, self.voltage_reference, self.capacitance_reference = references
        # C,gL,gE,gI,|EE|,|EI|,betaE,betaI,Lxyz,Rxyz: fourteen per-channel fields.
        self.log_parameters = nn.Linear(material_width, 14 * channels)
        # Retain Linear's fan-in initialization. Material starts at zero, so
        # coefficients still start uniformly at the declared unit references.
        # Zeroing BOTH maps would block the electrical-coefficient gradient to
        # material until another operator first changes that material.
        nn.init.zeros_(self.log_parameters.bias)
        self.opening = nn.Sequential(nn.Linear(4 * channels + material_width, hidden),
                                     nn.SiLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.opening[0].bias)
        nn.init.constant_(self.opening[-1].bias, math.log(math.expm1(1.0)))
        self.log_adaptation_gain = (nn.Linear(material_width, 2 * channels)
                                    if activity_adaptation else None)
        if self.log_adaptation_gain is not None:
            nn.init.zeros_(self.log_adaptation_gain.bias)

    def coefficients(self, material: torch.Tensor) -> ConductanceCoefficients:
        p = self.log_parameters(material).reshape(*material.shape[:-1], 14, self.channels).exp()
        torch._assert_async((torch.isfinite(p) & (p > 0)).all(),
                            'Conductance coefficients must remain finite and positive')
        gain = (None if self.log_adaptation_gain is None else
                self.log_adaptation_gain(material).reshape(*material.shape[:-1], 2, self.channels).exp())
        if gain is not None:
            torch._assert_async((torch.isfinite(gain) & (gain > 0)).all(),
                                'Activity adaptation gains must remain finite and positive')
        c0, t0, v0 = self.capacitance_reference, self.time_reference, self.voltage_reference
        return ConductanceCoefficients(
            capacitance=c0 * p[..., 0, :], leak=c0 / t0 * p[..., 1, :],
            maximum=c0 / t0 * p[..., 2:4, :],
            reversal=v0 * torch.stack((p[..., 4, :], -p[..., 5, :]), -2),
            closing=p[..., 6:8, :] / t0,
            inductance=t0 * t0 / c0 * p[..., 8:11, :],
            resistance=t0 / c0 * p[..., 11:14, :], material=material,
            adaptation_gain=gain)

    @staticmethod
    def processing_activity(field, flux):
        """Moving-signal energy fraction, independent of overall amplitude.

        Static field energy alone gives zero activity. Edge responses are the
        in-flight signal coordinates also changed by conservative collision.
        This is an engineered continuum activity measure, not a spike rate.
        """
        moving = sum(value.square().sum(-1, keepdim=True) for value in flux)
        total = moving + field.square().sum(-1, keepdim=True)
        return moving / total.clamp_min(torch.finfo(field.dtype).eps)

    @staticmethod
    def inhibition_history(gates):
        """Existing I-gate history, used for spatial competition and routing."""
        return gates[..., 1, :].mean(-1)

    def opening_rates(self, field, flux, coefficients, gates=None, *, local_partials=False):
        c = coefficients
        voltage = field / c.capacitance.sqrt()[None] / self.voltage_reference
        # Flux features are physical current measured in C0*V0/t0 units.
        currents = torch.stack(flux, -2) / c.inductance.sqrt()[None]
        currents = currents / (self.capacitance_reference * self.voltage_reference / self.time_reference)
        local = torch.cat((voltage, currents.flatten(-2),
                           c.material[None].expand(*field.shape[:-1], c.material.shape[-1])), -1)
        base = F.softplus(self.opening(local)).reshape(*field.shape[:-1], 2, self.channels) / self.time_reference
        if c.adaptation_gain is None:
            if local_partials:
                return base, torch.zeros_like(base[..., 0, :]), torch.zeros_like(base[..., 0, :])
            return base
        if gates is None:
            raise ValueError('Activity adaptation requires persisted receptor gates')
        activity = self.processing_activity(field, flux)
        baseline_i = base[..., 1, :] / (base[..., 1, :] + c.closing[None, ..., 1, :])
        # Extra inhibition over the current native equilibrium carries history.
        # Uniform tonic receptor opening alone does not suppress E everywhere.
        excess_i = (gates[..., 1, :] - baseline_i).clamp_min(0)
        alpha_e = base[..., 0, :] / (1 + c.adaptation_gain[None, ..., 0, :] * excess_i)
        alpha_i = base[..., 1, :] + c.adaptation_gain[None, ..., 1, :] * activity / self.time_reference
        opening = torch.stack((alpha_e, alpha_i), -2)
        if local_partials:
            # Partial derivatives of alpha_E at fixed voltage/current features.
            # They include the native I-gate feedback and its closing-dependent
            # baseline; equality at the ReLU kink uses PyTorch's zero derivative.
            active = (gates[..., 1, :] > baseline_i).to(field.dtype)
            d_e_d_i = (-base[..., 0, :] * c.adaptation_gain[None, ..., 0, :]
                      / (1 + c.adaptation_gain[None, ..., 0, :] * excess_i).square()) * active
            d_e_d_closing_i = d_e_d_i * base[..., 1, :] / (
                base[..., 1, :] + c.closing[None, ..., 1, :]).square()
            return opening, d_e_d_i, d_e_d_closing_i
        return opening

    @staticmethod
    def gates_step(gates, opening, closing, dt):
        rate = opening + closing[None]
        target = opening / rate
        fraction = -torch.expm1(-dt.unsqueeze(-1) * rate)
        return gates + fraction * (target - gates)

    @staticmethod
    def _moments(x):
        """Stable exponential moments for a frozen linear relaxation.

        p1=integral_0^1 exp(-x*s)ds, variance=p1(2x)-p1(x)^2.
        Small-x series avoids cancellation in the nonnegative heat integral.
        Branch point is a floating-point accuracy choice, not a physical rate.
        """
        denominator = torch.where(x == 0, torch.ones_like(x), x)
        p1 = -torch.expm1(-x) / denominator
        p1 = torch.where(x == 0, torch.ones_like(x), p1)
        p2 = -torch.expm1(-2 * x) / (2 * denominator)
        p2 = torch.where(x == 0, torch.ones_like(x), p2)
        # Exact variance expansion through x^6; verified against quadrature.
        # First omitted series term is order x^7. Target dtype absolute epsilon.
        cutoff = torch.finfo(x.dtype).eps ** (1/7)
        small = torch.where(x < cutoff, x, torch.zeros_like(x))
        series = small.square() * (1/12 - small/12 + 17*small.square()/360
                                    - 7*small.pow(3)/360 + 43*small.pow(4)/6720)
        variance = torch.where(x < cutoff, series, p2 - p1.square())
        return p1, variance

    def electrical_step(self, field, flux, gates, dt, coefficients, *, diagnostics=True):
        """Exact frozen-conductance membrane/RL step and integrated port ledger."""
        c = coefficients
        capacitance = c.capacitance[None]
        conductance = c.maximum[None] * gates
        total = c.leak[None] + conductance.sum(-2)
        reversal = c.reversal[None]
        equilibrium = (conductance * reversal).sum(-2) / total
        voltage = field / capacitance.sqrt()
        rate = total / capacitance
        fraction = -torch.expm1(-dt * rate)
        voltage_next = (-dt * rate).exp() * voltage + fraction * equilibrium
        field_next = capacitance.sqrt() * voltage_next
        edge_rate = c.resistance / c.inductance
        flux_next = tuple(value * (-dt * edge_rate[None, ..., axis, :]).exp()
                          for axis, value in enumerate(flux))
        if not diagnostics:
            return field_next, flux_next, {}
        p1, variance = self._moments(dt * rate)
        difference = voltage - equilibrium
        mean = equilibrium + difference * p1
        # integral(V-E)^2 = dt*((mean-E)^2 + difference^2*variance).
        heat = dt * c.leak[None] * (mean.square() + difference.square() * variance)
        branch_heat = dt.unsqueeze(-1) * conductance * (
            (mean.unsqueeze(-2) - reversal).square()
            + (difference.square() * variance).unsqueeze(-2))
        heat = heat + branch_heat.sum(-2)
        work = (dt.unsqueeze(-1) * conductance * reversal
                * (reversal - mean.unsqueeze(-2))).sum(-2)
        edge_heat = sum(0.5 * value.square() * (-torch.expm1(
            -2 * dt * edge_rate[None, ..., axis, :])) for axis, value in enumerate(flux))
        heat = heat + edge_heat
        volume = 1.0 / math.prod(field.shape[1:4])
        reduce = lambda value: volume * value.flatten(1).sum(1)
        return field_next, flux_next, {'response_source_work': reduce(work),
                                      'response_joule_heat': reduce(heat)}

    def forward(self, field, flux, gates, dt, coefficients, *, diagnostics=True):
        # Symmetric splitting: kinetics half, electrical full, kinetics half.
        observer = getattr(self, '_local_credit_observer', None)
        def kinetics(value, local_field, local_flux):
            if observer is None:
                opening = self.opening_rates(local_field, local_flux, coefficients, value)
                return self.gates_step(value, opening, coefficients.closing, 0.5 * dt)
            opening, feedback, baseline = self.opening_rates(
                local_field, local_flux, coefficients, value, local_partials=True)
            result = self.gates_step(value, opening, coefficients.closing, 0.5 * dt)
            observer(value, opening, coefficients.closing, 0.5 * dt, feedback, baseline)
            return result
        first = kinetics(gates, field, flux)
        field, flux, info = self.electrical_step(field, flux, first, dt, coefficients,
                                               diagnostics=diagnostics)
        next_gates = kinetics(first, field, flux)
        return field, flux, next_gates, info

    def field_rhs(self, field, gates, coefficients):
        """Instantaneous electrical response, without evaluating gate kinetics."""
        c = coefficients
        conductance = c.maximum[None] * gates
        total = c.leak[None] + conductance.sum(-2)
        return -total / c.capacitance[None] * field + (
            conductance * c.reversal[None]).sum(-2) / c.capacitance.sqrt()[None]

    def rhs(self, field, flux, gates, coefficients):
        c = coefficients
        df = self.field_rhs(field, gates, c)
        dj = tuple(-c.resistance[None, ..., axis, :] / c.inductance[None, ..., axis, :] * value
                   for axis, value in enumerate(flux))
        opening = self.opening_rates(field, flux, c, gates)
        ds = opening * (1 - gates) - c.closing[None] * gates
        return df, dj, ds
