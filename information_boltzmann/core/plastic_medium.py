"""Local, heterogeneous, passive 3-D medium; independent numerical candidate.

The fast state is a site field plus three oriented edge responses. Slow material
parameters are continuous Fourier fields; optional persistent edge log-speed
offsets carry activity-dependent structural experience between observations.
There is no vocabulary, task-specific rule, global FFT, or event-reset here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .conduction_plasticity import LocalConductionPlasticity
from .conductance_response import ConductanceCoefficients, LocalConductanceResponse
from .short_term_plasticity import LocalShortTermPlasticity
from .structural_resource import TransportCapacityBudget


def complete_periodic_modes(shape: Sequence[int]):
    """Minimal real Fourier basis spanning a reference grid, Nyquist included.

    The reference grid defines structural bandwidth, not the evaluation grid.
    Keeping this bandwidth fixed permits strict loading at finer resolutions.
    """
    import itertools

    if len(shape) != 3 or any(n < 2 for n in shape):
        raise ValueError("Three reference axes >= 2 required")
    cos_modes, sin_modes = [], []
    seen = {(0, 0, 0)}
    ranges = [range(-(n // 2), (n + 1) // 2) for n in shape]
    for mode in itertools.product(*ranges):
        if mode in seen:
            continue
        conjugate = tuple((-k + n // 2) % n - n // 2 for k, n in zip(mode, shape))
        seen.update((mode, conjugate))
        cos_modes.append(mode)
        if mode != conjugate:
            sin_modes.append(mode)
    return tuple(cos_modes), tuple(sin_modes)


@dataclass(frozen=True)
class MediumState:
    """field/flux: [B,X,Y,Z,D]; flux[a] lives on the positive axis-a edge."""

    field: torch.Tensor
    flux: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    elapsed: torch.Tensor  # [B], physical time, independent of observation count
    conduction: torch.Tensor | None = None  # [B,X,Y,Z,3], persistent log-speed offset
    receptors: torch.Tensor | None = None  # [B,X,Y,Z,2,D], E/I open fractions
    transmission: torch.Tensor | None = None  # [B,X,Y,Z,3,2], edge resource x/utilization u

    def with_field(self, field: torch.Tensor) -> "MediumState":
        """Assimilate a port event while retaining signals in transmission."""
        if field.shape != self.field.shape:
            raise ValueError("Port field shape must match the persistent field")
        return replace(self, field=field)

    def with_conduction(self, conduction: torch.Tensor) -> "MediumState":
        return replace(self, conduction=conduction)

    def detach(self) -> "MediumState":
        """Explicit gradient boundary; values and elapsed time are retained."""
        return MediumState(self.field.detach(), tuple(x.detach() for x in self.flux),
                           self.elapsed.detach(),
                           None if self.conduction is None else self.conduction.detach(),
                           None if self.receptors is None else self.receptors.detach(),
                           None if self.transmission is None else self.transmission.detach())


@dataclass(frozen=True)
class EvolutionCoefficients:
    """Fixed-parameter spatial coefficients, reusable within one weight version.

    Preserve their graph for a differentiable segment, or prepare under no_grad
    for deployment. Refresh after optimizer updates; runtime owns invalidation.
    """

    material: torch.Tensor
    baseline_log_speed: torch.Tensor
    plasticity: tuple[torch.Tensor, ...] | None
    response: ConductanceCoefficients | None = None
    short_term: tuple[torch.Tensor, ...] | None = None
    structural_factor: torch.Tensor | None = None
    structural_allocation: torch.Tensor | None = None


class ContinuousMaterial(nn.Module):
    """Periodic material a(x), evaluated at any grid resolution.

    Zero coefficients give a uniform medium. Inhomogeneous observations can
    produce different gradients for Fourier coefficients from that starting point.
    ``modes`` controls representational bandwidth, not evolution duration.
    """

    def __init__(self, width: int = 8,
                 modes: Sequence[Sequence[int]] = ((1, 0, 0), (0, 1, 0),
                                                  (0, 0, 1), (1, 1, 0),
                                                  (1, 0, 1), (0, 1, 1)),
                 sine_modes: Sequence[Sequence[int]] | None = None, *,
                 reference_shape: tuple[int, int, int] = (8, 8, 4)):
        super().__init__()
        if width < 1 or any(len(k) != 3 for k in modes):
            raise ValueError("Positive material width and three-component modes required")
        if len(reference_shape) != 3 or any(n < 2 for n in reference_shape):
            raise ValueError("Three material reference axes >= 2 required")
        self.width = int(width)
        self.reference_shape = tuple(reference_shape)
        self.register_buffer("modes", torch.tensor(modes, dtype=torch.float64).reshape(-1, 3))
        sine_modes = modes if sine_modes is None else sine_modes
        if any(len(k) != 3 for k in sine_modes):
            raise ValueError("Three-component sine modes required")
        self.register_buffer("sine_modes", torch.tensor(sine_modes, dtype=torch.float64).reshape(-1, 3))
        self.coefficients = nn.Parameter(torch.zeros(1 + len(modes) + len(sine_modes), width))

    def basis(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != 3:
            raise ValueError("Coordinates must end in xyz")
        phase = 2 * math.pi * (coordinates @ self.modes.to(coordinates).T)
        sine_phase = 2 * math.pi * (coordinates @ self.sine_modes.to(coordinates).T)
        return torch.cat((torch.ones_like(coordinates[..., :1]), phase.cos(), sine_phase.sin()), -1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.basis(coordinates) @ self.coefficients.to(coordinates)

    @torch.no_grad()
    def load_expanded_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """Explicitly embed an old Fourier field in a wider, zero-residual basis.

        Match continuous modes, including cos(-k)=cos(k), sin(-k)=-sin(k).
        This preserves the function at arbitrary coordinates, not just grid
        samples. An absent mode raises instead of aliasing or interpolating it.
        The caller owns migration of shape-dependent optimizer state.
        """
        modes, sine_modes = state_dict["modes"], state_dict["sine_modes"]
        coefficients = state_dict["coefficients"]
        if (modes.ndim != 2 or modes.shape[-1] != 3
                or sine_modes.ndim != 2 or sine_modes.shape[-1] != 3
                or coefficients.shape != (1 + len(modes) + len(sine_modes), self.width)):
            raise ValueError("Source material basis or coefficient width mismatch")
        expanded = torch.zeros_like(self.coefficients)
        expanded[0] = coefficients[0].to(expanded)
        lookups = ({tuple(mode): 1 + i for i, mode in enumerate(self.modes.tolist())},
                   {tuple(mode): 1 + len(self.modes) + i
                    for i, mode in enumerate(self.sine_modes.tolist())})
        offset = 1
        for is_sine, source_modes in enumerate((modes, sine_modes)):
            lookup = lookups[is_sine]
            for i, mode in enumerate(source_modes.tolist()):
                key = tuple(mode)
                if not any(key):
                    if not is_sine:
                        expanded[0] += coefficients[offset + i].to(expanded)
                    continue
                if key in lookup:
                    index, sign = lookup[key], 1
                elif tuple(-k for k in key) in lookup:
                    index, sign = lookup[tuple(-k for k in key)], -1 if is_sine else 1
                else:
                    raise ValueError(f"Expanded material basis is missing source mode {key}")
                expanded[index] += sign * coefficients[offset + i].to(expanded)
            offset += len(source_modes)
        return self.load_state_dict({"modes": self.modes.detach().clone(),
                                     "sine_modes": self.sine_modes.detach().clone(),
                                     "coefficients": expanded}, strict=True)

    @torch.no_grad()
    def initialize_spectral_xavier(self, reference_shape=None, field_std=1.0):
        """Xavier random directions, calibrated in material-field units.

        The DC component stays zero. Calibration uses a fixed reference grid,
        not the runtime grid, so refinement does not change the sampled medium.
        Unit variance matches the input scale assumed by downstream linear maps.
        This is an initialization prior, not a guarantee of task specialization.
        """
        if not math.isfinite(field_std) or field_std <= 0:
            raise ValueError('Positive finite material field standard deviation required')
        reference_shape = self.reference_shape if reference_shape is None else reference_shape
        axes = [torch.arange(n, device=self.coefficients.device,
                             dtype=self.coefficients.dtype) / n for n in reference_shape]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
        nn.init.xavier_uniform_(self.coefficients)
        self.coefficients[0].zero_()
        field = self(coordinates).reshape(-1, self.width)
        scale = field.std(0, unbiased=False).clamp_min(torch.finfo(field.dtype).eps)
        self.coefficients.mul_(field_std / scale)


class PlasticMedium3D(nn.Module):
    """Conservative local propagation + nonlinear scattering + selective outflow.

    Transport is linear for a fixed material. Rates are evaluated in physical
    units; solver substeps only refine the same requested duration. This class
    supports optional persistent local conduction adaptation. It does not
    implement a posterior over causes or autonomous optimizer updates.
    All material/operator parameters receive gradients from a caller's observed
    likelihood; online delayed credit is a separate learning-interface task.
    """

    def __init__(self, shape: tuple[int, int, int] = (8, 8, 4), channels: int = 128,
                 material_width: int = 8, hidden: int = 64, collision_layers: int = 2,
                 speed_reference: float = 1.0, bath_time_reference: float = 1.0,
                 modes: Sequence[Sequence[int]] | None = None,
                 sine_modes: Sequence[Sequence[int]] | None = None,
                 material_reference_shape: tuple[int, int, int] | None = (8, 8, 4),
                 adaptive_conduction: bool = False,
                 plasticity_time_reference: float = 1.0,
                 bath_type: str = 'quadratic', response_time_reference: float = 1.0,
                 voltage_reference: float = 1.0, capacitance_reference: float = 1.0,
                 activity_adaptation: bool = False, short_term_plasticity: bool = False,
                 execution_backend: str = 'native', anisotropic_transport: bool = False,
                 transport_capacity_budget: float | None = None,
                 structure_options: dict | None = None):
        super().__init__()
        if len(shape) != 3 or any(n < 2 or n % 2 for n in shape):
            raise ValueError("First local schedule requires three even axes >= 2")
        if channels < 2 or collision_layers < 1 or hidden < 1:
            raise ValueError("channels >= 2 and positive hidden/layer counts required")
        if not (math.isfinite(speed_reference) and speed_reference > 0
                and math.isfinite(bath_time_reference) and bath_time_reference > 0):
            raise ValueError("Reference scales must be finite and positive")
        self.shape, self.channels = tuple(shape), int(channels)
        if execution_backend not in ('native', 'fused'):
            raise ValueError('Medium execution backend must be native or fused')
        self.execution_backend = execution_backend
        self.anisotropic_transport = bool(anisotropic_transport)
        if transport_capacity_budget is not None and not anisotropic_transport:
            raise ValueError('Capacity constraint requires tensor transport')
        self.transport_capacity_limit = transport_capacity_budget
        self.transport_budget = (None if transport_capacity_budget is None else
                                 TransportCapacityBudget(transport_capacity_budget))
        self.collision_layers = int(collision_layers)
        self.speed_reference = float(speed_reference)
        if bath_type not in ('quadratic', 'conductance'):
            raise ValueError('bath_type must be quadratic or conductance')
        self.bath_type = bath_type
        if activity_adaptation and bath_type != 'conductance':
            raise ValueError('Activity adaptation reuses conductance receptor state')
        self.activity_adaptation = bool(activity_adaptation)
        self.short_term_plasticity = (LocalShortTermPlasticity(material_width, plasticity_time_reference)
                                      if short_term_plasticity else None)
        # None explicitly opts into a runtime-complete structural bandwidth.
        # The historical default remains fixed for strict fine-grid loading.
        self.material_reference_shape = (tuple(shape) if material_reference_shape is None
                                         else tuple(material_reference_shape))
        if modes is None:
            modes, sine_modes = complete_periodic_modes(self.material_reference_shape)
        self.material = ContinuousMaterial(material_width, modes, sine_modes,
                                          reference_shape=self.material_reference_shape)
        self.structural_posterior = None
        if structure_options is not None:
            if not anisotropic_transport or transport_capacity_budget is not None:
                raise ValueError('Structural posterior replaces the old tensor amplitude budget')
            from .structural_posterior import StructuralPosterior
            self.structural_posterior = StructuralPosterior(
                self.material.coefficients.shape[0], **structure_options)
        grid = torch.stack(torch.meshgrid(*[
            torch.arange(n, dtype=torch.float64) / n for n in shape
        ], indexing="ij"), -1)
        self.register_buffer("coordinates", grid, persistent=False)
        self.log_speed = nn.Linear(material_width, 3, bias=False)
        if self.structural_posterior is not None:
            # Retain the legacy key for explicit migration; it cannot compensate
            # for a capacity reduction and is absent from the learning graph.
            self.log_speed.weight.requires_grad_(False)
        self.transport_shear = (nn.Linear(material_width, 3, bias=True)
                                if self.anisotropic_transport else None)
        if self.transport_shear is not None:
            nn.init.zeros_(self.transport_shear.weight)
            nn.init.zeros_(self.transport_shear.bias)
        self.conduction_plasticity = (LocalConductionPlasticity(
            channels, material_width, plasticity_time_reference,
            spatial_metric=bath_type == 'conductance')
            if adaptive_conduction else None)
        self.free_width = 4 * (channels - 1)
        self.collision_rate = nn.Sequential(
            nn.Linear((6 if bath_type == 'conductance' else 4) * channels + material_width, hidden), nn.SiLU(),
            nn.Linear(hidden, collision_layers * (self.free_width // 2)))
        self.bath_rate = (nn.Sequential(
            nn.Linear(4 * channels + material_width, hidden), nn.SiLU(),
            nn.Linear(hidden, 4 * channels)) if bath_type == 'quadratic' else None)
        self.conductance_response = (LocalConductanceResponse(
            channels, material_width, hidden, response_time_reference,
            voltage_reference, capacitance_reference,
            activity_adaptation=activity_adaptation) if bath_type == 'conductance' else None)
        # Rate reference has units inverse time at unit dimensionless occupancy.
        if self.bath_rate is not None:
            nn.init.zeros_(self.bath_rate[-1].bias)
            self.register_buffer("bath_bias", torch.tensor(
                math.log(math.expm1(1.0 / bath_time_reference)), dtype=torch.float64))
        # Householder maps each group's channel mean to coordinate zero.
        direction = torch.zeros(channels, dtype=torch.float64)
        direction[0] = 1.0
        direction -= torch.ones(channels, dtype=torch.float64) / math.sqrt(channels)
        direction /= direction.norm()
        self.register_buffer("mean_reflector", direction, persistent=False)

    def initial_state(self, batch_size: int = 1, *, device=None, dtype=None) -> MediumState:
        if batch_size < 1:
            raise ValueError("Positive batch size required")
        reference = self.material.coefficients
        field = torch.zeros((batch_size, *self.shape, self.channels),
                            device=device or reference.device, dtype=dtype or reference.dtype)
        conduction = (field.new_zeros(batch_size, *self.shape, 3)
                      if self.conduction_plasticity is not None else None)
        transmission = (None if self.short_term_plasticity is None else
                        self.short_term_plasticity.initial_state(field,
                            self.short_term_plasticity.coefficients(self.material_field())))
        return MediumState(field, tuple(torch.zeros_like(field) for _ in range(3)),
                           field.new_zeros(batch_size, dtype=torch.float64), conduction,
                           field.new_zeros(batch_size, *self.shape, 2, self.channels)
                           if self.conductance_response is not None else None, transmission)

    def edge_log_speeds(self, state: MediumState,
                        material: torch.Tensor | None = None,
                        baseline_log_speed: torch.Tensor | None = None) -> torch.Tensor:
        """Linear propagation coefficients when material/conduction are frozen."""
        material = self.material_field() if material is None else material
        baseline = (self.log_speed(material) if baseline_log_speed is None
                    else baseline_log_speed)[None]
        if self.conduction_plasticity is None:
            if state.conduction is not None:
                raise ValueError("Adaptive state requires an adaptive-conduction model")
            return baseline
        if state.conduction is None or state.conduction.shape != (*state.field.shape[:-1], 3):
            raise ValueError("Persistent conduction state missing or mismatched")
        return baseline + state.conduction

    def adapt_conduction(self, state: MediumState, duration: float | torch.Tensor,
                         material: torch.Tensor | None = None,
                         coefficients: tuple[torch.Tensor, ...] | None = None) -> MediumState:
        """Physical-time structural adaptation; field and stored flux are retained."""
        if self.conduction_plasticity is None:
            return state
        if state.conduction is None:
            raise ValueError("Adaptive medium requires persistent conduction state")
        if coefficients is None:
            material = self.material_field() if material is None else material
            coefficients = self.conduction_plasticity.coefficients(material)
        updated = self.conduction_plasticity(
            state.conduction, state.field, state.flux,
            self._duration(duration, state.field), coefficients,
            inhibition=(self.conductance_response.inhibition_history(state.receptors)
                        if self.activity_adaptation else None))
        return state.with_conduction(updated)

    def material_field(self) -> torch.Tensor:
        return self.material(self.coordinates.to(self.material.coefficients))

    def prepare_evolution(self) -> EvolutionCoefficients:
        material = self.material_field()
        coefficients = (None if self.conduction_plasticity is None else
                        self.conduction_plasticity.coefficients(material))
        response = (None if self.conductance_response is None else
                    self.conductance_response.coefficients(material))
        short_term = (None if self.short_term_plasticity is None else
                      self.short_term_plasticity.coefficients(material))
        factor, allocation = None, None
        if self.structural_posterior is not None:
            posterior = self.structural_posterior
            basis = self.material.basis(self.coordinates.to(material))
            sample = posterior.sample_coefficients() if bool(posterior.window_active) else None
            allocation = posterior.allocation(basis, sample)
            rows = self.raw_transport_factor(material, material.new_ones(*material.shape[:-1], 3))
            directions = F.normalize(rows, dim=-1)
            factor = posterior.speed_reference.to(material) * allocation[..., :3, None] * directions
        return EvolutionCoefficients(material, self.log_speed(material), coefficients, response,
                                     short_term, factor, allocation)

    def energy(self, state: MediumState) -> torch.Tensor:
        """Quadrature energy [B] on the unit torus, including edge storage."""
        volume = 1.0 / math.prod(self.shape)
        return 0.5 * volume * sum(x.square().flatten(1).sum(1)
                                 for x in (state.field, *state.flux))

    @staticmethod
    def _duration(duration: float | torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        if not isinstance(duration, torch.Tensor) and not torch.compiler.is_compiling():
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Duration must be finite and nonnegative")
        dt = torch.as_tensor(duration, device=field.device, dtype=field.dtype)
        if torch.compiler.is_compiling():
            torch._assert_async((torch.isfinite(dt) & (dt >= 0)).all(),
                                "Duration must be finite and nonnegative")
        if dt.numel() not in (1, field.shape[0]):
            raise ValueError("Duration must be scalar or [B]")
        return dt.reshape(-1, 1, 1, 1, 1)

    @staticmethod
    def _lanes(value: torch.Tensor, axis: int, parity: int) -> torch.Tensor:
        key = [slice(None)] * value.ndim
        key[axis] = slice(parity, None, 2)
        return value[tuple(key)]

    @staticmethod
    def _interleave(left: torch.Tensor, right: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.stack((left, right), dim=axis + 1).flatten(axis, axis + 1)

    def transport_factor(self, material: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """B=diag(speed) T, A=B B^T positive definite for positive finite speeds.

        T is unit lower triangular. The energy metric stays Euclidean: B occurs
        in the field generator and its negative adjoint in the flux generator.
        Thus changing B does not change stored energy or require a moving metric.
        Flux components are edge-origin stores, approaching a collocated vector
        as spacing tends to zero. This is reciprocal propagation, not a diode.
        """
        factor = self.raw_transport_factor(material, speed)
        if self.transport_budget is None:
            return factor
        # Budget the slow structural field, never normalize global fast activity:
        # local conduction/STP changes must not instantly affect distant sites.
        base_speed = (self.speed_reference * self.log_speed(material).exp())[None]
        structural = self.raw_transport_factor(material, base_speed)
        return factor * self.transport_budget.allocation_scale(structural)

    def raw_transport_factor(self, material: torch.Tensor, speed: torch.Tensor):
        """Unconstrained factor, before the optional structural allocation scale."""
        zero = torch.zeros_like(speed[..., 0])
        one = torch.ones_like(zero)
        shear = (torch.zeros_like(material[..., :3]) if self.transport_shear is None
                 else self.transport_shear(material))
        shear = shear.expand(*speed.shape[:-1], 3)
        triangular = torch.stack((one, zero, zero,
                                  shear[..., 0], one, zero,
                                  shear[..., 1], shear[..., 2], one), -1)
        return speed[..., :, None] * triangular.reshape(*speed.shape[:-1], 3, 3)

    def current_transport_factor(self, state: MediumState,
                                 prepared: EvolutionCoefficients | None = None):
        """Actual factor including persistent conduction, STP and resource budget."""
        prepared = self.prepare_evolution() if prepared is None else prepared
        if prepared.structural_factor is not None:
            return self.utilized_structural_factor(state, prepared.structural_factor)
        speed = self.speed_reference * self.edge_log_speeds(
            state, prepared.material, prepared.baseline_log_speed).exp()
        if self.short_term_plasticity is not None:
            speed = speed * self.short_term_plasticity.transmission_gain(
                state.transmission, prepared.short_term)
        return self.transport_factor(prepared.material, speed)

    def utilized_structural_factor(self, state, installed):
        """Fast local utilization is bounded by the paid installed row capacity."""
        utilization = torch.ones_like(installed[..., 0])
        if state.conduction is not None:
            utilization = utilization * state.conduction.sigmoid()
        if state.transmission is not None:
            utilization = utilization * state.transmission[..., 0] * state.transmission[..., 1]
        return installed[None] * utilization[..., :, None]

    def transport_energy_current(self, state: MediumState, factor=None):
        """Oriented edge energy current [B,X,Y,Z,3] of the transport generator.

        Storage is assigned to each edge's origin, as in energy(). Hence outgoing
        J_a(i)=<f(i+e_a),sum_k B_ak(i)q_k(i)>. Its discrete negative divergence
        equals the derivative of local field-plus-store energy under transport.
        This is energy transport, not a decoded semantic or mass-flow velocity.
        """
        factor = self.current_transport_factor(state) if factor is None else factor
        return torch.stack([
            (torch.roll(state.field, -1, a + 1) * sum(
                factor[..., a, k, None] * state.flux[k] for k in range(3))).sum(-1)
            for a in range(3)], -1)

    def _tensor_transport(self, state: MediumState, dt: torch.Tensor,
                          factor: torch.Tensor) -> MediumState:
        """Exact skew edge flows, Lie split; conserves field sum and L2 energy."""
        field, flux = state.field, list(state.flux)
        for a in range(3):
            axis = a + 1
            row = factor[..., a, :]
            norm = row.norm(dim=-1, keepdim=True)
            direction = row / norm.clamp_min(torch.finfo(row.dtype).tiny)
            for colour in range(2):
                f = torch.roll(field, -colour, axis) if colour else field
                stores = [torch.roll(u, -colour, axis) if colour else u for u in flux]
                n = torch.roll(direction, -colour, axis) if colour else direction
                c = torch.roll(norm, -colour, axis) if colour else norm
                n = self._lanes(n, axis, 0)
                old = [self._lanes(u, axis, 0) for u in stores]
                projected = sum(n[..., k, None] * old[k] for k in range(3))
                left, right = self._lanes(f, axis, 0), self._lanes(f, axis, 1)
                difference, mean = (left - right) / math.sqrt(2.), (left + right) / 2
                angle = math.sqrt(2.) * self.shape[a] * self._lanes(c, axis, 0) * dt
                cosine, sine = angle.cos(), angle.sin()
                next_difference = cosine * difference - sine * projected
                delta = sine * difference + (cosine - 1) * projected
                f = self._interleave(mean + next_difference / math.sqrt(2.),
                                     mean - next_difference / math.sqrt(2.), axis)
                field = torch.roll(f, colour, axis) if colour else f
                for k in range(3):
                    u = self._interleave(old[k] + n[..., k, None] * delta,
                                         self._lanes(stores[k], axis, 1), axis)
                    flux[k] = torch.roll(u, colour, axis) if colour else u
        return replace(state, field=field, flux=tuple(flux))

    def transport(self, state: MediumState, duration: float | torch.Tensor,
                  material: torch.Tensor | None = None,
                  baseline_log_speed: torch.Tensor | None = None,
                  short_term_coefficients: tuple[torch.Tensor, ...] | None = None,
                  structural_factor: torch.Tensor | None = None) -> MediumState:
        """Six static edge colours, each an exact conservative three-way rotation.

        For i->j and its stored edge response u:
            df_i=-c*u/h, df_j=c*u/h, du=c*(f_i-f_j)/h.
        Pair sum is fixed; (difference/sqrt(2), u) rotates. No all-pairs matrix.
        The product is a splitting approximation, not an exact full PDE flow.
        """
        material = self.material_field() if material is None else material
        if self.structural_posterior is not None:
            if structural_factor is None:
                structural_factor = self.prepare_evolution().structural_factor
            return self._tensor_transport(state, self._duration(duration, state.field),
                self.utilized_structural_factor(state, structural_factor))
        log_speed = self.edge_log_speeds(state, material, baseline_log_speed)
        gain = None
        if self.short_term_plasticity is not None:
            if state.transmission is None:
                raise ValueError('STP transport requires persisted transmission state')
            short_term_coefficients = (self.short_term_plasticity.coefficients(material)
                                       if short_term_coefficients is None else short_term_coefficients)
            gain = self.short_term_plasticity.transmission_gain(state.transmission, short_term_coefficients)
        dt = self._duration(duration, state.field)
        if self.anisotropic_transport:
            speed = self.speed_reference * log_speed.exp()
            if gain is not None:
                speed = speed * gain
            return self._tensor_transport(state, dt, self.transport_factor(material, speed))
        field, flux = state.field, list(state.flux)
        for spatial_axis in range(3):
            axis = spatial_axis + 1
            # One stored edge property is used with opposite signs at its ends.
            # Endpoint averaging would erase checkerboard conductance modes.
            endpoint_log = log_speed[..., spatial_axis]
            edge_speed = self.speed_reference * torch.exp(endpoint_log)
            if gain is not None:
                edge_speed = edge_speed * gain[..., spatial_axis]
            edge_speed = edge_speed[..., None]
            for colour in range(2):
                f = torch.roll(field, -colour, axis) if colour else field
                u = torch.roll(flux[spatial_axis], -colour, axis) if colour else flux[spatial_axis]
                c = torch.roll(edge_speed, -colour, axis) if colour else edge_speed
                left, right = self._lanes(f, axis, 0), self._lanes(f, axis, 1)
                old_edge = self._lanes(u, axis, 0)
                angle = math.sqrt(2.0) * self.shape[spatial_axis] * self._lanes(c, axis, 0) * dt
                difference = (left - right) / math.sqrt(2.0)
                mean = 0.5 * (left + right)
                difference_next = angle.cos() * difference - angle.sin() * old_edge
                edge_next = angle.sin() * difference + angle.cos() * old_edge
                f = self._interleave(mean + difference_next / math.sqrt(2.0),
                                     mean - difference_next / math.sqrt(2.0), axis)
                u = self._interleave(edge_next, self._lanes(u, axis, 1), axis)
                field = torch.roll(f, colour, axis) if colour else f
                flux[spatial_axis] = torch.roll(u, colour, axis) if colour else u
        return replace(state, field=field, flux=tuple(flux))

    @staticmethod
    def _pack(state: MediumState) -> torch.Tensor:
        return torch.stack((state.field, *state.flux), -2)

    @staticmethod
    def _unpack(value: torch.Tensor, state: MediumState) -> MediumState:
        return replace(state, field=value[..., 0, :], flux=tuple(value[..., k, :] for k in (1, 2, 3)))

    def _features(self, packed: torch.Tensor, material: torch.Tensor,
                  receptors: torch.Tensor | None = None) -> torch.Tensor:
        values = packed.flatten(-2)
        local = F.rms_norm(values, (4 * self.channels,))
        parts = (local, material[None].expand(values.shape[0], -1, -1, -1, -1))
        if receptors is not None:
            parts += (receptors.flatten(-2),)
        return torch.cat(parts, -1)

    def _reflect(self, value: torch.Tensor) -> torch.Tensor:
        w = self.mean_reflector.to(value)
        return value - 2.0 * (value * w).sum(-1, keepdim=True) * w

    def collide(self, state: MediumState, duration: float | torch.Tensor,
                material: torch.Tensor | None = None) -> MediumState:
        """Content- and space-conditioned rotations with four local linear invariants.

        The invariant is the channel sum of each field/flux group. This is a
        signed wave-state model; these are NOT a kinetic D3Q8 momentum claim.
        Flux axes participate together, so scattering can change propagation direction.
        """
        material = self.material_field() if material is None else material
        value = self._pack(state)
        rates = self.collision_rate(self._features(value, material, state.receptors)).reshape(
            *state.field.shape[:-1], self.collision_layers, self.free_width // 2)
        dt = self._duration(duration, state.field).unsqueeze(-1)
        angles = rates * dt
        transformed = self._reflect(value)
        fixed = transformed[..., :1]
        free = transformed[..., 1:].flatten(-2)
        for layer in range(self.collision_layers):
            lanes = torch.roll(free, -layer, -1).reshape(*free.shape[:-1], -1, 2)
            left, right = lanes[..., 0], lanes[..., 1]
            angle = angles[..., layer, :]
            lanes = torch.stack((angle.cos() * left - angle.sin() * right,
                                 angle.sin() * left + angle.cos() * right), -1)
            free = torch.roll(lanes.flatten(-2), layer, -1)
        transformed = torch.cat((fixed, free.reshape(*value.shape[:-1], self.channels - 1)), -1)
        return self._unpack(self._reflect(transformed), state)

    def dissipate(self, state: MediumState, duration: float | torch.Tensor,
                  material: torch.Tensor | None = None, *,
                  account_energy: bool = True) -> tuple[MediumState, torch.Tensor | None]:
        """Positive directional rates with quadratic occupancy feedback.

        Frozen-rate rational update; exact radial quadratic flow when rates are
        equal. Directional state-dependent rates require integration refinement.
        Returns released energy [B], not a memory-importance proof.
        """
        if self.bath_rate is None:
            raise ValueError('Conductance response requires its source-work/Joule ledger; use respond()')
        material = self.material_field() if material is None else material
        value = self._pack(state)
        rates = F.softplus(self.bath_rate(self._features(value, material)) + self.bath_bias)
        rates = rates.reshape_as(value)
        occupancy = value.square().mean((-2, -1), keepdim=True)
        dt = self._duration(duration, state.field).unsqueeze(-1)
        output = self._unpack(value * torch.rsqrt(1.0 + 2.0 * dt * occupancy * rates), state)
        return output, (self.energy(state) - self.energy(output) if account_energy else None)

    def field_rhs(self, state: MediumState, *,
                  prepared: EvolutionCoefficients | None = None) -> torch.Tensor:
        """Analytic d(field)/d(time) at the current state; no state advance.

        This is the zero-duration derivative of the native splitting flow.
        The measurement at a site uses local state and its incident edge stores.
        Changing conduction, receptors or STP during the step contributes only
        at second order to the field. All parameter/state gradients stay live.
        """
        prepared = self.prepare_evolution() if prepared is None else prepared
        material = prepared.material
        transport_rhs = torch.zeros_like(state.field)
        factor = self.current_transport_factor(state, prepared) if self.anisotropic_transport else None
        if factor is None:
            speed = self.speed_reference * self.edge_log_speeds(
                state, material, prepared.baseline_log_speed).exp()
            if self.short_term_plasticity is not None:
                if state.transmission is None:
                    raise ValueError('Dynamic read requires persisted STP state')
                speed = speed * self.short_term_plasticity.transmission_gain(
                    state.transmission, prepared.short_term)
        for axis, flux in enumerate(state.flux):
            current = (speed[..., axis, None] * flux if factor is None else
                       sum(factor[..., axis, k, None] * state.flux[k] for k in range(3)))
            transport_rhs = transport_rhs + self.shape[axis] * (
                torch.roll(current, 1, axis + 1) - current)
        packed = self._pack(state)
        rates = self.collision_rate(self._features(packed, material, state.receptors)).reshape(
            *state.field.shape[:-1], self.collision_layers, self.free_width // 2)
        transformed = self._reflect(packed)
        free = transformed[..., 1:].flatten(-2)
        derivative = torch.zeros_like(free)
        # Every infinitesimal generator acts on the SAME initial vector.
        for layer in range(self.collision_layers):
            lanes = torch.roll(free, -layer, -1).reshape(*free.shape[:-1], -1, 2)
            rate = rates[..., layer, :]
            change = torch.stack((-rate * lanes[..., 1], rate * lanes[..., 0]), -1)
            derivative = derivative + torch.roll(change.flatten(-2), layer, -1)
        collision_rhs = self._reflect(torch.cat((
            torch.zeros_like(transformed[..., :1]),
            derivative.reshape(*packed.shape[:-1], self.channels - 1)), -1))[..., 0, :]
        if self.conductance_response is not None:
            if state.receptors is None:
                raise ValueError('Dynamic read requires persisted receptor state')
            response_rhs = self.conductance_response.field_rhs(
                state.field, state.receptors, prepared.response)
        else:
            rates = F.softplus(self.bath_rate(self._features(packed, material)) + self.bath_bias)
            rates = rates.reshape_as(packed)
            occupancy = packed.square().mean((-2, -1), keepdim=True)
            response_rhs = (-occupancy * rates * packed)[..., 0, :]
        return transport_rhs + collision_rhs + response_rhs

    def respond(self, state: MediumState, duration: float | torch.Tensor, *,
                coefficients: ConductanceCoefficients | None = None,
                diagnostics: bool = True):
        if self.conductance_response is None or state.receptors is None:
            raise ValueError('Conductance model and persisted receptor state required')
        if state.receptors.shape != (*state.field.shape[:-1], 2, self.channels):
            raise ValueError('Receptor state shape mismatch')
        if coefficients is None:
            coefficients = self.conductance_response.coefficients(self.material_field())
        field, flux, receptors, info = self.conductance_response(
            state.field, state.flux, state.receptors, self._duration(duration, state.field),
            coefficients, diagnostics=diagnostics)
        return replace(state, field=field, flux=flux, receptors=receptors), info

    def adapt_transmission(self, state, duration, coefficients):
        if self.short_term_plasticity is None:
            return state
        updated = self.short_term_plasticity(state.transmission, state.field, state.flux,
                                             duration, coefficients)
        return replace(state, transmission=updated)

    def validate_transmission(self, state):
        if (state.transmission is None) != (self.short_term_plasticity is None):
            raise ValueError('STP continuation state must match architecture')
        if state.transmission is not None:
            if state.transmission.shape != (*state.field.shape[:-1], 3, 2):
                raise ValueError('STP continuation shape mismatch')
            torch._assert_async((torch.isfinite(state.transmission) & (state.transmission >= 0)
                                 & (state.transmission <= 1)).all(), 'Invalid STP fractions')

    def advance(self, state: MediumState, duration: float | torch.Tensor, *, substeps: int = 1,
                transport: bool = True, collision: bool = True,
                bath: bool = True, prepared: EvolutionCoefficients | None = None,
                diagnostics: bool = True,
                activation_checkpointing: bool = False) -> tuple[MediumState, dict[str, torch.Tensor]]:
        """Execute the same flow, optionally fusing quiet CUDA FP32 evolution.

        The native path remains the numerical/diagnostic reference. AOT
        autograd owns the complete fused backward, including all material,
        rate-network and persistent-state derivatives. No state is detached.
        """
        use_fused = (self.execution_backend == 'fused' and state.field.is_cuda
                     and state.field.dtype == torch.float32
                     and not torch.compiler.is_compiling())
        if use_fused:
            # AOT's reverse-mode wrapper does not implement forward-mode AD.
            # Fourth-pillar JVPs use the identical native physical operator.
            from torch.autograd.forward_ad import unpack_dual
            leaves = (state.field, *state.flux, state.conduction, state.receptors,
                      state.transmission, duration)
            use_fused = not any(isinstance(x, torch.Tensor) and unpack_dual(x).tangent is not None
                                for x in leaves)
        if not use_fused:
            return self.native_advance(state, duration, substeps=substeps, transport=transport,
                                       collision=collision, bath=bath, prepared=prepared,
                                       diagnostics=diagnostics)
        if prepared is None:
            prepared = self.prepare_evolution()
        if not hasattr(self, '_compiled_advance'):
            import torch._inductor.config as config
            config.compile_threads = 1
            if os.name == 'nt':
                config.use_static_cuda_launcher = False
            self._compiled_advance = torch.compile(self.native_advance, fullgraph=True, dynamic=False)
        return self._fused_substeps(self._compiled_advance, state, duration,
            substeps=substeps, transport=transport, collision=collision, bath=bath,
            prepared=prepared, diagnostics=diagnostics,
            activation_checkpointing=activation_checkpointing)

    def _fused_substeps(self, operation, state, duration, *, substeps=1,
                        transport=True, collision=True, bath=True, prepared=None,
                        diagnostics=True, activation_checkpointing=False):
        """Reuse one compiled physical step, including the real health ledger.

        Unrolling numerical refinement in the compiler duplicates its complete
        forward/backward graph. Here refinement remains a differentiable loop;
        each call executes the identical native subflow, without detached state.
        The clock is assigned from the original FP64 interval once, avoiding
        repeated low-precision duration accumulation.
        """
        if substeps < 1:
            raise ValueError('Positive solver resolution required')
        if substeps == 1 and not activation_checkpointing:
            return operation(state, duration, substeps=1, transport=transport,
                collision=collision, bath=bath, prepared=prepared, diagnostics=diagnostics)
        dt = duration if substeps == 1 else self._duration(duration, state.field).flatten() / substeps
        totals, info, result = {}, {}, state
        summed = {'bath_out_energy', 'response_source_work', 'response_joule_heat',
                  'response_energy_residual'}
        def step(current, step_duration):
            return operation(current, step_duration, substeps=1, transport=transport,
                collision=collision, bath=bath, prepared=prepared, diagnostics=diagnostics)
        for index in range(substeps):
            if activation_checkpointing and torch.is_grad_enabled():
                from .state_checkpoint import checkpoint_state
                result, part = checkpoint_state(step, result, dt)
            else:
                result, part = step(result, dt)
            if diagnostics:
                for key, value in part.items():
                    if key in summed or key.endswith('_change'):
                        totals[key] = totals.get(key, 0) + value
                    elif key != 'energy_before' or index == 0:
                        info[key] = value
        elapsed = state.elapsed + torch.as_tensor(duration, device=state.elapsed.device,
                                                   dtype=state.elapsed.dtype).flatten()
        result = replace(result, elapsed=elapsed)
        info.update(totals)
        return result, info

    def native_advance(self, state: MediumState, duration: float | torch.Tensor, *, substeps: int = 1,
                       transport: bool = True, collision: bool = True,
                       bath: bool = True, prepared: EvolutionCoefficients | None = None,
                       diagnostics: bool = True) -> tuple[MediumState, dict[str, torch.Tensor]]:
        """Continue with no observation; reads/writes need not wait for convergence.

        The caller chooses real input/output timestamps. ``substeps`` is solely
        numerical resolution. Tensor durations must be nonnegative finite [B].
        """
        if substeps < 1:
            raise ValueError("Positive solver resolution required")
        expected = (state.field.shape[0], *self.shape, self.channels)
        if state.field.shape != expected or any(x.shape != expected for x in state.flux):
            raise ValueError("State does not match this grid and content width")
        if (state.receptors is None) != (self.conductance_response is None):
            raise ValueError('Receptor continuation state must match the selected architecture')
        self.validate_transmission(state)
        dt = self._duration(duration, state.field) / substeps
        prepared = self.prepare_evolution() if prepared is None else prepared
        material, coefficients = prepared.material, prepared.plasticity
        # Validate even at zero duration rather than silently dropping learned structure.
        self.edge_log_speeds(state, material, prepared.baseline_log_speed)
        energy_before = self.energy(state) if diagnostics else None
        released = torch.zeros_like(energy_before) if diagnostics else None
        source_work = torch.zeros_like(energy_before) if diagnostics else None
        stage_changes = {}
        if diagnostics:
            stage_changes = {f'{stage}_{quantity}_change': torch.zeros_like(energy_before)
                             for stage in ('transport', 'collision', 'bath')
                             for quantity in ('energy', 'spatial_energy')}
        def stage_measure(value):
            dc = 0.5 * sum(x.mean((1, 2, 3)).square().sum(-1)
                           for x in (value.field, *value.flux))
            total = self.energy(value)
            return total, total - dc
        def record_stage(stage, before, after):
            for quantity, left, right in zip(('energy', 'spatial_energy'), before, stage_measure(after)):
                key = f'{stage}_{quantity}_change'
                stage_changes[key] = stage_changes[key] + right - left
        result = state
        for _ in range(substeps):
            result = self.adapt_transmission(result, 0.5 * dt, prepared.short_term)
            result = self.adapt_conduction(result, 0.5 * dt, material, coefficients)
            if transport:
                before_stage = stage_measure(result) if diagnostics else None
                result = self.transport(result, dt, material, prepared.baseline_log_speed,
                                        prepared.short_term, prepared.structural_factor)
                if diagnostics:
                    record_stage('transport', before_stage, result)
            if collision:
                before_stage = stage_measure(result) if diagnostics else None
                result = self.collide(result, dt, material)
                if diagnostics:
                    record_stage('collision', before_stage, result)
            if bath:
                before_stage = stage_measure(result) if diagnostics else None
                if self.conductance_response is not None:
                    result, response_info = self.respond(
                        result, dt, coefficients=prepared.response, diagnostics=diagnostics)
                    if diagnostics:
                        released = released + response_info['response_joule_heat']
                        source_work = source_work + response_info['response_source_work']
                else:
                    result, out = self.dissipate(result, dt, material, account_energy=diagnostics)
                    if diagnostics:
                        released = released + out
                if diagnostics:
                    record_stage('bath', before_stage, result)
            result = self.adapt_conduction(result, 0.5 * dt, material, coefficients)
            result = self.adapt_transmission(result, 0.5 * dt, prepared.short_term)
        elapsed = state.elapsed + torch.as_tensor(
            duration, device=state.elapsed.device, dtype=state.elapsed.dtype).flatten()
        result = replace(result, elapsed=elapsed)
        if not diagnostics:
            return result, {}
        diagnostics = {"energy_before": energy_before, "energy_after": self.energy(result),
                        "bath_out_energy": released,
                        "material_spatial_variance": material.var((0, 1, 2), unbiased=False).mean()}
        diagnostics.update(stage_changes)
        if result.conduction is not None:
            diagnostics.update({"conduction_offset_rms": result.conduction.square().mean().sqrt(),
                                "conduction_offset_max": result.conduction.abs().amax(),
                                "conduction_spatial_variance": result.conduction.var(
                                    (1, 2, 3), unbiased=False).mean()})
        if result.receptors is not None:
            diagnostics.update({'response_source_work': source_work,
                                'response_joule_heat': released,
                                'response_energy_residual': self.energy(result) - energy_before - source_work + released,
                                'receptor_open_mean': result.receptors.mean((0, 1, 2, 3, 5))})
        if result.transmission is not None:
            gain = self.short_term_plasticity.transmission_gain(result.transmission, prepared.short_term)
            diagnostics.update({'stp_resource_mean': result.transmission[..., 0].mean(),
                                'stp_utilization_mean': result.transmission[..., 1].mean(),
                                'stp_gain_mean': gain.mean(), 'stp_gain_min': gain.amin(),
                                'stp_gain_max': gain.amax(),
                                'stp_spatial_variance': gain.var((1, 2, 3), unbiased=False).mean()})
        return result, diagnostics

    def forward(self, state: MediumState, duration: float | torch.Tensor, *, substeps: int = 1):
        """Module entry point for full-graph compilation of the actual update."""
        return self.advance(state, duration, substeps=substeps)


def boundary_scatter(state: MediumState, packet: torch.Tensor,
                     angle: torch.Tensor) -> tuple[MediumState, torch.Tensor]:
    """Observation-independent port law; the write agent supplies packet/angle.

    Field plus outgoing packet closes the incoming energy ledger. Signals already
    on edges and physical elapsed time are retained through the boundary event.
    """
    if packet.shape != state.field.shape:
        raise ValueError("Incoming packet must have the field shape")
    field = angle.cos() * state.field + angle.sin() * packet
    outgoing = -angle.sin() * state.field + angle.cos() * packet
    return state.with_field(field), outgoing
