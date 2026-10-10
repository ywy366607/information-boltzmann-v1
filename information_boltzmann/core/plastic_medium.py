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
    junction_material_logits: torch.Tensor | None = None
    junction_route_mask: torch.Tensor | None = None


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


# Named collision operators (opt-in; the unnamed default stays the legacy MLP so existing individuals load unchanged).
#   boltzmann      pair collisions: rate = timescale_k * kappa_k * bilinear coincidence of the head (0 for vacuum / one packet),
#                  position-free, bounded, plus the three cross-group layers (field<->flux, flux<->flux).
#   boltzmann-mlp  the same guard rails around a learned MLP operator over the pair's Gram invariants (switch for long runs).
COLLISION_PRESETS = {
    'boltzmann': dict(timescale=True, heads=8, kerr_std=3.0, coincidence=True, cross_groups=True),
    'boltzmann-mlp': dict(timescale=True, gram_input=8, pair_gate=True, cross_groups=True, mod_gain=4.0),
}


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
                 structure_options: dict | None = None,
                 hopf_recomposition: bool = False,
                 collision_options: dict | None = None, plasticity_options: dict | None = None):
        super().__init__()
        collision_options = dict(collision_options or {})
        unknown = set(collision_options) - {'center_receptors', 'position_rates', 'bed_rates', 'kerr_rates', 'kerr_only',
                                            'heads', 'rate_scale', 'kerr_std', 'timescale', 'phi_min', 'phi_max', 'amplitude_input', 'mod_gain', 'mod_bound', 'gram_input', 'coincidence', 'cross_groups', 'pair_gate', 'gravity', 'gravity_std', 'cell_gain', 'cell_scale'}
        if unknown:
            raise ValueError(f'Unknown collision options {sorted(unknown)}')
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
        # cross_groups: three extra rotation layers that pair the four state groups (field, flux x/y/z) at the SAME channel
        # (the three perfect matchings of the 4 groups), so scattering can turn field into flux and redirect propagation.
        # The in-group layers pair only adjacent channels of one group (at most ~4 of the pairs ever straddle two groups).
        self.collision_inner_layers = int(collision_layers)
        self.collision_cross_groups = bool(collision_options.get('cross_groups', False))
        # Scattering cross-section field sigma_s(x): how strongly collisions act at each site (the local tissue), while the
        # collision rule itself stays local, instantaneous and pair-based. Conserved resource: sigma = S * softmax(scale * theta)
        # has lattice mean exactly 1, so sites compete (hubs vs conduits). Birth: a smooth random field spanning
        # [1/cell_gain, cell_gain]. scale makes one Adam step (~lr) move log sigma by scale*lr (function-space calibration).
        self.cell_theta = None
        if collision_options.get('cell_gain'):
            spread = float(collision_options['cell_gain'])
            if not math.isfinite(spread) or spread <= 1:
                raise ValueError('cell_gain is the birth spread of sigma and must exceed 1')
            self.cell_scale = float(collision_options.get('cell_scale', 50.0))
            n = tuple(int(x) for x in shape)
            noise = torch.randn(n, dtype=torch.float64)
            spectrum = torch.fft.fftn(noise)
            k = torch.stack(torch.meshgrid(*[torch.fft.fftfreq(m) * m for m in n], indexing='ij')).square().sum(0).sqrt()
            smooth = torch.fft.ifftn(spectrum * (k <= 2)).real
            smooth = smooth - smooth.mean()
            smooth = smooth / smooth.abs().max().clamp_min(1e-12) * math.log(spread)
            self.cell_theta = nn.Parameter((smooth / self.cell_scale).float())
        self.collision_layers = int(collision_layers) + (3 if self.collision_cross_groups else 0)
        if self.collision_cross_groups:     # device index buffers: host index lists would copy H2D inside a CUDA-graph capture
            matchings = self._MATCHINGS
            self.register_buffer('collision_cross_left', torch.tensor([[a, c] for (a, _), (c, _) in matchings]), persistent=False)
            self.register_buffer('collision_cross_right', torch.tensor([[b, d] for (_, b), (_, d) in matchings]), persistent=False)
            self.register_buffer('collision_cross_inverse', torch.stack([torch.argsort(torch.tensor([a, c, b, d]))
                                                                         for (a, b), (c, d) in matchings]), persistent=False)
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
        self.hopf_pathway = None
        if hopf_recomposition:
            if self.structural_posterior is None:
                raise ValueError('Physical junctions require explicit installed structural capacity')
            from .medium_junction import LocalFluxJunction
            self.hopf_pathway = LocalFluxJunction(material_width, self.material_reference_shape)
            self.register_load_state_dict_pre_hook(self._validate_junction_load)
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
        # amplitude_input: the state enters the rate MLP rms-normalised (direction only), which erases amplitude. This adds
        # the amplitude back as its own input: per-head log energy relative to the lattice mean (a site contrast, stateless
        # and scale-free), so amplitude-dependent (Kerr-type) scattering is expressible by the MLP.
        amp = collision_options.get('amplitude_input', False)
        self.collision_amplitude_heads = 0 if not amp else (8 if amp is True else int(amp))
        if self.collision_amplitude_heads and channels % self.collision_amplitude_heads:
            raise ValueError('Amplitude heads must divide the channel count')
        # gram_input: scattering depends only on the colliding wave packets, not on where they are. Per head the four channel
        # vectors (field, 3 fluxes) have a 4x4 Gram matrix: invariant under any orthogonal change of channel basis, it holds
        # every relative amplitude/direction/overlap of the pair. Inputs are the 10 trace-normalised entries plus the log
        # energy log(1 + E/T_h) with a learnable local reference T_h, per head. No material code, no receptor gates, no raw state.
        gram = collision_options.get('gram_input', False)
        # coincidence: Boltzmann-type pair collisions. No static background rotation: the angle is proportional to the
        # bilinear coupling of the two partners, (|field|^2 - sum_a |flux_a|^2) per head = 2..4 f1.f2 for two packets and
        # exactly 0 for vacuum or a single travelling wave, saturated by 1/(E + T) so it grows ~amplitude^2 when small and
        # cannot blow up. rate_k = exp(log_phi_k)/T_x * kappa_k * coincidence_head(k). Position-free by construction.
        self.collision_coincidence = bool(collision_options.get('coincidence', False))
        if self.collision_coincidence and not (collision_options.get('timescale', False) and 'heads' in collision_options):
            raise ValueError('coincidence collisions need timescale and heads')
        self.collision_gram_heads = 0 if not gram else (8 if gram is True else int(gram))
        if self.collision_gram_heads and (channels % self.collision_gram_heads or amp):
            raise ValueError('Gram heads must divide the channel count and replace the amplitude input')
        if self.collision_gram_heads:      # per-head reference energy of the amplitude input (local, learnable)
            self.collision_gram_energy = nn.Parameter(torch.full(
                (self.collision_gram_heads,), math.log(0.04 * channels / self.collision_gram_heads)))
            rows, cols = torch.triu_indices(4, 4)
            self.register_buffer('collision_gram_rows', rows, persistent=False)
            self.register_buffer('collision_gram_cols', cols, persistent=False)
        self.collision_mod_gain = float(collision_options.get('mod_gain', 1.0))
        # The relative modulation is bounded: rate stays within exp(+-mod_bound) of its timescale (an unbounded exp(g) with
        # a large gain sent single rates to ~1e6 rad/time and overflowed the structural credit in the first window).
        self.collision_mod_bound = float(collision_options.get('mod_bound', 2.0))
        if not self.collision_mod_bound > 0:
            raise ValueError('mod_bound must be positive')
        self.collision_rate = nn.Sequential(
            nn.Linear(11 * self.collision_gram_heads if self.collision_gram_heads else
                      (6 if bath_type == 'conductance' else 4) * channels + material_width
                      + self.collision_amplitude_heads, hidden), nn.SiLU(),
            nn.Linear(hidden, self.collision_layers * (self.free_width // 2)))
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
        # Opt-in collision activity (default off: legacy keys, arithmetic and RNG stream unchanged).
        #   center_receptors: remove the per-site channel mean of the 2C receptor gates before the rate MLP.
        #     Those gates are ~constant and uncentred, so Adam moves all 2C of their first-layer weights
        #     together and they act as a ~2C-times-faster bias that pins the hidden units.
        #   position_rates / bed_rates: zero-initialised additive rate maps from the material code / from the
        #     installed capacity allocation, so position and the bed can steer scattering directly.
        # New parameters are plain zeros (no RNG draw) and are created last, so every legacy tensor of a
        # control model with the same seed is bitwise identical.
        self.collision_center_receptors = bool(collision_options.get('center_receptors', False))
        rate_width = self.collision_layers * (self.free_width // 2)
        self.collision_position = (nn.Parameter(torch.zeros(rate_width, material_width))
                                   if collision_options.get('position_rates', False) else None)
        # kerr_rates: amplitude-aware, bounded drives of the rotation angle. total/(total+E0) is Kerr self-modulation;
        # (field^2 - flux^2)/(total+E0) vanishes for any single travelling wave and equals 4 F1 F2 for two counter-
        # propagating ones, i.e. it is the pure wave-wave coincidence signal (their cross terms cancel in total energy).
        self.collision_kerr = self.collision_kerr_energy = None
        if collision_options.get('kerr_rates', False):
            self.collision_kerr = nn.Parameter(torch.zeros(2, rate_width))
            self.collision_kerr_energy = nn.Parameter(torch.tensor(math.log(0.04 * channels)))
        # kerr_only: no MLP. rate = s * (base_pair + sum_k kappa_{k,pair} * drive_k[head(pair)]) with H heads over the
        # channels; drive_k use each head's own channel slice (content), and the head temperature
        # E_h(x) = exp(a_h + w_h . material(x)) is per head AND per point (position). Everything starts non-zero
        # (rates ~ s, i.e. O(1) rad per event) so the nonlinearity is alive from birth instead of identity-zero.
        self.collision_kerr_heads = 0
        kerr_only = collision_options.get('kerr_only', False)
        timescale = collision_options.get('timescale', False)
        if kerr_only and timescale:
            raise ValueError('timescale already fixes the rate scale; combine it with heads, not kerr_only')
        if kerr_only or (timescale and 'heads' in collision_options):
            heads = int(collision_options.get('heads', 8))
            if channels % heads:
                raise ValueError('Heads must divide the channel count')
            scale0 = float(collision_options.get('rate_scale', 10.0))
            kstd = float(collision_options.get('kerr_std', 1.0))
            self.collision_kerr_heads = heads
            if kerr_only:
                self.collision_scale = nn.Parameter(torch.tensor(math.log(scale0)))
                self.collision_base = nn.Parameter(torch.randn(rate_width))
            self.collision_kerr = nn.Parameter(kstd * torch.randn(2, rate_width))
            self.collision_head_energy = nn.Parameter(torch.full((heads,), math.log(0.04 * channels / heads)))
            self.collision_head_position = nn.Parameter(0.3 * torch.randn(heads, material_width))
            pairs = self.free_width // 2
            flat = torch.arange(rate_width)
            layer, pair = flat // pairs, flat % pairs
            coordinate = (2 * pair + layer) % self.free_width
            channel = torch.where(layer < self.collision_inner_layers, coordinate % (channels - 1), pair % (channels - 1))
            self.register_buffer('collision_pair_head', (channel * heads // (channels - 1)).clamp_max(heads - 1),
                                 persistent=False)
        # timescale: scale-free rates. Every rotation pair k gets its own timescale, log-uniform over
        # Phi in [phi_min, phi_max] radians per domain-crossing time T_x = 1/speed_reference (the torus has unit length),
        # with a random sign; the state dependence is a relative (log) modulation g, so
        #   rate_k = sign_k * exp(log_phi_k + g_k(state)) / T_x,    angle per event = rate_k * dt.
        # Nothing here depends on channel count, lattice size, hidden width or the event length: the pairs are drawn
        # independently, g starts O(0.2-0.4) from variance-preserving inits, and a parameter step changes the angle by a
        # fraction of itself instead of a fraction of the tiny absolute rate. g comes from the rate MLP, or from the
        # per-head energy drives when 'heads' is given.
        self.collision_log_phi = None
        if timescale:
            low, high = math.log(float(collision_options.get('phi_min', 0.3))), math.log(float(collision_options.get('phi_max', 10.0)))
            if not low < high:
                raise ValueError('phi_min must be below phi_max')
            self.collision_log_phi = nn.Parameter(low + (high - low) * torch.rand(rate_width))
            self.register_buffer('collision_sign', torch.where(torch.rand(rate_width) < 0.5, -1.0, 1.0))
            self.collision_crossing_time = 1.0 / self.speed_reference
        # pair_gate: the repaired MLP collision operator. The MLP (fed only the pair's Gram invariants) sets each pair's
        # bounded relative modulation; the whole rate is multiplied by the bilinear coincidence of its head, so no MLP
        # output can create a collision when fewer than two packets are present.
        self.collision_pair_gate = bool(collision_options.get('pair_gate', False))
        if self.collision_pair_gate:
            if not (timescale and self.collision_gram_heads) or 'heads' in collision_options:
                raise ValueError('pair_gate needs timescale and gram_input (and not the kerr heads)')
            gram_heads = self.collision_gram_heads
            pairs = self.free_width // 2
            flat = torch.arange(rate_width)
            layer, pair = flat // pairs, flat % pairs
            coordinate = (2 * pair + layer) % self.free_width
            channel = torch.where(layer < self.collision_inner_layers, coordinate % (channels - 1), pair % (channels - 1))
            self.register_buffer('collision_pair_head', (channel * gram_heads // (channels - 1)).clamp_max(gram_heads - 1),
                                 persistent=False)
        # gravity: a long-range force on the waves. The per-head energy fraction e_h = E_h/(E_h+T_h) is the mass density;
        # its periodic Poisson potential gives a force F_a = -d_a Phi at every site from ALL the energy in the lattice.
        # The force enters as an extra rotation angle of the (field, flux_a) pair of each channel -- exactly the cross-group
        # plane that turns a packet toward axis a -- so waves are deflected toward (or away from) energy concentrations without
        # any local partner. Orthogonal, vacuum -> 0, translation-covariant, G=0 identity. Real (non-FFT) Green matrices:
        # compile-friendly on small lattices (S x S per axis).
        self.collision_gravity = bool(collision_options.get('gravity', False))
        if self.collision_gravity:
            if not (self.collision_coincidence and self.collision_cross_groups):
                raise ValueError('gravity needs coincidence and cross_groups (it rotates the field-flux planes)')
            self.register_buffer('collision_green', self._green_gradients(tuple(self.shape)), persistent=False)
            self.register_buffer('collision_channel_head', (torch.arange(channels - 1) * self.collision_kerr_heads // (channels - 1)).clamp_max(self.collision_kerr_heads - 1), persistent=False)
            self.collision_gravity_gain = nn.Parameter(
                float(collision_options.get('gravity_std', 1.0)) * torch.randn(3, channels - 1))
        self.collision_bed = None
        if collision_options.get('bed_rates', False):
            if self.structural_posterior is None:
                raise ValueError('bed_rates require an installed structural capacity allocation')
            self.collision_bed = nn.Parameter(torch.zeros(rate_width, 4))
        # plasticity_options (opt-in): a log-uniform SPECTRUM of time constants for the use-dependent fast layers, plus an
        # initial-speed compensation so that switching them on does not change the nominal propagation speed.
        #   spectrum    (tau_min, tau_max) in physical time: ln(tau) of every site/axis starts ~ uniform over the range (Mamba-dt
        #               style; language statistics are scale-free, so equal value per octave of lag) instead of a delta at the
        #               reference time. Applied to the conduction law's rate and the STP recovery rate.
        #   depression  STP activity rate (default 0.1): how hard traffic depresses an edge per unit time.
        #   compensation  'auto' (default): multiply the transport factor by 1/(sigmoid(p0)*U0) so the initial effective speed is
        #               the nominal one; the fast layers can then throttle (down to 0) or boost (up to that factor).
        plasticity_options = dict(plasticity_options or {})
        unknown_p = set(plasticity_options) - {'spectrum', 'depression', 'compensation'}
        if unknown_p:
            raise ValueError(f'Unknown plasticity options {sorted(unknown_p)}')
        if 'spectrum' in plasticity_options:
            low, high = plasticity_options['spectrum']
            if not 0 < low < high:
                raise ValueError('plasticity spectrum needs 0 < tau_min < tau_max')
        self._plasticity_options = plasticity_options
        # Unified riverbed: absorption rate (1/time) of a pure-soil (idle) site; 0 = idle is inert (legacy).
        self.soil_absorption = 0.0
        self.utilization_compensation = 1.0
        if plasticity_options and plasticity_options.get('compensation', 'auto') == 'auto':
            u0 = 1.0
            if self.conduction_plasticity is not None:
                u0 *= 0.5                                           # sigmoid(0): the conduction offset starts at zero
            if self.short_term_plasticity is not None:
                u0 *= 0.5                                           # U = sigmoid(0) at rest, x = 1
            self.utilization_compensation = 1.0 / u0

    @torch.no_grad()
    def initialize_plasticity_spectrum(self) -> None:
        """Call AFTER the material code is initialised (its scale sets the weight scale). Idempotent per material draw."""
        options = self._plasticity_options
        if not options or (self.conduction_plasticity is None and self.short_term_plasticity is None):
            return
        code = self.material_field()
        code = code.reshape(-1, code.shape[-1])
        rms = float(code.square().sum(-1).mean().sqrt())
        if 'spectrum' in options:
            if rms == 0:
                raise ValueError('Initialise the material code before the plasticity spectrum')
            low, high = options['spectrum']
            mu, sigma = 0.5 * (math.log(low) + math.log(high)), (math.log(high) - math.log(low)) / math.sqrt(12.0)
            if self.conduction_plasticity is not None:
                layer = self.conduction_plasticity
                layer.log_rate.weight.normal_(std=sigma / rms)
                layer.log_rate.bias.fill_(-(mu - math.log(layer.time_reference)))
            if self.short_term_plasticity is not None:
                layer = self.short_term_plasticity
                for axis in range(3):
                    layer.parameters_map.weight[4 * axis].normal_(std=sigma / rms)
                    layer.parameters_map.bias[4 * axis] = -(mu - math.log(layer.time_reference))
        if self.short_term_plasticity is not None:
            depression = float(options.get('depression', 0.1))
            layer = self.short_term_plasticity
            for axis in range(3):
                layer.parameters_map.bias[4 * axis + 2] = math.log(depression) + math.log(layer.time_reference)
                layer.parameters_map.weight[4 * axis + 3].zero_()          # resting utilisation U = sigmoid(0) = 1/2 at every site,
                layer.parameters_map.bias[4 * axis + 3] = 0.0              # so the speed compensation is exact at birth

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
            sample = posterior.sample_coefficients() if posterior.window_is_active() else None
            allocation = posterior.allocation(basis, sample)
            rows = self.raw_transport_factor(material, material.new_ones(*material.shape[:-1], 3))
            directions = F.normalize(rows, dim=-1)
            factor = posterior.speed_reference.to(material) * allocation[..., :3, None] * directions
        junction = (None if self.hopf_pathway is None else
                    self.hopf_pathway.material_logits(material))
        mask = None if self.hopf_pathway is None else self.hopf_pathway.route_mask.clone()
        return EvolutionCoefficients(material, self.log_speed(material), coefficients, response,
                                     short_term, factor, allocation, junction, mask)

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

    def _validate_junction_load(self, module, saved, prefix, *args):
        self.hopf_pathway.validate_checkpoint(saved, prefix + 'hopf_pathway.')

    def current_transport_factor(self, state: MediumState,
                                 prepared: EvolutionCoefficients | None = None):
        """Actual factor including persistent conduction, STP and resource budget."""
        prepared = self.prepare_evolution() if prepared is None else prepared
        if prepared.structural_factor is not None:
            factor = self.utilized_structural_factor(state, prepared.structural_factor)
            if self.hopf_pathway is not None:
                factor = self.hopf_pathway.allocate(state, factor,
                    material_logits=prepared.junction_material_logits,
                    route_mask=prepared.junction_route_mask).factor
            return factor
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
        return installed[None] * (utilization * self.utilization_compensation)[..., :, None]

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
                  structural_factor: torch.Tensor | None = None,
                  junction_material_logits: torch.Tensor | None = None,
                  junction_route_mask: torch.Tensor | None = None,
                  stage_ledger: dict | None = None) -> MediumState:
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
            factor = self.utilized_structural_factor(state, structural_factor)
            dt = self._duration(duration, state.field)
            if self.hopf_pathway is None:
                return self._tensor_transport(state, dt, factor)
            allocation = self.hopf_pathway.allocate(state, factor, material=material,
                                                    material_logits=junction_material_logits,
                                                    route_mask=junction_route_mask)
            def measure(value):
                total = self.energy(value)
                dc = 0.5 * sum(x.mean((1, 2, 3)).square().sum(-1)
                               for x in (value.field, *value.flux))
                return total, total - dc
            before = measure(state) if stage_ledger is not None else None
            routed = self.hopf_pathway.rotate(state, allocation.rates, 0.5 * dt)
            first = measure(routed) if stage_ledger is not None else None
            routed = self._tensor_transport(routed, dt, allocation.factor)
            middle = measure(routed) if stage_ledger is not None else None
            routed = self.hopf_pathway.rotate(routed, allocation.rates, 0.5 * dt)
            if stage_ledger is not None:
                after = measure(routed)
                for quantity, a, b, c, d in zip(('energy', 'spatial_energy'), before, first, middle, after):
                    stage_ledger[f'junction_{quantity}_change'] = (b - a) + (d - c)
            return routed
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

    def _gram_features(self, packed: torch.Tensor) -> torch.Tensor:
        return self._gram_terms(packed)[0]

    def _gram_terms(self, packed: torch.Tensor):
        """(MLP features [..., 11H], bilinear coincidence per head [..., H]) from the per-head Gram matrices."""
        heads = self.collision_gram_heads
        v = packed.reshape(*packed.shape[:-1], heads, self.channels // heads).movedim(-2, -3)     # [B,X,Y,Z,H,4,w]
        gram = v @ v.transpose(-1, -2)                                                          # [B,X,Y,Z,H,4,4]
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)                                          # [B,X,Y,Z,H]
        rows, cols = self.collision_gram_rows, self.collision_gram_cols
        shape = gram[..., rows, cols] / (trace[..., None] + 1e-12)                               # [B,X,Y,Z,H,10]
        # centre on equipartition (G = trace/4 * identity): the inputs are then ~zero-mean instead of near-constant
        shape = shape - (rows == cols).to(shape) * 0.25
        temperature = self.collision_gram_energy.exp().to(trace)
        amplitude = torch.log1p(trace / temperature)                                             # local: 0 in vacuum
        coincidence = (gram[..., 0, 0] - gram[..., 1, 1] - gram[..., 2, 2] - gram[..., 3, 3]) / (trace + temperature)
        return torch.cat((shape, amplitude[..., None]), -1).flatten(-2), coincidence             # [B,X,Y,Z,11H], [B,X,Y,Z,H]

    def _features(self, packed: torch.Tensor, material: torch.Tensor,
                  receptors: torch.Tensor | None = None, *, center_receptors: bool = False) -> torch.Tensor:
        if self.collision_gram_heads:
            return self._gram_features(packed)
        values = packed.flatten(-2)
        local = F.rms_norm(values, (4 * self.channels,))
        parts = (local, material[None].expand(values.shape[0], -1, -1, -1, -1))
        if self.collision_amplitude_heads:
            heads = self.collision_amplitude_heads
            energy = packed.reshape(*packed.shape[:-1], heads, self.channels // heads).square().sum((-3, -1))   # [B,X,Y,Z,H]
            mean = energy.mean((1, 2, 3), keepdim=True)
            parts += ((energy + 1e-12).log() - (mean + 1e-12).log(),)
        if receptors is not None:
            gates = receptors.flatten(-2)
            if center_receptors:
                gates = gates - gates.mean(-1, keepdim=True)
            parts += (gates,)
        return torch.cat(parts, -1)

    @staticmethod
    def _green_gradients(shape) -> torch.Tensor:
        """[3, S, S]: M[a, s, j] = 2*pi * d_a Phi(s) for a unit mass at j, Laplace Phi = delta_j - mean, unit periodic box."""
        S = 1
        for n in shape:
            S *= n
        freqs = [torch.fft.fftfreq(n, d=1.0 / n).double() for n in shape]            # integer wave numbers
        grids = torch.meshgrid(*freqs, indexing='ij')
        k2 = sum((2 * math.pi * g) ** 2 for g in grids)
        inverse = torch.where(k2 > 0, 1.0 / k2.clamp_min(1e-30), torch.zeros_like(k2))
        delta = torch.eye(S, dtype=torch.float64).reshape(S, *shape)
        spectrum = torch.fft.fftn(delta, dim=(1, 2, 3))
        out = []
        for a, (n, g) in enumerate(zip(shape, grids)):
            derivative = (2 * math.pi * g).clone()
            if n % 2 == 0:
                derivative[g.abs() == n // 2] = 0.0                              # no sine-aliased Nyquist derivative
            field = torch.fft.ifftn(1j * derivative * (-inverse) * spectrum, dim=(1, 2, 3)).real      # [S(source), X, Y, Z]
            out.append((2 * math.pi) * field.reshape(S, S).t())                  # [S(site), S(source)]
        return torch.stack(out).to(torch.get_default_dtype())

    @staticmethod
    def _lattice_poisson(shape, screening: float = 0.0) -> torch.Tensor:
        """[S, S] inverse of (7-point periodic lattice Laplacian - screening) on zero-mean sources (unit box, spacing 1/n_a).

        screening = kappa^2 > 0 gives a finite-range (Yukawa) potential: the mode growth factor lambda/(lambda+kappa^2) suppresses
        long wavelengths, which together with a weak diffusion selects a finite structure scale instead of a box-scale monolith."""
        S = 1
        for n in shape:
            S *= n
        freqs = [torch.fft.fftfreq(n, d=1.0 / n).double() for n in shape]
        grids = torch.meshgrid(*freqs, indexing='ij')
        eigen = sum(-2.0 * n * n * (1 - torch.cos(2 * math.pi * g / n)) for n, g in zip(shape, grids))
        mean_mode = eigen == 0
        eigen = eigen - screening * (~mean_mode)                      # (lambda_lattice + kappa^2) on every non-mean mode
        inverse = torch.where(eigen != 0, 1.0 / torch.where(eigen != 0, eigen, torch.ones_like(eigen)), torch.zeros_like(eigen))
        delta = torch.eye(S, dtype=torch.float64).reshape(S, *shape)
        field = torch.fft.ifftn(inverse * torch.fft.fftn(delta, dim=(1, 2, 3)), dim=(1, 2, 3)).real
        return field.reshape(S, S).t()

    def _gravity_coupling(self, packed: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        """Add the long-range force to the (field, flux_a) pairs of the three cross layers.  coupling [..., rate_width]."""
        heads = self.collision_kerr_heads
        width = self.channels // heads
        field = packed[..., 0, :].reshape(*packed.shape[:-2], heads, width).square().sum(-1)
        flux = packed[..., 1:, :].reshape(*packed.shape[:-2], 3, heads, width).square().sum((-3, -1))
        total = field + flux
        density = total / (total + self.collision_head_energy.exp().to(total))          # bounded mass per head, [B,X,Y,Z,H]
        flat = density.flatten(1, 3)                                                      # [B,S,H]
        flat = flat - flat.mean(1, keepdim=True)                                          # neutralising background
        force = torch.stack([self.collision_green[a].to(flat) @ flat for a in range(3)], 1)       # [B,3,S,H]
        force = force.index_select(-1, self.collision_channel_head)                       # [B,3,S,C-1]
        extra = force * self.collision_gravity_gain.to(force)[None, :, None, :]           # [B,3,S,C-1]
        pairs = self.free_width // 2
        per_layer = torch.cat((extra, torch.zeros_like(extra)), -1)                       # pairs j=0 are (field, flux_a)
        inner = extra.new_zeros(extra.shape[0], self.collision_inner_layers, extra.shape[2], pairs)
        layers = torch.cat((inner, per_layer), 1)                                         # [B,L,S,pairs]
        added = layers.permute(0, 2, 1, 3).reshape(*packed.shape[:4], self.collision_layers * pairs)
        return coupling + added

    @torch.no_grad()
    def structural_gravity_step(self, rate: float, screening: float = 0.0, diffusion: float = 0.0) -> dict:
        """One window-boundary step of structural self-gravity on the INSTALLED capacity (slow state, not used conductance).

        Like attracts like: each axis capacity c_a(x) is a species with its own neutral lattice-Poisson potential
        (Laplace Phi_a = c_a - mean) and force F_a = -grad Phi_a.  Species a flows along F_a by a continuity equation with
        receiver-limited (exclusion) flux on the TOTAL fill,
            J^a_face = eta * c_a,donor * (1 - m_receiver / resource_density) * F^a_face,
        so axis-dominated regions grow instead of being averaged away; each species' total is conserved (a global limiter keeps
        every site positive and the total below the cap).  `rate` is the linear e-folding rate PER WINDOW of every species
        density mode at the current mean fill (one constant, no scale: eta = rate / (S * mean(c) * (1 - mean fill)), exact for
        every mode because the discrete Laplacian is used throughout).  Optional scale selection: `screening` (kappa^2, lattice
        Laplacian units) makes the force finite-ranged so the mode growth is rate*lambda/(lambda+kappa^2), and `diffusion` D adds
        -D*lambda: the net linear rate has a maximum at an intermediate wavenumber.  The logit change is projected onto the band-limited chart
        and added to the posterior mean.  rate == 0 returns before touching anything (bitwise identity).
        """
        if rate == 0:
            return {}
        if not math.isfinite(rate) or rate < 0:
            raise ValueError('Finite nonnegative gravity rate required')
        posterior = self.structural_posterior
        if posterior is None:
            raise ValueError('Structural gravity needs the structural posterior')
        shape = tuple(self.shape)
        S = int(math.prod(shape))
        basis = self.material.basis(self.coordinates).to(posterior.mean).double()            # [X,Y,Z,N]
        flat_basis = basis.reshape(S, -1)
        cache = self.__dict__.setdefault('_structural_gravity_cache', {})
        key = (str(flat_basis.device), flat_basis.shape, float(screening))
        if key not in cache:
            cache[key] = (torch.linalg.pinv(flat_basis), self._lattice_poisson(shape, float(screening)).to(flat_basis))
        pinv, poisson = cache[key]
        alloc = posterior.allocation(basis.to(posterior.mean)).double().reshape(S, 4)
        rho = float(posterior.resource_density)
        c = alloc[:, :3].reshape(*shape, 3)                                                   # species densities
        m = c.sum(-1)
        species_mean = c.reshape(S, 3).mean()
        fill = m.mean() / rho
        potential = (poisson @ (c.reshape(S, 3) - c.reshape(S, 3).mean(0, keepdim=True))).reshape(*shape, 3)
        eta = rate / (S * species_mean * (1 - fill))
        flux, forwards = [], []
        for axis in range(3):
            face_force = -S * shape[axis] * (torch.roll(potential, -1, axis) - potential)      # F^a at the face x -> x+e, [X,Y,Z,3]
            forward = face_force > 0
            donor = torch.where(forward, c, torch.roll(c, -1, axis))
            receiver_total = torch.where(forward, torch.roll(m, -1, axis)[..., None], m[..., None])
            q = eta * shape[axis] * donor * (1 - receiver_total / rho).clamp_min(0) * face_force
            if diffusion:                                                                         # linear diffusion of every species
                q = q - diffusion * shape[axis] ** 2 * (torch.roll(c, -1, axis) - c)
                forward = q > 0
            flux.append(q)
            forwards.append(forward)
        out = torch.zeros_like(c)
        for axis in range(3):                                                                  # outflow demanded of every donor
            q = flux[axis]
            out = out + torch.where(forwards[axis], q, torch.zeros_like(q)).abs()
            out = out + torch.roll(torch.where(~forwards[axis], q, torch.zeros_like(q)).abs(), 1, axis)
        limit = torch.minimum(torch.ones_like(c), 0.2 * c / out.clamp_min(1e-300))
        delta = torch.zeros_like(c)
        for axis in range(3):
            scale = torch.where(forwards[axis], limit, torch.roll(limit, -1, axis))            # the donor's limiter
            transfer = flux[axis] * scale
            delta = delta - transfer + torch.roll(transfer, 1, axis)
        new = (c + delta).clamp_min(1e-9 * rho)
        total = new.sum(-1, keepdim=True)
        new = new * torch.minimum(torch.ones_like(total), 0.98 * rho / total)                  # stay below the cap
        idle = (rho - new.sum(-1, keepdim=True)).clamp_min(1e-9 * rho)
        old_idle = alloc[:, 3].reshape(*shape, 1)
        delta_logits = (new.log() - idle.log()) - (c.log() - old_idle.log())                   # [X,Y,Z,3]
        posterior.mean.add_((pinv @ delta_logits.reshape(S, 3)).to(posterior.mean))
        shares_old = c / m[..., None]
        shares_new = new / new.sum(-1, keepdim=True)
        spread = lambda s: (s.max(-1).values - s.min(-1).values).mean()
        return {'mass_change': float((new.sum() - c.sum()) / c.sum()),
                'mass_cv_before': float(m.std() / m.mean()), 'mass_cv_after': float(new.sum(-1).std() / new.sum(-1).mean()),
                'anisotropy_before': float(spread(shares_old)), 'anisotropy_after': float(spread(shares_new)),
                'max_limiter': float(1 - limit.min())}

    def _kerr_drive(self, packed, material, *, pair_only=False):
        """sum_k kappa_k * drive_k[head(pair)] per rotation pair, [..., rate_width]."""
        heads, width = self.collision_kerr_heads, self.channels // self.collision_kerr_heads
        field = packed[..., 0, :].reshape(*packed.shape[:-2], heads, width).square().sum(-1)
        flux = packed[..., 1:, :].reshape(*packed.shape[:-2][:], 3, heads, width).square().sum((-3, -1))
        total = field + flux
        # [B,X,Y,Z,H] temperature = per-head baseline times a position-dependent factor from the material code
        log_temperature = self.collision_head_energy + (0.0 if pair_only else F.linear(material, self.collision_head_position))
        temperature = (log_temperature if pair_only else log_temperature[None]).exp()
        scale = total + temperature.to(total)
        if pair_only:       # only the bilinear coincidence column: no self-energy drive, no stack, one gather
            coincidence = ((field - flux) / scale).index_select(-1, self.collision_pair_head)
            return coincidence * self.collision_kerr[1].to(total), total
        drive = torch.stack((total / scale, (field - flux) / scale), -1)        # [..., H, 2]
        per_pair = drive.index_select(-2, self.collision_pair_head)            # [..., rate_width, 2]
        return (per_pair * self.collision_kerr.t().to(total)).sum(-1), total

    def _kerr_only_rates(self, packed, material, allocation=None):
        modulation, total = self._kerr_drive(packed, material)
        rates = self.collision_scale.exp().to(total) * (self.collision_base.to(total) + modulation)
        if self.collision_position is not None:
            rates = rates + F.linear(material, self.collision_position.to(material))[None]
        return rates.reshape(*packed.shape[:-2], self.collision_layers, self.free_width // 2)

    def cell_cross_section(self) -> torch.Tensor | None:
        """sigma_s(x) [X,Y,Z], lattice mean exactly 1; None when the field is off."""
        if self.cell_theta is None:
            return None
        logits = self.cell_scale * self.cell_theta
        return logits.numel() * torch.softmax(logits.flatten(), 0).reshape(logits.shape)

    def collision_rates(self, packed: torch.Tensor, material: torch.Tensor,
                        receptors: torch.Tensor | None = None,
                        allocation: torch.Tensor | None = None) -> torch.Tensor:
        """Rotation rates [B,X,Y,Z,layers,pairs] shared by the flow and its derivative measurement."""
        rates = self._collision_rates_rule(packed, material, receptors, allocation)
        sigma = self.cell_cross_section()
        if sigma is None:
            return rates
        return rates * sigma.to(rates)[None, ..., None, None]

    def _collision_rates_rule(self, packed: torch.Tensor, material: torch.Tensor,
                              receptors: torch.Tensor | None = None,
                              allocation: torch.Tensor | None = None) -> torch.Tensor:
        """The local pair rule (identical at every site)."""
        if self.collision_log_phi is not None:
            if self.collision_coincidence:
                coupling, _ = self._kerr_drive(packed, material, pair_only=True)
                if self.collision_gravity:
                    coupling = self._gravity_coupling(packed, coupling)
                rates = self.collision_log_phi.to(coupling).exp() * coupling / self.collision_crossing_time
                return rates.reshape(*packed.shape[:-2], self.collision_layers, self.free_width // 2)
            if self.collision_kerr_heads:
                modulation, _ = self._kerr_drive(packed, material)
            elif self.collision_pair_gate:
                features, coincidence = self._gram_terms(packed)
                modulation = self.collision_rate(features)
            else:
                modulation = self.collision_rate(self._features(packed, material, receptors,
                                                                center_receptors=self.collision_center_receptors))
            rates = (self.collision_sign.to(modulation)
                     * (self.collision_log_phi.to(modulation) + self.collision_mod_bound * torch.tanh(
                         self.collision_mod_gain * modulation / self.collision_mod_bound)).exp()
                     / self.collision_crossing_time)
            if self.collision_pair_gate:
                rates = rates * coincidence.index_select(-1, self.collision_pair_head)
            return rates.reshape(*packed.shape[:-2], self.collision_layers, self.free_width // 2)
        if self.collision_kerr_heads:
            return self._kerr_only_rates(packed, material, allocation)
        rates = self.collision_rate(self._features(packed, material, receptors,
                                                   center_receptors=self.collision_center_receptors))
        if self.collision_position is not None:
            rates = rates + F.linear(material, self.collision_position.to(material))[None]
        if self.collision_kerr is not None:
            field_energy = packed[..., 0, :].square().sum(-1)
            flux_energy = packed[..., 1:, :].square().sum((-2, -1))
            total = field_energy + flux_energy
            scale = total + self.collision_kerr_energy.exp().to(total)
            drive = torch.stack((total / scale, (field_energy - flux_energy) / scale), -1)
            rates = rates + drive @ self.collision_kerr.to(drive)
        if self.collision_bed is not None:
            if allocation is None:
                raise ValueError('Bed-driven collisions need the prepared structural allocation')
            reference = self.structural_posterior.resource_density.to(allocation) / 4
            rates = rates + F.linear(allocation - reference, self.collision_bed.to(allocation))[None]
        return rates.reshape(*packed.shape[:-2], self.collision_layers, self.free_width // 2)

    def _reflect(self, value: torch.Tensor) -> torch.Tensor:
        w = self.mean_reflector.to(value)
        return value - 2.0 * (value * w).sum(-1, keepdim=True) * w

    _MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))

    def _pair_view(self, free: torch.Tensor, layer: int):
        """The two coordinates of every rotation pair of a layer, each [..., free_width // 2]."""
        if layer < self.collision_inner_layers:
            lanes = torch.roll(free, -layer, -1).reshape(*free.shape[:-1], -1, 2)
            return lanes[..., 0], lanes[..., 1]
        k = layer - self.collision_inner_layers
        groups = free.reshape(*free.shape[:-1], 4, self.channels - 1)
        return (groups.index_select(-2, self.collision_cross_left[k]).flatten(-2),
                groups.index_select(-2, self.collision_cross_right[k]).flatten(-2))

    def _pair_restore(self, left: torch.Tensor, right: torch.Tensor, layer: int) -> torch.Tensor:
        if layer < self.collision_inner_layers:
            return torch.roll(torch.stack((left, right), -1).flatten(-2), layer, -1)
        width = self.channels - 1
        stacked = torch.cat((left.reshape(*left.shape[:-1], 2, width), right.reshape(*right.shape[:-1], 2, width)), -2)
        return stacked.index_select(-2, self.collision_cross_inverse[layer - self.collision_inner_layers]).flatten(-2)

    def _rotate_layers(self, free: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """All rotation layers in order on the free coordinates (a pure tensor function, so it can be compiled)."""
        for layer in range(self.collision_layers):
            left, right = self._pair_view(free, layer)
            angle = angles[..., layer, :]
            free = self._pair_restore(angle.cos() * left - angle.sin() * right,
                                      angle.sin() * left + angle.cos() * right, layer)
        return free

    def collide(self, state: MediumState, duration: float | torch.Tensor,
                material: torch.Tensor | None = None,
                allocation: torch.Tensor | None = None) -> MediumState:
        """Content- and space-conditioned rotations with four local linear invariants.

        The invariant is the channel sum of each field/flux group. This is a
        signed wave-state model; these are NOT a kinetic D3Q8 momentum claim.
        Flux axes participate together, so scattering can change propagation direction.
        """
        material = self.material_field() if material is None else material
        value = self._pack(state)
        rates = self.collision_rates(value, material, state.receptors, allocation)
        dt = self._duration(duration, state.field).unsqueeze(-1)
        angles = rates * dt
        transformed = self._reflect(value)
        fixed = transformed[..., :1]
        free = transformed[..., 1:].flatten(-2)
        free = self._rotate_layers(free, angles)
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
        """Optional fused derivative measurement, with exact native AD fallback."""
        if (not getattr(self, 'segment_graph_enabled', False) or not state.field.is_cuda
                or state.field.dtype != torch.float32 or torch.compiler.is_compiling()):
            return self.native_field_rhs(state, prepared=prepared)
        from torch.autograd.forward_ad import unpack_dual
        leaves = (state.field, *state.flux, state.conduction, state.receptors,
                  state.transmission)
        if any(isinstance(x, torch.Tensor) and unpack_dual(x).tangent is not None for x in leaves):
            return self.native_field_rhs(state, prepared=prepared)
        prepared = self.prepare_evolution() if prepared is None else prepared
        if not hasattr(self, '_compiled_rhs'):
            import torch._inductor.config as config
            config.compile_threads = 1
            if os.name == 'nt':
                config.use_static_cuda_launcher = False
            self._compiled_rhs = torch.compile(self.native_field_rhs, fullgraph=True, dynamic=False)
        operation = self._compiled_rhs
        if not torch.is_grad_enabled():
            from .segment_graph import NoGradSegmentGraph
            if not hasattr(self, '_segment_rhs'):
                self._segment_rhs = NoGradSegmentGraph(operation, max_variants=1,
                                                       parameters=self.parameters())
            operation = self._segment_rhs
        return operation(state, prepared=prepared)

    def native_field_rhs(self, state: MediumState, *,
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
        rates = self.collision_rates(packed, material, state.receptors, prepared.structural_allocation)
        transformed = self._reflect(packed)
        free = transformed[..., 1:].flatten(-2)
        derivative = torch.zeros_like(free)
        # Every infinitesimal generator acts on the SAME initial vector.
        for layer in range(self.collision_layers):
            left, right = self._pair_view(free, layer)
            rate = rates[..., layer, :]
            derivative = derivative + self._pair_restore(-rate * right, rate * left, layer)
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
        if self.soil_absorption and prepared.structural_allocation is not None:
            response_rhs = response_rhs - self.soil_absorption * self.soil_share(
                prepared.structural_allocation).to(state.field)[None] * state.field
        return transport_rhs + collision_rhs + response_rhs

    def soil_share(self, allocation: torch.Tensor) -> torch.Tensor:
        """Idle (uninstalled) share of the site budget, [X,Y,Z,1]: tissue without channels that absorbs."""
        return allocation[..., 3:4] / self.structural_posterior.resource_density.to(allocation)

    def absorb_soil(self, state: MediumState, dt: torch.Tensor, allocation: torch.Tensor) -> MediumState:
        """Soil absorbs: field and the three edge stores decay at soil_absorption * soil(x) (unified riverbed + bath).

        Channels (installed axis shares) conduct with high fidelity; the idle share is soil that dissipates what reaches it.
        The idle total is a fixed share of the conserved site budget at birth, so the dissipation budget is set by the bed.
        """
        decay = torch.exp(-self.soil_absorption * self.soil_share(allocation).to(state.field)[None] * dt)
        return replace(state, field=state.field * decay, flux=tuple(x * decay for x in state.flux))

    @torch.no_grad()
    def through_flow(self, state: MediumState, prepared: EvolutionCoefficients | None = None) -> torch.Tensor:
        """Signed wave-energy through-flow per site and axis, [X,Y,Z,3], summed over the batch.

        T_a(i) = |J_a(i) + J_a(i - e_a)| / 2 with J the transport energy current: flow passing THROUGH a site along an
        axis adds up, converging inflows from both sides cancel (a river is carved along its discharge, not its inflow).
        """
        prepared = self.prepare_evolution() if prepared is None else prepared
        current = self.transport_energy_current(state, self.current_transport_factor(state, prepared)).sum(0)
        return torch.stack([(current[..., a] + torch.roll(current[..., a], 1, a)).abs() / 2 for a in range(3)], -1)

    @torch.no_grad()
    def channel_growth_step(self, through_flow: torch.Tensor, rate: float, mu: float = 2.0) -> dict:
        """Unified riverbed growth (Tero / river rule) on the installed shares (x, y, z, soil) of the posterior mean.

        Target channel share per axis tau_a = T_a^mu / (T_a^mu + mean(T)^mu); target vector (tau_x, tau_y, tau_z, 1)
        normalised; shares relax toward it at `rate` per call; the soil total is then rescaled to its current mean
        (conserved dissipation budget) and kept inside [0.02, 0.98] of the site. Where flow passes, channels grow out of
        soil; where no flow passes, channels return to soil. The logit change is projected onto the band-limited chart.
        rate == 0 returns before touching anything (bitwise identity).
        """
        if rate == 0:
            return {}
        if not math.isfinite(rate) or not 0 < rate <= 1 or not math.isfinite(mu) or mu <= 0:
            raise ValueError('Channel growth needs 0 < rate <= 1 and a positive finite exponent')
        posterior = self.structural_posterior
        if posterior is None:
            raise ValueError('Channel growth needs the structural posterior')
        shape = tuple(self.shape)
        S = int(math.prod(shape))
        basis = self.material.basis(self.coordinates).to(posterior.mean).double()
        flat_basis = basis.reshape(S, -1)
        cache = self.__dict__.setdefault('_structural_gravity_cache', {})
        key = (str(flat_basis.device), flat_basis.shape, 'pinv')
        if key not in cache:
            cache[key] = torch.linalg.pinv(flat_basis)
        pinv = cache[key]
        shares = (posterior.allocation(basis) / posterior.resource_density.to(basis)).double().reshape(S, 4)
        T = through_flow.to(shares).reshape(S, 3)
        scale = T.mean().clamp_min(torch.finfo(T.dtype).tiny)
        tau = (T / scale).pow(mu)
        tau = tau / (tau + 1)
        target = torch.cat((tau, torch.ones_like(tau[:, :1])), -1)
        target = target / target.sum(-1, keepdim=True)
        new = shares + rate * (target - shares)
        soil = (new[:, 3] * shares[:, 3].mean() / new[:, 3].mean()).clamp(0.02, 0.98)
        channels = new[:, :3] / new[:, :3].sum(-1, keepdim=True) * (1 - soil[:, None])
        delta_logits = (channels.log() - soil.log()[:, None]) - (shares[:, :3].log() - shares[:, 3:].log())
        posterior.mean.add_((pinv @ delta_logits).to(posterior.mean))
        after = (posterior.allocation(basis) / posterior.resource_density.to(basis)).double().reshape(S, 4)
        spread = lambda s: ((s[:, :3].max(-1).values - s[:, :3].min(-1).values) / s[:, :3].sum(-1)).mean()
        return {'soil_mean': float(after[:, 3].mean()), 'soil_cv': float(after[:, 3].std() / after[:, 3].mean()),
                'channel_anisotropy': float(spread(after)), 'flow_cv': float(T.sum(-1).std() / T.sum(-1).mean()),
                'projection_residual': float((after - torch.cat((channels, soil[:, None]), -1)).abs().mean())}

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
        operation = self._compiled_advance
        if getattr(self, 'segment_graph_enabled', False) and not torch.is_grad_enabled():
            from .segment_graph import NoGradSegmentGraph
            if not hasattr(self, '_segment_advance'):
                def segment_operation(current, interval, **options):
                    return self.native_advance(current, interval, **options)
                compiled_segment = torch.compile(segment_operation, fullgraph=True, dynamic=False)
                self._segment_advance = NoGradSegmentGraph(compiled_segment,
                                                           parameters=self.parameters())
            operation = self._segment_advance
            # Values vary with intrinsic time; shapes and graph addresses stay fixed.
            duration = torch.as_tensor(duration, device=state.field.device,
                                       dtype=state.elapsed.dtype).flatten()
        return self._fused_substeps(operation, state, duration,
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
            stages = ('transport', 'collision', 'bath') + (() if self.hopf_pathway is None else ('junction',))
            stage_changes = {f'{stage}_{quantity}_change': torch.zeros_like(energy_before)
                             for stage in stages
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
                junction_ledger = {} if diagnostics and self.hopf_pathway is not None else None
                result = self.transport(result, dt, material, prepared.baseline_log_speed,
                                        prepared.short_term, prepared.structural_factor,
                                        prepared.junction_material_logits,
                                        prepared.junction_route_mask, stage_ledger=junction_ledger)
                if diagnostics:
                    record_stage('transport', before_stage, result)
                    for key, value in (junction_ledger or {}).items():
                        stage_changes[key] = stage_changes[key] + value
                        transport_key = key.replace('junction_', 'transport_')
                        stage_changes[transport_key] = stage_changes[transport_key] - value
            if collision:
                before_stage = stage_measure(result) if diagnostics else None
                result = self.collide(result, dt, material, prepared.structural_allocation)
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
                if self.soil_absorption and prepared.structural_allocation is not None:
                    before_soil = self.energy(result) if diagnostics else None
                    result = self.absorb_soil(result, dt, prepared.structural_allocation)
                    if diagnostics:
                        released = released + before_soil - self.energy(result)
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
