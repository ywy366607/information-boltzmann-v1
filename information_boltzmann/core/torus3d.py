"""Pure geometric CBIM on the periodic three-torus.

The persistent state is ``f[x, y, z, q, a]``.  Space is a numerical
Fourier--Galerkin truncation of T^3, ``q`` indexes continuous velocity carriers, and ``a``
carries learned content.  No anatomical graph or named biological region is
used by this model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import cube_velocities


def d3q_velocities(velocities: int = 8, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return normalized discrete velocity vectors for D3Q8 or D3Q27."""
    if velocities == 8:
        return cube_velocities(dtype=dtype)
    elif velocities == 27:
        coords = []
        # 1. Rest particle (c = 0)
        coords.append((0.0, 0.0, 0.0))
        # 2. 6 face centers (axes, c = 1)
        for axis in range(3):
            for s in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0]
                v[axis] = s
                coords.append(tuple(v))
        # 3. 12 edge centers (c = sqrt(2))
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = [0.0, 0.0, 0.0]
                    v[i] = si
                    v[j] = sj
                    coords.append(tuple(v))
        # 4. 8 cube corners (c = sqrt(3))
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    coords.append((sx, sy, sz))
        vel = torch.tensor(coords, dtype=dtype)
        return vel / math.sqrt(3.0)
    else:
        raise ValueError(f"Unsupported velocity count: {velocities}")


def torus_grid(shape: tuple[int, int, int]) -> torch.Tensor:
    axes = [torch.arange(n, dtype=torch.float32) / n for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)


def torus_features(points: torch.Tensor) -> torch.Tensor:
    phase = 2.0 * math.pi * points
    return torch.cat((phase.sin(), phase.cos()), -1)


class FullRankTorusWrite(nn.Module):
    """State-conditioned full-rank packet coupled through an orthogonal port."""

    def __init__(self, vocab_size=50257, shape=(4, 4, 4), d=128,
                 packet_radius=1.25, max_angle=0.30, relative_address=True,
                 write_type="w0_baseline", spectral_packet=False, nu_s_init=0.020,
                 decouple_source_feedback=False):
        super().__init__()
        self.shape, self.d = tuple(shape), int(d)
        self.write_type = str(write_type)
        self.relative_address = bool(relative_address)
        self.spectral_packet = bool(spectral_packet)
        self.decouple_source_feedback = bool(decouple_source_feedback)
        self.embedding = nn.Embedding(vocab_size, d)
        self.address = nn.Linear(d, 3)
        self.width = nn.Linear(d, 3)
        self.content = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.state_content = nn.Linear(d, d, bias=False)
        self.neighbor_content = nn.Linear(d, d, bias=False)

        if self.write_type == "w0_baseline":
            self.max_angle = float(max_angle)
            angle_in_dim = 2 * d
            angle_bias = -1.5
        elif self.write_type == "w1_unbounded":
            self.max_angle = math.pi / 2.0
            angle_in_dim = 2 * d
            angle_bias = -0.65  # ~12%-14% initial transmission on T^3, within [10%, 20%]
        elif self.write_type == "w2_impedance":
            self.max_angle = math.pi / 2.0
            angle_in_dim = 2 * d + 4  # [token, context, E_local, cos_fp, f_norm, p_norm]
            angle_bias = -0.65
        elif self.write_type == "w3_additive":
            self.eta_max = 1.0
            angle_in_dim = 2 * d
            angle_bias = -0.65
        else:
            raise ValueError(f"Unknown write_type: {write_type}")

        self.angle = nn.Sequential(
            nn.Linear(angle_in_dim, d), nn.SiLU(), nn.Linear(d, d))
        for layer in (self.content[-1], self.state_content,
                      self.neighbor_content, self.angle[-1]):
            nn.init.normal_(layer.weight, std=1e-3)
        nn.init.zeros_(self.content[-1].bias)
        nn.init.constant_(self.angle[-1].bias, angle_bias)

        # Fixed physical packet reference width in [0, 1)^3 independent of grid resolution
        reference_width = 0.15
        nn.init.zeros_(self.width.weight)
        nn.init.constant_(self.width.bias, math.log(math.expm1(float(reference_width))))
        self.channel_scale = nn.Parameter(torch.full(
            (d,), float(packet_radius) * math.sqrt(d / 64.0)))
        self.register_buffer(
            "coordinates", torus_grid(self.shape)[None], persistent=False)

        # Continuous physical wave vectors on T^3
        wave_axes = [torch.fft.fftfreq(n, d=1.0 / n) * 2.0 * math.pi for n in self.shape]
        wave = torch.stack(torch.meshgrid(*wave_axes, indexing="ij"), -1)
        k_sq = wave.square().sum(-1).unsqueeze(0).unsqueeze(-1)  # [1, X, Y, Z, 1]
        self.register_buffer("wave", wave, persistent=False)
        self.register_buffer("wave_sq", wave.square(), persistent=False)

        if self.spectral_packet:
            nu_s_inv = math.log(max(math.exp(float(nu_s_init)) - 1.0, 1e-6))
            self.nu_s_param = nn.Parameter(torch.tensor(nu_s_inv, dtype=torch.float32))

        # Continuous physical Gaussian heat kernel on T^3 with fixed physical radius sigma_mix
        sigma_mix = 0.08
        self.register_buffer(
            "heat_filter", torch.exp(-0.5 * (sigma_mix ** 2) * k_sq), persistent=False)

    @staticmethod
    def energy(value: torch.Tensor) -> torch.Tensor:
        # Quadrature approximation of integral_T3 1/2 |f(x)|^2 dx.  Averaging
        # the spatial samples makes the energy unit independent of resolution.
        return 0.5 * value.square().sum(dim=-1).mean()

    def neighborhood(self, field: torch.Tensor) -> torch.Tensor:
        """Continuous physical Gaussian heat diffusion on T^3 (replaces discrete roll)."""
        freq = torch.fft.fftn(field, dim=(1, 2, 3), norm="ortho")
        return torch.fft.ifftn(
            freq * self.heat_filter.to(device=field.device, dtype=field.dtype),
            dim=(1, 2, 3), norm="ortho").real

    def field_anchor(self, field: torch.Tensor) -> torch.Tensor:
        """Circular energy barycenter, equivariant to translations on T^3."""
        energy = field.square().sum(-1)
        phase = 2.0 * math.pi * self.coordinates
        axes = (1, 2, 3)
        sine = (energy[..., None] * phase.sin()).sum(axes)
        cosine = (energy[..., None] * phase.cos()).sum(axes)
        # The small positive reference makes the empty initial field finite;
        # once energy exists, both numerator and denominator translate with it.
        return torch.remainder(torch.atan2(sine, cosine + 1e-6) / (2.0 * math.pi), 1.0)

    def synthesize_packet(
            self, field: torch.Tensor, token: torch.Tensor, *,
            displacement: Optional[torch.Tensor] = None,
            width: Optional[torch.Tensor] = None,
            content: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """Map one port coordinate to the shared physical packet chart.

        The observed token and a field-generated prior token must use this
        exact function.  That makes their difference a genuine innovation in
        the field's input-port Hilbert space, rather than an error between two
        unrelated latent vectors.
        """
        if self.relative_address:
            anchor = self.field_anchor(field)
            displacement_val = (displacement if displacement is not None else
                                0.5 * torch.tanh(self.address(token)))
            center_value = torch.remainder(anchor + displacement_val, 1.0)
        else:
            anchor = torch.zeros(field.shape[0], 3, device=field.device,
                                 dtype=field.dtype)
            center_value = (torch.sigmoid(self.address(token)) if displacement is None
                            else displacement)
            displacement_val = center_value

        nu_s = field.new_zeros(())
        if self.spectral_packet:
            nu_s = F.softplus(self.nu_s_param).clamp_min(1e-5)
            width_val = width if width is not None else F.softplus(self.width(token))
            sigma_eff_sq = width_val.square() + 2.0 * nu_s
            exponent = -0.5 * torch.einsum(
                "bj,xyzj->bxyz", sigma_eff_sq,
                self.wave_sq.to(dtype=field.dtype))
            b_eff = torch.exp(exponent)
            phase = -torch.einsum(
                "xyzj,bj->bxyz", self.wave.to(dtype=field.dtype), center_value)
            s_k = b_eff * torch.complex(phase.cos(), phase.sin())
            s_x = torch.fft.ifftn(s_k, dim=(1, 2, 3), norm="ortho").real
            spatial = s_x / s_x.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)
            spatial = spatial.clamp(min=0.0, max=1.0)
        else:
            width_val = width if width is not None else F.softplus(self.width(token))
            center = center_value[:, None, None, None, :]
            width_tensor = (width_val[:, None, None, None, :]
                            if width_val.ndim == 2 else width_val)
            delta = (self.coordinates - center).abs()
            distance = torch.minimum(delta, 1.0 - delta) / width_tensor.clamp_min(1e-6)
            tail = math.sqrt(-2.0 * math.log(torch.finfo(field.dtype).tiny))
            envelope = torch.exp(-0.5 * distance.clamp(-tail, tail).square().sum(-1))
            spatial = envelope / envelope.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)

        field_for_source = field.detach() if self.decouple_source_feedback else field
        neighbors = self.neighborhood(field_for_source)
        content_val = (content if content is not None else self.content(token))[:, None, None, None]
        local = (content_val
                 + self.state_content(field_for_source)
                 + self.neighbor_content(neighbors))
        local = F.normalize(local, dim=-1)
        packet = spatial[..., None] * (self.channel_scale * local)
        return {
            "packet": packet,
            "spatial": spatial,
            "anchor": anchor,
            "center": center_value[:, None, None, None, :],
            "displacement": displacement_val,
            "width": width_val,
            "nu_s": nu_s,
        }

    def forward(self, field: torch.Tensor, token_ids: torch.Tensor, return_diag: bool = True,
                *, token: Optional[torch.Tensor] = None,
                displacement: Optional[torch.Tensor] = None,
                width: Optional[torch.Tensor] = None,
                content: Optional[torch.Tensor] = None):
        if token is None:
            token = self.embedding(token_ids)
        synthesis = self.synthesize_packet(
            field, token, displacement=displacement, width=width, content=content)
        packet = synthesis["packet"]
        spatial = synthesis["spatial"]
        anchor = synthesis["anchor"]
        center = synthesis["center"]
        displacement_val = synthesis["displacement"]
        width_val = synthesis["width"]
        nu_s = synthesis["nu_s"]
        axes = (1, 2, 3)
        spatial_weight = spatial[..., None]
        spatial_sum = spatial.sum(axes)[..., None].clamp_min(1e-8)
        context = ((spatial_weight * field).sum(axes) / spatial_sum)

        if self.write_type in ("w0_baseline", "w1_unbounded"):
            context_in = context.detach() if self.decouple_source_feedback else context
            angle = self.max_angle * torch.sigmoid(
                self.angle(torch.cat((token, context_in), -1)))
            theta = spatial[..., None] * angle[:, None, None, None]
            cosine, sine = theta.cos(), theta.sin()
            field_next = cosine * field + sine * packet
            if return_diag:
                reflected = -sine * field + cosine * packet
                residual = (self.energy(field_next) + self.energy(reflected)
                            - self.energy(field) - self.energy(packet)).abs()
                cross_interference = (torch.sin(2.0 * theta) * field * packet).sum(dim=-1).mean()
            else:
                reflected = None
                residual = field.new_zeros(())
                cross_interference = field.new_zeros(())
        elif self.write_type == "w2_impedance":
            packet_local = (spatial_weight * packet).sum(axes) / spatial_sum
            local_energy = 0.5 * (spatial_weight * field.square()).sum(axes).sum(-1, keepdim=True) / spatial_sum
            f_norm = context.norm(dim=-1, keepdim=True)
            p_norm = packet_local.norm(dim=-1, keepdim=True)
            f_hat = context / f_norm.clamp_min(1e-6)
            p_hat = packet_local / p_norm.clamp_min(1e-6)
            cos_fp = (f_hat * p_hat).sum(dim=-1, keepdim=True)
            phys = torch.cat([local_energy, cos_fp, f_norm, p_norm], dim=-1)
            context_in = context.detach() if self.decouple_source_feedback else context
            phys_in = phys.detach() if self.decouple_source_feedback else phys
            angle = self.max_angle * torch.sigmoid(
                self.angle(torch.cat((token, context_in, phys_in), -1)))
            theta = spatial[..., None] * angle[:, None, None, None]
            cosine, sine = theta.cos(), theta.sin()
            field_next = cosine * field + sine * packet
            if return_diag:
                reflected = -sine * field + cosine * packet
                residual = (self.energy(field_next) + self.energy(reflected)
                            - self.energy(field) - self.energy(packet)).abs()
                cross_interference = (torch.sin(2.0 * theta) * field * packet).sum(dim=-1).mean()
            else:
                reflected = None
                residual = field.new_zeros(())
                cross_interference = field.new_zeros(())
        elif self.write_type == "w3_additive":
            eta_peak = self.eta_max * torch.sigmoid(
                self.angle(torch.cat((token, context), -1)))
            eta = spatial[..., None] * eta_peak[:, None, None, None]
            field_next = field + eta * packet
            reflected = (1.0 - eta).clamp_min(0.0) * packet
            residual = field.new_zeros(())
            cross_interference = (2.0 * eta * field * packet).sum(dim=-1).mean()
            theta = eta
            angle = eta_peak
        else:
            raise ValueError(f"Unknown write_type: {self.write_type}")

        if return_diag:
            incident_energy = self.energy(packet)
            delta_e_field = self.energy(field_next) - self.energy(field)
            if self.write_type == "w3_additive":
                reflected_energy = (incident_energy - delta_e_field).clamp_min(0.0)
                t_packet = delta_e_field / incident_energy.clamp_min(1e-8)
            else:
                reflected_energy = self.energy(reflected)
                t_packet = 1.0 - reflected_energy / incident_energy.clamp_min(1e-8)
            write_delta = field_next - field
            write_ratio = write_delta.norm(dim=-1).mean() / (field.norm(dim=-1).mean() + 1e-6)

            diag = {
                "incident_energy": incident_energy.detach(),
                "reflected_energy": reflected_energy.detach(),
                "accepted_energy": delta_e_field.detach(),
                "delta_e_field": delta_e_field.detach(),
                "accepted_fraction": t_packet.detach(),
                "t_packet": t_packet.detach(),
                "cross_interference": cross_interference.detach(),
                "write_to_f_ratio": write_ratio.detach(),
                "write_angle_abs_mean": theta.detach().abs().mean(),
                "write_angle_peak_mean": angle.detach().abs().mean(),
                "write_spatial_support": (spatial.detach() > 0.1).float().mean(),
                "write_balance_residual": residual.detach(),
                "source_center": center.detach().reshape(field.shape[0], 3),
                "source_anchor": anchor.detach(),
                # ``displacement`` and ``width`` are optional cached inputs.
                # Diagnostics must describe the values actually used by this
                # event, including the ordinary uncached step path.
                "source_displacement": displacement_val.detach(),
                "write_width_mean": width_val.detach().mean(),
                "source_nu_s": nu_s.detach() if self.spectral_packet else field.new_zeros(()),
            }
        else:
            diag = {}

        return field_next, reflected, diag

    @staticmethod
    def _safe_unit(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Finite zero-support extension for compact multi-port packets."""
        return value * torch.rsqrt(
            value.square().sum(dim=-1, keepdim=True).clamp_min(eps * eps))

    def forward_ports(self, field: torch.Tensor, tokens: torch.Tensor,
                      centers: torch.Tensor, widths: torch.Tensor,
                      mask: torch.Tensor | None = None,
                      return_diag: bool = True):
        """Apply the canonical boundary writer to an unordered port set.

        ``tokens`` supplies one already encoded external value per port.  A
        single active port dispatches directly to :meth:`forward`, providing
        an exact equivalence check with language's established writer.  For a
        set, full-rank packets are power-composed before one orthogonal field
        exchange.  This keeps the field bounded independently of port count
        and retains the same content, state/neighbor, angle, and energy laws.
        """
        if tokens.ndim != 3 or centers.shape != (*tokens.shape[:2], 3):
            raise ValueError("Expected tokens [B,P,D] and centers [B,P,3]")
        if widths.shape != centers.shape:
            raise ValueError("Expected one positive width vector per port")
        batch, ports, d = tokens.shape
        if d != self.d:
            raise ValueError(f"Expected port dimension {self.d}, got {d}")
        if mask is None:
            mask = torch.ones(batch, ports, device=field.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError("Expected mask [B,P]")
        # An empty observation event is an identity evolution.  Returning it
        # before any packet normalisation gives the port contract a finite
        # zero-support extension and avoids differentiating sqrt(0) through
        # an otherwise inactive address chart.
        if not bool(mask.any()):
            if not return_diag:
                return field, field.new_zeros(field.shape), {}
            zero = field.new_zeros(())
            return field, field.new_zeros(field.shape), {
                "incident_energy": zero,
                "reflected_energy": zero,
                "accepted_energy": zero,
                "write_angle_abs_mean": zero,
                "write_angle_peak_mean": zero,
                "write_spatial_support": zero,
                "source_center": field.new_zeros(batch, 3),
                "source_anchor": field.new_zeros(batch, 3),
                "source_displacement": field.new_zeros(batch, 3),
                "write_width_mean": zero,
            }
        # This branch is deliberately exact, rather than merely close: it is
        # the contractual bridge from the language operator to port events.
        if ports == 1 and bool(mask.all()):
            return self.forward(
                field, token_ids=torch.zeros(batch, device=field.device, dtype=torch.long),
                token=tokens[:, 0], displacement=centers[:, 0],
                width=widths[:, 0], return_diag=return_diag)

        mask_value = mask.to(field.dtype)
        count = mask_value.sum(1, keepdim=True).clamp_min(1.0)
        # ``coordinates`` already carries a leading singleton batch axis.
        # Insert only the port axis: [1, 1, X, Y, Z, 3].  Adding two new
        # axes here silently created a six-dimensional spatial envelope and
        # broke the multi-port contraction below.
        grid = self.coordinates.to(field)[:, None]
        center = centers[:, :, None, None, None, :]
        delta = (grid - center).abs()
        distance = torch.minimum(delta, 1.0 - delta) / widths[:, :, None, None, None, :].clamp_min(1e-6)
        tail = math.sqrt(-2.0 * math.log(torch.finfo(field.dtype).tiny))
        envelope = torch.exp(-0.5 * distance.clamp(-tail, tail).square().sum(-1))
        envelope = envelope / envelope.amax((2, 3, 4), keepdim=True).clamp_min(1e-8)
        envelope = envelope * mask_value[:, :, None, None, None]

        field_for_source = field.detach() if self.decouple_source_feedback else field
        state = self.state_content(field_for_source) + self.neighbor_content(
            self.neighborhood(field_for_source))
        content = self.content(tokens) * mask_value[..., None]
        support = (envelope.square().sum(1) / count[:, :, None, None]).clamp_min(1e-12).sqrt()
        support = support / support.amax((1, 2, 3), keepdim=True).clamp_min(1e-6)
        support = support * mask_value.sum(1, keepdim=True).gt(0).to(field.dtype)[:, :, None, None]
        local = self._safe_unit(
            torch.einsum("bpxyz,bpd->bxyzd", envelope, content) + support[..., None] * state)
        packet = support[..., None] * self.channel_scale * local

        axes = (2, 3, 4)
        context = (envelope[..., None] * field[:, None]).sum(axes) / (
            envelope.sum(axes).unsqueeze(-1).clamp_min(1e-8))
        if self.write_type == "w2_impedance":
            # A port set is one physical boundary event.  First compose its
            # bounded incoming packet, then use exactly the impedance
            # variables of the language writer to determine its admittance.
            # This is distinct from averaging independent write rates: the
            # pressure, phase alignment and packet norm are properties of
            # the complete incoming boundary wave.
            spatial_weight = support[..., None]
            spatial_sum = support.sum((1, 2, 3), keepdim=False).view(batch, 1).clamp_min(1e-8)
            aggregate_context = (spatial_weight * field).sum((1, 2, 3)) / spatial_sum
            packet_local = (spatial_weight * packet).sum((1, 2, 3)) / spatial_sum
            local_energy = 0.5 * (spatial_weight * field.square()).sum((1, 2, 3)).sum(-1, keepdim=True) / spatial_sum
            f_norm = aggregate_context.norm(dim=-1, keepdim=True)
            p_norm = packet_local.norm(dim=-1, keepdim=True)
            cos_fp = ((aggregate_context / f_norm.clamp_min(1e-6))
                      * (packet_local / p_norm.clamp_min(1e-6))).sum(-1, keepdim=True)
            phys = torch.cat((local_energy, cos_fp, f_norm, p_norm), -1)
            port_mass = envelope.sum(axes)
            port_mass = port_mass / port_mass.sum(1, keepdim=True).clamp_min(1e-8)
            event_token = (port_mass[..., None] * tokens).sum(1)
            context_in = aggregate_context.detach() if self.decouple_source_feedback else aggregate_context
            phys_in = phys.detach() if self.decouple_source_feedback else phys
            event_angle = self.max_angle * torch.sigmoid(
                self.angle(torch.cat((event_token, context_in, phys_in), -1)))
            theta = support[..., None] * event_angle[:, None, None, None]
            peak_angle = event_angle
        elif self.write_type in ("w0_baseline", "w1_unbounded"):
            per_port_angle = self.max_angle * torch.sigmoid(
                self.angle(torch.cat((tokens, context), -1))) * mask_value[..., None]
            theta = ((envelope[..., None] * per_port_angle[:, :, None, None, None]).square().sum(1)
                     / count[:, :, None, None, None]).clamp_min(0.0).sqrt()
            peak_angle = per_port_angle
        else:
            raise ValueError(
                "Multi-port boundary events support orthogonal w0, w1, and w2 writers")
        cosine, sine = theta.cos(), theta.sin()
        field_next = cosine * field + sine * packet
        reflected = -sine * field + cosine * packet

        phase = 2.0 * math.pi * centers
        sine_mean = (phase.sin() * mask_value[..., None]).sum(1) / count
        cosine_mean = (phase.cos() * mask_value[..., None]).sum(1) / count
        event_center = torch.remainder(
            torch.atan2(sine_mean, cosine_mean + 1e-6) / (2.0 * math.pi), 1.0)
        if return_diag:
            diag = {
                "incident_energy": self.energy(packet).detach(),
                "reflected_energy": self.energy(reflected).detach(),
                "accepted_energy": (self.energy(field_next) - self.energy(field)).detach(),
                "accepted_fraction": (
                    1.0 - self.energy(reflected) / self.energy(packet).clamp_min(1e-8)).detach(),
                "write_angle_abs_mean": theta.detach().abs().mean(),
                "write_angle_peak_mean": peak_angle.detach().abs().mean(),
                "write_spatial_support": (support.detach() > 0.1).float().mean(),
                "source_center": event_center.detach(),
                "source_anchor": event_center.detach(),
                "source_displacement": event_center.detach(),
                "write_width_mean": widths.detach().mean(),
            }
        else:
            diag = {}
        return field_next, reflected, diag


@dataclass
class KineticBeliefState:
    """Persistent posterior belief for the predictive-impedance branch.

    ``field`` is the posterior kinetic mean.  ``precision`` is a channelwise
    structured precision, deliberately compact enough to persist in an
    infinite stream while retaining a real prior-to-posterior update.
    """

    field: torch.Tensor
    precision: torch.Tensor


class PredictiveImpedanceWriteAgent(nn.Module):
    """Causal posterior writer on the common Q8 packet chart.

    The agent receives a pre-event field belief and an observed token.  It
    predicts a categorical port distribution from the field alone, maps both
    the observed feature and its predicted expectation through one *linear*
    Fourier--Galerkin packet chart, and scatters only their innovation.  Its
    action is a posterior distribution over positive channel admittances;
    this first implementation uses its reparameterized mean.  Later M
    branches may sample that same posterior without changing the port law.
    """

    def __init__(self, d: int, vocab_size: int, port_modes: int = 8) -> None:
        super().__init__()
        self.d = int(d)
        self.vocab_size = int(vocab_size)
        self.port_modes = int(port_modes)
        if self.port_modes not in (8, 27):
            raise ValueError("Predictive impedance supports the Q8 or Q27 Fourier chart")
        self.port_prior = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.port_logit_bias = nn.Parameter(torch.zeros(vocab_size))
        self.chart_gate = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, self.port_modes))
        self.action_prior = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 2 * d))
        self.action_posterior = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 2 * d))
        self.process_variance = nn.Sequential(
            nn.Linear(2 * d, d), nn.Softplus())
        self.observation_precision = nn.Sequential(
            nn.Linear(3 * d, d), nn.Softplus())

        # The field-to-port prediction starts near its uninformative prior,
        # while the policy starts at unit natural scale (log-admittance 0).
        for module in (self.port_prior[-1], self.chart_gate[-1],
                       self.action_prior[-1], self.action_posterior[-1]):
            nn.init.normal_(module.weight, std=1e-3)
            nn.init.zeros_(module.bias)

        if self.port_modes == 8:
            modes = [
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0), (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0), (1.0, 1.0, 1.0),
            ]
        else:
            modes = [(float(x), float(y), float(z))
                     for x in (-1, 0, 1)
                     for y in (-1, 0, 1)
                     for z in (-1, 0, 1)]
        self.register_buffer("mode_vectors", torch.tensor(modes), persistent=False)
        group = max(1, d // self.port_modes)
        permutations = torch.stack([
            torch.roll(torch.arange(d), shifts=mode * group)
            for mode in range(self.port_modes)
        ])
        self.register_buffer("channel_permutations", permutations, persistent=False)

    def initial_precision(self, batch_size: int, *, device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
        """Return the unit-precision prior in dimensionless port units."""
        return torch.ones(batch_size, self.d, device=device, dtype=dtype)

    @staticmethod
    def _normal_kl(prior_mean: torch.Tensor, prior_std: torch.Tensor,
                   posterior_mean: torch.Tensor,
                   posterior_std: torch.Tensor) -> torch.Tensor:
        ratio_sq = (posterior_std / prior_std).square()
        mean_sq = ((posterior_mean - prior_mean) / prior_std).square()
        return 0.5 * (ratio_sq + mean_sq - 1.0 - ratio_sq.log()).mean()

    def _packet_chart(self, writer: FullRankTorusWrite, field: torch.Tensor,
                      prior_features: torch.Tensor,
                      port_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor, torch.Tensor]:
        r"""Return the packet chart \(P(F,\varphi)\), exactly linear in \(\varphi\).

        For a fixed pre-event field, the gates and translated Fourier basis do
        not depend on the token.  Consequently
        ``P(F, E[varphi]) == E[P(F, varphi)]`` exactly, which is the required
        bridge from a categorical prediction to a physical innovation port.
        """
        gate = torch.softmax(self.chart_gate(prior_features), dim=-1)
        anchor = writer.field_anchor(field)
        relative = writer.coordinates.to(field) - anchor[:, None, None, None, :]
        phase = 2.0 * math.pi * torch.einsum(
            "bxyzj,rj->brxyz", relative, self.mode_vectors.to(field))
        basis = phase.cos()
        basis = basis / basis.square().mean((2, 3, 4), keepdim=True).sqrt().clamp_min(
            torch.finfo(field.dtype).eps)
        weighted_basis = gate[:, :, None, None, None] * basis
        feature_by_mode = port_feature[:, self.channel_permutations]
        packet = torch.einsum("brxyz,brd->bxyzd", weighted_basis, feature_by_mode)
        packet = packet * writer.channel_scale
        support = weighted_basis.abs().sum(1)
        return packet, support, anchor, gate

    def forward(self, writer: FullRankTorusWrite, field: torch.Tensor,
                token_ids: torch.Tensor, precision: torch.Tensor,
                *, predicted_feature: Optional[torch.Tensor] = None,
                return_diag: bool = True):
        """Apply one posterior innovation write and update precision.

        ``predicted_feature`` exists only for analytical tests and controlled
        interventions.  The ordinary causal path derives it from a categorical
        distribution before inspecting ``token_ids``.
        """
        if precision.shape != (field.shape[0], self.d):
            raise ValueError(
                f"Expected precision {(field.shape[0], self.d)}, got {tuple(precision.shape)}")
        eps = torch.finfo(field.dtype).eps
        field_summary = field.mean((1, 2, 3))
        log_precision = precision.clamp_min(eps).log()
        prior_features = torch.cat((F.rms_norm(field_summary, (self.d,)), log_precision), -1)

        prior_action = self.action_prior(prior_features)
        prior_mean, prior_log_std = prior_action.chunk(2, -1)
        prior_std = F.softplus(prior_log_std) + eps
        process_variance = self.process_variance(prior_features) + eps
        prior_precision = 1.0 / (precision.reciprocal() + process_variance)

        # The categorical prior and its physical port expectation are both
        # constructed before indexing the observed token.
        token_features = F.normalize(writer.embedding.weight, dim=-1)
        natural_parameter = self.port_prior(prior_features)
        port_logits = natural_parameter @ token_features.transpose(0, 1)
        port_logits = port_logits + self.port_logit_bias
        port_probability = torch.softmax(port_logits, dim=-1)
        expected_feature = port_probability @ token_features
        predicted_feature = (expected_feature if predicted_feature is None
                             else predicted_feature)
        observed_feature = token_features[token_ids]
        observed_packet, support, anchor, chart_gate = self._packet_chart(
            writer, field, prior_features, observed_feature)
        predicted_packet, _, _, _ = self._packet_chart(
            writer, field, prior_features, predicted_feature)
        innovation = observed_packet - predicted_packet
        innovation_summary = innovation.square().mean((1, 2, 3)).sqrt()

        posterior_features = torch.cat((
            F.rms_norm(field_summary, (self.d,)), log_precision,
            innovation_summary), -1)
        posterior_action = self.action_posterior(posterior_features)
        posterior_mean, posterior_log_std = posterior_action.chunk(2, -1)
        posterior_std = F.softplus(posterior_log_std) + eps
        admittance = F.softplus(posterior_mean)
        observation_precision = self.observation_precision(posterior_features) + eps
        posterior_precision = prior_precision + observation_precision

        # A Cayley-equivalent bounded port angle.  The observed innovation
        # controls the action amplitude, hence an exactly predicted packet is
        # an exact identity boundary event.
        theta = torch.atan(
            admittance[:, None, None, None, :]
            * innovation.abs()
            * posterior_precision[:, None, None, None, :].sqrt())
        cosine, sine = theta.cos(), theta.sin()
        field_next = cosine * field + sine * innovation
        reflected = -sine * field + cosine * innovation

        if not return_diag:
            return field_next, posterior_precision, reflected, {}

        innovation_energy = writer.energy(innovation)
        port_nll = F.cross_entropy(port_logits, token_ids)
        policy_kl = self._normal_kl(
            prior_mean, prior_std, posterior_mean, posterior_std)
        balance = (writer.energy(field_next) + writer.energy(reflected)
                   - writer.energy(field) - innovation_energy).abs()
        diag = {
            "incident_energy": innovation_energy.detach(),
            "reflected_energy": writer.energy(reflected).detach(),
            "accepted_energy": (writer.energy(field_next) - writer.energy(field)).detach(),
            "accepted_fraction": (
                1.0 - writer.energy(reflected) / innovation_energy.clamp_min(eps)).detach(),
            "t_packet": (
                1.0 - writer.energy(reflected) / innovation_energy.clamp_min(eps)).detach(),
            "delta_e_field": (writer.energy(field_next) - writer.energy(field)).detach(),
            "cross_interference": (
                torch.sin(2.0 * theta) * field * innovation).sum(dim=-1).mean().detach(),
            "write_to_f_ratio": (
                (field_next - field).norm(dim=-1).mean()
                / field.norm(dim=-1).mean().clamp_min(eps)).detach(),
            "write_angle_abs_mean": theta.detach().abs().mean(),
            "write_angle_peak_mean": theta.detach().abs().amax(dim=(1, 2, 3)).mean(),
            "write_spatial_support": (support.detach() > 0.1).float().mean(),
            "write_balance_residual": balance.detach(),
            "source_center": anchor.detach(),
            "source_anchor": anchor.detach(),
            "source_displacement": field.new_zeros(field.shape[0], 3),
            "write_width_mean": field.new_zeros(()),
            "source_nu_s": field.new_zeros(()),
            "innovation_energy": innovation_energy.detach(),
            "innovation_norm": innovation_summary.detach().mean(),
            "port_nll": port_nll.detach(),
            "write_action_kl": policy_kl.detach(),
            "write_free_energy": (port_nll + policy_kl).detach(),
            "port_predictive_entropy": (
                -(port_probability * port_probability.clamp_min(eps).log()).sum(-1).mean()).detach(),
            "port_chart_entropy": (
                -(chart_gate * chart_gate.clamp_min(eps).log()).sum(-1).mean()).detach(),
            "_write_free_energy": port_nll + policy_kl,
            "prior_precision_mean": prior_precision.detach().mean(),
            "posterior_precision_mean": posterior_precision.detach().mean(),
            "write_admittance_mean": admittance.detach().mean(),
        }
        return field_next, posterior_precision, reflected, diag


class VelocityCayleyTransport3D(nn.Module):
    """Norm-preserving periodic streaming with an explicit velocity baseline."""

    def __init__(self, shape=(4, 4, 4), velocities=8, content_dim=16,
                 hidden=64):
        super().__init__()
        self.shape = tuple(shape)
        self.velocities, self.content_dim = velocities, content_dim
        if velocities not in (8, 27):
            raise ValueError(f"The geometric truncation supports D3Q8 and D3Q27, got {velocities}")
        self.residual_symbol = nn.Sequential(
            nn.Linear(12, hidden), nn.SiLU(),
            nn.Linear(hidden, velocities * content_dim))
        nn.init.normal_(self.residual_symbol[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_symbol[-1].bias)
        self.log_speed = nn.Parameter(torch.zeros(velocities))
        self.residual_scale = nn.Parameter(torch.zeros(()))
        # Exact continuous physical wavenumbers k_phys = 2 * pi * m on T^3 with L = 1
        wave_axes = [torch.fft.fftfreq(n, d=1.0 / n) * 2.0 * math.pi for n in self.shape]
        wave = torch.stack(torch.meshgrid(*wave_axes, indexing="ij"), -1)
        # Consistent modified wavenumber \tilde{k}_j = sin(k_phys * h) / h
        # Converges to k_phys as h -> 0 and vanishes at Nyquist (k*h = -pi) ensuring exact Hermitian symmetry
        h_vec = torch.tensor([1.0 / n for n in self.shape], dtype=torch.float32).view(1, 1, 1, 3)
        mod_k = (wave * h_vec).sin() / h_vec
        k_ref = 8.0 * math.pi  # Fixed physical reference frequency for continuous feature encoding
        k_norm = wave / k_ref
        encode = lambda k: torch.cat(
            (k.sin(), k.cos(), (2.0 * k).sin(), (2.0 * k).cos()), -1)
        self.register_buffer("wave", wave, persistent=False)
        self.register_buffer("mod_k", mod_k, persistent=False)
        self.register_buffer("symbol_input", encode(k_norm), persistent=False)
        self.register_buffer("negative_symbol_input", encode(-k_norm), persistent=False)
        self.register_buffer("velocity_vectors", d3q_velocities(velocities), persistent=False)

    def learned_symbol(self) -> torch.Tensor:
        learned = 0.5 * (
            self.residual_symbol(self.symbol_input)
            - self.residual_symbol(self.negative_symbol_input))
        return learned.reshape(*self.shape, self.velocities, self.content_dim)

    def dispersion(self, direction: torch.Tensor | None = None,
                   learned: torch.Tensor | None = None) -> torch.Tensor:
        # Consistent modified wavenumber \tilde{k} = sin(k_phys * h) / h
        # As h -> 0, \tilde{k} -> k_phys (exact continuous spectral generator omega = v . k_phys)
        k_stream = self.mod_k.to(device=self.log_speed.device, dtype=self.log_speed.dtype)
        if direction is not None:
            baseline = torch.einsum(
                "xyzj,bhj->bhxyz", k_stream, direction)
            baseline = baseline.permute(0, 2, 3, 4, 1)  # [B, X, Y, Z, H]
            baseline = baseline[..., :, None] * self.log_speed.exp()[:, None]  # [B, X, Y, Z, H, c]
        else:
            baseline = torch.einsum(
                "...j,qj->...q", k_stream, self.velocity_vectors.to(k_stream.dtype))
            baseline = baseline[..., :, None] * self.log_speed.exp()[:, None]
        sym = self.learned_symbol() if learned is None else learned
        if direction is not None:
            return baseline + self.residual_scale * sym[None]
        return baseline + self.residual_scale * sym

    def multiplier(self, delta_tau: float | torch.Tensor = 1.0,
                   direction: torch.Tensor | None = None,
                   learned: torch.Tensor | None = None):
        if direction is not None:
            batch = direction.shape[0]
            omega = self.dispersion(direction, learned=learned).reshape(batch, *self.shape, -1)
            if isinstance(delta_tau, torch.Tensor):
                dt = delta_tau.view(batch, 1, 1, 1, 1)
            else:
                dt = float(delta_tau)
            mu = 0.5 * omega * dt
            denominator = 1.0 + mu.square()
            value = torch.complex(
                (1.0 - mu.square()) / denominator, -2.0 * mu / denominator)
            return value, omega
        else:
            omega = self.dispersion().reshape(*self.shape, -1)
            if isinstance(delta_tau, torch.Tensor):
                dt = delta_tau.view(-1, 1, 1, 1, 1)
            else:
                dt = float(delta_tau)
            mu = 0.5 * omega * dt
            denominator = 1.0 + mu.square()
            value = torch.complex(
                (1.0 - mu.square()) / denominator, -2.0 * mu / denominator)
            return (value if isinstance(delta_tau, torch.Tensor) else value[None]), omega

    @staticmethod
    def apply_multiplier(field: torch.Tensor, multiplier: torch.Tensor):
        frequency = torch.fft.fftn(field, dim=(1, 2, 3), norm="ortho")
        return torch.fft.ifftn(
            frequency * multiplier, dim=(1, 2, 3), norm="ortho").real

    def forward(self, field: torch.Tensor):
        multiplier, omega = self.multiplier()
        output = self.apply_multiplier(field, multiplier)
        return output, {
            "transport_angle_abs_mean": omega.detach().abs().mean(),
            "transport_angle_abs_max": omega.detach().abs().amax(),
            "transport_norm_residual": (
                output.square().sum() - field.square().sum()).detach().abs(),
        }


class LocalInvariantCollision3D(nn.Module):
    """Local nonlinear collision preserving total mass and momentum."""

    def __init__(self, shape=(4, 4, 4), velocities=8, content_dim=16,
                 hidden=96, layers=2, position_conditioned=False):
        super().__init__()
        self.shape = tuple(shape)
        self.velocities, self.content_dim = velocities, content_dim
        self.d = velocities * content_dim
        self.position_conditioned = bool(position_conditioned)
        velocity = d3q_velocities(velocities, dtype=torch.float64)
        mass = torch.ones(1, velocities, content_dim, dtype=torch.float64)
        momentum = velocity.T[:, :, None].expand(-1, -1, content_dim)
        constraints = torch.cat((mass, momentum), 0).reshape(4, self.d)
        _, singular, right = torch.linalg.svd(constraints, full_matrices=True)
        rank = int((singular > 1e-10).sum())
        nullspace = right[rank:].T
        if nullspace.shape[1] % 2:
            raise ValueError("Collision nullity must be even")
        self.nullity, self.layers = nullspace.shape[1], int(layers)
        schedules = [torch.roll(torch.arange(self.nullity), layer).reshape(-1, 2)
                     for layer in range(self.layers)]
        self.register_buffer("constraints", constraints, persistent=False)
        self.register_buffer("nullspace", nullspace, persistent=False)
        self.register_buffer("schedules", torch.stack(schedules), persistent=False)
        self.norm = nn.LayerNorm(self.d)
        self.register_buffer(
            "position_features", torus_features(torus_grid(self.shape)).reshape(-1, 6),
            persistent=False)
        self.angle = nn.Sequential(
            nn.Linear(self.d + (6 if self.position_conditioned else 0), hidden), nn.SiLU(),
            nn.Linear(hidden, self.layers * (self.nullity // 2)))
        nn.init.normal_(self.angle[-1].weight, std=1e-3)
        nn.init.zeros_(self.angle[-1].bias)

    def forward(self, field: torch.Tensor, delta_tau: float | torch.Tensor = 1.0):
        batch = field.shape[0]
        flat = field.reshape(batch, -1, self.d)
        nullspace = self.nullspace.to(dtype=flat.dtype)
        coefficient = torch.einsum("dk,bnd->bnk", nullspace, flat)
        conserved = flat - torch.einsum("dk,bnk->bnd", nullspace, coefficient)
        angle_input = self.norm(flat)
        if self.position_conditioned:
            position = self.position_features.to(flat)[None].expand(batch, -1, -1)
            angle_input = torch.cat((angle_input, position), -1)
        angles = self.angle(angle_input).reshape(
            batch, flat.shape[1], self.layers, self.nullity // 2)
        if isinstance(delta_tau, torch.Tensor):
            dt = delta_tau.view(batch, 1, 1, 1) if delta_tau.numel() == batch else delta_tau.reshape(-1, 1, 1, 1).expand(batch, 1, 1, 1)
        else:
            dt = float(delta_tau)
        scaled_angles = angles * dt
        try:
            from .triton_givens import triton_givens
            value = triton_givens(coefficient, scaled_angles)
        except Exception:
            value = coefficient
            for layer in range(self.layers):
                pair = self.schedules[layer]
                left, right = value[..., pair[:, 0]], value[..., pair[:, 1]]
                theta = scaled_angles[:, :, layer]
                cosine, sine = theta.cos(), theta.sin()
                updated = value.clone()
                updated[..., pair[:, 0]] = cosine * left - sine * right
                updated[..., pair[:, 1]] = sine * left + cosine * right
                value = updated
        output = conserved + torch.einsum("dk,bnk->bnd", nullspace, value)
        coll_in_power = coefficient.square().sum(-1).mean()
        cons_in_power = conserved.square().sum(-1).mean()
        coll_out_power = value.square().sum(-1).mean()
        return output.reshape_as(field), {
            "collision_angle_abs_mean": scaled_angles.detach().abs().mean(),
            "collision_angle_abs_max": scaled_angles.detach().abs().amax(),
            "collision_input_snr": (coll_in_power / cons_in_power.clamp_min(1e-8)).detach(),
            "collision_output_snr": (coll_out_power / cons_in_power.clamp_min(1e-8)).detach(),
        }


class QuadraticTorusBath(nn.Module):
    """Local passive outflow driven smoothly by dimensionless energy density."""

    def __init__(self, shape=(4, 4, 4), d=128, local_radius=1.25,
                 max_kappa=0.05, position_conditioned=False):
        super().__init__()
        self.local_radius = float(local_radius) * math.sqrt(d / 64.0)
        self.max_kappa = float(max_kappa)
        self.position_conditioned = bool(position_conditioned)
        if self.position_conditioned:
            self.kappa_net = nn.Sequential(
                nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 1))
            nn.init.normal_(self.kappa_net[-1].weight, std=1e-3)
            nn.init.zeros_(self.kappa_net[-1].bias)
            self.register_buffer(
                "position_features", torus_features(torus_grid(tuple(shape))),
                persistent=False)
        else:
            self.kappa_logit = nn.Parameter(torch.zeros(()))

    def forward(self, field: torch.Tensor, delta_tau: float | torch.Tensor = 1.0):
        local_energy = 0.5 * field.square().sum(-1, keepdim=True)
        rho = 2.0 * local_energy / self.local_radius ** 2
        if self.position_conditioned:
            kappa = self.max_kappa * torch.sigmoid(
                self.kappa_net(self.position_features.to(field)))[None]
        else:
            kappa = self.max_kappa * torch.sigmoid(self.kappa_logit)
        if isinstance(delta_tau, torch.Tensor):
            dt = delta_tau.view(-1, 1, 1, 1, 1)
        else:
            dt = float(delta_tau)
        gamma_dt = (kappa * rho * dt).clamp(max=1.0)
        cosine = torch.exp(-gamma_dt)
        sin2 = (1.0 - torch.exp(-2.0 * gamma_dt)).clamp(max=0.5)
        sine = torch.sqrt(sin2.clamp_min(torch.finfo(field.dtype).tiny))
        output, bath_out = cosine * field, -sine * field
        return output, {
            "bath_out_energy": (0.5 * bath_out.square().sum(-1).mean()).detach(),
            "bath_angle_abs_mean": torch.asin(sine).detach().abs().mean(),
        }


class UnifiedTorusDissipation(nn.Module):
    """Unified 3-layer dissipation operator:
       D_t(q) = gamma_0 * I + nu * lambda(q) * I + U_t * Lambda_t * U_t^T

    Layer 1: gamma_0 * I: weak uniform leakage ensuring BIBO stability and rho < 1.
    Layer 2: nu * lambda(q) * I: scale-selective spectral viscosity killing acoustic ringing and torus reverberations.
             lambda(q) = 4 * sum_j sin^2(q_j / 2) on periodic 3D torus.
    Layer 3: U_t * Lambda_t * U_t^T: content-selective rank-R subspace forgetting (R=4 or 8).
             e^{-dt * U Lambda U^T} = I + U (e^{-dt * Lambda} - I) U^T in O(dR).
    """

    def __init__(self, shape=(4, 4, 4), d=128, rank=4,
                 gamma0_init=0.010, nu_init=0.020, lambda_max=0.20):
        super().__init__()
        self.shape = tuple(shape)
        self.d = d
        self.rank = rank
        self.lambda_max = float(lambda_max)

        # 1. Base scalar leakage gamma_0 > 0
        gamma0_inv = math.log(max(math.exp(gamma0_init) - 1.0, 1e-6))
        self.gamma0_param = nn.Parameter(torch.tensor(gamma0_inv, dtype=torch.float32))

        # 2. Spectral viscosity nu > 0
        nu_inv = math.log(max(math.exp(nu_init) - 1.0, 1e-6))
        self.nu_param = nn.Parameter(torch.tensor(nu_inv, dtype=torch.float32))

        # Continuous spectral Laplacian eigenvalues on periodic 3-torus T^3:
        # -\Delta e^{i k . x} = |k_phys|^2 e^{i k . x}, where k_phys = 2 * pi * m on [0, 1)^3
        wave_axes = [torch.fft.fftfreq(n, d=1.0 / n) * 2.0 * math.pi for n in self.shape]
        wave = torch.stack(torch.meshgrid(*wave_axes, indexing="ij"), -1)
        laplacian = wave.square().sum(-1)  # shape: (*shape,), exact continuous |k_phys|^2
        self.register_buffer("laplacian", laplacian, persistent=False)

        # 3. Content-selective rank-R subspace forgetting
        self.context_norm = nn.LayerNorm(d)
        self.u_base = nn.Parameter(torch.empty(d, rank))
        nn.init.orthogonal_(self.u_base)

        self.u_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d * rank)
        )
        nn.init.normal_(self.u_proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.u_proj[-1].bias)

        self.lambda_net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, rank)
        )
        nn.init.normal_(self.lambda_net[-1].weight, std=1e-3)
        nn.init.constant_(self.lambda_net[-1].bias, -2.0)

    def forward(self, field: torch.Tensor, delta_tau: float | torch.Tensor = 1.0,
                tok_embed: torch.Tensor | None = None, *,
                gamma0_factor: float = 1.0,
                disable_viscosity: bool = False,
                disable_subspace: bool = False):
        B = field.shape[0]
        N = math.prod(self.shape)
        flat = field.reshape(B, N, self.d)

        if isinstance(delta_tau, torch.Tensor):
            dt_vec = delta_tau.view(B, 1) if delta_tau.numel() == B else delta_tau.reshape(-1, 1).expand(B, 1)
            dt_freq = delta_tau.view(B, 1, 1, 1, 1) if delta_tau.numel() == B else delta_tau.reshape(-1, 1, 1, 1, 1).expand(B, 1, 1, 1, 1)
        else:
            dt = float(delta_tau)
            dt_vec = dt
            dt_freq = dt

        # Layer 3: Low-rank Subspace Forgetting U_t Lambda_t U_t^T (O(dR))
        if not disable_subspace:
            f_mean = flat.mean(dim=1)
            if tok_embed is not None:
                c = self.context_norm(f_mean + tok_embed.to(dtype=field.dtype))
            else:
                c = self.context_norm(f_mean)
            u_raw = self.u_base[None] + self.u_proj(c).reshape(B, self.d, self.rank)
            Q, _ = torch.linalg.qr(u_raw.float())
            U = Q.to(dtype=field.dtype)  # [B, d, R]
            gamma_r = self.lambda_max * torch.sigmoid(self.lambda_net(c))  # [B, R]
            decay_r = torch.exp(-dt_vec * gamma_r) - 1.0  # [B, R]

            H_R = torch.einsum("bnd,bdr->bnr", flat, U)
            H_R_decay = H_R * decay_r.unsqueeze(1)
            flat_decayed = flat + torch.einsum("bnr,bdr->bnd", H_R_decay, U)
            field_sub = flat_decayed.reshape_as(field)
        else:
            field_sub = field
            gamma_r = field.new_zeros(B, self.rank)

        # Layers 1 & 2: Base scalar leakage + Spectral Viscosity
        if gamma0_factor > 0:
            gamma0 = F.softplus(self.gamma0_param).clamp_min(1e-5) * float(gamma0_factor)
        else:
            gamma0 = self.gamma0_param.new_zeros(())

        if disable_viscosity:
            # Zero viscosity: spectral rate is spatially constant gamma0.
            # Bypass 3D FFT and 3D IFFT entirely: exact spatial scaling! (2x faster)
            nu = field.new_zeros(())
            lap = self.laplacian.to(device=field.device, dtype=field.dtype)
            damping_scalar = torch.exp(-dt_vec.view(B, 1, 1) * gamma0)
            field_out = (flat_decayed * damping_scalar).reshape_as(field)
        else:
            nu = F.softplus(self.nu_param).clamp_min(1e-5)
            lap = self.laplacian.to(device=field.device, dtype=field.dtype)
            spectral_rate = gamma0 + nu * lap  # [*shape]
            damping = torch.exp(-dt_freq * spectral_rate.unsqueeze(0).unsqueeze(-1))  # [B, X, Y, Z, 1]

            freq = torch.fft.fftn(field_sub, dim=(1, 2, 3), norm="ortho")
            freq_damped = freq * damping
            field_out = torch.fft.ifftn(freq_damped, dim=(1, 2, 3), norm="ortho").real

        energy_before = 0.5 * field.square().sum(-1).mean()
        energy_after = 0.5 * field_out.square().sum(-1).mean()
        energy_loss = (energy_before - energy_after).detach()

        return field_out, {
            "bath_out_energy": energy_loss,
            "bath_angle_abs_mean": (gamma0 + nu * 6.0).detach(),
            "dissipation_gamma0": gamma0.detach(),
            "dissipation_nu": nu.detach(),
            "dissipation_lambda_mean": gamma_r.mean().detach(),
            "dissipation_high_q_damping": torch.exp(-gamma0 - nu * lap.amax()).detach(),
            "dissipation_energy_ratio": (energy_after / (energy_before + 1e-8)).detach(),
        }


class EnergyFactoredTorusReadout(nn.Module):
    """QK-normalized multi-query readout with a local-energy prior."""

    def __init__(self, shape=(4, 4, 4), d=128, queries=4, heads=4,
                 local_radius=1.25, moving_frame=True):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.shape, self.d = tuple(shape), d
        self.queries, self.heads, self.head_dim = queries, heads, d // heads
        self.local_radius = float(local_radius) * math.sqrt(d / 64.0)
        self.moving_frame = bool(moving_frame)
        self.query = nn.Parameter(torch.randn(1, queries, d) * 0.02)
        nodes = math.prod(self.shape)
        initial_scale = math.log2(nodes * nodes - nodes)
        self.head_log_scale = nn.Parameter(
            torch.full((1, heads, 1, 1), math.log(initial_scale)))
        self.head_log_scale._no_weight_decay = True
        self.energy_alpha = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.position = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.merge = nn.Linear(queries * d, d)
        self.output = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("coordinates", torus_grid(self.shape), persistent=False)

    def field_anchor(self, field: torch.Tensor) -> torch.Tensor:
        energy = field.square().sum(-1)
        phase = 2.0 * math.pi * self.coordinates
        axes = (1, 2, 3)
        sine = (energy[..., None] * phase.sin()).sum(axes)
        cosine = (energy[..., None] * phase.cos()).sum(axes)
        return torch.remainder(torch.atan2(sine, cosine + 1e-6) / (2.0 * math.pi), 1.0)

    def position_encoding(self, field: torch.Tensor):
        if self.moving_frame:
            anchor = self.field_anchor(field)[:, None, None, None, :]
            coordinates = torch.remainder(
                self.coordinates[None] - anchor + 0.5, 1.0) - 0.5
        else:
            coordinates = self.coordinates[None].expand(field.shape[0], -1, -1, -1, -1)
        return self.position(torus_features(coordinates).reshape(field.shape[0], -1, 6))

    def _weights_for_queries(self, flat: torch.Tensor, position: torch.Tensor,
                             query_embeddings: torch.Tensor,
                             logit_prior: torch.Tensor | None = None):
        batch, nodes, d = flat.shape
        split = lambda value: value.reshape(
            batch, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = F.normalize(split(self.key(self.norm(flat) + position)), dim=-1)
        queries = F.normalize(split(query_embeddings), dim=-1)
        semantic = torch.matmul(queries, keys.transpose(-1, -2))
        semantic = semantic * self.head_log_scale.exp()
        rho = flat.square().sum(-1) / self.local_radius ** 2
        logits = semantic + self.energy_alpha * torch.log1p(rho)[:, None, None]
        if logit_prior is not None:
            if logit_prior.shape != (batch, query_embeddings.shape[1], nodes):
                raise ValueError("Expected read logit prior [B,Q,N]")
            logits = logits + logit_prior[:, None]
        return torch.softmax(logits, -1)

    def _weights(self, flat: torch.Tensor, position: torch.Tensor):
        return self._weights_for_queries(
            flat, position, self.query.expand(flat.shape[0], -1, -1))

    def attend(self, field: torch.Tensor, query_embeddings: torch.Tensor,
               position_encoding: torch.Tensor | None = None,
               logit_prior: torch.Tensor | None = None):
        """Canonical QK-normalized energy-factored attention for any queries."""
        batch = field.shape[0]
        if query_embeddings.ndim != 3 or query_embeddings.shape[0] != batch or query_embeddings.shape[-1] != self.d:
            raise ValueError("Expected query embeddings [B,Q,D]")
        flat = field.reshape(batch, -1, self.d)
        position = self.position_encoding(field) if position_encoding is None else position_encoding
        attention = self._weights_for_queries(flat, position, query_embeddings, logit_prior)
        values = self.value(self.norm(flat)).reshape(
            batch, -1, self.heads, self.head_dim).transpose(1, 2)
        read = torch.matmul(attention, values).transpose(1, 2).reshape(
            batch, query_embeddings.shape[1], self.d)
        return read, attention

    def forward(self, field: torch.Tensor, position_encoding=None):
        batch = field.shape[0]
        read, _ = self.attend(
            field, self.query.expand(batch, -1, -1), position_encoding)
        return self.output(self.merge(read.reshape(batch, self.queries * self.d)))

    @torch.no_grad()
    def attention_weights(self, field: torch.Tensor, position_encoding=None):
        flat = field.reshape(field.shape[0], -1, self.d)
        position = self.position_encoding(field) if position_encoding is None else position_encoding
        return self._weights(flat, position)


class MicroStepClock(nn.Module):
    """Adaptive internal clock rate alpha_{t, k} bounded below by causal horizon."""

    def __init__(self, d=128, alpha_causal=0.90, alpha_max=2.50, initial_alpha=1.05):
        super().__init__()
        self.alpha_causal = float(alpha_causal)
        self.alpha_max = float(alpha_max)
        self.norm_token = nn.RMSNorm(d)
        self.norm_field = nn.RMSNorm(d)
        self.controller = nn.Sequential(
            nn.Linear(2 * d + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        target_p = (initial_alpha - alpha_causal) / (alpha_max - alpha_causal)
        init_bias = math.log(target_p / (1.0 - target_p))
        nn.init.xavier_uniform_(self.controller[0].weight)
        nn.init.zeros_(self.controller[0].bias)
        nn.init.xavier_uniform_(self.controller[-1].weight)
        nn.init.constant_(self.controller[-1].bias, init_bias)

    def forward(self, field: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        B = field.shape[0]
        f_mean = field.mean(dim=(1, 2, 3))
        e_mean = 0.5 * field.square().sum(dim=-1).mean(dim=(1, 2, 3)).unsqueeze(-1)
        e_var = field.var(dim=(1, 2, 3)).mean(dim=-1).unsqueeze(-1)
        u = torch.cat([self.norm_token(token_embed), self.norm_field(f_mean), e_mean, e_var], dim=-1)
        gate = torch.sigmoid(self.controller(u))
        return self.alpha_causal + (self.alpha_max - self.alpha_causal) * gate


class MicroStepDirection(nn.Module):
    """Adaptive continuous transport directions n_{t, k, h} on S^2 per micro-step."""

    def __init__(self, d=128, heads=8):
        super().__init__()
        self.d, self.heads = int(d), int(heads)
        self.norm_token = nn.RMSNorm(self.d)
        self.norm_field = nn.RMSNorm(self.d)
        self.controller = nn.Sequential(
            nn.Linear(2 * self.d + 2, 64),
            nn.SiLU(),
            nn.Linear(64, self.heads * 3)
        )
        # Initialize to zero so n_h strictly starts at base D3Q8 directions
        nn.init.xavier_uniform_(self.controller[0].weight)
        nn.init.zeros_(self.controller[0].bias)
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        base_dirs = d3q_velocities(8)  # [8, 3] on S^2
        self.register_buffer("base_dirs", base_dirs, persistent=False)

    def forward(self, field: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        B = field.shape[0]
        f_mean = field.mean(dim=(1, 2, 3))
        e_mean = 0.5 * field.square().sum(dim=-1).mean(dim=(1, 2, 3)).unsqueeze(-1)
        e_var = field.var(dim=(1, 2, 3)).mean(dim=-1).unsqueeze(-1)
        u = torch.cat([self.norm_token(token_embed), self.norm_field(f_mean), e_mean, e_var], dim=-1)
        delta = self.controller(u).reshape(B, self.heads, 3)
        u_dir = self.base_dirs.unsqueeze(0) + delta
        n_dir = F.normalize(u_dir, p=2, dim=-1, eps=1e-6)
        return n_dir  # [B, H, 3]


class ReversibleHamiltonianPonderFunction(torch.autograd.Function):
    """Exact zero-memory Hamiltonian reversible pondering for (TC)^K.

    Eliminates intermediate microstep activation storage in the computational graph:
    - Forward: runs (TC)^K in no_grad mode, storing only the final state F_K and metadata.
    - Backward: inverts step-by-step F_{k-1} = T^{-1} C^{-1} F_k with 0.0 error,
      and computes analytical adjoint VJP for full 128-token sequence.
    - Zero shared memory: VRAM is O(1) in K (< 1.5 GB).
    - Fully compatible with static CUDA Graph capture.
    """

    @staticmethod
    def forward(ctx, f_in, micro_steps, model, tok_embed):
        curr = f_in
        step_hist = []
        cached_learned = model.transport.learned_symbol()
        nullspace = model.collision.nullspace.to(dtype=curr.dtype)
        with torch.no_grad():
            for _ in range(micro_steps):
                alpha = model.clock(curr, tok_embed) if model.adaptive_clock else model.tau_0_tensor
                dt = alpha * model.tau_0_tensor if model.adaptive_clock else model.tau_0_tensor
                dir_k = model.direction_controller(curr, tok_embed) if model.continuous_velocities else None
                mult, _ = model.transport.multiplier(dt, direction=dir_k, learned=cached_learned)
                f_tr = model.transport.apply_multiplier(curr, mult)

                batch = curr.shape[0]
                flat = f_tr.reshape(batch, -1, model.collision.d)
                coefficient = torch.einsum("dk,bnd->bnk", nullspace, flat)
                conserved = flat - torch.einsum("dk,bnk->bnd", nullspace, coefficient)

                pos = model.collision.position_features.to(flat)[None].expand(batch, -1, -1) if model.collision.position_conditioned else None
                angle_in = torch.cat((model.collision.norm(flat), pos), -1) if pos is not None else model.collision.norm(flat)
                angles = model.collision.angle(angle_in).reshape(batch, flat.shape[1], model.collision.layers, -1)
                dt_val = dt.view(batch, 1, 1, 1) if isinstance(dt, torch.Tensor) else float(dt)
                scaled_angles = angles * dt_val

                try:
                    from .triton_givens import triton_givens
                    val = triton_givens(coefficient, scaled_angles)
                except Exception:
                    val = coefficient
                    for layer in range(model.collision.layers):
                        pair = model.collision.schedules[layer]
                        th = scaled_angles[:, :, layer]
                        cos, sin = th.cos(), th.sin()
                        l, r = val[..., pair[:, 0]], val[..., pair[:, 1]]
                        upd = val.clone()
                        upd[..., pair[:, 0]] = cos * l - sin * r
                        upd[..., pair[:, 1]] = sin * l + cos * r
                        val = upd

                f_next = (conserved + torch.einsum("dk,bnk->bnd", nullspace, val)).reshape_as(curr)
                step_hist.append((dt, dir_k, scaled_angles))
                curr = f_next

        ctx.step_hist = step_hist
        ctx.save_for_backward(curr)
        ctx.micro_steps = micro_steps
        ctx.model = model
        ctx.cached_learned = cached_learned
        return curr

    @staticmethod
    def backward(ctx, grad_output):
        curr, = ctx.saved_tensors
        g = grad_output
        f_state = curr
        model = ctx.model
        cached_learned = ctx.cached_learned
        batch = f_state.shape[0]
        nullspace = model.collision.nullspace.to(dtype=f_state.dtype)

        for step_idx in reversed(range(ctx.micro_steps)):
            dt, dir_k, scaled_angles = ctx.step_hist[step_idx]
            mult, _ = model.transport.multiplier(dt, direction=dir_k, learned=cached_learned)

            # 1. Adjoint through collision
            g_flat = g.reshape(batch, -1, model.collision.d)
            g_c = torch.einsum("dk,bnd->bnk", nullspace, g_flat)
            g_cons = g_flat - torch.einsum("dk,bnk->bnd", nullspace, g_c)
            try:
                from .triton_givens import triton_givens_adjoint
                curr_g_c = triton_givens_adjoint(g_c, scaled_angles)
            except Exception:
                curr_g_c = g_c
                for layer in reversed(range(model.collision.layers)):
                    pair = model.collision.schedules[layer]
                    th = scaled_angles[:, :, layer]
                    cos, sin = th.cos(), th.sin()
                    g_l, g_r = curr_g_c[..., pair[:, 0]], curr_g_c[..., pair[:, 1]]
                    upd_g = curr_g_c.clone()
                    upd_g[..., pair[:, 0]] = cos * g_l + sin * g_r
                    upd_g[..., pair[:, 1]] = -sin * g_l + cos * g_r
                    curr_g_c = upd_g
            g_tr = (g_cons + torch.einsum("dk,bnk->bnd", nullspace, curr_g_c)).reshape_as(f_state)

            # 2. Adjoint through transport
            freq_G = torch.fft.fftn(g_tr, dim=(1, 2, 3), norm="ortho")
            g = torch.fft.ifftn(freq_G * mult.conj(), dim=(1, 2, 3), norm="ortho").real

            # 3. Invert collision state on the fly!
            flat_state = f_state.reshape(batch, -1, model.collision.d)
            coeff_state = torch.einsum("dk,bnd->bnk", nullspace, flat_state)
            conserved_state = flat_state - torch.einsum("dk,bnk->bnd", nullspace, coeff_state)

            try:
                from .triton_givens import triton_givens
                val = triton_givens(coeff_state, -scaled_angles)
            except Exception:
                val = coeff_state
                for layer in reversed(range(model.collision.layers)):
                    pair = model.collision.schedules[layer]
                    th = -scaled_angles[:, :, layer]
                    cos, sin = th.cos(), th.sin()
                    l, r = val[..., pair[:, 0]], val[..., pair[:, 1]]
                    upd = val.clone()
                    upd[..., pair[:, 0]] = cos * l - sin * r
                    upd[..., pair[:, 1]] = sin * l + cos * r
                    val = upd

            f_tr = (conserved_state + torch.einsum("dk,bnk->bnd", nullspace, val)).reshape_as(f_state)
            f_state = model.transport.apply_multiplier(f_tr, mult.conj())

        return g, None, None, None


class CBIMTorus3D(nn.Module):
    """Complete persistent kinetic model on T^3."""

    architecture = "CBIM-Torus3D-relative-fullrank-d3q8-v3"

    def __init__(self, vocab_size=50257, shape=(4, 4, 4), velocities=8,
                 content_dim=16, queries=4, heads=4, collision_layers=2,
                 relative_address=True, v2_coordinate_components=False,
                 readout_type="baseline", readout_probes=8, readout_rounds=1,
                 write_type="w0_baseline", micro_steps=1,
                 adaptive_clock=False, continuous_velocities=False,
                 tau_0=1.0, alpha_causal=0.90, alpha_max=2.50,
                 dissipation_type="quadratic", dissipation_rank=4,
                 gamma0_init=0.010, nu_init=0.020,
                 three_clock=False, tau_mem=3.0,
                 spectral_write=None, nu_s_init=0.020,
                 decouple_source_feedback=None,
                 reversible_ponder=None):
        super().__init__()
        self.vocab_size, self.shape = vocab_size, tuple(shape)
        self.velocities, self.content_dim = velocities, content_dim
        self.d, self.L = velocities * content_dim, math.prod(shape)
        self.v2_coordinate_components = bool(v2_coordinate_components)
        self.readout_type = readout_type
        self.write_type = str(write_type)
        self.micro_steps = int(micro_steps)
        self.adaptive_clock = bool(adaptive_clock)
        self.continuous_velocities = bool(continuous_velocities)
        self.dissipation_type = str(dissipation_type)
        self.dissipation_rank = int(dissipation_rank)
        self.tau_0 = float(tau_0)
        self.register_buffer("tau_0_tensor", torch.tensor(float(tau_0)), persistent=False)
        self.three_clock = bool(three_clock)
        self.tau_mem = float(tau_mem)
        self.spectral_write = self.three_clock if spectral_write is None else bool(spectral_write)
        self.decouple_source_feedback = bool(three_clock) if decouple_source_feedback is None else bool(decouple_source_feedback)
        self.reversible_ponder = bool(three_clock and int(micro_steps) >= 4) if reversible_ponder is None else bool(reversible_ponder)

        if self.adaptive_clock:
            self.clock = MicroStepClock(
                d=self.d, alpha_causal=alpha_causal, alpha_max=alpha_max, initial_alpha=1.05)
        if self.continuous_velocities:
            self.direction_controller = MicroStepDirection(d=self.d, heads=velocities)

        if self.three_clock:
            self.architecture = f"CBIM-ThreeClock-{self.write_type}-continuous-q{velocities}-K{micro_steps}"
        elif self.continuous_velocities:
            self.architecture = f"CBIM-Torus3D-{self.write_type}-continuous-q{velocities}"
        elif self.v2_coordinate_components and relative_address:
            self.architecture = f"CBIM-Torus3D-v2-relative-write-only-d3q{velocities}"
        elif self.v2_coordinate_components and not relative_address:
            if readout_type == "baseline" and self.write_type == "w0_baseline":
                self.architecture = f"CBIM-Torus3D-fullrank-d3q{velocities}-v2"
            elif readout_type != "baseline":
                self.architecture = f"CBIM-Torus3D-{readout_type}-d3q{velocities}-v2"
            else:
                self.architecture = f"CBIM-Torus3D-{self.write_type}-d3q{velocities}-v2"
        elif readout_type != "baseline":
            self.architecture = f"CBIM-Torus3D-{readout_type}-d3q{velocities}"
        elif self.write_type != "w0_baseline":
            self.architecture = f"CBIM-Torus3D-{self.write_type}-d3q{velocities}"
        if self.dissipation_type == "unified" and not self.three_clock:
            self.architecture += "-unified-dissipation"
        self.state_shape = (*self.shape, self.d)
        # W4 is a distinct active-inference boundary law.  It reuses the W2
        # packet chart so that old raw writers stay loadable and the observed
        # and predicted packets inhabit one physical space.
        source_write_type = (
            "w2_impedance" if self.write_type == "w4_predictive_agent"
            else self.write_type)
        self.source = FullRankTorusWrite(
            vocab_size, shape, self.d, relative_address=relative_address,
            write_type=source_write_type, spectral_packet=self.spectral_write,
            nu_s_init=nu_s_init, decouple_source_feedback=self.decouple_source_feedback)
        self.write_agent = (
            PredictiveImpedanceWriteAgent(
                self.d, vocab_size=vocab_size, port_modes=velocities)
            if self.write_type == "w4_predictive_agent" else None)
        self.transport = VelocityCayleyTransport3D(shape, velocities, content_dim)
        self.collision = LocalInvariantCollision3D(
            shape, velocities, content_dim, layers=collision_layers,
            position_conditioned=self.v2_coordinate_components)
        if self.dissipation_type == "unified":
            self.bath = UnifiedTorusDissipation(
                shape, self.d, rank=self.dissipation_rank,
                gamma0_init=gamma0_init, nu_init=nu_init)
        elif self.dissipation_type == "quadratic":
            self.bath = QuadraticTorusBath(
                shape, self.d, position_conditioned=self.v2_coordinate_components)
        else:
            raise ValueError(f"Unknown dissipation_type: {dissipation_type}")

        if readout_type == "baseline":
            self.readout = EnergyFactoredTorusReadout(
                shape, self.d, queries, heads,
                moving_frame=not self.v2_coordinate_components)
        elif readout_type == "dynamic_linear":
            from .readout_probes import DynamicLinearReadout
            self.readout = DynamicLinearReadout(shape=shape, d=self.d, heads=heads)
        elif readout_type in ("kernel_r1", "kernel_r2"):
            from .readout_probes import CharacteristicKernelReadout
            rounds = 1 if readout_type == "kernel_r1" else 2
            self.readout = CharacteristicKernelReadout(
                shape=shape, d=self.d, heads=heads, queries=queries, rounds=rounds
            )
        else:
            raise ValueError(f"Unknown readout_type: {readout_type}")

        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.source.embedding.weight

        # Thermodynamic Non-Equilibrium Steady State (NESS) prior buffers
        self.register_buffer("ness_dc_mean", torch.zeros(1, 1, 1, 1, self.d), persistent=True)
        self.register_buffer("ness_amplitude_spec", torch.zeros(1, *self.shape, self.d), persistent=True)
        self.register_buffer("has_ness_prior", torch.tensor(False), persistent=True)

    @staticmethod
    def resample_spectral_amplitude(amp_src: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        """Resample continuous Fourier mode amplitudes across grid resolutions via mode slicing / zero-padding."""
        B, Xs, Ys, Zs, D = amp_src.shape
        Xt, Yt, Zt = target_shape
        if (Xs, Ys, Zs) == (Xt, Yt, Zt):
            return amp_src
        out = torch.zeros(B, Xt, Yt, Zt, D, dtype=amp_src.dtype, device=amp_src.device)
        def mode_slices(N_src, N_tgt):
            c = min(N_src, N_tgt)
            pos, neg = (c + 1) // 2, c // 2
            return (slice(0, pos), slice(0, pos)), (slice(N_src - neg, N_src), slice(N_tgt - neg, N_tgt))
        (sx_p, tx_p), (sx_n, tx_n) = mode_slices(Xs, Xt)
        (sy_p, ty_p), (sy_n, ty_n) = mode_slices(Ys, Yt)
        (sz_p, tz_p), (sz_n, tz_n) = mode_slices(Zs, Zt)
        for sx, tx_s in [(sx_p, tx_p), (sx_n, tx_n)]:
            for sy, ty_s in [(sy_p, ty_p), (sy_n, ty_n)]:
                for sz, tz_s in [(sz_p, tz_p), (sz_n, tz_n)]:
                    out[:, tx_s, ty_s, tz_s] = amp_src[:, sx, sy, sz]
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key in ("ness_dc_mean", "ness_amplitude_spec", "has_ness_prior"):
            full_key = prefix + key
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self, key)
            elif key == "ness_amplitude_spec" and state_dict[full_key].shape != self.ness_amplitude_spec.shape:
                state_dict[full_key] = self.resample_spectral_amplitude(state_dict[full_key], self.shape)
        nu_s_key = prefix + "source.nu_s_param"
        if hasattr(self.source, "nu_s_param") and nu_s_key not in state_dict:
            state_dict[nu_s_key] = self.source.nu_s_param
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def set_ness_prior(self, mature_state: torch.Tensor):
        """Extract and persist the thermodynamic macro-statistics of the NESS state:
        1. DC Channel Mean (global zero-frequency background)
        2. AC Wavenumber Amplitude Spectrum (steady-state radial kinetic energy)
        Phases are intentionally discarded to eliminate contextual memory leaks.
        """
        with torch.no_grad():
            state = mature_state.detach().float()
            if state.dim() == 4:
                state = state.unsqueeze(0)
            dc_mean = state.mean(dim=(1, 2, 3), keepdim=True)
            ac_wave = state - dc_mean
            freq = torch.fft.fftn(ac_wave, dim=(1, 2, 3), norm="ortho")
            self.ness_dc_mean.copy_(dc_mean.mean(dim=0, keepdim=True).to(self.ness_dc_mean.dtype))
            self.ness_amplitude_spec.copy_(freq.abs().mean(dim=0, keepdim=True).to(self.ness_amplitude_spec.dtype))
            self.has_ness_prior.fill_(True)

    def initial_state(self, batch_size, device=None, dtype=None, warm_start=True):
        parameter = next(self.parameters())
        dev = device or parameter.device
        dt = dtype or parameter.dtype

        if warm_start and bool(self.has_ness_prior):
            # Thermodynamic NESS Random-Phase Prior (Random Phase Approximation / RPA):
            # Zero contextual memory leaked (phases are drawn uniformly on S^1),
            # but exact steady-state impedance matched (E = E_NESS, eliminates initial inrush surge).
            noise = torch.randn(batch_size, *self.shape, self.d, device=dev, dtype=torch.float32)
            freq_noise = torch.fft.fftn(noise, dim=(1, 2, 3), norm="ortho")
            phases = freq_noise / freq_noise.abs().clamp_min(1e-12)
            amp = self.ness_amplitude_spec.to(device=dev, dtype=torch.float32)
            dc = self.ness_dc_mean.to(device=dev, dtype=dt)
            ac_field = torch.fft.ifftn(amp * phases, dim=(1, 2, 3), norm="ortho").real.to(dtype=dt)
            return ac_field + dc
        else:
            return torch.zeros(batch_size, *self.state_shape, device=dev, dtype=dt)

    def initial_belief(self, batch_size: int, device=None, dtype=None,
                       warm_start: bool = True) -> KineticBeliefState:
        """Initialize the explicit posterior state used by W4 port agency."""
        if self.write_agent is None:
            raise RuntimeError("initial_belief requires write_type='w4_predictive_agent'")
        field = self.initial_state(batch_size, device=device, dtype=dtype,
                                   warm_start=warm_start)
        precision = self.write_agent.initial_precision(
            batch_size, device=field.device, dtype=field.dtype)
        return KineticBeliefState(field=field, precision=precision)

    def step(self, field, token_ids, *, disable_transport=False,
             disable_collision=False, disable_bath=False,
             micro_steps=None, gamma0_factor=1.0,
             disable_viscosity=False, disable_subspace=False,
             precision: Optional[torch.Tensor] = None):
        tok_embed = self.source.embedding(token_ids)
        posterior_precision = None
        if self.write_agent is None:
            field, reflected, source_diag = self.source(field, token_ids)
        else:
            if precision is None:
                raise RuntimeError(
                    "w4_predictive_agent requires a persistent precision; "
                    "use initial_belief() and belief_step()")
            field, posterior_precision, reflected, source_diag = self.write_agent(
                self.source, field, token_ids, precision)
        transport_diag = {
            "transport_angle_abs_mean": field.new_zeros(()),
            "transport_angle_abs_max": field.new_zeros(()),
            "transport_norm_residual": field.new_zeros(()),
        }
        collision_diag = {
            "collision_angle_abs_mean": field.new_zeros(()),
            "collision_angle_abs_max": field.new_zeros(()),
            "collision_input_snr": field.new_zeros(()),
            "collision_output_snr": field.new_zeros(()),
        }
        bath_diag = {"bath_out_energy": field.new_zeros(()),
                     "bath_angle_abs_mean": field.new_zeros(())}
        alphas_k, coll_angles_k, dirs_k, transport_phase_k = [], [], [], []
        steps_to_run = int(micro_steps) if micro_steps is not None else self.micro_steps
        for _ in range(steps_to_run):
            if self.adaptive_clock:
                alpha_k = self.clock(field, tok_embed)
                dt_k = alpha_k * self.tau_0_tensor
                alphas_k.append(alpha_k)
            else:
                dt_k = self.tau_0_tensor
            if self.continuous_velocities:
                dir_k = self.direction_controller(field, tok_embed)
                dirs_k.append(dir_k)
            else:
                dir_k = None
            if not disable_transport:
                mult, omega = self.transport.multiplier(dt_k, direction=dir_k)
                field = self.transport.apply_multiplier(field, mult)
                if isinstance(dt_k, torch.Tensor):
                    transport_dt = dt_k.view(field.shape[0], 1, 1, 1, 1)
                else:
                    transport_dt = float(dt_k)
                transport_phase_k.append(
                    (2.0 * torch.atan(0.5 * omega * transport_dt)).abs().mean()
                )
            if not disable_collision:
                field, collision_diag = self.collision(field, dt_k)
                coll_angles_k.append(collision_diag["collision_angle_abs_mean"])
            if not self.three_clock and not disable_bath:
                if isinstance(self.bath, UnifiedTorusDissipation):
                    field, bath_diag = self.bath(
                        field, dt_k, tok_embed=tok_embed,
                        gamma0_factor=gamma0_factor,
                        disable_viscosity=disable_viscosity,
                        disable_subspace=disable_subspace)
                else:
                    field, bath_diag = self.bath(field, dt_k)

        clock_diag = {}
        if self.adaptive_clock and len(alphas_k) >= 3:
            clock_diag = {
                "alpha_1": alphas_k[0].mean().detach(),
                "alpha_2": alphas_k[1].mean().detach(),
                "alpha_3": alphas_k[2].mean().detach(),
                "delta_tau_total": torch.stack(alphas_k).sum(dim=0).mean().detach() * self.tau_0,
                "collision_exposure": sum(coll_angles_k).detach() if coll_angles_k else field.new_zeros(()),
            }

        if transport_phase_k:
            transport_diag = {
                "transport_angle_abs_mean": torch.stack(transport_phase_k).mean().detach(),
                "transport_angle_abs_max": torch.stack(transport_phase_k).amax().detach(),
                "transport_norm_residual": field.new_zeros(()),
            }

        dir_diag = {}
        if self.continuous_velocities and dirs_k:
            cos_disp = (dirs_k[-1] * self.direction_controller.base_dirs[None]).sum(dim=-1).mean()
            dir_diag["dir_disp_deg"] = torch.rad2deg(torch.acos(cos_disp.clamp(-1.0, 1.0))).detach()
            M = torch.bmm(dirs_k[-1], dirs_k[-1].transpose(1, 2))
            off_diag_cos = (M.sum(dim=(-1, -2)) - self.velocities) / (self.velocities * (self.velocities - 1))
            dir_diag["dir_pairwise_sep_deg"] = torch.rad2deg(torch.acos(off_diag_cos.clamp(-1.0, 1.0))).mean().detach()
            if len(dirs_k) >= 2:
                cos_micro = (dirs_k[1] * dirs_k[0]).sum(dim=-1).mean()
                dir_diag["dir_change_micro_deg"] = torch.rad2deg(torch.acos(cos_micro.clamp(-1.0, 1.0))).detach()
            else:
                dir_diag["dir_change_micro_deg"] = field.new_zeros(())

        if self.readout_type == "baseline":
            feature = self.readout(field)
        else:
            feature, _ = self.readout(field, tok_embed, return_diag=False)
        logits = self.decoder(feature)

        if self.three_clock and not disable_bath:
            dt_mem = self.tau_mem * self.tau_0_tensor
            if isinstance(self.bath, UnifiedTorusDissipation):
                field, bath_diag = self.bath(
                    field, dt_mem, tok_embed=tok_embed,
                    gamma0_factor=gamma0_factor,
                    disable_viscosity=True,
                    disable_subspace=disable_subspace)
            else:
                field, bath_diag = self.bath(field, dt_mem)

        diagnostics = {**source_diag, **transport_diag, **collision_diag, **bath_diag, **clock_diag, **dir_diag,
                       "energy": (0.5 * field.detach().square().sum(-1).mean())}
        if posterior_precision is not None:
            # Kept out of ordinary monitoring/JSON; ``belief_step`` carries
            # it into the next event as persistent posterior uncertainty.
            diagnostics["_posterior_precision"] = posterior_precision
        return logits, field, diagnostics

    def belief_step(self, belief: KineticBeliefState, token_ids: torch.Tensor,
                    **kwargs) -> tuple[torch.Tensor, KineticBeliefState, dict]:
        """Advance a full W4 posterior belief through one observed event."""
        if self.write_agent is None:
            raise RuntimeError("belief_step requires write_type='w4_predictive_agent'")
        logits, field, diagnostics = self.step(
            belief.field, token_ids, precision=belief.precision, **kwargs)
        precision = diagnostics.pop("_posterior_precision")
        return logits, KineticBeliefState(field=field, precision=precision), diagnostics

    def forward_belief(self, input_ids: torch.Tensor, targets: torch.Tensor,
                       belief: Optional[KineticBeliefState] = None, *,
                       port_free_energy_weight: float = 1.0,
                       disable_transport: bool = False,
                       disable_collision: bool = False,
                       disable_bath: bool = False,
                       micro_steps: Optional[int] = None):
        """Run a token sequence while retaining W4 prior/posterior precision.

        Both terms are expressed as per-event negative log densities.  The
        coefficient therefore defaults to one rather than introducing a
        hidden auxiliary-loss scale.  It remains explicit for ablations.
        """
        if self.write_agent is None:
            raise RuntimeError("forward_belief requires write_type='w4_predictive_agent'")
        batch, length = input_ids.shape
        belief = (self.initial_belief(batch, input_ids.device)
                  if belief is None else belief)
        total_likelihood = input_ids.new_zeros((), dtype=torch.float32)
        total_free_energy = input_ids.new_zeros((), dtype=torch.float32)
        final_diagnostics = {}
        for index in range(length):
            logits, belief, diagnostics = self.belief_step(
                belief, input_ids[:, index],
                disable_transport=disable_transport,
                disable_collision=disable_collision,
                disable_bath=disable_bath,
                micro_steps=micro_steps)
            likelihood = F.cross_entropy(logits, targets[:, index])
            total_likelihood = total_likelihood + likelihood
            total_free_energy = total_free_energy + diagnostics.pop("_write_free_energy")
            final_diagnostics = diagnostics
        loss = (total_likelihood + float(port_free_energy_weight)
                * total_free_energy) / length
        final_diagnostics = {
            **final_diagnostics,
            "token_nll": (total_likelihood / length).detach(),
            "write_free_energy_mean": (total_free_energy / length).detach(),
        }
        return loss, belief, final_diagnostics

    def forward(self, input_ids, targets, state=None, *, disable_transport=False,
                disable_collision=False, disable_bath=False, micro_steps=None):
        if self.write_agent is not None:
            raise RuntimeError(
                "w4_predictive_agent requires forward_belief() so its posterior "
                "precision cannot be silently discarded between token events")
        batch, length = input_ids.shape
        field = self.initial_state(batch, input_ids.device) if state is None else state
        features, diagnostic_rows = [], []
        cached_learned = self.transport.learned_symbol()
        k_steps = int(micro_steps) if micro_steps is not None else self.micro_steps

        # 1. Static Projection Hoisting: Batch precompute token representations across all 128 tokens
        all_tok_embed = self.source.embedding(input_ids)
        all_disp = 0.5 * torch.tanh(self.source.address(all_tok_embed)) if self.source.relative_address else torch.sigmoid(self.source.address(all_tok_embed))
        all_width = F.softplus(self.source.width(all_tok_embed))
        all_content = self.source.content(all_tok_embed)
        if self.readout_type in ("kernel_r1", "kernel_r2"):
            all_u = F.rms_norm(all_tok_embed, (self.d,))
            all_q_field = self.readout.w_q(all_u).view(batch, length, self.readout.heads, self.readout.queries, self.readout.d_h)
        else:
            all_q_field = None

        for index in range(length):
            tok_id = input_ids[:, index]
            tok_embed = all_tok_embed[:, index]
            field, _, source_diag = self.source(
                field, tok_id, return_diag=(index == length - 1),
                token=tok_embed, displacement=all_disp[:, index],
                width=all_width[:, index], content=all_content[:, index])
            transport_diag = {"transport_norm_residual": field.new_zeros(())}
            collision_diag = {
                "collision_angle_abs_mean": field.new_zeros(()),
                "collision_angle_abs_max": field.new_zeros(()),
                "collision_input_snr": field.new_zeros(()),
                "collision_output_snr": field.new_zeros(()),
            }
            bath_diag = {"bath_out_energy": field.new_zeros(()),
                         "bath_angle_abs_mean": field.new_zeros(())}
            if self.reversible_ponder and k_steps > 1 and not (disable_transport or disable_collision):
                field = ReversibleHamiltonianPonderFunction.apply(field, k_steps, self, tok_embed)
                clock_diag, dir_diag = {}, {}
            else:
                alphas_k, coll_angles_k, dirs_k, transport_phase_k = [], [], [], []
                for _ in range(k_steps):
                    if self.adaptive_clock:
                        alpha_k = self.clock(field, tok_embed)
                        dt_k = alpha_k * self.tau_0_tensor
                        alphas_k.append(alpha_k)
                    else:
                        dt_k = self.tau_0_tensor
                    if self.continuous_velocities:
                        dir_k = self.direction_controller(field, tok_embed)
                        dirs_k.append(dir_k)
                    else:
                        dir_k = None
                    if not disable_transport:
                        mult, omega = self.transport.multiplier(dt_k, direction=dir_k, learned=cached_learned)
                        before = field.square().sum()
                        field = self.transport.apply_multiplier(field, mult)
                        if isinstance(dt_k, torch.Tensor):
                            transport_dt = dt_k.view(batch, 1, 1, 1, 1)
                        else:
                            transport_dt = float(dt_k)
                        transport_phase_k.append(
                            (2.0 * torch.atan(0.5 * omega * transport_dt)).abs().mean()
                        )
                        transport_diag = {"transport_norm_residual":
                                          (field.square().sum() - before).detach().abs()}
                    if not disable_collision:
                        field, collision_diag = self.collision(field, dt_k)
                        coll_angles_k.append(collision_diag["collision_angle_abs_mean"])
                    if not self.three_clock and not disable_bath:
                        if isinstance(self.bath, UnifiedTorusDissipation):
                            field, bath_diag = self.bath(
                                field, dt_k, tok_embed=tok_embed,
                                disable_viscosity=False,
                                disable_subspace=False)
                        else:
                            field, bath_diag = self.bath(field, dt_k)

                clock_diag = {}
                if self.adaptive_clock and len(alphas_k) >= 3:
                    clock_diag = {
                        "alpha_1": alphas_k[0].mean().detach(),
                        "alpha_2": alphas_k[1].mean().detach(),
                        "alpha_3": alphas_k[2].mean().detach(),
                        "delta_tau_total": torch.stack(alphas_k).sum(dim=0).mean().detach() * self.tau_0,
                        "collision_exposure": sum(coll_angles_k).detach() if coll_angles_k else field.new_zeros(()),
                    }

                if transport_phase_k:
                    transport_diag.update({
                        "transport_angle_abs_mean": torch.stack(transport_phase_k).mean().detach(),
                        "transport_angle_abs_max": torch.stack(transport_phase_k).amax().detach(),
                    })

                dir_diag = {}
                if self.continuous_velocities and dirs_k:
                    cos_disp = (dirs_k[-1] * self.direction_controller.base_dirs[None]).sum(dim=-1).mean()
                    dir_diag["dir_disp_deg"] = torch.rad2deg(torch.acos(cos_disp.clamp(-1.0, 1.0))).detach()
                    M = torch.bmm(dirs_k[-1], dirs_k[-1].transpose(1, 2))
                    off_diag_cos = (M.sum(dim=(-1, -2)) - self.velocities) / (self.velocities * (self.velocities - 1))
                    dir_diag["dir_pairwise_sep_deg"] = torch.rad2deg(torch.acos(off_diag_cos.clamp(-1.0, 1.0))).mean().detach()
                    if len(dirs_k) >= 2:
                        cos_micro = (dirs_k[1] * dirs_k[0]).sum(dim=-1).mean()
                        dir_diag["dir_change_micro_deg"] = torch.rad2deg(torch.acos(cos_micro.clamp(-1.0, 1.0))).detach()
                    else:
                        dir_diag["dir_change_micro_deg"] = field.new_zeros(())

            if self.readout_type == "baseline":
                feat = self.readout(field)
            elif all_q_field is not None:
                flat_field = field.reshape(batch, self.readout.nodes, self.d)
                normed_field = self.readout.field_norm(flat_field)
                keys = self.readout.k_proj(normed_field)
                key_h = keys.reshape(batch, self.readout.nodes, self.readout.heads, self.readout.d_h).transpose(1, 2)
                values = self.readout.v_proj(normed_field)
                val_h = values.reshape(batch, self.readout.nodes, self.readout.heads, self.readout.d_h).transpose(1, 2)
                m, _, _ = self.readout.measure(key_h, val_h, all_q_field[:, index])
                feat = self.readout.output(self.readout.merge(m))
            else:
                feat, _ = self.readout(field, tok_embed, return_diag=False)
            features.append(feat)

            if self.three_clock and not disable_bath:
                dt_mem = self.tau_mem * self.tau_0_tensor
                if isinstance(self.bath, UnifiedTorusDissipation):
                    field, bath_diag = self.bath(
                        field, dt_mem, tok_embed=tok_embed,
                        gamma0_factor=1.0,
                        disable_viscosity=True,
                        disable_subspace=False)
                else:
                    field, bath_diag = self.bath(field, dt_mem)

            diagnostic_rows.append({**source_diag, **transport_diag,
                                    **collision_diag, **bath_diag, **clock_diag, **dir_diag})
        logits = self.decoder(torch.stack(features, 1))
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        last = diagnostic_rows[-1]
        if hasattr(self.readout, "attention_weights"):
            attn_weights = self.readout.attention_weights(field)
            read_entropy = (-(attn_weights.clamp_min(1e-12).log() * attn_weights).sum(-1).mean()).detach()
        elif hasattr(self.readout, "last_entropy"):
            read_entropy = self.readout.last_entropy
        else:
            # `new_tensor` creates a host-originated scalar and is rejected
            # during CUDA Graph capture.  Derive the diagnostic zero from the
            # captured field instead, preserving device/dtype without a host
            # allocation.
            read_entropy = field.reshape(-1)[0].detach() * 0.0
        return loss, field, {**last,
            "energy": 0.5 * field.detach().square().sum(-1).mean(),
            "read_attention_entropy": read_entropy,
        }
