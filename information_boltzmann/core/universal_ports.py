"""One D3Q8 kinetic medium with task-independent observation and query ports.

The internal state always lives on a periodic three-torus.  A task supplies a
set of observed values with external addresses and a set of query addresses;
these addresses are *ports*, never the geometry of the kinetic medium.  The
only nonlinear state reorganisation after a write is the local invariant
collision.  In particular, this module intentionally contains no
post-collision corrective field network.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .torus3d import (
    EnergyFactoredTorusReadout,
    FullRankTorusWrite,
    LocalInvariantCollision3D,
    QuadraticTorusBath,
    VelocityCayleyTransport3D,
    torus_features,
)


class PortCoordinateChart(nn.Module):
    """Shared continuous map from external port addresses to the 3D torus.

    An external coordinate is an address, rather than a field coordinate.  The
    chart starts as the identity on the torus and learns one smooth warp that
    both the boundary writer and query reader use.  Consequently an observed
    port and a request for that port refer to the same internal neighbourhood;
    the categorical value only affects the packet's content and velocity
    composition.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        hidden = max(32, d // 2)
        self.warp = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.width_net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        # Identity coordinates are a geometric prior, not a learned arbitrary
        # source/readout alignment.  The chart may subsequently deform them.
        nn.init.zeros_(self.warp[-1].weight)
        nn.init.zeros_(self.warp[-1].bias)
        nn.init.zeros_(self.width_net[-1].weight)
        nn.init.constant_(self.width_net[-1].bias, -1.6)

    @staticmethod
    def features(coordinates: torch.Tensor) -> torch.Tensor:
        return torus_features(torch.remainder(coordinates, 1.0))

    def address(self, coordinates: torch.Tensor) -> torch.Tensor:
        displacement = 0.25 * torch.tanh(self.warp(self.features(coordinates)))
        return torch.remainder(coordinates + displacement, 1.0)

    def width(self, coordinates: torch.Tensor) -> torch.Tensor:
        return 0.04 + 0.21 * torch.sigmoid(self.width_net(self.features(coordinates)))


class PortSetBoundaryWrite3D(nn.Module):
    """Port adapter for the language model's canonical full-rank writer."""

    def __init__(self, vocab_size: int, shape: tuple[int, int, int], d: int,
                 chart: PortCoordinateChart, packet_radius: float = 1.25) -> None:
        super().__init__()
        self.shape, self.d = tuple(shape), int(d)
        self.chart = chart
        self.boundary = FullRankTorusWrite(
            vocab_size=vocab_size, shape=self.shape, d=self.d,
            packet_radius=packet_radius, relative_address=False,
            write_type="w2_impedance")

    def forward(self, field: torch.Tensor, observation_values: torch.Tensor,
                observation_coordinates: torch.Tensor,
                observation_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Apply one energy-accounted port exchange.

        ``observation_values`` is ``[B, P]`` categorical evidence and
        ``observation_coordinates`` is an arbitrary ``[B, P, 3]`` external
        address.  The latter is encoded by the port, rather than being used as
        a coordinate of the kinetic field.
        """
        if observation_values.ndim != 2 or observation_coordinates.shape != (*observation_values.shape, 3):
            raise ValueError("Expected categorical [B,P] values and [B,P,3] port coordinates")
        batch, ports = observation_values.shape
        if observation_mask is None:
            observation_mask = torch.ones_like(observation_values, dtype=torch.bool)
        mask = observation_mask.to(field.dtype)
        count = mask.sum(1, keepdim=True).clamp_min(1.0)
        # Values define packet composition.  Addresses only define where the
        # boundary event enters the medium; they cannot create a private
        # coordinate-to-content shortcut around the shared field.
        port = self.boundary.embedding(observation_values)
        centers = self.chart.address(observation_coordinates)
        widths = self.chart.width(observation_coordinates)
        field_next, reflected, diagnostics = self.boundary.forward_ports(
            field, port, centers, widths, observation_mask)
        diagnostics["write_center"] = diagnostics.pop("source_center")
        diagnostics["event_context"] = (port * mask[..., None]).sum(1) / count
        return field_next, reflected, diagnostics


class ControlledInvariantCollision3D(nn.Module):
    """State- and hypothesis-conditioned local D3Q8 invariant scattering."""

    def __init__(self, shape: tuple[int, int, int], velocities: int,
                 content_dim: int, control_dim: int, hidden: int = 96,
                 layers: int = 2) -> None:
        super().__init__()
        self.base = LocalInvariantCollision3D(
            shape, velocities, content_dim, hidden=hidden, layers=layers,
            position_conditioned=False)
        self.control_dim = control_dim
        self.control_angle = nn.Sequential(
            nn.Linear(self.base.d + control_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, layers * (self.base.nullity // 2)))
        nn.init.normal_(self.control_angle[-1].weight, std=1e-3)
        nn.init.zeros_(self.control_angle[-1].bias)

    def forward(self, field: torch.Tensor, control: torch.Tensor,
                delta_tau: float | torch.Tensor = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = field.shape[0]
        flat = field.reshape(batch, -1, self.base.d)
        nullspace = self.base.nullspace.to(dtype=flat.dtype)
        coefficient = torch.einsum("dk,bnd->bnk", nullspace, flat)
        conserved = flat - torch.einsum("dk,bnk->bnd", nullspace, coefficient)
        local_state = self.base.norm(flat)
        expanded = control[:, None].expand(-1, flat.shape[1], -1)
        angles = self.base.angle(local_state) + self.control_angle(
            torch.cat((local_state, expanded), -1))
        angles = angles.reshape(batch, flat.shape[1], self.base.layers, self.base.nullity // 2)
        if isinstance(delta_tau, torch.Tensor):
            dt = delta_tau.reshape(batch, 1, 1, 1)
        else:
            dt = float(delta_tau)
        scaled = angles * dt
        value = coefficient
        for layer in range(self.base.layers):
            pair = self.base.schedules[layer]
            left, right = value[..., pair[:, 0]], value[..., pair[:, 1]]
            theta = scaled[:, :, layer]
            updated = value.clone()
            updated[..., pair[:, 0]] = theta.cos() * left - theta.sin() * right
            updated[..., pair[:, 1]] = theta.sin() * left + theta.cos() * right
            value = updated
        output = conserved + torch.einsum("dk,bnk->bnd", nullspace, value)
        return output.reshape_as(field), {
            "collision_angle_abs_mean": scaled.detach().abs().mean(),
            "collision_angle_abs_max": scaled.detach().abs().amax(),
            "collision_invariant_residual": (
                torch.einsum("rd,bnd->bnr", self.base.constraints.to(flat), output - flat)
                .detach().abs().amax()),
        }


class PortQueryReadout3D(nn.Module):
    """Address-query adapter for language's energy-factored QK readout."""

    def __init__(self, shape: tuple[int, int, int], d: int,
                 chart: PortCoordinateChart, heads: int = 4) -> None:
        super().__init__()
        if d % heads:
            raise ValueError("d must divide heads")
        self.shape, self.d, self.heads = tuple(shape), d, heads
        self.chart = chart
        self.query_position = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.operator = EnergyFactoredTorusReadout(
            self.shape, d, queries=1, heads=heads, moving_frame=True)

    def forward(self, field: torch.Tensor, query_coordinates: torch.Tensor,
                return_attention: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query_coordinates.ndim != 3 or query_coordinates.shape[-1] != 3:
            raise ValueError("Expected [B,Q,3] query coordinates")
        batch, queries = query_coordinates.shape[:2]
        anchors = self.chart.address(query_coordinates)
        # A query address labels the requested observation.  It does not
        # prescribe a local field aperture: global QK competition decides
        # which internal sites carry the evidence, as it does for language.
        query_features = self.chart.features(anchors)
        read, attention = self.operator.attend(
            field, self.query_position(query_features))
        diagnostics: dict[str, torch.Tensor] = {
            "read_attention_entropy": (-(attention * attention.clamp_min(1e-12).log()).sum(-1).mean()).detach(),
            "read_head_scale": self.operator.head_log_scale.exp().detach().mean(),
            "read_anchor": anchors.detach().mean((0, 1)),
        }
        if return_attention:
            diagnostics["attention"] = attention.detach()
        return self.operator.output(read), diagnostics


class CBIMUniversalPorts3D(nn.Module):
    """Task-independent D3Q8 field with observation and query port contracts."""

    architecture = "cbim-universal-ports-d3q8-v4-canonical-five-operators"

    def __init__(self, vocab_size: int = 11, shape: tuple[int, int, int] = (8, 8, 4),
                 velocities: int = 8, content_dim: int = 16, branches: int = 1,
                 micro_steps: int = 4, heads: int = 4) -> None:
        super().__init__()
        if branches < 1:
            raise ValueError("branches must be positive")
        self.vocab_size, self.shape = int(vocab_size), tuple(shape)
        self.velocities, self.content_dim = int(velocities), int(content_dim)
        self.d, self.branches, self.micro_steps = velocities * content_dim, int(branches), int(micro_steps)
        self.port_chart = PortCoordinateChart(self.d)
        self.source = PortSetBoundaryWrite3D(vocab_size, self.shape, self.d, self.port_chart)
        self.transport = VelocityCayleyTransport3D(self.shape, velocities, content_dim)
        # The one-field model is the reference physical medium.  It uses the
        # exact language collision law: local field -> invariant nullspace
        # rotations.  A hypothesis ensemble may add its separate controller,
        # but it must never silently alter the B=1 causal experiment.
        if self.branches == 1:
            self.collision = LocalInvariantCollision3D(
                self.shape, velocities, content_dim, position_conditioned=False)
            self.branch_embedding = None
            self.branch_control = None
            self.branch_router = None
        else:
            self.collision = ControlledInvariantCollision3D(
                self.shape, velocities, content_dim, self.d)
            self.branch_embedding = nn.Parameter(torch.randn(branches, self.d) * 0.02)
            self.branch_control = nn.Sequential(
                nn.LayerNorm(self.d), nn.Linear(self.d, self.d), nn.SiLU(),
                nn.Linear(self.d, self.d))
            self.branch_router = nn.Sequential(
                nn.LayerNorm(2 * self.d), nn.Linear(2 * self.d, self.d),
                nn.SiLU(), nn.Linear(self.d, 1))
            nn.init.zeros_(self.branch_router[-1].weight)
            nn.init.zeros_(self.branch_router[-1].bias)
        self.bath = QuadraticTorusBath(self.shape, self.d)
        self.readout = PortQueryReadout3D(self.shape, self.d, self.port_chart, heads)
        self.decoder = nn.Linear(self.d, vocab_size)

    def initial_state(self, batch: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        device = device or self.decoder.weight.device
        return torch.zeros(batch, *self.shape, self.d, device=device, dtype=self.decoder.weight.dtype)

    def _evolve(self, fields: torch.Tensor, controls: torch.Tensor | None,
                duration: float | torch.Tensor, disable_transport: bool,
                disable_collision: bool, disable_bath: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = fields.shape[0]
        if isinstance(duration, torch.Tensor):
            delta = duration.reshape(batch, 1) / self.micro_steps
        else:
            delta = float(duration) / self.micro_steps
        multiplier, _ = self.transport.multiplier(delta)
        diagnostics: dict[str, torch.Tensor] = {
            "transport_norm_residual": fields.new_zeros(())}
        for _ in range(self.micro_steps):
            if not disable_transport:
                before = fields.square().sum()
                fields = self.transport.apply_multiplier(fields, multiplier)
                diagnostics["transport_norm_residual"] = (
                    (fields.square().sum() - before).detach().abs()
                    / before.detach().abs().clamp_min(1e-12))
            if not disable_collision:
                if controls is None:
                    fields, collision_diag = self.collision(fields, delta)
                else:
                    fields, collision_diag = self.collision(fields, controls, delta)
                diagnostics.update(collision_diag)
            if not disable_bath:
                fields, bath_diag = self.bath(fields, delta)
                diagnostics.update(bath_diag)
        return fields, diagnostics

    def event(self, field: torch.Tensor, observation_values: torch.Tensor,
              observation_coordinates: torch.Tensor, query_coordinates: torch.Tensor,
              observation_mask: torch.Tensor | None = None, duration: float | torch.Tensor = 1.0,
              *, disable_transport: bool = False, disable_collision: bool = False,
              disable_bath: bool = False, return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Assimilate observations, evolve M complete fields, and decode queries."""
        batch = field.shape[0]
        written, _, source_diag = self.source(field, observation_values, observation_coordinates, observation_mask)
        context = source_diag.pop("event_context")
        if self.branches == 1:
            evolved, dynamics_diag = self._evolve(
                written, None, duration, disable_transport,
                disable_collision, disable_bath)
            read, read_diag = self.readout(
                evolved, query_coordinates, return_attention=return_attention)
            logits = self.decoder(read)
            diagnostics = {
                **source_diag, **dynamics_diag, **read_diag,
                "field_energy": (0.5 * evolved.detach().square().sum(-1).mean()).detach(),
                "branch_effective_count": evolved.new_ones(()),
                "branch_field_spread": evolved.new_zeros(()),
                "selected_branch": torch.zeros(batch, dtype=torch.long, device=field.device),
            }
            return logits.log_softmax(-1), evolved, diagnostics

        assert self.branch_control is not None
        assert self.branch_embedding is not None
        assert self.branch_router is not None
        controls = self.branch_control(context[:, None] + self.branch_embedding[None])
        fields = written[:, None].expand(-1, self.branches, -1, -1, -1, -1).reshape(
            batch * self.branches, *self.shape, self.d).contiguous()
        if isinstance(duration, torch.Tensor) and duration.numel() == batch:
            branch_duration: float | torch.Tensor = duration[:, None].expand(
                -1, self.branches).reshape(-1)
        else:
            branch_duration = duration
        evolved, dynamics_diag = self._evolve(
            fields, controls.reshape(batch * self.branches, self.d), branch_duration,
            disable_transport, disable_collision, disable_bath)
        branch_fields = evolved.reshape(batch, self.branches, *self.shape, self.d)
        read, read_diag = self.readout(
            evolved, query_coordinates[:, None].expand(-1, self.branches, -1, -1).reshape(
                batch * self.branches, query_coordinates.shape[1], 3),
            return_attention=return_attention)
        branch_logits = self.decoder(read).reshape(batch, self.branches, query_coordinates.shape[1], self.vocab_size)
        branch_summary = evolved.reshape(batch, self.branches, -1, self.d).mean(2)
        route_input = torch.cat((branch_summary, context[:, None].expand_as(branch_summary)), -1)
        route_logits = self.branch_router(route_input).squeeze(-1)
        weights = route_logits.softmax(-1)
        log_probs = branch_logits.log_softmax(-1)
        mixture = torch.logsumexp(weights.log()[:, :, None, None] + log_probs, 1)
        selected = route_logits.argmax(-1)
        next_field = branch_fields[torch.arange(batch, device=field.device), selected]
        spread = (branch_fields - branch_fields[:, :1]).square().mean().sqrt()
        diagnostics = {
            **source_diag, **dynamics_diag, **read_diag,
            "field_energy": (0.5 * next_field.detach().square().sum(-1).mean()).detach(),
            "branch_effective_count": ((-(weights * weights.clamp_min(1e-12).log()).sum(-1)).exp().mean()).detach(),
            "branch_field_spread": spread.detach(),
            "selected_branch": selected.detach(),
        }
        return mixture, next_field, diagnostics

    @staticmethod
    def sudoku_coordinates(batch: int, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        row, col = torch.meshgrid(
            torch.arange(9, device=device, dtype=dtype) / 9.0,
            torch.arange(9, device=device, dtype=dtype) / 9.0,
            indexing="ij")
        return torch.stack((row, col, torch.full_like(row, 0.5)), -1).reshape(1, 81, 3).expand(batch, -1, -1)

    def forward_sudoku(self, inputs: torch.Tensor, *, field: torch.Tensor | None = None,
                       duration: float | torch.Tensor = 1.0, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if inputs.ndim != 2 or inputs.shape[1] != 81:
            raise ValueError("Sudoku is only an external port adapter: expected [B,81] categorical observations")
        state = self.initial_state(inputs.shape[0], device=inputs.device) if field is None else field
        coordinates = self.sudoku_coordinates(inputs.shape[0], device=inputs.device)
        # Sudoku-Extreme reserves ``1`` for an unobserved cell.  This choice
        # is isolated in the adapter; the physical core only receives a
        # generic observation mask.
        return self.event(state, inputs, coordinates, coordinates, inputs.ne(1), duration, **kwargs)

    def forward_language(self, input_ids: torch.Tensor, *, field: torch.Tensor | None = None,
                         duration: float | torch.Tensor = 1.0, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Language adapter using the same port model; no lexical private field."""
        if input_ids.ndim != 2:
            raise ValueError("Expected [B,T] language token ids")
        state = self.initial_state(input_ids.shape[0], device=input_ids.device) if field is None else field
        outputs = []
        length = max(input_ids.shape[1], 1)
        for index in range(input_ids.shape[1]):
            coordinate = torch.zeros(input_ids.shape[0], 1, 3, device=input_ids.device)
            # Language supplies a temporal event address through the same
            # port contract.  The physical field remains periodic; a streaming
            # caller can supply its own persistent event phase.
            coordinate[..., 0] = (index + 0.5) / length
            coordinate[..., 1:] = 0.5
            logits, state, _ = self.event(
                state, input_ids[:, index:index + 1], coordinate, coordinate,
                duration=duration, **kwargs)
            outputs.append(logits)
        return torch.cat(outputs, 1), state
