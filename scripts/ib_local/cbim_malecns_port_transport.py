"""MaleCNS CBIM with anatomical ports and geometric D3Q8 transport.

The persistent state is a signed kinetic feature field ``h[x,q,a]``.  This
version is deliberately described as a conservative scattering medium rather
than a complete nonnegative Boltzmann population: its local Givens collision is
reversible and has no discrete H theorem.  Irreversible energy export occurs at
a fixed cold output boundary; there is no global gamma or controller state.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.ib_local.cbim_malecns import load_malecns_graph
from scripts.ib_local.cbim_malecns_v3 import FullStateKineticCollision
from scripts.ib_local.geometric_transport import DirectionalCayleyTransport


def _port_masks(weights: torch.Tensor, topk: int,
                exclude: torch.Tensor | None = None):
    """Return energy-normalized, read-normalized and max-one sparse masks."""
    weights = weights.float().clamp_min(0)
    sparse = torch.zeros_like(weights)
    for row_index, row in enumerate(weights):
        candidate = row.clone()
        if exclude is not None:
            candidate[exclude] = 0
        positive = int((candidate > 0).sum())
        if positive == 0:
            raise ValueError(f"port {row_index} has no available parcel")
        indices = candidate.topk(min(int(topk), positive)).indices
        sparse[row_index, indices] = candidate[indices].sqrt()
    energy = sparse / sparse.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    read = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-12)
    gate = sparse / sparse.amax(-1, keepdim=True).clamp_min(1e-12)
    return energy, read, gate


class AnatomicalThreePortBoundary(nn.Module):
    """Token input, persistent field and zero-input cold bath scattering."""

    def __init__(self, graph, vocab_size=50257, d=64, input_topk=16,
                 output_topk=16, packet_radius=1.25, min_write_angle=.03,
                 max_write_angle=.30, min_bath_angle=.025,
                 max_bath_angle=.18):
        super().__init__()
        if "input_port_weights" not in graph or "output_port_weights" not in graph:
            raise ValueError("a port-aware MaleCNS graph is required")
        self.d = int(d)
        self.packet_radius = float(packet_radius)
        self.min_write_angle = float(min_write_angle)
        self.max_write_angle = float(max_write_angle)
        self.min_bath_angle = float(min_bath_angle)
        self.max_bath_angle = float(max_bath_angle)

        output_energy, output_read, output_gate = _port_masks(
            graph["output_port_weights"], output_topk)
        output_support = output_gate.amax(0) > 0
        input_energy, input_read, input_gate = _port_masks(
            graph["input_port_weights"], input_topk, exclude=output_support)
        if torch.any((input_gate.amax(0) > 0) & output_support):
            raise AssertionError("input and output port supports must be disjoint")

        self.input_ports = input_energy.shape[0]
        self.output_ports = output_energy.shape[0]
        self.input_port_names = tuple(
            str(name) for name in graph.get("input_port_names", (
                "central_brain_sensory", "optic_lobe_sensory",
                "vnc_sensory", "sensory_relay")))
        self.output_port_names = tuple(
            str(name) for name in graph.get("output_port_names", (
                "motor", "efferent", "endocrine", "other_exit")))
        self.embedding = nn.Embedding(vocab_size, d)
        self.route = nn.Linear(d, self.input_ports)
        self.incident = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.write_coupling = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.normal_(self.incident[-1].weight, std=1e-3)
        nn.init.zeros_(self.incident[-1].bias)
        nn.init.normal_(self.write_coupling[-1].weight, std=1e-3)
        nn.init.constant_(self.write_coupling[-1].bias, -1.5)

        self.register_buffer("input_energy_masks", input_energy, persistent=True)
        self.register_buffer("input_read_masks", input_read, persistent=True)
        self.register_buffer("input_gate_masks", input_gate, persistent=True)
        self.register_buffer("output_energy_masks", output_energy, persistent=True)
        self.register_buffer("output_read_masks", output_read, persistent=True)
        self.register_buffer("output_gate", output_gate.amax(0), persistent=True)

    @staticmethod
    def _energy(value):
        return .5 * value.square().sum(dim=(-1, -2)).mean()

    def write(self, field, token_ids):
        token = self.embedding(token_ids)
        route = F.softmax(self.route(token), -1)
        spatial = torch.einsum("bp,pn->bn", route, self.input_energy_masks)
        spatial = spatial / spatial.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        gate = torch.einsum("bp,pn->bn", route, self.input_gate_masks)
        local = torch.einsum("pn,bnd->bpd", self.input_read_masks, field)
        context = (route[..., None] * local).sum(1)
        content = self.packet_radius * F.normalize(self.incident(token), dim=-1)
        incoming = spatial[..., None] * content[:, None]
        raw_angle = self.write_coupling(torch.cat((token, context), -1))
        channel_angle = self.min_write_angle + (
            self.max_write_angle - self.min_write_angle) * torch.sigmoid(raw_angle)
        theta = gate[..., None] * channel_angle[:, None]
        cosine, sine = theta.cos(), theta.sin()
        field_next = cosine * field + sine * incoming
        token_out = -sine * field + cosine * incoming
        residual = (self._energy(field_next) + self._energy(token_out)
                    - self._energy(field) - self._energy(incoming)).abs()
        return field_next, token_out, {
            "incident_energy": self._energy(incoming).detach(),
            "token_out_energy": self._energy(token_out).detach(),
            "write_angle_abs_mean": theta.detach().abs().mean(),
            "write_balance_residual": residual.detach(),
            "input_port_entropy": (-(route * route.clamp_min(1e-12).log()).sum(-1)
                                   .mean().detach()),
        }

    def read_and_absorb(self, field):
        # A state-local conductance increases monotonically with local energy.
        # Its sign, support and nonzero floor are structural, not learned by CE.
        local_energy = field.square().mean(-1)
        activation = local_energy / (1.0 + local_energy)
        theta = self.output_gate[None] * (
            self.min_bath_angle
            + (self.max_bath_angle - self.min_bath_angle) * activation)
        cosine, sine = theta[..., None].cos(), theta[..., None].sin()
        field_next = cosine * field
        bath_out = -sine * field
        response = torch.einsum("pn,bnd->bpd", self.output_read_masks, bath_out)
        residual = (self._energy(field_next) + self._energy(bath_out)
                    - self._energy(field)).abs()
        return field_next, response, {
            "bath_out_energy": self._energy(bath_out).detach(),
            "bath_angle_abs_mean": theta.detach().abs().mean(),
            "bath_angle_abs_max": theta.detach().abs().amax(),
            "bath_balance_residual": residual.detach(),
        }


class CBIMMaleCNSPortTransport(nn.Module):
    """No-global-readout CBIM with anatomical I/O ports."""

    architecture = "CBIM-MaleCNS-anatomical-port-geometric-transport"

    def __init__(self, graph_path, vocab_size=50257, velocities=8,
                 content_dim=8, input_topk=16, output_topk=16,
                 checkpoint_tokens=8, stream_fraction=.5):
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path = str(graph_path)
        self.velocities = int(velocities)
        self.content_dim = int(content_dim)
        self.d = self.velocities * self.content_dim
        self.L = graph["coordinates"].shape[0]
        self.state_shape = (self.L, self.d)
        self.checkpoint_tokens = int(checkpoint_tokens)
        self.boundary = AnatomicalThreePortBoundary(
            graph, vocab_size=vocab_size, d=self.d,
            input_topk=input_topk, output_topk=output_topk)
        self.collision = FullStateKineticCollision(
            graph["node_features"], velocities, content_dim)
        self.transport = DirectionalCayleyTransport(
            graph["coordinates"], graph["adjacency"],
            coordinate_scale=graph["coordinate_scale"],
            stream_fraction=stream_fraction)
        self.port_readout = nn.Sequential(
            nn.Linear(self.boundary.output_ports * self.d, 2 * self.d),
            nn.SiLU(), nn.Linear(2 * self.d, self.d))
        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.boundary.embedding.weight

    def initial_state(self, batch_size, device=None, dtype=None):
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size, self.L, self.d,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype)

    def evolve(self, state, token_ids, disable_collision=False,
               disable_transport=False):
        state, _, write = self.boundary.write(state, token_ids)
        if disable_collision:
            collision = {"collision_angle_abs_mean": state.new_zeros(())}
        else:
            state, collision = self.collision(state)
        transport_energy_before = .5 * state.square().sum(-1).mean()
        if disable_transport:
            transport = {
                "transport_norm_residual": state.new_zeros(()),
                "transport_directional_alignment": state.new_zeros(()),
            }
        else:
            shaped = state.reshape(
                state.shape[0], self.L, self.velocities, self.content_dim)
            shaped, transport = self.transport(shaped)
            state = shaped.reshape(state.shape[0], self.L, self.d)
        transport_energy_after = .5 * state.square().sum(-1).mean()
        state, response, bath = self.boundary.read_and_absorb(state)
        diagnostics = {
            **write, **collision, **transport, **bath,
            "transport_energy_change": (
                transport_energy_after - transport_energy_before).detach(),
        }
        return state, response, diagnostics

    def forward(self, input_ids, targets, state=None, disable_collision=False,
                disable_transport=False):
        batch, tokens = input_ids.shape
        state = (self.initial_state(batch, input_ids.device)
                 if state is None else state)
        features, sums = [], []

        def segment(current, ids):
            segment_features, segment_diagnostics = [], []
            for index in range(ids.shape[1]):
                current, response, diagnostics = self.evolve(
                    current, ids[:, index], disable_collision, disable_transport)
                segment_features.append(self.port_readout(response.flatten(1)))
                segment_diagnostics.append(torch.stack((
                    diagnostics["incident_energy"],
                    diagnostics["token_out_energy"],
                    diagnostics["bath_out_energy"],
                    diagnostics["write_balance_residual"],
                    diagnostics["bath_balance_residual"],
                    diagnostics["write_angle_abs_mean"],
                    diagnostics["bath_angle_abs_mean"],
                    diagnostics["bath_angle_abs_max"],
                    diagnostics["input_port_entropy"],
                    diagnostics["collision_angle_abs_mean"],
                    diagnostics["transport_norm_residual"],
                    diagnostics["transport_directional_alignment"],
                    diagnostics["transport_energy_change"])))
            return (current, torch.stack(segment_features, 1),
                    torch.stack(segment_diagnostics).sum(0))

        for start in range(0, tokens, self.checkpoint_tokens):
            ids = input_ids[:, start:start + self.checkpoint_tokens]
            if self.training and torch.is_grad_enabled():
                state, feature, diagnostics = checkpoint(
                    segment, state, ids, use_reentrant=False,
                    preserve_rng_state=False)
            else:
                state, feature, diagnostics = segment(state, ids)
            features.append(feature)
            sums.append(diagnostics)
        logits = self.decoder(torch.cat(features, 1))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        diagnostic = torch.stack(sums).sum(0) / tokens
        return loss, state, {
            "final_energy": .5 * state.square().sum(-1).mean().detach(),
            "incident_energy": diagnostic[0],
            "token_out_energy": diagnostic[1],
            "bath_out_energy": diagnostic[2],
            "write_balance_residual": diagnostic[3],
            "bath_balance_residual": diagnostic[4],
            "write_angle_abs_mean": diagnostic[5],
            "bath_angle_abs_mean": diagnostic[6],
            "bath_angle_abs_max": diagnostic[7],
            "input_port_entropy": diagnostic[8],
            "collision_angle_abs_mean": diagnostic[9],
            "transport_norm_residual": diagnostic[10],
            "transport_directional_alignment": diagnostic[11],
            "transport_energy_change": diagnostic[12],
        }
