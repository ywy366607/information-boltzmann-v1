"""Passive-port Information Boltzmann field on the MaleCNS graph."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.ib_local.cbim_malecns import load_malecns_graph
from scripts.ib_local.cbim_malecns_v3 import (
    ContinuousGraphTransport, FullStateKineticCollision, GraphStateReadout)


class PassiveBoundaryScattering(nn.Module):
    """Local orthogonal scattering between the field and a bounded token port."""

    def __init__(self, graph, vocab_size=50257, d=64, packet_radius=1.25,
                 min_angle=.02, max_angle=.35):
        super().__init__()
        self.d = d
        self.packet_radius = float(packet_radius)
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        node_features = graph["node_features"].float()
        self.embedding = nn.Embedding(vocab_size, d)
        self.address = nn.Linear(d, 3)
        self.width = nn.Linear(d, 3)
        self.token = nn.Linear(d, d)
        self.state = nn.Linear(d, d)
        self.neighbor = nn.Linear(d, d)
        self.position = nn.Linear(node_features.shape[-1], d)
        self.incoming = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.coupling = nn.Linear(d, d)
        nn.init.normal_(self.incoming[-1].weight, std=1e-3)
        nn.init.zeros_(self.incoming[-1].bias)
        nn.init.normal_(self.coupling.weight, std=1e-3)
        nn.init.constant_(self.coupling.bias, -2.0)
        self.register_buffer(
            "coordinates", graph["coordinates"].float()[None], persistent=False)
        self.register_buffer(
            "node_features", node_features[None], persistent=False)
        self.register_buffer(
            "neighbor_indices", graph["neighbor_indices"].long(), persistent=False)
        self.register_buffer(
            "neighbor_weights", graph["neighbor_weights"].float(), persistent=False)

    def forward(self, field, token_ids):
        token = self.embedding(token_ids)
        center = torch.sigmoid(self.address(token))[:, None]
        width = (.04 + .21 * torch.sigmoid(self.width(token)))[:, None]
        distance = (self.coordinates - center) / width
        envelope = torch.exp(-.5 * distance.square().sum(-1))
        neighbors = field[:, self.neighbor_indices]
        neighborhood = (
            neighbors * self.neighbor_weights[None, ..., None]).sum(2)
        context = F.silu(
            self.token(token)[:, None] + self.state(field)
            + self.neighbor(neighborhood) + self.position(self.node_features))
        incident = (self.packet_radius * envelope[..., None]
                    * F.normalize(self.incoming(context), dim=-1))
        local_angle = self.min_angle + (
            self.max_angle - self.min_angle) * torch.sigmoid(
                self.coupling(context))
        theta = envelope[..., None] * local_angle
        cosine, sine = theta.cos(), theta.sin()
        field_next = cosine * field + sine * incident
        outgoing = -sine * field + cosine * incident
        # Remove the free incident reflection.  This response contains only
        # the pre-existing field, so the decoder has no token-only shortcut.
        response = -sine * field
        field_before = .5 * field.square().sum(-1).mean()
        field_after = .5 * field_next.square().sum(-1).mean()
        incident_energy = .5 * incident.square().sum(-1).mean()
        outgoing_energy = .5 * outgoing.square().sum(-1).mean()
        residual = field_after - field_before - incident_energy + outgoing_energy
        weight = envelope / envelope.sum(-1, keepdim=True).clamp_min(1e-8)
        response_read = (weight[..., None] * response).sum(1)
        return field_next, response_read, {
            "field_energy_before": field_before.detach(),
            "field_energy_after_boundary": field_after.detach(),
            "incident_energy": incident_energy.detach(),
            "outgoing_energy": outgoing_energy.detach(),
            "boundary_work": (field_after - field_before).detach(),
            "boundary_balance_residual": residual.detach().abs(),
            "coupling_angle_abs_mean": theta.detach().abs().mean(),
            "reflection_ratio": (outgoing_energy /
                                 incident_energy.clamp_min(1e-8)).detach(),
        }


class CBIMMaleCNSV4(nn.Module):
    architecture = "CBIM-MaleCNS-passive-port-v4"

    def __init__(self, graph_path, vocab_size=50257, velocities=8,
                 content_dim=8, queries=4, heads=4, checkpoint_tokens=8):
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path = str(graph_path)
        self.velocities, self.content_dim = velocities, content_dim
        self.d = velocities * content_dim
        self.L = graph["coordinates"].shape[0]
        self.state_shape = (self.L, self.d)
        self.checkpoint_tokens = int(checkpoint_tokens)
        self.boundary = PassiveBoundaryScattering(
            graph, vocab_size=vocab_size, d=self.d)
        self.collision = FullStateKineticCollision(
            graph["node_features"], velocities, content_dim)
        self.transport = ContinuousGraphTransport(
            graph["laplacian_basis"], graph["laplacian_eigenvalues"],
            velocities, content_dim)
        self.readout = GraphStateReadout(
            graph["node_features"], self.d, queries, heads)
        self.merge = nn.Linear(2 * self.d, self.d)
        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.boundary.embedding.weight

    def initial_state(self, batch_size, device=None, dtype=None):
        p = next(self.parameters())
        return torch.zeros(batch_size, self.L, self.d,
                           device=p.device if device is None else device,
                           dtype=p.dtype if dtype is None else dtype)

    def evolve(self, state, token_ids, disable_collision=False,
               disable_transport=False):
        state, response, boundary = self.boundary(state, token_ids)
        if disable_collision:
            collision = {"collision_angle_abs_mean": state.new_zeros(())}
        else:
            state, collision = self.collision(state)
        if disable_transport:
            transport = {"transport_angle_abs_mean": state.new_zeros(())}
        else:
            state, transport = self.transport(state)
        return state, response, boundary, collision, transport

    def forward(self, input_ids, targets, state=None, disable_collision=False,
                disable_transport=False):
        batch, tokens = input_ids.shape
        state = self.initial_state(batch, input_ids.device) if state is None else state
        position = self.readout.position_encoding()
        features, sums = [], []

        def segment(s, ids, pos):
            fs, ds = [], []
            for i in range(ids.shape[1]):
                s, response, boundary, collision, transport = self.evolve(
                    s, ids[:, i], disable_collision, disable_transport)
                deep = self.readout(s, pos)
                fs.append(self.merge(torch.cat((deep, response), -1)))
                ds.append(torch.stack((
                    boundary["incident_energy"], boundary["outgoing_energy"],
                    boundary["boundary_work"],
                    boundary["boundary_balance_residual"],
                    boundary["coupling_angle_abs_mean"],
                    boundary["reflection_ratio"],
                    collision["collision_angle_abs_mean"],
                    transport["transport_angle_abs_mean"])))
            return s, torch.stack(fs, 1), torch.stack(ds).sum(0)

        for start in range(0, tokens, self.checkpoint_tokens):
            ids = input_ids[:, start:start + self.checkpoint_tokens]
            if self.training and torch.is_grad_enabled():
                state, fs, ds = checkpoint(
                    segment, state, ids, position, use_reentrant=False,
                    preserve_rng_state=False)
            else:
                state, fs, ds = segment(state, ids, position)
            features.append(fs); sums.append(ds)
        logits = self.decoder(torch.cat(features, 1))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        d = torch.stack(sums).sum(0) / tokens
        return loss, state, {
            "final_energy": .5 * state.square().sum(-1).mean().detach(),
            "incident_energy": d[0], "outgoing_energy": d[1],
            "boundary_work": d[2], "boundary_balance_residual": d[3],
            "coupling_angle_abs_mean": d[4], "reflection_ratio": d[5],
            "collision_angle_abs_mean": d[6],
            "transport_angle_abs_mean": d[7],
        }
