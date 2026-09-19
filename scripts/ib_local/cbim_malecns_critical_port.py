"""Passive two-port CBIM with bounded critical accommodation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.ib_local.cbim_malecns import load_malecns_graph
from scripts.ib_local.cbim_malecns_v3 import (
    ContinuousGraphTransport, FullStateKineticCollision, GraphStateReadout)


class PassiveCriticalPort(nn.Module):
    """Orthogonal environment exchange with explicit incident/outgoing fields."""

    def __init__(self, graph, vocab_size=50257, d=64, packet_radius=1.25,
                 max_write=0.25):
        super().__init__()
        nf = graph["node_features"].float()
        self.packet_radius, self.max_write = float(packet_radius), float(max_write)
        self.max_write_angle = float(torch.asin(torch.tensor(max_write).sqrt()))
        self.embedding = nn.Embedding(vocab_size, d)
        self.address, self.width = nn.Linear(d, 3), nn.Linear(d, 3)
        self.token, self.state = nn.Linear(d, d), nn.Linear(d, d)
        self.neighbor, self.position = nn.Linear(d, d), nn.Linear(nf.shape[-1], d)
        self.proposal = nn.Sequential(nn.Linear(d, 2*d), nn.SiLU(), nn.Linear(2*d, d))
        self.write_rate = nn.Linear(d, 1)
        nn.init.normal_(self.proposal[-1].weight, std=1e-3)
        nn.init.zeros_(self.proposal[-1].bias)
        nn.init.normal_(self.write_rate.weight, std=1e-3)
        nn.init.zeros_(self.write_rate.bias)
        self.register_buffer("coordinates", graph["coordinates"].float()[None], persistent=False)
        self.register_buffer("node_features", nf[None], persistent=False)
        self.register_buffer("neighbor_indices", graph["neighbor_indices"].long(), persistent=False)
        self.register_buffer("neighbor_weights", graph["neighbor_weights"].float(), persistent=False)

    def forward(self, field, token_ids, critical_a):
        emb = self.embedding(token_ids)
        center = torch.sigmoid(self.address(emb))[:, None]
        width = (.04 + .21*torch.sigmoid(self.width(emb)))[:, None]
        envelope = torch.exp(-.5*((self.coordinates-center)/width).square().sum(-1))
        neighbors = field[:, self.neighbor_indices]
        neighborhood = (neighbors*self.neighbor_weights[None, ..., None]).sum(2)
        context = F.silu(self.token(emb)[:, None] + self.state(field)
                         + self.neighbor(neighborhood) + self.position(self.node_features))
        packet = self.packet_radius*F.normalize(self.proposal(context), dim=-1)
        incident = envelope[..., None]*packet
        write_angle = self.max_write_angle*envelope[..., None]*torch.sigmoid(
            self.write_rate(context))
        critical_angle = torch.asin(critical_a.detach().clamp(0, 1).sqrt())[..., None]
        total_angle = write_angle+critical_angle
        cosine, sine = torch.cos(total_angle), torch.sin(total_angle)
        total_a = sine.square()
        output = cosine*field + sine*incident
        outgoing = -sine*field + cosine*incident
        field_change = .5*(output.square()-field.square()).sum(-1).mean()
        node_work = .5*(output.square()-field.square()).sum(-1)
        port_change = .5*(outgoing.square()-incident.square()).sum(-1).mean()
        return output, {
            "boundary_work": field_change.detach(),
            "port_balance_residual": (field_change+port_change).detach(),
            "incident_energy": (.5*incident.square().sum(-1).mean()).detach(),
            "outgoing_energy": (.5*outgoing.square().sum(-1).mean()).detach(),
            "accommodation_mean": total_a.detach().mean(),
            "accommodation_max": total_a.detach().amax(),
            "node_boundary_work": node_work.detach(),
        }


class CBIMMaleCNSCriticalPort(nn.Module):
    architecture = "CBIM-MaleCNS-critical-passive-port-v2"

    def __init__(self, graph_path, vocab_size=50257, velocities=8,
                 content_dim=8, queries=4, heads=4, checkpoint_tokens=32,
                 probe_epsilon=1e-3, controller_ema=.05,
                 controller_lr=.02, flux_ema=.01, flux_lr=.02,
                 max_critical_a=.25):
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path, self.velocities = str(graph_path), velocities
        self.content_dim, self.d = content_dim, velocities*content_dim
        self.L = graph["coordinates"].shape[0]
        self.state_shape = (self.L, 2*self.d+3)
        self.checkpoint_tokens = int(checkpoint_tokens)
        self.probe_epsilon, self.controller_ema = float(probe_epsilon), float(controller_ema)
        self.controller_lr, self.max_critical_a = float(controller_lr), float(max_critical_a)
        self.flux_ema, self.flux_lr = float(flux_ema), float(flux_lr)
        self.port = PassiveCriticalPort(graph, vocab_size, self.d)
        self.collision = FullStateKineticCollision(graph["node_features"], velocities, content_dim)
        self.transport = ContinuousGraphTransport(
            graph["laplacian_basis"], graph["laplacian_eigenvalues"], velocities, content_dim)
        self.readout = GraphStateReadout(graph["node_features"], self.d, queries, heads)
        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.port.embedding.weight
        probe = torch.sin(torch.arange(self.L*self.d, dtype=torch.float32)*1.618).reshape(self.L, self.d)
        self.register_buffer("initial_probe", probe/probe.norm().clamp_min(1e-12), persistent=False)

    def initial_state(self, batch_size, device=None, dtype=None):
        p = next(self.parameters())
        device = p.device if device is None else device
        dtype = p.dtype if dtype is None else dtype
        field = torch.zeros(batch_size, self.L, self.d, device=device, dtype=dtype)
        delta = self.initial_probe.to(device=device, dtype=dtype)[None].expand(batch_size, -1, -1)
        zeros = torch.zeros(batch_size, self.L, 1, device=device, dtype=dtype)
        return torch.cat((field, delta, zeros, zeros, zeros), -1)

    def unpack(self, state):
        return (state[..., :self.d], state[..., self.d:2*self.d],
                state[..., 2*self.d], state[..., 2*self.d+1],
                state[..., 2*self.d+2])

    def kinetic(self, field, token_ids, critical_a, disable_collision=False,
                disable_transport=False):
        field, boundary = self.port(field, token_ids, critical_a)
        if disable_collision:
            collision = {"collision_angle_abs_mean": field.new_zeros(())}
        else:
            field, collision = self.collision(field)
        if disable_transport:
            transport = {"transport_angle_abs_mean": field.new_zeros(())}
        else:
            field, transport = self.transport(field)
        return field, boundary, collision, transport

    def evolve(self, state, token_ids, measure_critical=False,
               disable_collision=False, disable_transport=False):
        field, delta, critical_a, lambda_ema, flux_ema = self.unpack(state)
        output, boundary, collision, transport = self.kinetic(
            field, token_ids, critical_a, disable_collision, disable_transport)
        with torch.no_grad():
            flux_ema = ((1-self.flux_ema)*flux_ema
                        + self.flux_ema*boundary["node_boundary_work"])
            control_step = self.flux_lr*flux_ema
            if measure_critical:
                shadow, _, _, _ = self.kinetic(
                    field.detach()+self.probe_epsilon*delta, token_ids,
                    critical_a.detach(), disable_collision, disable_transport)
                tangent = (shadow-output.detach())/self.probe_epsilon
                after_square = tangent.square().sum((-1, -2)).clamp_min(1e-14)
                gain = torch.log(after_square.sqrt()/delta.flatten(1).norm(dim=-1).clamp_min(1e-7)).clamp(-1, 1)
                energy_share = tangent.square().sum(-1)/after_square[:, None]
                pressure = .5 + .5*self.L*energy_share
                local_signal = gain[:, None]*pressure
                lambda_ema = (1-self.controller_ema)*lambda_ema + self.controller_ema*local_signal
                control_step = control_step+self.controller_lr*lambda_ema
                delta = tangent/after_square.sqrt()[:, None, None]
            critical_a = (critical_a+control_step).clamp(0, self.max_critical_a)
            gain_ema = lambda_ema.mean()
        packed = torch.cat((output, delta, critical_a[..., None], lambda_ema[..., None],
                            flux_ema[..., None]), -1)
        return packed, {**boundary,
            "field_energy": (.5*output.square().sum(-1).mean()).detach(),
            "critical_a_mean": critical_a.mean(), "critical_a_max": critical_a.max(),
            "lambda_mean": gain_ema, "lambda_max": lambda_ema.max(),
            "flux_ema_mean": flux_ema.mean(), "flux_ema_max": flux_ema.max(),
            "collision_angle_abs_mean": collision["collision_angle_abs_mean"],
            "transport_angle_abs_mean": transport["transport_angle_abs_mean"]}

    def forward(self, input_ids, targets, state=None, disable_collision=False,
                disable_transport=False):
        batch, tokens = input_ids.shape
        state = self.initial_state(batch, input_ids.device) if state is None else state
        position = self.readout.position_encoding()
        features, sums = [], []

        def segment(s, ids, pos):
            fs, ds = [], []
            for i in range(ids.shape[1]):
                s, d = self.evolve(s, ids[:, i], measure_critical=(i == 0),
                                   disable_collision=disable_collision,
                                   disable_transport=disable_transport)
                field, _, _, _, _ = self.unpack(s)
                fs.append(self.readout(field, pos))
                ds.append(torch.stack((d["field_energy"], d["boundary_work"],
                    d["port_balance_residual"], d["incident_energy"], d["outgoing_energy"],
                    d["accommodation_mean"], d["accommodation_max"], d["critical_a_mean"],
                    d["critical_a_max"], d["lambda_mean"], d["lambda_max"],
                    d["flux_ema_mean"], d["flux_ema_max"],
                    d["collision_angle_abs_mean"], d["transport_angle_abs_mean"])))
            return s, torch.stack(fs, 1), torch.stack(ds).sum(0)

        for start in range(0, tokens, self.checkpoint_tokens):
            ids = input_ids[:, start:start+self.checkpoint_tokens]
            if self.training and torch.is_grad_enabled():
                state, fs, ds = checkpoint(segment, state, ids, position,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                state, fs, ds = segment(state, ids, position)
            features.append(fs); sums.append(ds)
        logits = self.decoder(torch.cat(features, 1))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        values = torch.stack(sums).sum(0)/tokens
        names = ("energy", "boundary_work", "port_balance_residual", "incident_energy",
                 "outgoing_energy", "accommodation_mean", "accommodation_max",
                 "critical_a_mean", "critical_a_max", "lambda_mean", "lambda_max",
                 "flux_ema_mean", "flux_ema_max", "collision_angle_abs_mean",
                 "transport_angle_abs_mean")
        diagnostics = {name: values[i] for i, name in enumerate(names)}
        field, _, critical_a, lam, flux = self.unpack(state)
        diagnostics["final_energy"] = .5*field.square().sum(-1).mean().detach()
        diagnostics["node_critical_a"] = critical_a.detach()
        diagnostics["node_lambda"] = lam.detach()
        diagnostics["node_flux"] = flux.detach()
        return loss, state, diagnostics
