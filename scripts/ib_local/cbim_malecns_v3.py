"""Open-boundary, self-organizing kinetic CBIM on the MaleCNS graph.

The persistent state packs ``f[x, q, a]``, a local write resource and a slow
fatigue field into one CUDA-graph-friendly tensor.  Internal transport and
collision are conservative.  Only localized exchange with the moving token
boundary changes total field energy.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.ib_local.cbim_malecns import load_malecns_graph


def _cube_velocities(dtype=torch.float32):
    velocity = torch.tensor(
        [(x, y, z) for x in (-1.0, 1.0)
         for y in (-1.0, 1.0) for z in (-1.0, 1.0)], dtype=dtype)
    return velocity / math.sqrt(3.0)


class FullRankBoundaryExchange(nn.Module):
    """State/neighborhood-aware full-rank exchange with the token boundary."""

    def __init__(self, graph, vocab_size=50257, d=64, packet_radius=1.25,
                 max_write_rate=.25, recovery_rate=2e-3,
                 depletion_rate=.25):
        super().__init__()
        coordinates = graph["coordinates"].float()
        node_features = graph["node_features"].float()
        self.d = d
        self.packet_radius = float(packet_radius)
        self.max_write_rate = float(max_write_rate)
        self.recovery_rate = float(recovery_rate)
        self.depletion_rate = float(depletion_rate)
        self.embedding = nn.Embedding(vocab_size, d)
        self.address = nn.Linear(d, 3)
        self.width = nn.Linear(d, 3)
        self.token = nn.Linear(d, d)
        self.state = nn.Linear(d, d)
        self.neighbor = nn.Linear(d, d)
        self.position = nn.Linear(node_features.shape[-1], d)
        self.proposal = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.amplitude = nn.Linear(d, 1)
        self.write_rate = nn.Linear(d + 2, 1)
        nn.init.normal_(self.proposal[-1].weight, std=1e-3)
        nn.init.zeros_(self.proposal[-1].bias)
        nn.init.normal_(self.write_rate.weight, std=1e-3)
        nn.init.zeros_(self.write_rate.bias)
        nn.init.constant_(self.amplitude.bias, -1.5)
        self.register_buffer("coordinates", coordinates[None], persistent=False)
        self.register_buffer("node_features", node_features[None], persistent=False)
        self.register_buffer(
            "neighbor_indices", graph["neighbor_indices"].long(), persistent=False)
        self.register_buffer(
            "neighbor_weights", graph["neighbor_weights"].float(), persistent=False)

    def forward(self, field, resource, fatigue, token_ids):
        embedding = self.embedding(token_ids)
        center = torch.sigmoid(self.address(embedding))[:, None]
        width = (.04 + .21 * torch.sigmoid(self.width(embedding)))[:, None]
        distance = (self.coordinates - center) / width
        envelope = torch.exp(-.5 * distance.square().sum(-1))
        neighbors = field[:, self.neighbor_indices]
        neighborhood = (
            neighbors * self.neighbor_weights[None, ..., None]).sum(2)
        context = F.silu(
            self.token(embedding)[:, None] + self.state(field)
            + self.neighbor(neighborhood) + self.position(self.node_features))
        raw_proposal = self.proposal(context)
        direction = F.normalize(raw_proposal, dim=-1)
        amplitude = torch.sigmoid(self.amplitude(context))
        proposal = self.packet_radius * amplitude * direction
        local = torch.cat((context, resource[..., None], fatigue[..., None]), -1)
        rate = (self.max_write_rate * envelope[..., None] * resource[..., None]
                * torch.sigmoid(self.write_rate(local)))
        output = field + rate * (proposal - field)

        novelty = (proposal - field).square().mean(-1)
        recovered = 1.0 - (1.0 - resource) * math.exp(-self.recovery_rate)
        cost = self.depletion_rate * rate.squeeze(-1) * novelty
        resource_next = recovered * torch.exp(-cost)
        energy_delta = .5 * (output.square() - field.square()).sum(-1).mean()
        return output, resource_next, envelope, {
            "boundary_input_work": energy_delta.detach(),
            "write_rate_mean": rate.detach().mean(),
            "write_rate_max": rate.detach().amax(),
            "resource_mean": resource_next.detach().mean(),
            "resource_min": resource_next.detach().amin(),
            "source_center": center.detach().squeeze(1),
        }


class ContinuousGraphTransport(nn.Module):
    """Resolution-independent orthogonal graph transport with full channels."""

    def __init__(self, basis, eigenvalues, velocities=8, content_dim=8,
                 hidden=64):
        super().__init__()
        if basis.shape[1] % 2:
            raise ValueError("An even number of graph modes is required")
        basis = torch.linalg.qr(basis.float(), mode="reduced").Q
        values = eigenvalues.float()
        mode = torch.stack((values[0::2], values[1::2],
                            values[1::2] - values[0::2],
                            .5 * (values[0::2] + values[1::2])), -1)
        velocity = _cube_velocities()[None].expand(mode.shape[0], -1, -1)
        mode = mode[:, None].expand(-1, velocities, -1)
        symbol_input = torch.cat((mode, velocity), -1)
        self.velocities = velocities
        self.content_dim = content_dim
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("symbol_input", symbol_input, persistent=False)
        self.symbol = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(), nn.Linear(hidden, content_dim))
        nn.init.normal_(self.symbol[-1].weight, std=1e-3)
        nn.init.zeros_(self.symbol[-1].bias)

    def angles(self):
        return self.symbol(self.symbol_input)

    def forward(self, field):
        batch, nodes, d = field.shape
        values = field.reshape(
            batch, nodes, self.velocities, self.content_dim)
        coefficients = torch.einsum("mr,bmqa->brqa", self.basis, values)
        theta = self.angles()[None]
        left, right = coefficients[:, 0::2], coefficients[:, 1::2]
        cosine, sine = theta.cos(), theta.sin()
        rotated = torch.stack((cosine * left - sine * right,
                               sine * left + cosine * right), 2).flatten(1, 2)
        transported = values + torch.einsum(
            "mr,brqa->bmqa", self.basis, rotated - coefficients)
        return transported.reshape(batch, nodes, d), {
            "transport_angle_abs_mean": theta.detach().abs().mean(),
            "transport_angle_abs_max": theta.detach().abs().max(),
        }


class FullStateKineticCollision(nn.Module):
    """Local D3Q8 collision coupling content in the full invariant nullspace."""

    def __init__(self, node_features, velocities=8, content_dim=8,
                 hidden=96, layers=2):
        super().__init__()
        if velocities != 8:
            raise ValueError("The first implementation uses D3Q8")
        self.velocities = velocities
        self.content_dim = content_dim
        self.d = velocities * content_dim
        velocity = _cube_velocities(dtype=torch.float64)
        mass = torch.ones(1, velocities, content_dim, dtype=torch.float64)
        momentum = velocity.T[:, :, None].expand(-1, -1, content_dim)
        constraints = torch.cat((mass, momentum), 0).reshape(4, self.d)
        _, singular, right = torch.linalg.svd(constraints, full_matrices=True)
        rank = int((singular > 1e-10).sum())
        nullspace = right[rank:].T
        if nullspace.shape[1] % 2:
            raise ValueError("Collision nullity must be even")
        self.nullity = nullspace.shape[1]
        self.layers = int(layers)
        schedules = []
        for layer in range(self.layers):
            order = torch.roll(torch.arange(self.nullity), layer)
            schedules.append(order.reshape(-1, 2))
        self.register_buffer("constraints", constraints, persistent=False)
        self.register_buffer("nullspace", nullspace, persistent=False)
        self.register_buffer("schedules", torch.stack(schedules), persistent=False)
        self.register_buffer(
            "node_features", node_features.float()[None], persistent=False)
        self.norm = nn.LayerNorm(self.d)
        self.angle = nn.Sequential(
            nn.Linear(self.d + node_features.shape[-1], hidden), nn.SiLU(),
            nn.Linear(hidden, self.layers * (self.nullity // 2)))
        nn.init.normal_(self.angle[-1].weight, std=1e-3)
        nn.init.zeros_(self.angle[-1].bias)

    def forward(self, field):
        nullspace = self.nullspace.to(dtype=field.dtype)
        coefficient = torch.einsum("dk,bnd->bnk", nullspace, field)
        conserved = field - torch.einsum("dk,bnk->bnd", nullspace, coefficient)
        features = self.node_features.expand(field.shape[0], -1, -1)
        angles = self.angle(torch.cat((self.norm(field), features), -1)).reshape(
            field.shape[0], field.shape[1], self.layers, self.nullity // 2)
        value = coefficient
        for layer in range(self.layers):
            pair = self.schedules[layer]
            left, right = value[..., pair[:, 0]], value[..., pair[:, 1]]
            theta = angles[:, :, layer]
            cosine, sine = theta.cos(), theta.sin()
            left_new = cosine * left - sine * right
            right_new = sine * left + cosine * right
            updated = value.clone()
            updated[..., pair[:, 0]] = left_new
            updated[..., pair[:, 1]] = right_new
            value = updated
        output = conserved + torch.einsum("dk,bnk->bnd", nullspace, value)
        return output, {
            "collision_angle_abs_mean": angles.detach().abs().mean(),
            "collision_angle_abs_max": angles.detach().abs().amax(),
        }


class LocalBoundaryOutflow(nn.Module):
    """Remove energy only where the moving token boundary is active."""

    def __init__(self, node_features, d=64, hidden=64, max_outflow=.05):
        super().__init__()
        self.max_outflow = float(max_outflow)
        self.register_buffer(
            "node_features", node_features.float()[None], persistent=False)
        self.rate = nn.Sequential(
            nn.Linear(d + node_features.shape[-1] + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        nn.init.normal_(self.rate[-1].weight, std=1e-3)
        nn.init.constant_(self.rate[-1].bias, -4.0)

    def forward(self, field, resource, fatigue, envelope):
        features = self.node_features.expand(field.shape[0], -1, -1)
        context = torch.cat(
            (field, features, resource[..., None], fatigue[..., None]), -1)
        rate = self.max_outflow * envelope[..., None] * torch.sigmoid(
            self.rate(context))
        output = (1.0 - rate) * field
        dissipated = .5 * (field.square() - output.square()).sum(-1).mean()
        return output, {
            "boundary_outflow_rate": rate.detach().mean(),
            "boundary_output_work": dissipated.detach(),
        }


class GraphStateReadout(nn.Module):
    """State-only multi-query attention over anatomical parcels."""

    def __init__(self, node_features, d=64, queries=4, heads=4):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d, self.queries, self.heads = d, queries, heads
        self.query = nn.Parameter(torch.randn(1, queries, d) * .02)
        # QKNorm initializes g from attention-set length:
        # g0 = log2(L^2-L). We optimize log(g) per head so scale stays positive
        # without imposing an arbitrary upper bound.
        nodes = int(node_features.shape[0])
        initial_scale = math.log2(nodes * nodes - nodes)
        self.head_log_scale = nn.Parameter(torch.full(
            (1, heads, 1, 1), math.log(initial_scale)))
        self.head_log_scale._no_weight_decay = True
        self.position = nn.Sequential(
            nn.Linear(node_features.shape[-1], d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.merge = nn.Linear(queries * d, d)
        self.output = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer(
            "node_features", node_features.float()[None], persistent=False)

    def position_encoding(self):
        return self.position(self.node_features)

    def forward(self, field, position_encoding=None):
        batch, _, d = field.shape
        position = (self.position_encoding() if position_encoding is None
                    else position_encoding)
        keys = self.key(self.norm(field) + position)
        values = self.value(self.norm(field))
        queries = self.query.expand(batch, -1, -1)

        def split(value):
            return value.reshape(
                batch, -1, self.heads, d // self.heads).transpose(1, 2)

        query_heads = F.normalize(split(queries), dim=-1)
        key_heads = F.normalize(split(keys), dim=-1)
        value_heads = split(values)
        scale = torch.exp(self.head_log_scale)
        attention = torch.softmax(
            torch.matmul(query_heads, key_heads.transpose(-1, -2)) * scale,
            dim=-1)
        read = torch.matmul(attention, value_heads)
        read = read.transpose(1, 2).reshape(batch, self.queries * d)
        return self.output(self.merge(read))

    @torch.no_grad()
    def attention_weights(self, field, position_encoding=None):
        """Return attention weights as ``[batch, head, query, node]``."""
        batch, _, d = field.shape
        position = (self.position_encoding() if position_encoding is None
                    else position_encoding)
        keys = F.normalize(self.key(self.norm(field) + position).reshape(
            batch, -1, self.heads, d // self.heads).transpose(1, 2)
        , dim=-1)
        queries = F.normalize(self.query.expand(batch, -1, -1).reshape(
            batch, self.queries, self.heads, d // self.heads).transpose(1, 2)
        , dim=-1)
        scale = torch.exp(self.head_log_scale)
        weights = torch.softmax(
            torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1)
        return weights

    @torch.no_grad()
    def attention_map(self, field, position_encoding=None):
        """Return mean-head attention per query for scalar diagnostics."""
        return self.attention_weights(field, position_encoding).mean(1)

    def logit_scales(self):
        return torch.exp(self.head_log_scale)


class CBIMMaleCNSV3(nn.Module):
    architecture = "CBIM-MaleCNS-open-boundary-kinetic-v3"

    def __init__(self, graph_path, vocab_size=50257, velocities=8,
                 content_dim=8, queries=4, heads=4, fatigue_decay=.995,
                 checkpoint_tokens=8):
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path = str(graph_path)
        self.velocities = velocities
        self.content_dim = content_dim
        self.d = velocities * content_dim
        self.L = graph["coordinates"].shape[0]
        self.state_shape = (self.L, self.d + 2)
        self.fatigue_decay = float(fatigue_decay)
        self.checkpoint_tokens = int(checkpoint_tokens)
        self.source = FullRankBoundaryExchange(
            graph, vocab_size=vocab_size, d=self.d)
        self.transport = ContinuousGraphTransport(
            graph["laplacian_basis"], graph["laplacian_eigenvalues"],
            velocities, content_dim)
        self.collision = FullStateKineticCollision(
            graph["node_features"], velocities, content_dim)
        self.outflow = LocalBoundaryOutflow(graph["node_features"], self.d)
        self.readout = GraphStateReadout(
            graph["node_features"], self.d, queries, heads)
        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.source.embedding.weight

    def initial_state(self, batch_size, device=None, dtype=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        state = torch.zeros(
            batch_size, self.L, self.d + 2, device=device, dtype=dtype)
        state[..., self.d] = 1.0
        return state

    def unpack(self, state):
        return state[..., :self.d], state[..., self.d], state[..., self.d + 1]

    def evolve(self, state, token_ids, disable_collision=False):
        field, resource, fatigue = self.unpack(state)
        energy_before = .5 * field.square().sum(-1).mean()
        field, resource, envelope, source = self.source(
            field, resource, fatigue, token_ids)
        field, transport = self.transport(field)
        if disable_collision:
            collision = {
                "collision_angle_abs_mean": field.new_zeros(()),
                "collision_angle_abs_max": field.new_zeros(()),
            }
        else:
            field, collision = self.collision(field)
        field, outflow = self.outflow(field, resource, fatigue, envelope)
        local_energy = .5 * field.square().sum(-1)
        fatigue = (self.fatigue_decay * fatigue
                   + (1.0 - self.fatigue_decay) * local_energy)
        energy_after = local_energy.mean()
        ledger = (energy_after - energy_before - source["boundary_input_work"]
                  + outflow["boundary_output_work"])
        next_state = torch.cat(
            (field, resource[..., None], fatigue[..., None]), -1)
        return next_state, {
            "source": source, "transport": transport,
            "collision": collision,
            "outflow": outflow,
            "energy_balance_residual": ledger.detach(),
            "field_energy": energy_after.detach(),
            "fatigue_mean": fatigue.detach().mean(),
            "fatigue_max": fatigue.detach().amax(),
        }

    def step(self, state, token_ids, disable_collision=False):
        state, diagnostics = self.evolve(
            state, token_ids, disable_collision)
        field, _, _ = self.unpack(state)
        return self.decoder(self.readout(field)), state, diagnostics

    def forward(self, input_ids, targets, state=None, disable_collision=False):
        batch, tokens = input_ids.shape
        if state is None:
            state = self.initial_state(batch, input_ids.device)
        position = self.readout.position_encoding()
        features, diagnostic_sums = [], []

        def run_segment(segment_state, segment_ids, position_encoding):
            segment_features = []
            segment_diagnostics = []
            for index in range(segment_ids.shape[1]):
                segment_state, diagnostics = self.evolve(
                    segment_state, segment_ids[:, index], disable_collision)
                field, _, _ = self.unpack(segment_state)
                segment_features.append(self.readout(field, position_encoding))
                segment_diagnostics.append(torch.stack((
                    diagnostics["source"]["boundary_input_work"],
                    diagnostics["outflow"]["boundary_output_work"],
                    diagnostics["source"]["write_rate_mean"],
                    diagnostics["outflow"]["boundary_outflow_rate"],
                    diagnostics["transport"]["transport_angle_abs_mean"],
                    diagnostics["collision"]["collision_angle_abs_mean"],
                    diagnostics["energy_balance_residual"].abs(),
                )))
            return (segment_state, torch.stack(segment_features, 1),
                    torch.stack(segment_diagnostics).sum(0))

        for start in range(0, tokens, self.checkpoint_tokens):
            segment_ids = input_ids[:, start:start + self.checkpoint_tokens]
            if self.training and torch.is_grad_enabled():
                state, segment_features, diagnostic_sum = checkpoint(
                    run_segment, state, segment_ids, position,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                state, segment_features, diagnostic_sum = run_segment(
                    state, segment_ids, position)
            features.append(segment_features)
            diagnostic_sums.append(diagnostic_sum)
        logits = self.decoder(torch.cat(features, 1))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        diagnostic_mean = torch.stack(diagnostic_sums).sum(0) / tokens
        field, resource, fatigue = self.unpack(state)
        return loss, state, {
            "final_energy": .5 * field.square().sum(-1).mean().detach(),
            "boundary_input_work": diagnostic_mean[0],
            "boundary_output_work": diagnostic_mean[1],
            "write_rate_mean": diagnostic_mean[2],
            "outflow_rate_mean": diagnostic_mean[3],
            "resource_mean": resource.detach().mean(),
            "resource_min": resource.detach().amin(),
            "fatigue_mean": fatigue.detach().mean(),
            "fatigue_max": fatigue.detach().amax(),
            "transport_angle_abs_mean": diagnostic_mean[4],
            "collision_angle_abs_mean": diagnostic_mean[5],
            "energy_balance_residual": diagnostic_mean[6],
        }
