"""Self-organizing MaleCNS kinetic field with an explicit internal clock.

One external token performs one boundary exchange, followed by ``micro_steps``
shared transport/collision steps.  Observation is independent of the cold bath.
The persistent signed field has shape ``[batch, node, velocity, content]``.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.ib_local.cbim_malecns import load_malecns_graph
from scripts.ib_local.cbim_malecns_v3 import FullStateKineticCollision, GraphStateReadout
from scripts.ib_local.geometric_transport import cube_velocities


class LearnedBoundaryWrite(nn.Module):
    """Learned localized, energy-accounted token/field scattering."""

    def __init__(self, graph, vocab_size=50257, d=64, packet_radius=1.25,
                 max_angle=.30):
        super().__init__()
        self.d = d
        self.packet_radius = float(packet_radius)
        self.max_angle = float(max_angle)
        self.embedding = nn.Embedding(vocab_size, d)
        self.address = nn.Linear(d, 3)
        self.width = nn.Linear(d, 3)
        coordinates = graph["coordinates"].float()
        pair_distance = torch.cdist(coordinates, coordinates)
        pair_distance.fill_diagonal_(float("inf"))
        reference_width = pair_distance.amin(-1).median().item()
        inverse_softplus = math.log(math.expm1(reference_width))
        nn.init.zeros_(self.width.weight)
        nn.init.constant_(self.width.bias, inverse_softplus)
        self.content = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.state_content = nn.Linear(d, d, bias=False)
        self.neighbor_content = nn.Linear(d, d, bias=False)
        self.angle = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.normal_(self.content[-1].weight, std=1e-3)
        nn.init.zeros_(self.content[-1].bias)
        nn.init.normal_(self.state_content.weight, std=1e-3)
        nn.init.normal_(self.neighbor_content.weight, std=1e-3)
        nn.init.normal_(self.angle[-1].weight, std=1e-3)
        nn.init.constant_(self.angle[-1].bias, -1.5)
        self.register_buffer("coordinates", coordinates[None], persistent=False)
        if "neighbor_indices" in graph:
            neighbor_indices = graph["neighbor_indices"].long()
            weights = graph["neighbor_weights"].float()
        else:
            adjacency = graph["adjacency"].float()
            count = min(8, adjacency.shape[-1] - 1)
            weights, neighbor_indices = adjacency.topk(count, dim=-1)
        self.register_buffer("neighbor_indices", neighbor_indices, persistent=False)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("neighbor_weights", weights, persistent=False)
        # Node-local capacity with learnable per-channel scale (RMSNorm style):
        # packet at each active node has amplitude determined by envelope and channel_scale,
        # decoupling active node amplitude from total graph size and channel count.
        self.channel_scale = nn.Parameter(
            torch.full((d,), float(packet_radius) * math.sqrt(d / 64.0)))

    @staticmethod
    def energy(x):
        return .5 * x.square().sum(dim=(-1, -2)).mean()

    def forward(self, field, token_ids):
        token = self.embedding(token_ids)
        center = torch.sigmoid(self.address(token))[:, None]
        width = F.softplus(self.width(token))[:, None]
        width = width.clamp_min(torch.finfo(width.dtype).eps)
        distance = (self.coordinates - center) / width
        # A learned width can become much smaller than the graph spacing for a
        # particular token.  Squaring that distance is finite in the forward
        # pass often enough to underflow exp() to zero, but its backward can
        # form 0 * inf and poison every source gradient.  Saturate only beyond
        # the representable Gaussian tail; this changes no resolvable FP32
        # value and gives that numerically absent tail exactly zero gradient.
        tail_limit = math.sqrt(-2.0 * math.log(torch.finfo(distance.dtype).tiny))
        safe_distance = distance.clamp(min=-tail_limit, max=tail_limit)
        envelope = torch.exp(-.5 * safe_distance.square().sum(-1))
        # Peak-normalization: guarantees that the closest parcel always receives
        # the peak amplitude 1.0, eliminating coordinate-void signal starvation.
        envelope_peak = envelope.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        spatial = envelope / envelope_peak
        neighbors = field[:, self.neighbor_indices]
        neighborhood = (neighbors * self.neighbor_weights[None, ..., None]).sum(2)
        local_content = (self.content(token)[:, None]
                         + self.state_content(field)
                         + self.neighbor_content(neighborhood))
        local_content = F.normalize(local_content, dim=-1)
        context = (spatial[..., None] * field).sum(1) / spatial.sum(-1, keepdim=True).clamp_min(1e-8)
        packet = spatial[..., None] * (self.channel_scale * local_content)
        theta_channel = self.max_angle * torch.sigmoid(self.angle(torch.cat((token, context), -1)))
        theta = spatial[..., None] * theta_channel[:, None]
        cosine, sine = theta.cos(), theta.sin()
        field_next = cosine * field + sine * packet
        reflected = -sine * field + cosine * packet
        residual = (self.energy(field_next) + self.energy(reflected)
                    - self.energy(field) - self.energy(packet)).abs()
        incident_energy = self.energy(packet)
        reflected_energy = self.energy(reflected)
        return field_next, reflected, {
            "incident_energy": incident_energy.detach(),
            "reflected_energy": reflected_energy.detach(),
            "accepted_energy": (incident_energy - reflected_energy).detach(),
            "accepted_fraction": ((incident_energy - reflected_energy)
                                  / incident_energy.clamp_min(1e-8)).detach(),
            "write_angle_abs_mean": theta.detach().abs().mean(),
            "write_angle_peak_mean": theta_channel.detach().abs().mean(),
            "write_spatial_support": (spatial.detach() > 0.1).float().mean(),
            "write_balance_residual": residual.detach(),
            "source_center": center.detach().squeeze(1),
        }


class LearnedDirectionalEdgeTransport(nn.Module):
    """Geometry-aligned learned Givens transport on disjoint graph edges."""

    def __init__(self, graph, velocities=8, content_dim=8,
                 base_angle=.18, residual_angle=.17):
        super().__init__()
        pairs = graph["collision_pairs"].long()
        edge_features = graph["collision_features"].float()
        coordinates = graph["coordinates"].float()
        scale = graph.get("coordinate_scale", torch.ones(3)).float()
        physical = coordinates * scale
        displacement = physical[pairs[..., 1]] - physical[pairs[..., 0]]
        direction = displacement / displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        velocity = cube_velocities()
        projection = torch.einsum("lpk,qk->lpq", direction, velocity)
        geometry = torch.cat((edge_features[..., None, :].expand(-1, -1, velocities, -1),
                              projection[..., None]), -1)
        self.layers = pairs.shape[0]
        self.velocities = velocities
        self.content_dim = content_dim
        self.base_angle = float(base_angle)
        self.residual_angle = float(residual_angle)
        self.register_buffer("pairs", pairs, persistent=False)
        self.register_buffer("geometry", geometry, persistent=False)
        self.register_buffer("velocity_vectors", velocity, persistent=False)
        self.angle = nn.Sequential(nn.Linear(geometry.shape[-1], 32), nn.SiLU(),
                                   nn.Linear(32, content_dim))
        nn.init.normal_(self.angle[-1].weight, std=1e-3)
        nn.init.zeros_(self.angle[-1].bias)

    def forward(self, field):
        result = field
        all_angles = []
        for layer in range(self.layers):
            left_index, right_index = self.pairs[layer].unbind(-1)
            left, right = result[:, left_index], result[:, right_index]
            # Free streaming has a nonzero structural velocity and therefore
            # cannot collapse to identity. Learning supplies dispersion around
            # that physical baseline; state-dependent interaction remains the
            # responsibility of the local collision operator.
            projection = self.geometry[layer, ..., -1:]
            baseline = self.base_angle * projection
            residual = self.residual_angle * torch.tanh(
                self.angle(self.geometry[layer]))
            theta = baseline + residual
            theta = theta[None]
            cosine, sine = theta.cos(), theta.sin()
            left_new = cosine * left - sine * right
            right_new = sine * left + cosine * right
            updated = result.clone()
            updated[:, left_index] = left_new
            updated[:, right_index] = right_new
            result = updated
            all_angles.append(theta)
        angles = torch.stack(all_angles)
        before = field.square().sum()
        after = result.square().sum()
        return result, {
            "transport_angle_abs_mean": angles.detach().abs().mean(),
            "transport_angle_abs_max": angles.detach().abs().amax(),
            "transport_baseline_angle_abs_mean": (
                self.base_angle * self.geometry[..., -1]).detach().abs().mean(),
            "transport_norm_residual": (after - before).detach().abs(),
        }


class DirectionalSpectralCayleyTransport(nn.Module):
    """Velocity-aligned, norm-preserving transport in a Galerkin subspace."""

    def __init__(self, graph, velocities=8, content_dim=8):
        super().__init__()
        basis = torch.linalg.qr(graph["laplacian_basis"].float(), mode="reduced").Q
        coordinates = graph["coordinates"].float() * graph.get("coordinate_scale", torch.ones(3)).float()
        adjacency = graph["adjacency"].float()
        adjacency = .5 * (adjacency + adjacency.T)
        delta = coordinates[None] - coordinates[:, None]
        direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        velocity = cube_velocities()
        generators = []
        for v in velocity:
            generator = adjacency * torch.einsum("ijk,k->ij", direction, v)
            generator = .5 * (generator - generator.T)
            reduced = basis.T @ generator @ basis
            reduced = reduced / torch.linalg.matrix_norm(reduced, ord=2).clamp_min(1e-8)
            generators.append(reduced)
        self.velocities, self.content_dim = velocities, content_dim
        self.register_buffer("basis", basis, persistent=False)
        generators = torch.stack(generators)
        # iA is Hermitian for real skew A.  Diagonalize once so the learned
        # Cayley step needs only elementwise unit-modulus factors at runtime;
        # cuSOLVER itself is not permitted during CUDA Graph capture.
        eigenvalues, eigenvectors = torch.linalg.eigh(1j * generators.to(torch.complex64))
        self.register_buffer("generator_eigenvalues", eigenvalues.float(), persistent=False)
        self.register_buffer("generator_eigenvectors", eigenvectors, persistent=False)
        self.register_buffer("velocity_vectors", velocity, persistent=False)
        self.log_step = nn.Parameter(torch.zeros(velocities))

    def forward(self, field):
        coefficients = torch.einsum("nr,bnqa->bqra", self.basis, field)
        step = self.log_step.exp()[:, None]
        mu = self.generator_eigenvalues
        phase = (1.0 - .5j * step * mu) / (1.0 + .5j * step * mu)
        vectors = self.generator_eigenvectors
        modal = torch.einsum("qrs,bqsa->bqra", vectors.conj().transpose(-2, -1),
                             coefficients.to(torch.complex64))
        moved = torch.einsum("qrs,bqsa->bqra", vectors, phase[None, ..., None] * modal).real
        output = field + torch.einsum("nr,bqra->bnqa", self.basis, moved - coefficients)
        return output, {
            "transport_angle_abs_mean": self.log_step.exp().detach().mean(),
            "transport_angle_abs_max": self.log_step.exp().detach().max(),
            "transport_baseline_angle_abs_mean": self.log_step.exp().detach().mean(),
            "transport_norm_residual": (output.square().sum() - field.square().sum()).detach().abs(),
        }


class QuadraticPassiveBath(nn.Module):
    """Threshold-free quadratic passive radiation bath J_bath = 2 * kappa * E^2 / R^2."""

    def __init__(
        self,
        node_features: torch.Tensor,
        d: int = 64,
        local_radius: float = 1.25,
        max_kappa: float = 0.05,
    ) -> None:
        super().__init__()
        self.d = d
        # Dimension-invariant capacity scale: R_d = R_0 * sqrt(d / 64)
        self.local_radius = float(local_radius) * math.sqrt(d / 64.0)
        self.max_kappa = float(max_kappa)
        node_dim = node_features.shape[-1]
        self.kappa_net = nn.Sequential(
            nn.Linear(node_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        nn.init.normal_(self.kappa_net[-1].weight, std=1e-3)
        nn.init.constant_(self.kappa_net[-1].bias, 0.0)
        self.register_buffer("node_features", node_features.float()[None], persistent=False)

    def forward(self, field: torch.Tensor):
        local_energy = 0.5 * field.square().sum(dim=-1, keepdim=True)
        rho = 2.0 * local_energy / (self.local_radius ** 2)

        kappa = self.max_kappa * torch.sigmoid(self.kappa_net(self.node_features))
        sin2_theta = torch.clamp(kappa * rho, max=0.5)
        cos_theta = torch.sqrt(1.0 - sin2_theta)
        sin_theta = torch.sqrt(sin2_theta.clamp_min(torch.finfo(field.dtype).tiny))

        field_next = cos_theta * field
        bath_out = -sin_theta * field
        j_bath = 0.5 * bath_out.square().sum(dim=-1).mean()
        theta = torch.asin(sin_theta)

        return field_next, {
            "bath_out_energy": j_bath.detach(),
            "bath_angle_abs_mean": theta.detach().abs().mean(),
            "bath_active_fraction": (rho > 1.0).float().mean().detach(),
        }


class ColdBoundaryBath(QuadraticPassiveBath):
    """Alias for backwards compatibility."""
    pass


class EnergyFactoredReadout(nn.Module):
    """Energy-factored decoupled spatial attention over anatomical parcels."""

    def __init__(self, node_features, d=64, queries=4, heads=4, local_radius=1.25):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d, self.queries, self.heads = d, queries, heads
        self.head_dim = d // heads
        self.local_radius = float(local_radius) * math.sqrt(d / 64.0)
        self.query = nn.Parameter(torch.randn(1, queries, d) * .02)
        nodes = int(node_features.shape[0])
        initial_scale = math.log2(nodes * nodes - nodes)
        self.head_log_scale = nn.Parameter(torch.full((1, heads, 1, 1), math.log(initial_scale)))
        self.head_log_scale._no_weight_decay = True
        self.energy_alpha = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.position = nn.Sequential(
            nn.Linear(node_features.shape[-1], d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.merge = nn.Linear(queries * d, d)
        self.output = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("node_features", node_features.float()[None], persistent=False)

    def position_encoding(self):
        return self.position(self.node_features)

    def logit_scales(self):
        return torch.exp(self.head_log_scale)

    def forward(self, field, position_encoding=None):
        batch, nodes, d = field.shape
        position = (self.position_encoding() if position_encoding is None
                    else position_encoding)
        keys = self.key(self.norm(field) + position)
        values = self.value(self.norm(field))
        queries = self.query.expand(batch, -1, -1)

        def split(value):
            return value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        query_heads = F.normalize(split(queries), dim=-1)
        key_heads = F.normalize(split(keys), dim=-1)
        value_heads = split(values)

        scale = torch.exp(self.head_log_scale)
        semantic_score = torch.matmul(query_heads, key_heads.transpose(-1, -2)) * scale

        local_energy = 0.5 * field.square().sum(dim=-1)
        rho = 2.0 * local_energy / (self.local_radius ** 2)
        energy_prior = torch.log1p(rho)[:, None, None, :]
        scores = semantic_score + self.energy_alpha * energy_prior
        attention = torch.softmax(scores, dim=-1)

        read = torch.matmul(attention, value_heads)
        read = read.transpose(1, 2).reshape(batch, self.queries * d)
        return self.output(self.merge(read))

    @torch.no_grad()
    def attention_weights(self, field, position_encoding=None):
        batch, nodes, d = field.shape
        position = (self.position_encoding() if position_encoding is None
                    else position_encoding)
        keys = F.normalize(self.key(self.norm(field) + position).reshape(
            batch, -1, self.heads, self.head_dim).transpose(1, 2), dim=-1)
        queries = F.normalize(self.query.expand(batch, -1, -1).reshape(
            batch, self.queries, self.heads, self.head_dim).transpose(1, 2), dim=-1)
        scale = torch.exp(self.head_log_scale)
        semantic_score = torch.matmul(queries, keys.transpose(-1, -2)) * scale
        local_energy = 0.5 * field.square().sum(dim=-1)
        rho = 2.0 * local_energy / (self.local_radius ** 2)
        energy_prior = torch.log1p(rho)[:, None, None, :]
        scores = semantic_score + self.energy_alpha * energy_prior
        return torch.softmax(scores, dim=-1)

    @torch.no_grad()
    def attention_map(self, field, position_encoding=None):
        """Return mean-head attention per query for scalar diagnostics."""
        return self.attention_weights(field, position_encoding).mean(1)


class CBIMMaleCNSInternalTime(nn.Module):
    architecture = "CBIM-MaleCNS-derived-scale-qknorm-v4"

    def __init__(self, graph_path, vocab_size=50257, velocities=8,
                 content_dim=8, queries=4, heads=4, micro_steps=4,
                 checkpoint_tokens=4, spectral_transport=False):
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path = str(graph_path)
        self.velocities, self.content_dim = velocities, content_dim
        self.d = velocities * content_dim
        self.L = graph["coordinates"].shape[0]
        self.micro_steps = int(micro_steps)
        self.checkpoint_tokens = int(checkpoint_tokens)
        self.architecture = ("CBIM-MaleCNS-fullrank-spectral-cayley-v5"
                             if spectral_transport else
                             "CBIM-MaleCNS-fullrank-edge-v5")
        self.state_shape = (self.L, self.d)
        self.source = LearnedBoundaryWrite(graph, vocab_size, self.d)
        self.transport = (DirectionalSpectralCayleyTransport(graph, velocities, content_dim)
                          if spectral_transport else
                          LearnedDirectionalEdgeTransport(graph, velocities, content_dim))
        self.collision = FullStateKineticCollision(graph["node_features"], velocities, content_dim)
        self.bath = QuadraticPassiveBath(graph["node_features"], self.d)
        self.readout = EnergyFactoredReadout(graph["node_features"], self.d, queries, heads)
        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.source.embedding.weight

    def initial_state(self, batch_size, device=None, dtype=None):
        parameter = next(self.parameters())
        return torch.zeros(batch_size, self.L, self.d,
                           device=parameter.device if device is None else device,
                           dtype=parameter.dtype if dtype is None else dtype)

    def evolve(self, state, token_ids, disable_transport=False, disable_collision=False):
        state, _, source = self.source(state, token_ids)
        transport_angle = state.new_zeros(())
        collision_angle = state.new_zeros(())
        transport_residual = state.new_zeros(())
        for _ in range(self.micro_steps):
            shaped = state.reshape(state.shape[0], self.L, self.velocities, self.content_dim)
            if not disable_transport:
                shaped, transport = self.transport(shaped)
                transport_angle = transport_angle + transport["transport_angle_abs_mean"]
                transport_residual = torch.maximum(transport_residual, transport["transport_norm_residual"])
            state = shaped.reshape(state.shape[0], self.L, self.d)
            if not disable_collision:
                state, collision = self.collision(state)
                collision_angle = collision_angle + collision["collision_angle_abs_mean"]
        state, bath = self.bath(state)
        return state, {
            **source, **bath,
            "transport_angle_abs_mean": transport_angle / self.micro_steps,
            "collision_angle_abs_mean": collision_angle / self.micro_steps,
            "transport_norm_residual": transport_residual,
        }

    def forward(self, input_ids, targets, state=None, disable_transport=False,
                disable_collision=False):
        batch, tokens = input_ids.shape
        state = self.initial_state(batch, input_ids.device) if state is None else state
        position = self.readout.position_encoding()
        features, sums = [], []

        def segment(current, ids, position_encoding):
            local_features, local_diagnostics = [], []
            for index in range(ids.shape[1]):
                current, diagnostics = self.evolve(current, ids[:, index], disable_transport, disable_collision)
                local_features.append(self.readout(current, position_encoding))
                local_diagnostics.append(torch.stack((
                    diagnostics["incident_energy"], diagnostics["reflected_energy"],
                    diagnostics["bath_out_energy"], diagnostics["write_angle_abs_mean"],
                    diagnostics["bath_angle_abs_mean"], diagnostics["bath_active_fraction"],
                    diagnostics["transport_angle_abs_mean"],
                    diagnostics["collision_angle_abs_mean"], diagnostics["write_balance_residual"],
                    diagnostics["transport_norm_residual"])))
            return current, torch.stack(local_features, 1), torch.stack(local_diagnostics).sum(0)

        for start in range(0, tokens, self.checkpoint_tokens):
            ids = input_ids[:, start:start + self.checkpoint_tokens]
            if self.training and torch.is_grad_enabled():
                state, feature, diagnostic = checkpoint(segment, state, ids, position,
                                                        use_reentrant=False, preserve_rng_state=False)
            else:
                state, feature, diagnostic = segment(state, ids, position)
            features.append(feature)
            sums.append(diagnostic)
        logits = self.decoder(torch.cat(features, 1))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        mean = torch.stack(sums).sum(0) / tokens
        return loss, state, {
            "final_energy": .5 * state.square().sum(-1).mean().detach(),
            "incident_energy": mean[0], "reflected_energy": mean[1],
            "bath_out_energy": mean[2], "write_angle_abs_mean": mean[3],
            "bath_angle_abs_mean": mean[4], "bath_active_fraction": mean[5],
            "transport_angle_abs_mean": mean[6],
            "collision_angle_abs_mean": mean[7], "write_balance_residual": mean[8],
            "transport_norm_residual": mean[9],
        }
