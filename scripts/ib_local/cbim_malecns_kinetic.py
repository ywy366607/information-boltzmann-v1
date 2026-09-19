"""Role-separated kinetic CBIM on the coarsened MaleCNS graph.

The recurrent field is ``f[x, q, a]``: ``x`` is a graph parcel, ``q`` is one
of eight discrete velocity channels and ``a`` is continuous carried content.
Source, spatial transport, local collision and dissipation are separate
operators so that their causal roles can be tested after training.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.cbim_malecns import load_malecns_graph


def _cube_velocities(device=None, dtype=torch.float32):
    values = [-1.0, 1.0]
    velocity = torch.tensor(
        [(x, y, z) for x in values for y in values for z in values],
        device=device, dtype=dtype,
    )
    return velocity / math.sqrt(3.0)


class GraphBoundarySource(nn.Module):
    """Inject a bounded token packet without moving or erasing old state."""

    def __init__(self, coordinates, vocab_size=50257, velocities=8,
                 content_dim=8, max_injection=.20):
        super().__init__()
        self.velocities = velocities
        self.content_dim = content_dim
        self.d = velocities * content_dim
        self.max_injection = float(max_injection)
        self.embedding = nn.Embedding(vocab_size, self.d)
        self.content = nn.Sequential(
            nn.Linear(self.d, self.d), nn.SiLU(),
            nn.Linear(self.d, content_dim),
        )
        self.velocity_route = nn.Linear(self.d, velocities)
        self.address = nn.Linear(self.d, 3)
        self.width = nn.Linear(self.d, 3)
        self.amplitude = nn.Linear(self.d, 1)
        nn.init.constant_(self.amplitude.bias, -2.0)
        self.register_buffer(
            "coordinates", coordinates.float()[None], persistent=False)

    def forward(self, h, token_ids):
        batch, nodes, _ = h.shape
        emb = self.embedding(token_ids)
        center = torch.sigmoid(self.address(emb))[:, None]
        sigma = (.04 + .21 * torch.sigmoid(self.width(emb)))[:, None]
        distance = (self.coordinates - center) / sigma
        envelope = torch.exp(-.5 * distance.square().sum(-1))
        # L2 normalization makes the injected energy independent of graph size.
        envelope = envelope / envelope.square().sum(1, keepdim=True).sqrt().clamp_min(1e-8)
        content = F.normalize(self.content(emb), dim=-1)
        route = F.softmax(self.velocity_route(emb), dim=-1).sqrt()
        packet = (route[..., None] * content[:, None]).reshape(batch, self.d)
        amplitude = self.max_injection * torch.sigmoid(self.amplitude(emb))
        injection = amplitude[:, None, :] * envelope[..., None] * packet[:, None]
        output = h + injection
        energy_delta = .5 * (output.square() - h.square()).sum(-1).mean()
        return output, {
            "injection_norm": injection.flatten(1).norm(dim=-1).detach().mean(),
            "source_energy_delta": energy_delta.detach(),
            "source_center": center.detach().squeeze(1),
            "source_width": sigma.detach().squeeze(1),
        }


class GraphVelocityTransport(nn.Module):
    """Norm-preserving spatial transport that never mixes velocity channels."""

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
        velocity = _cube_velocities(dtype=torch.float32)
        mode = mode[:, None].expand(-1, velocities, -1)
        velocity = velocity[None].expand(mode.shape[0], -1, -1)
        symbol_input = torch.cat((mode, velocity), -1)
        self.velocities = velocities
        self.content_dim = content_dim
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("symbol_input", symbol_input, persistent=False)
        self.symbol = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.symbol[-1].weight, std=1e-3)
        nn.init.zeros_(self.symbol[-1].bias)

    def angles(self):
        # Each velocity owns a dispersion law; content is carried unchanged.
        return self.symbol(self.symbol_input).squeeze(-1)

    def forward(self, h):
        batch, nodes, _ = h.shape
        field = h.reshape(batch, nodes, self.velocities, self.content_dim)
        coefficients = torch.einsum("mr,bmqa->brqa", self.basis, field)
        theta = self.angles()[None, ..., None]
        cosine, sine = theta.cos(), theta.sin()
        left, right = coefficients[:, 0::2], coefficients[:, 1::2]
        rotated = torch.stack((cosine * left - sine * right,
                               sine * left + cosine * right), 2).flatten(1, 2)
        transported = field + torch.einsum(
            "mr,brqa->bmqa", self.basis, rotated - coefficients)
        return transported.reshape_as(h), {
            "transport_angle_abs_mean": theta.detach().abs().mean(),
            "transport_angle_abs_max": theta.detach().abs().max(),
        }


class LocalVelocityCollision(nn.Module):
    """State-aware collision in velocity space, independently at every node.

    A precomputed nullspace removes the mass-like and three momentum-like
    moments. Learned Givens rotations act only inside that nullspace, exactly
    preserving those four linear moments and the quadratic state energy at
    every graph parcel.
    """

    def __init__(self, velocities=8, content_dim=8, hidden=64):
        super().__init__()
        if velocities != 8:
            raise ValueError("The first implementation uses the D3Q8 cube")
        self.velocities = velocities
        self.content_dim = content_dim
        self.d = velocities * content_dim
        velocity = _cube_velocities(dtype=torch.float64)
        constraints = torch.cat((torch.ones(1, velocities, dtype=torch.float64),
                                 velocity.T), 0)
        _, singular, right = torch.linalg.svd(constraints, full_matrices=True)
        rank = int((singular > 1e-10).sum())
        nullspace = right[rank:].T
        self.nullity = nullspace.shape[1]
        self.rotations = [(i, j) for i in range(self.nullity)
                          for j in range(i + 1, self.nullity)]
        self.register_buffer("velocities_q3", velocity, persistent=False)
        self.register_buffer("constraints", constraints, persistent=False)
        self.register_buffer("nullspace", nullspace, persistent=False)
        self.angle = nn.Sequential(
            nn.LayerNorm(self.d), nn.Linear(self.d, hidden), nn.SiLU(),
            nn.Linear(hidden, len(self.rotations)),
        )
        nn.init.normal_(self.angle[-1].weight, std=1e-3)
        nn.init.zeros_(self.angle[-1].bias)

    def forward(self, h):
        batch, nodes, _ = h.shape
        field = h.reshape(batch, nodes, self.velocities, self.content_dim)
        nullspace = self.nullspace.to(dtype=field.dtype)
        coefficient = torch.einsum("qk,bnqa->bnka", nullspace, field)
        conserved = field - torch.einsum(
            "qk,bnka->bnqa", nullspace, coefficient)
        angles = self.angle(h)
        values = list(coefficient.unbind(2))
        for index, (left_index, right_index) in enumerate(self.rotations):
            cosine = angles[..., index].cos()[..., None]
            sine = angles[..., index].sin()[..., None]
            left, right = values[left_index], values[right_index]
            values[left_index] = cosine * left - sine * right
            values[right_index] = sine * left + cosine * right
        rotated = torch.stack(values, 2)
        output = conserved + torch.einsum(
            "qk,bnka->bnqa", nullspace, rotated)
        return output.reshape_as(h), {
            "collision_angle_abs_mean": angles.detach().abs().mean(),
            "collision_angle_abs_max": angles.detach().abs().max(),
        }


class GraphAdaptiveDissipation(nn.Module):
    """Node-, state- and channel-local decay with analytic energy feedback."""

    def __init__(self, node_features, d=64, hidden=64, target_energy=.70,
                 feedback_sharpness=8.0, base_gamma=1.5e-4):
        super().__init__()
        self.target_energy = float(target_energy)
        self.feedback_sharpness = float(feedback_sharpness)
        self.base_gamma = float(base_gamma)
        self.norm = nn.LayerNorm(d)
        node_dim = node_features.shape[-1]
        self.node_rate = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.SiLU(), nn.Linear(hidden, d),
        )
        self.rate = nn.Sequential(
            nn.Linear(d + node_dim + 1, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, d, bias=False),
        )
        nn.init.normal_(self.node_rate[-1].weight, std=1e-3)
        nn.init.zeros_(self.node_rate[-1].bias)
        nn.init.normal_(self.rate[-1].weight, std=1e-3)
        self.local_rate = nn.Parameter(
            torch.zeros(node_features.shape[0], d, dtype=torch.float32))
        self.local_rate._no_weight_decay = True
        self.register_buffer(
            "node_features", node_features.float()[None], persistent=False)

    def forward(self, h):
        energy_before = .5 * h.square().sum(dim=-1, keepdim=True)
        relative_energy = energy_before / self.target_energy
        node_features = self.node_features.expand(h.shape[0], -1, -1)
        context = torch.cat(
            (self.norm(h), relative_energy, node_features), -1)
        raw = self.local_rate[None] + self.node_rate(
            self.node_features) + self.rate(context)
        learned_gamma = (
            self.base_gamma * F.softplus(raw) / math.log(2.0))
        learned_output = torch.exp(-learned_gamma) * h
        learned_energy = .5 * learned_output.square().sum(dim=-1, keepdim=True)
        log_ratio = (learned_energy / self.target_energy).clamp_min(1e-12).log()
        feedback_gamma = .5 / self.feedback_sharpness * F.softplus(
            self.feedback_sharpness * log_ratio)
        output = torch.exp(-feedback_gamma) * learned_output
        energy_after = .5 * output.square().sum(dim=-1, keepdim=True)
        total_gamma = learned_gamma + feedback_gamma
        return output, {
            "alpha_mean": torch.exp(-learned_gamma).detach().mean(),
            "gamma_mean": total_gamma.detach().mean(),
            "gamma_spatial_std": total_gamma.detach().mean(-1).std(unbiased=False),
            "gamma_channel_std": total_gamma.detach().std(dim=(-3, -2), unbiased=False).mean(),
            "gamma_min": total_gamma.detach().amin(),
            "gamma_max": total_gamma.detach().amax(),
            "gamma_node_mean": total_gamma.detach().mean(dim=(0, 2)),
            "learned_gamma_mean": learned_gamma.detach().mean(),
            "feedback_gamma_mean": feedback_gamma.detach().mean(),
            "feedback_active_rate": (
                learned_energy.detach() > self.target_energy).float().mean(),
            "dissipated_energy": (energy_before - energy_after).detach().mean(),
            "max_local_energy": energy_after.detach().amax(),
            "safety_projection_rate": h.new_zeros(()),
        }


class GraphKineticReadout(nn.Module):
    """State-valued multi-query readout over graph parcels."""

    def __init__(self, node_features, d=64, queries=4, heads=4):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d, self.queries, self.heads = d, queries, heads
        self.query = nn.Parameter(torch.randn(1, queries, d) * .02)
        self.position = nn.Sequential(
            nn.Linear(node_features.shape[-1], d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.merge = nn.Linear(queries * d, d)
        self.output = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("node_features", node_features.float()[None],
                             persistent=False)

    def position_encoding(self):
        return self.position(self.node_features)

    def forward(self, h, position_encoding=None):
        batch, _, d = h.shape
        state = self.norm(h)
        position = (self.position_encoding() if position_encoding is None
                    else position_encoding)
        keys = self.key(state + position)
        values = self.value(state)
        queries = self.query.expand(batch, -1, -1)

        def split(value):
            return value.reshape(batch, -1, self.heads, d // self.heads).transpose(1, 2)

        read = F.scaled_dot_product_attention(
            split(queries), split(keys), split(values), dropout_p=0.0)
        read = read.transpose(1, 2).reshape(batch, self.queries * d)
        return self.output(self.merge(read))


class CBIMMaleCNSKinetic(nn.Module):
    """MaleCNS CBIM with non-overlapping source/transport/collision/outflow."""

    architecture = "CBIM-MaleCNS-kinetic-local-gamma-v2.2"

    def __init__(self, graph_path, vocab_size=50257, velocities=8,
                 content_dim=8, queries=4, heads=4):
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path = str(graph_path)
        self.velocities = velocities
        self.content_dim = content_dim
        self.d = velocities * content_dim
        self.L = graph["coordinates"].shape[0]
        self.state_shape = (self.L, self.d)
        self.source = GraphBoundarySource(
            graph["coordinates"], vocab_size, velocities, content_dim)
        self.transport = GraphVelocityTransport(
            graph["laplacian_basis"], graph["laplacian_eigenvalues"],
            velocities, content_dim)
        self.collision = LocalVelocityCollision(velocities, content_dim)
        self.dissipation = GraphAdaptiveDissipation(
            graph["node_features"], self.d)
        self.readout = GraphKineticReadout(graph["node_features"], self.d,
                                           queries, heads)
        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.source.embedding.weight

    def evolve(self, h, token_ids, disable_collision=False):
        energy_before = .5 * h.square().sum(-1).mean()
        h, source = self.source(h, token_ids)
        h, transport = self.transport(h)
        if disable_collision:
            collision = {"collision_angle_abs_mean": h.new_zeros(()),
                         "collision_angle_abs_max": h.new_zeros(())}
        else:
            h, collision = self.collision(h)
        h, dissipation = self.dissipation(h)
        energy_after = .5 * h.square().sum(-1).mean()
        energy_balance_residual = (
            energy_after - energy_before - source["source_energy_delta"]
            + dissipation["dissipated_energy"])
        return h, {"source": source, "transport": transport,
                   "collision": collision, "dissipation": dissipation,
                   "energy_balance_residual": energy_balance_residual.detach()}

    def step(self, h, token_ids, disable_collision=False):
        h, diagnostics = self.evolve(h, token_ids, disable_collision)
        return self.decoder(self.readout(h)), h, diagnostics

    def forward(self, input_ids, targets, state=None, disable_collision=False):
        batch, tokens = input_ids.shape
        h = (torch.zeros(batch, *self.state_shape, device=input_ids.device)
             if state is None else state)
        features = []
        injection, source_energy, collision_angle = [], [], []
        alpha, gamma, gamma_std, gamma_channel_std = [], [], [], []
        gamma_min, gamma_max, gamma_node = [], [], []
        learned_gamma = []
        feedback_gamma, feedback_active, dissipated, balance = [], [], [], []
        position = self.readout.position_encoding()
        for index in range(tokens):
            h, diagnostics = self.evolve(
                h, input_ids[:, index], disable_collision)
            features.append(self.readout(h, position))
            injection.append(diagnostics["source"]["injection_norm"])
            source_energy.append(diagnostics["source"]["source_energy_delta"])
            collision_angle.append(
                diagnostics["collision"]["collision_angle_abs_mean"])
            alpha.append(diagnostics["dissipation"]["alpha_mean"])
            gamma.append(diagnostics["dissipation"]["gamma_mean"])
            gamma_std.append(diagnostics["dissipation"]["gamma_spatial_std"])
            gamma_channel_std.append(
                diagnostics["dissipation"]["gamma_channel_std"])
            gamma_min.append(diagnostics["dissipation"]["gamma_min"])
            gamma_max.append(diagnostics["dissipation"]["gamma_max"])
            gamma_node.append(diagnostics["dissipation"]["gamma_node_mean"])
            learned_gamma.append(
                diagnostics["dissipation"]["learned_gamma_mean"])
            feedback_gamma.append(
                diagnostics["dissipation"]["feedback_gamma_mean"])
            feedback_active.append(
                diagnostics["dissipation"]["feedback_active_rate"])
            dissipated.append(diagnostics["dissipation"]["dissipated_energy"])
            balance.append(diagnostics["energy_balance_residual"])
        logits = self.decoder(torch.stack(features, 1))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return loss, h, {
            "final_energy": .5 * h.square().sum(-1).mean().detach(),
            "injection_norm": torch.stack(injection).mean(),
            "source_energy_delta": torch.stack(source_energy).mean(),
            "collision_angle_abs_mean": torch.stack(collision_angle).mean(),
            "alpha_mean": torch.stack(alpha).mean(),
            "gamma_mean": torch.stack(gamma).mean(),
            "gamma_spatial_std": torch.stack(gamma_std).mean(),
            "gamma_channel_std": torch.stack(gamma_channel_std).mean(),
            "gamma_min": torch.stack(gamma_min).amin(),
            "gamma_max": torch.stack(gamma_max).amax(),
            "gamma_node_mean": torch.stack(gamma_node).mean(0),
            "learned_gamma_mean": torch.stack(learned_gamma).mean(),
            "feedback_gamma_mean": torch.stack(feedback_gamma).mean(),
            "feedback_active_rate": torch.stack(feedback_active).mean(),
            "dissipated_energy": torch.stack(dissipated).mean(),
            "energy_balance_residual": torch.stack(balance).abs().mean(),
            "safety_projection_rate": h.new_zeros(()),
        }
