"""Information-Boltzmann field on a coarsened MaleCNS connectome.

The connectome supplies geometry, graph-Laplacian modes and collision edges.
Token learning still acts through an open-system source/outflow, conservative
transport and learned state-aware pair scattering.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_malecns_graph(path):
    with np.load(Path(path)) as archive:
        return {name: torch.from_numpy(archive[name]) for name in archive.files}


class GraphSpectralTransport(nn.Module):
    """Orthogonal transport through paired graph-Laplacian modes."""

    def __init__(self, basis, eigenvalues, d=64, hidden=64):
        super().__init__()
        if basis.shape[1] % 2:
            raise ValueError("An even number of graph modes is required")
        # QR removes float serialization error and makes the norm invariant exact
        # for the stored truncation.
        basis = torch.linalg.qr(basis.float(), mode="reduced").Q
        values = eigenvalues.float()
        mode_input = torch.stack((values[0::2], values[1::2],
                                  values[1::2] - values[0::2],
                                  .5 * (values[0::2] + values[1::2])), -1)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("mode_input", mode_input, persistent=False)
        self.symbol = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, d))
        nn.init.normal_(self.symbol[-1].weight, std=1e-3)
        nn.init.zeros_(self.symbol[-1].bias)

    def angles(self):
        return self.symbol(self.mode_input)

    def forward(self, h):
        coefficients = torch.einsum("mr,bmd->brd", self.basis, h)
        theta = self.angles()[None]
        cosine, sine = theta.cos(), theta.sin()
        left, right = coefficients[:, 0::2], coefficients[:, 1::2]
        rotated = torch.stack((cosine * left - sine * right,
                               sine * left + cosine * right), 2).flatten(1, 2)
        output = h + torch.einsum(
            "mr,brd->bmd", self.basis, rotated - coefficients)
        return output, {
            "transport_angle_abs_mean": theta.detach().abs().mean(),
            "transport_angle_abs_max": theta.detach().abs().max(),
        }


class GraphLocalSourceOutflow(nn.Module):
    """State-aware token exchange localized in anatomical 3D coordinates."""

    def __init__(self, graph, vocab_size=50257, d=64, cell_radius=1.25,
                 max_write_rate=.25, max_outflow_rate=.05):
        super().__init__()
        coordinates = graph["coordinates"].float()
        self.nodes, self.d = coordinates.shape[0], d
        self.cell_radius = float(cell_radius)
        self.max_write_rate = float(max_write_rate)
        self.max_outflow_rate = float(max_outflow_rate)
        self.embedding = nn.Embedding(vocab_size, d)
        self.content = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.address = nn.Linear(d, 3)
        self.width = nn.Linear(d, 3)
        self.token_context = nn.Linear(d, d)
        self.state_context = nn.Linear(d, d)
        self.neighbor_context = nn.Linear(d, d)
        self.content_correction = nn.Linear(d, d)
        self.context_write_rate = nn.Linear(d, 1)
        self.outflow_rate = nn.Linear(d, 1)
        for layer in (self.content_correction, self.context_write_rate,
                      self.outflow_rate):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.outflow_rate.bias, -4.)
        self.register_buffer("coordinates", coordinates[None], persistent=False)
        self.register_buffer("neighbor_indices",
                             graph["neighbor_indices"].long(), persistent=False)
        self.register_buffer("neighbor_weights",
                             graph["neighbor_weights"].float(), persistent=False)

    def forward(self, h, token_ids):
        emb = self.embedding(token_ids)
        center = torch.sigmoid(self.address(emb))[:, None, :]
        sigma = (.04 + .21 * torch.sigmoid(self.width(emb)))[:, None, :]
        distance = (self.coordinates - center) / sigma
        envelope = torch.exp(-.5 * distance.square().sum(-1))
        neighbors = h[:, self.neighbor_indices]
        neighborhood = (neighbors * self.neighbor_weights[None, ..., None]).sum(2)
        context = F.silu(self.token_context(emb)[:, None] +
                         self.state_context(h) +
                         self.neighbor_context(neighborhood))
        outflow = self.max_outflow_rate * torch.sigmoid(self.outflow_rate(context))
        retained = (1 - outflow) * h
        rate = envelope[..., None] * self.max_write_rate * torch.sigmoid(
            self.context_write_rate(context))
        proposal = (1 - rate) * retained + rate * (
            self.content(emb)[:, None] + self.content_correction(context))
        norm = proposal.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.cell_radius / norm.clamp_min(1e-8), max=1.)
        result = proposal * scale
        return result, {
            "write_rate_mean": rate.detach().mean(),
            "outflow_rate_mean": outflow.detach().mean(),
            "local_projection_rate": (scale.detach() < 1).float().mean(),
        }


class GraphConservativeScattering(nn.Module):
    """Learned nonlinear scattering on disjoint MaleCNS edge matchings."""

    def __init__(self, graph, d=64):
        super().__init__()
        if d % 2:
            raise ValueError("Even feature dimension required")
        pairs = graph["collision_pairs"].long()
        features = graph["collision_features"].float()
        self.d, self.layers = d, pairs.shape[0]
        self.register_buffer("pairs", pairs, persistent=False)
        self.register_buffer("edge_features", features, persistent=False)
        self.angle = nn.Sequential(
            nn.Linear(2 * d + features.shape[-1], d), nn.SiLU(),
            nn.Linear(d, d // 2))
        nn.init.normal_(self.angle[-1].weight, std=1e-3)
        nn.init.zeros_(self.angle[-1].bias)

    def forward(self, h):
        result = h
        for layer in range(self.layers):
            left_index, right_index = self.pairs[layer].unbind(-1)
            shifted = torch.roll(result, -layer, -1)
            left, right = shifted[:, left_index], shifted[:, right_index]
            edge = self.edge_features[layer][None].expand(h.shape[0], -1, -1)
            theta = self.angle(torch.cat((left, right, edge), -1))
            mean = (left + right) / math.sqrt(2)
            difference = (left - right).reshape(
                *left.shape[:-1], self.d // 2, 2) / math.sqrt(2)
            cosine, sine = theta.cos(), theta.sin()
            rotated = torch.stack((cosine * difference[..., 0] -
                                   sine * difference[..., 1],
                                   sine * difference[..., 0] +
                                   cosine * difference[..., 1]), -1).flatten(-2)
            left_new = (mean + rotated) / math.sqrt(2)
            right_new = (mean - rotated) / math.sqrt(2)
            updated = shifted.clone()
            updated[:, left_index] = left_new
            updated[:, right_index] = right_new
            result = torch.roll(updated, layer, -1)
        return result, {"scattering_layers_applied": self.layers}


class GraphOperatorReadout(nn.Module):
    """Multi-query state-only readout over anatomical parcels."""

    def __init__(self, graph, d=64, queries=4, heads=4):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        features = graph["node_features"].float()
        self.d, self.queries, self.heads = d, queries, heads
        self.query = nn.Parameter(torch.randn(1, queries, d) * .02)
        self.position = nn.Sequential(
            nn.Linear(features.shape[-1], d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.merge = nn.Linear(queries * d, d)
        self.output = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("node_features", features[None], persistent=False)

    def position_encoding(self):
        return self.position(self.node_features)

    def forward(self, h, position_encoding=None):
        batch, nodes, d = h.shape
        state = self.norm(h)
        encoding = self.position_encoding() if position_encoding is None else position_encoding
        keys = self.key(state + encoding)
        values = self.value(state)
        queries = self.query.expand(batch, -1, -1)
        split = lambda x: x.reshape(batch, -1, self.heads, d // self.heads).transpose(1, 2)
        read = F.scaled_dot_product_attention(
            split(queries), split(keys), split(values), dropout_p=0.)
        read = read.transpose(1, 2).reshape(batch, self.queries * d)
        return self.output(self.merge(read))


class CBIMMaleCNS(nn.Module):
    """Open-system CBIM whose internal space is the MaleCNS parcel graph."""

    def __init__(self, graph_path, vocab_size=50257, d=64, queries=4, heads=4):
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path = str(graph_path)
        self.L, self.d = graph["coordinates"].shape[0], d
        self.state_shape = (self.L, d)
        self.source = GraphLocalSourceOutflow(graph, vocab_size, d)
        self.transport = GraphSpectralTransport(
            graph["laplacian_basis"], graph["laplacian_eigenvalues"], d)
        self.scattering = GraphConservativeScattering(graph, d)
        self.readout = GraphOperatorReadout(graph, d, queries, heads)
        self.decoder = nn.Linear(d, vocab_size)
        self.decoder.weight = self.source.embedding.weight

    def step(self, h, token_ids, disable_scattering=False):
        h, source_diag = self.source(h, token_ids)
        h, transport_diag = self.transport(h)
        if disable_scattering:
            scattering_diag = {"scattering_layers_applied": 0}
        else:
            h, scattering_diag = self.scattering(h)
        logits = self.decoder(self.readout(h))
        return logits, h, {"source": source_diag, "transport": transport_diag,
                           "scattering": scattering_diag}

    def forward(self, input_ids, targets, state=None, disable_scattering=False):
        batch, tokens = input_ids.shape
        h = (torch.zeros(batch, *self.state_shape, device=input_ids.device)
             if state is None else state)
        features, projections = [], []
        position = self.readout.position_encoding()
        for index in range(tokens):
            h, source_diag = self.source(h, input_ids[:, index])
            h, _ = self.transport(h)
            if not disable_scattering:
                h, _ = self.scattering(h)
            features.append(self.readout(h, position))
            projections.append(source_diag["local_projection_rate"])
        logits = self.decoder(torch.stack(features, 1))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return loss, h, {
            "final_energy": .5 * h.square().sum(-1).mean().detach(),
            "local_projection_rate": torch.stack(projections).mean(),
        }
