"""State-addressed transition memory on the existing TableGDN.

This is the representation/recall part of navigation, not a navigation policy.
Reads use the SAME body's causal hidden state; writes contain observed successor
features, never imagined branches. A chunk reads an immutable table snapshot and
commits after backward. Its write latency is therefore at most one bounded chunk.

Persistent entries store source/successor *features*, re-encoded with the current
shared linear encoder on every read. This avoids mixing old/new projector charts;
drift of the underlying GDN feature representation remains an empirical question.
The first implementation uses bounded FIFO replacement, not unproved utility or
eligibility-based eviction. Byte memory and its trained readout are preserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def sigreg(z: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Projected characteristic-function discrepancy from N(0,1).

    Follows the Epps--Pulley quadrature in LeJEPA's official MINIMAL.md.
    z is [samples, latent_dim], directions [latent_dim, projections]. Samples
    here are actual observed states, not forced-diverse imaginary outcomes.
    Correlated stream samples do not inherit an IID statistical guarantee.
    """
    if z.ndim != 2 or z.shape[0] < 2 or directions.shape[0] != z.shape[1]:
        raise ValueError('SIGReg needs at least two observed samples and matching projections')
    t = torch.linspace(0, 3, 17, device=z.device, dtype=z.dtype)
    phi = torch.exp(-t.square() / 2)
    quadrature = torch.ones_like(t) * (6 / 16)
    quadrature[0] /= 2
    quadrature[-1] /= 2
    phase = (z @ F.normalize(directions, dim=0)).unsqueeze(-1) * t
    error = (phase.cos().mean(0) - phi).square() + phase.sin().mean(0).square()
    return (error @ (quadrature * phi)).mean() * z.shape[0]


@dataclass
class LatentRead:
    value: torch.Tensor
    slots: torch.Tensor
    weights: torch.Tensor


class LatentTransitionTable(nn.Module):
    """Fixed-capacity observed transitions plus a bounded pending horizon.

    No global decay, no history-sized graph, no semantic labels. Replacement is
    a local one-hot write with beta=1 (existing tied-write Lean instance).
    Counts and a reward account are not required for this initial carrier.
    """
    def __init__(self, width: int, capacity: int = 4096, horizon: int = 32,
                 topk: int = 8, temperature: float = .2):
        super().__init__()
        if min(width, capacity, horizon, topk) <= 0 or topk > capacity or temperature <= 0:
            raise ValueError('invalid finite transition-table dimensions')
        self.width, self.capacity, self.horizon = width, capacity, horizon
        self.topk, self.temperature = topk, temperature
        self.register_buffer('source', torch.zeros(capacity, width))
        self.register_buffer('successor', torch.zeros(capacity, width))
        self.register_buffer('occupied', torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer('cursor', torch.zeros((), dtype=torch.long))
        self.register_buffer('pending', torch.zeros(horizon, width))
        # Python count is bounded and serialized through extra_state.
        self.pending_count = 0

    def get_extra_state(self):
        return dict(width=self.width, capacity=self.capacity, horizon=self.horizon,
                    topk=self.topk, temperature=self.temperature,
                    pending_count=self.pending_count)

    def set_extra_state(self, state):
        for key in ('width', 'capacity', 'horizon', 'topk', 'temperature'):
            if state[key] != getattr(self, key):
                raise ValueError(f'transition-table restore mismatch: {key}')
        if not 0 <= state['pending_count'] <= self.horizon:
            raise ValueError('invalid pending horizon')
        self.pending_count = state['pending_count']

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.buffers())

    def read(self, features: torch.Tensor, encoder: nn.Module) -> LatentRead:
        q = encoder(features)
        keys = encoder(self.source)
        values = encoder(self.successor)
        score = F.normalize(q, dim=-1) @ F.normalize(keys, dim=-1).T
        score = score.masked_fill(~self.occupied, -1e4)
        selected, slots = score.topk(self.topk, dim=-1)
        valid = self.occupied[slots]
        weights = (selected / self.temperature).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        value = (weights.unsqueeze(-1) * values[slots]).sum(-2)
        return LatentRead(value, slots, weights)

    def pairs(self, features: torch.Tensor):
        if features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError('one continuous individual: features must be [time,width]')
        # clone: callers may commit after backward without retaining mutable aliases.
        joined = torch.cat((self.pending[:self.pending_count].clone(), features), 0)
        if len(joined) <= self.horizon:
            return joined[:0], joined[:0]
        return joined[:-self.horizon], joined[self.horizon:]

    @torch.no_grad()
    def append_pairs(self, source: torch.Tensor, successor: torch.Tensor):
        if source.shape != successor.shape or source.ndim != 2 or source.shape[1] != self.width:
            raise ValueError('paired source/successor feature shapes must agree')
        total = len(source)
        retained = min(total, self.capacity)
        if retained:
            # Keep last capacity writes; no duplicate scatter destinations.
            offset = torch.arange(total-retained, total, device=source.device)
            slots = (self.cursor + offset) % self.capacity
            self.source.index_copy_(0, slots, source[-retained:].detach())
            self.successor.index_copy_(0, slots, successor[-retained:].detach())
            self.occupied.index_fill_(0, slots, True)
        self.cursor.copy_((self.cursor + total) % self.capacity)

    @torch.no_grad()
    def commit(self, features: torch.Tensor):
        """Call only after scoring/backward on the observed chunk, once per chunk."""
        source, successor = self.pairs(features.detach())
        self.append_pairs(source, successor)
        joined = torch.cat((self.pending[:self.pending_count], features.detach()), 0)
        tail = joined[-self.horizon:].clone()
        self.pending.zero_()
        self.pending[:len(tail)].copy_(tail)
        self.pending_count = len(tail)


class LatentNavigationBody(nn.Module):
    """Small optional continuation of TableGDN, including its language readout.

    Zero residual gain exactly restores original logits and recurrent transition.
    Both JEPA encoder branches receive gradients within the declared window.
    The long-lived table stores detached features, not graphs or exact old credit.
    """
    def __init__(self, body: nn.Module, latent_dim: int = 64, capacity: int = 4096,
                 horizon: int = 32, topk: int = 8, seed: int = 17, max_chunk: int = 512):
        super().__init__()
        self.body = body
        if max_chunk < 2:
            raise ValueError('bounded training chunk must contain at least two observations')
        self.max_chunk = max_chunk
        width = body.base.d
        # Do not consume the continuing individual's RNG during extension.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = nn.Linear(width, latent_dim, bias=False)
            self.predictor = nn.Sequential(nn.Linear(2*latent_dim, latent_dim),
                                           nn.SiLU(), nn.Linear(latent_dim, latent_dim))
            self.to_hidden = nn.Linear(latent_dim, width, bias=False)
        self.read_gain = nn.Parameter(torch.zeros(()))
        self.memory = LatentTransitionTable(width, capacity, horizon, topk)
        self.to(device=next(body.parameters()).device, dtype=next(body.parameters()).dtype)

    @property
    def base(self):
        return self.body.base

    def initial_state(self, batch, device=None):
        return self.body.initial_state(batch, device)

    def predict_observed(self, ids, state, table_logp):
        if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= self.max_chunk:
            raise ValueError('one continuous individual per latent table')
        features, reads = [], []
        # At the existing table injection boundary, use the actual causal state
        # entering this GDN layer. Feed the recall into that SAME layer, so its
        # recurrent state, later predictions and eventual actions can change.
        # Previous chunks alone populate memory; it is immutable during the call.
        def inject(module, args):
            hidden = args[0]
            h = hidden[0]
            read = self.memory.read(h, self.encoder)
            residual = self.to_hidden(read.value) * self.read_gain.tanh()
            features.append(h)
            reads.append(read)
            return (hidden + residual[None], *args[1:])

        handle = self.base.layers[self.body.inject_layer].register_forward_pre_hook(inject)
        try:
            logits, next_state, diagnostics = self.body.predict(ids, state, table_logp)
        finally:
            handle.remove()
        h = features[0]
        return logits, next_state, diagnostics, h, reads[0]

    def predict(self, ids, state, table_logp):
        logits, next_state, diagnostics, _, _ = self.predict_observed(ids, state, table_logp)
        return logits, next_state, diagnostics

    def objective(self, ids, targets, state, table_logp, directions,
                  prediction_weight=.1, regularization_weight=.01):
        logits, next_state, diagnostics, h, read = self.predict_observed(ids, state, table_logp)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        # A pending source from the previous chunk cannot be trained by reading
        # today's table snapshot: that snapshot already contains observations
        # later than that source. The pending queue still commits correctly
        # aligned observed pairs; the loss uses only current-chunk sources.
        source, successor = h[:-self.memory.horizon], h[self.memory.horizon:]
        if len(source):
            recalled = self.memory.read(source, self.encoder).value
            prediction = self.predictor(torch.cat((self.encoder(source), recalled), -1))
            # LeWM has no stop-gradient on observed targets inside this window.
            latent_loss = F.mse_loss(prediction, self.encoder(successor))
        else:
            latent_loss = h.sum() * 0
        gaussian = sigreg(self.encoder(h), directions)
        loss = ce + prediction_weight*latent_loss + regularization_weight*gaussian
        return loss, next_state, h, dict(ce=ce.detach(), latent=latent_loss.detach(),
                                        sigreg=gaussian.detach(), **diagnostics)

    @torch.no_grad()
    def commit_observed(self, features):
        self.memory.commit(features)

    def predict_successor(self, causal_features):
        """For a state AFTER self-hearing an utterance, before its response.

        This provides an action-conditioned prediction interface. Its accuracy
        under dialogue interventions must still be learned and evaluated.
        """
        read = self.memory.read(causal_features, self.encoder)
        return self.predictor(torch.cat((self.encoder(causal_features), read.value), -1))


class LatentGraphTrainer:
    """Capture a fixed window; write observed transitions only after backward.

    GPU inputs, projections, recurrent state and fixed memory live at stable
    addresses across replays. Random directions are supplied by the caller and
    the full table changes between replays; nothing from a future chunk is read.
    The optimizer may already contain restored moments of the base individual.
    """
    def __init__(self, model, optimizer, tokens=512, projections=32, capture=True,
                 prediction_weight=.1, regularization_weight=.01):
        if tokens < 2 or tokens > model.max_chunk or projections < 1:
            raise ValueError('invalid bounded window or projection count')
        self.model, self.optimizer, self.tokens = model, optimizer, tokens
        device = next(model.parameters()).device
        self.state = model.initial_state(1, device)
        self.ids = torch.zeros(1, tokens, device=device, dtype=torch.long)
        self.targets = torch.zeros_like(self.ids)
        self.table_logp = torch.full((1, tokens, 256), -5.54517744448, device=device)
        self.directions = torch.ones(model.encoder.out_features, projections, device=device)
        self.prediction_weight = prediction_weight
        self.regularization_weight = regularization_weight
        self.capture = capture
        if capture:
            if device.type != 'cuda':
                raise ValueError('CUDA graph requires CUDA')
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=False)
                    self._backward()
            torch.cuda.current_stream().wait_stream(stream)
            optimizer.zero_grad(set_to_none=False)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loss, self.next_state, self.features, self.diagnostics = self._backward()
            optimizer.zero_grad(set_to_none=False)

    def _backward(self):
        loss, state, features, diagnostics = self.model.objective(
            self.ids, self.targets, self.state, self.table_logp, self.directions,
            self.prediction_weight, self.regularization_weight)
        loss.backward()
        return loss, state, features, diagnostics

    def step(self, ids, targets, table_logp, directions):
        self.optimizer.zero_grad(set_to_none=False)
        with torch.no_grad():
            self.ids.copy_(ids)
            self.targets.copy_(targets)
            self.table_logp.copy_(table_logp)
            self.directions.copy_(directions)
        if self.capture:
            self.graph.replay()
        else:
            self.loss, self.next_state, self.features, self.diagnostics = self._backward()
        if not torch.isfinite(self.loss):
            raise FloatingPointError('nonfinite latent objective')
        # Same original lexical/other clipping, plus the additional encoder
        # group. Group definitions live with the training protocol.
        return self.loss.detach(), self.next_state.detach(), self.features.detach(), self.diagnostics

    @torch.no_grad()
    def finish(self):
        # Call after gradient checks/clipping, before optimizer.step().
        self.model.commit_observed(self.features)
        self.state.copy_(self.next_state.detach())
