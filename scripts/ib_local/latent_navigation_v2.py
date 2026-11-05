"""Finite two-horizon transition memory for the existing language individual.

Only observed states are committed, after the scored window. Approximate merge
is an empirical compression hypothesis, not a theorem of semantic equivalence.
All counters, timestamps, pending features and replay samples have fixed size.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from scripts.ib_local.latent_navigation import LatentGraphTrainer, LatentNavigationBody


def weighted_sigreg(current, historical, valid, directions):
    """LeJEPA quadrature over current observations and a masked fixed buffer."""
    z = torch.cat((current, historical.detach()), 0)
    weight = torch.cat((torch.ones(len(current), device=z.device, dtype=z.dtype), valid.to(z.dtype)))
    count = weight.sum().clamp_min(1)
    t = torch.linspace(0, 3, 17, device=z.device, dtype=z.dtype)
    phi = torch.exp(-t.square()/2)
    quadrature = torch.ones_like(t) * (6/16)
    quadrature[0] /= 2; quadrature[-1] /= 2
    phase = (z @ F.normalize(directions, dim=0)).unsqueeze(-1) * t
    cosine = (phase.cos()*weight[:, None, None]).sum(0)/count
    sine = (phase.sin()*weight[:, None, None]).sum(0)/count
    return (((cosine-phi).square()+sine.square()) @ (quadrature*phi)).mean()*count


@dataclass
class TransitionRead:
    value: torch.Tensor
    long_value: torch.Tensor
    slots: torch.Tensor
    weights: torch.Tensor


class TransitionMemory(nn.Module):
    """Shared-address short/long successors, snapshot reads and batched writes.

    Merging compares both source and both successors in the CURRENT encoder
    geometry. A batch matches the prior snapshot; brand-new entries created in
    the same commit are not merged together. This bounded lag is explicit.
    Read age is tracked separately from write/merge age. Each commit evicts the
    least recently read slots, protecting slots merged in the same transaction.
    """
    def __init__(self, width, capacity=4096, short=32, long=512, topk=8,
                 threshold=.98, temperature=.2, mode='merge', history_size=256):
        super().__init__()
        if not 0 < short < long or capacity < topk or mode not in ('merge', 'fifo'):
            raise ValueError('invalid bounded memory configuration')
        if not -1 <= threshold <= 1 or temperature <= 0 or history_size < 2:
            raise ValueError('invalid merge or sample settings')
        self.width, self.capacity, self.short, self.long = width, capacity, short, long
        self.topk, self.threshold, self.temperature, self.mode = topk, threshold, temperature, mode
        self.history_size = history_size
        for name in ('source', 'successor', 'long_successor'):
            self.register_buffer(name, torch.zeros(capacity, width))
        for name in ('count', 'birth', 'last_write', 'last_touch'):
            self.register_buffer(name, torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('occupied', torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer('cursor', torch.zeros((), dtype=torch.long))
        self.register_buffer('clock', torch.zeros((), dtype=torch.long))
        self.register_buffer('pending', torch.zeros(long, width))
        self.register_buffer('pending_view', torch.zeros(long, width))
        self.register_buffer('pending_valid', torch.zeros(long, dtype=torch.bool))
        self.register_buffer('history', torch.zeros(history_size, width))
        self.register_buffer('history_clock', torch.zeros(history_size, dtype=torch.long))
        self.register_buffer('history_valid', torch.zeros(history_size, dtype=torch.bool))
        self.register_buffer('merged_total', torch.zeros((), dtype=torch.long))
        self.register_buffer('admitted_total', torch.zeros((), dtype=torch.long))
        self.register_buffer('dropped_total', torch.zeros((), dtype=torch.long))
        self.register_buffer('evicted_total', torch.zeros((), dtype=torch.long))
        self.pending_count = 0
        self.history_count = 0
        self.history_cursor = 0

    def get_extra_state(self):
        return dict(width=self.width, capacity=self.capacity, short=self.short, long=self.long,
                    topk=self.topk, threshold=self.threshold, temperature=self.temperature,
                    mode=self.mode, history_size=self.history_size, pending_count=self.pending_count,
                    history_count=self.history_count, history_cursor=self.history_cursor)

    def set_extra_state(self, state):
        for name in ('width', 'capacity', 'short', 'long', 'topk', 'threshold',
                     'temperature', 'mode', 'history_size'):
            if state[name] != getattr(self, name):
                raise ValueError(f'incompatible memory restore: {name}')
        if not (0 <= state['pending_count'] <= self.long and
                0 <= state['history_count'] <= self.history_size and
                0 <= state['history_cursor'] < self.history_size):
            raise ValueError('invalid bounded memory metadata')
        self.pending_count = state['pending_count']
        self.history_count = state['history_count']
        self.history_cursor = state['history_cursor']

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.buffers())

    def read(self, features, encoder):
        q = F.normalize(encoder(features), dim=-1)
        k = F.normalize(encoder(self.source), dim=-1)
        score = (q @ k.T).masked_fill(~self.occupied, -1e4)
        selected, slots = score.topk(self.topk, dim=-1)
        valid = self.occupied[slots]
        weights = (selected / self.temperature).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        short = (weights[..., None] * encoder(self.successor)[slots]).sum(-2)
        long = (weights[..., None] * encoder(self.long_successor)[slots]).sum(-2)
        return TransitionRead(short, long, slots, weights)

    def pairs(self, features):
        if features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError('features must be [time,width] for one individual')
        previous = self.pending[-self.pending_count:].clone() if self.pending_count else self.pending[:0]
        joined = torch.cat((previous, features), 0)
        return joined[:-self.long], joined[self.short:-self.long+self.short], joined[self.long:]

    @torch.no_grad()
    def ages(self, read, length):
        """Weighted youngest/latest and oldest contribution ages at each read."""
        event = self.clock + torch.arange(length, device=self.clock.device)
        old = (event[:, None] - self.birth[read.slots]).clamp_min(0)
        recent = (event[:, None] - self.last_write[read.slots]).clamp_min(0)
        return ((read.weights * old).sum(-1), (read.weights * recent).sum(-1))

    @torch.no_grad()
    def commit(self, features, encoder, read=None):
        if not 0 < features.shape[0] <= min(self.capacity, self.long):
            raise ValueError('each atomic chunk must fit the declared capacity and horizon')
        # Capture read-time values BEFORE any merge or eviction mutates slots.
        if read is None:
            current_view = torch.zeros_like(features)
        else:
            current_view = (read.weights[..., None] * self.long_successor[read.slots]).sum(-2)
        current_view = current_view.detach().clone()
        source, short, long = self.pairs(features.detach())
        length = len(source)
        next_clock = self.clock + len(features)
        if int(next_clock) < int(self.clock):
            raise OverflowError('event clock exhausted')
        if read is not None:
            touched = read.slots[read.weights > 0]
            if touched.numel():
                self.last_touch.index_fill_(0, touched.unique(), next_clock)

        if length:
            # Pair/source timestamps refer to observed source events, not commit.
            stamps = self.clock - self.pending_count + torch.arange(length, device=source.device)
            if self.mode == 'merge':
                q = F.normalize(encoder(source), dim=-1)
                old = F.normalize(encoder(self.source), dim=-1)
                sk = F.normalize(encoder(short), dim=-1) @ F.normalize(encoder(self.successor), dim=-1).T
                lk = F.normalize(encoder(long), dim=-1) @ F.normalize(encoder(self.long_successor), dim=-1).T
                # Search jointly: the closest source can have a conflicting
                # successor while another source-close entry matches both.
                joint = torch.minimum(q @ old.T, torch.minimum(sk, lk)).masked_fill(~self.occupied, -2)
                match_score, nearest = joint.max(-1)
                matched = (match_score >= self.threshold) & self.occupied[nearest]
            else:
                nearest = torch.zeros(length, device=source.device, dtype=torch.long)
                matched = torch.zeros(length, device=source.device, dtype=torch.bool)
            slots = nearest[matched]
            self.merged_total.add_(len(slots))
            # A CUDA index_add over repeated slots uses floating atomic updates
            # whose order can change after resume. Sort stably and segment-reduce
            # each group in a fixed input order instead.
            permutation = torch.argsort(slots, stable=True)
            sorted_slots = slots[permutation]
            merged_slots, group_size = torch.unique_consecutive(sorted_slots, return_counts=True)
            delta = torch.zeros(self.capacity, device=source.device, dtype=torch.long)
            delta[merged_slots] = group_size
            if merged_slots.numel():
                old_count = self.count[merged_slots].to(source.dtype)
                new_count = group_size.to(source.dtype)
                for field, incoming in (('source', source), ('successor', short), ('long_successor', long)):
                    accumulated = torch.segment_reduce(incoming[matched][permutation], 'sum', lengths=group_size)
                    values = ((getattr(self, field)[merged_slots] * old_count[:, None] + accumulated) /
                              (old_count + new_count)[:, None])
                    getattr(self, field).index_copy_(0, merged_slots, values)
                self.count[merged_slots] += delta[merged_slots]
                self.last_write[merged_slots] = next_clock

            fresh = torch.nonzero(~matched, as_tuple=False).flatten()
            if self.mode == 'fifo':
                destinations = (self.cursor + torch.arange(len(fresh), device=source.device)) % self.capacity
                self.cursor.copy_((self.cursor + len(fresh)) % self.capacity)
            else:
                protected = delta > 0
                eligible = torch.nonzero(~protected, as_tuple=False).flatten()
                priority = torch.where(self.occupied[eligible], self.last_touch[eligible], -1)
                order = eligible[torch.argsort(priority, stable=True)]
                # If every entry was touched by a merge, some admissions are
                # dropped, rather than silently overwriting the same transaction.
                fresh = fresh[:len(order)]
                destinations = order[:len(fresh)]
            if destinations.numel():
                self.evicted_total.add_(self.occupied[destinations].sum())
                self.admitted_total.add_(len(destinations))
                for field, incoming in (('source', source), ('successor', short), ('long_successor', long)):
                    getattr(self, field).index_copy_(0, destinations, incoming[fresh])
                self.occupied[destinations] = True
                self.count[destinations] = 1
                self.birth[destinations] = stamps[fresh]
                self.last_write[destinations] = next_clock
                self.last_touch[destinations] = next_clock
            self.dropped_total.add_(length-len(slots)-len(destinations))

        previous = self.pending[-self.pending_count:] if self.pending_count else self.pending[:0]
        previous_view = self.pending_view[-self.pending_count:] if self.pending_count else self.pending_view[:0]
        joined = torch.cat((previous, features.detach()), 0)
        joined_view = torch.cat((previous_view, current_view), 0)
        tail = joined[-self.long:].clone()
        tail_view = joined_view[-self.long:].clone()
        self.pending.zero_(); self.pending[-len(tail):].copy_(tail)
        self.pending_view.zero_(); self.pending_view[-len(tail):].copy_(tail_view)
        self.pending_valid.zero_(); self.pending_valid[-len(tail):] = True
        self.pending_count = len(tail)
        # One actual observed feature per commit; no history-sized buffer.
        if len(features):
            self.history[self.history_cursor].copy_(features[-1])
            self.history_clock[self.history_cursor] = next_clock
            self.history_valid[self.history_cursor] = True
            self.history_cursor = (self.history_cursor + 1) % self.history_size
            self.history_count = min(self.history_count + 1, self.history_size)
        self.clock.copy_(next_clock)


class LatentNavigationV2(LatentNavigationBody):
    """V1 body with two observed horizons and bounded state-addressed memory."""
    def __init__(self, body, latent_dim=64, capacity=4096, short=32, long=512,
                 topk=8, seed=17, max_chunk=512, threshold=.98, mode='merge', history_size=256):
        if max_chunk > long or max_chunk > capacity:
            raise ValueError('chunk must fit both finite capacity and long horizon')
        # V1 allocates its temporary memory; it is replaced immediately before use.
        super().__init__(body, latent_dim, capacity, short, topk, seed, max_chunk)
        self.memory = TransitionMemory(body.base.d, capacity, short, long, topk,
                                       threshold=threshold, mode=mode, history_size=history_size)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+1)
            self.long_predictor = nn.Sequential(nn.Linear(2*latent_dim, latent_dim), nn.SiLU(),
                                                nn.Linear(latent_dim, latent_dim))
        self.to(device=next(body.parameters()).device, dtype=next(body.parameters()).dtype)

    def objective(self, ids, targets, state, table_logp, directions,
                  prediction_weight=.1, regularization_weight=.01, score_mask=None):
        logits, next_state, diagnostics, h, read = self.predict_observed(ids, state, table_logp)
        token_ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none')
        if score_mask is None:
            ce = token_ce.mean()
        else:
            mask = score_mask.reshape(-1).to(token_ce.dtype)
            ce = (token_ce*mask).sum()/mask.sum().clamp_min(1)
        short = self.memory.short
        if len(h) > short:
            source = h[:-short]
            recall = self.memory.read(source, self.encoder)
            pred = self.predictor(torch.cat((self.encoder(source), recall.value), -1))
            short_loss = F.mse_loss(pred, self.encoder(h[short:]))
        else:
            short_loss = h.sum()*0
        # Delayed targets read the bounded source and actual table view saved at
        # source time. Never re-query past sources with the newer table. Values
        # are re-encoded now; old routing derivatives are not reconstructed.
        prior = self.memory.pending[:len(h)]
        prior_view = self.memory.pending_view[:len(h)]
        valid = self.memory.pending_valid[:len(h)].to(h.dtype)
        long_pred = self.long_predictor(torch.cat((self.encoder(prior), self.encoder(prior_view)), -1))
        long_error = (long_pred-self.encoder(h)).square().mean(-1)
        long_loss = (long_error*valid).sum()/valid.sum().clamp_min(1)
        observed = self.encoder(h)
        old = self.encoder(self.memory.history).detach()
        regularizer = weighted_sigreg(observed, old, self.memory.history_valid, directions)
        loss = ce + prediction_weight*(short_loss+long_loss) + regularization_weight*regularizer
        return loss, next_state, h, dict(ce=ce.detach(), latent=short_loss.detach(),
            latent_long=long_loss.detach(), sigreg=regularizer.detach(),
            read_slots=read.slots.detach(), read_weights=read.weights.detach(),
            logits=logits, **diagnostics)

    @torch.no_grad()
    def commit_observed(self, features, read=None):
        self.memory.commit(features, self.encoder, read)

    def predict_successors(self, features):
        recall = self.memory.read(features, self.encoder)
        z = self.encoder(features)
        return (self.predictor(torch.cat((z, recall.value), -1)),
                self.long_predictor(torch.cat((z, recall.long_value), -1)))


class V2GraphTrainer(LatentGraphTrainer):
    @torch.no_grad()
    def finish(self):
        receipt = TransitionRead(None, None, self.diagnostics['read_slots'],
                                 self.diagnostics['read_weights'])
        self.model.commit_observed(self.features, receipt)
        self.state.copy_(self.next_state.detach())
