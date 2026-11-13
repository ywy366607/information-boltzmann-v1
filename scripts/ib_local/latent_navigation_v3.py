"""Bounded action-experience storage for a continuous language body.

Callers supply observed body features at the source, after self-hearing the
action, and at both successor horizons. The v4 loop commits these episodes.
Measured progress remains an observation; read-time credit is settled from
actual prediction derivatives without changing that observation's identity.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from scripts.ib_local.latent_navigation_v2 import LatentNavigationV2, TransitionMemory, TransitionRead


class ExperienceMemory(TransitionMemory):
    """One finite shared table; byte events update context, episodes write it.

    The existing GDN feature encoder addresses source states, re-encodes the
    experienced action and successors, and feeds recalled action features back
    through the same body read path. A stored post-action feature is not proof
    of reconstructing an exact sentence or the original causal gradient.
    """
    def __init__(self, width, capacity=4096, short=32, long=512, topk=8,
                 threshold=.7, temperature=.2, mode='merge', history_size=256):
        super().__init__(width, capacity, short, long, topk, threshold,
                         temperature, mode, history_size)
        self.register_buffer('action', torch.zeros(capacity, width))
        self.register_buffer('intervening', torch.zeros(capacity, width))
        self.register_buffer('intervention_count', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('progress', torch.zeros(capacity))
        self.register_buffer('progress_valid', torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer('episodes', torch.zeros((), dtype=torch.long))
        # A record's measured outcome and the usefulness of reading that record
        # are different quantities. Reuse the existing record capacity.
        self.register_buffer('read_credit_mean', torch.zeros(capacity))
        self.register_buffer('read_credit_count', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('read_credit_version', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('read_credit_first', torch.zeros(capacity))
        self.register_buffer('read_credit_second', torch.zeros(capacity))
        self.register_buffer('read_credit_steps', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('retention_utility_first', torch.zeros(capacity))
        self.register_buffer('retention_utility_second', torch.zeros(capacity))
        self.register_buffer('retention_utility_steps', torch.zeros(capacity, dtype=torch.long))
        self.read_credit_rule = 'mean'
        self.read_credit_lr = 3e-4
        self.read_credit_betas = (.9, .999)
        self.read_credit_eps = 1e-8
        self.read_credit_enabled = False
        # Opt-in for the pretrained replay candidate; legacy individuals keep
        # their original selection/replacement until explicitly migrated.
        self.consolidation_enabled = False
        self.consolidation_rule = 'preference'
        self._read_participation = []
        self._read_credit_active = False
        self._read_credit_entries = 0

    def get_extra_state(self):
        return dict(super().get_extra_state(), read_credit_enabled=self.read_credit_enabled,
                    consolidation_enabled=self.consolidation_enabled,
                    consolidation_rule=self.consolidation_rule,
                    read_credit_rule=self.read_credit_rule, read_credit_lr=self.read_credit_lr,
                    read_credit_betas=self.read_credit_betas, read_credit_eps=self.read_credit_eps)

    def set_extra_state(self, state):
        self.read_credit_enabled = state.get('read_credit_enabled', self.read_credit_enabled)
        self.consolidation_enabled = state.get('consolidation_enabled', False)
        self.consolidation_rule = state.get('consolidation_rule', 'preference')
        if self.consolidation_rule not in ('preference', 'evidence', 'retention', 'utility'):
            raise ValueError('unknown bounded consolidation rule')
        for name in ('read_credit_rule','read_credit_lr','read_credit_betas','read_credit_eps'):
            if name in state:
                setattr(self, name, state[name])
        super().set_extra_state(state)

    def _load_from_state_dict(self, state, prefix, *args):
        # Existing complete individuals migrate with neutral read credit.
        for name in ('read_credit_mean', 'read_credit_count', 'read_credit_version',
                     'read_credit_first', 'read_credit_second', 'read_credit_steps',
                     'retention_utility_first', 'retention_utility_second', 'retention_utility_steps'):
            if prefix + name not in state:
                state[prefix + name] = torch.zeros_like(getattr(self, name))
        self._read_participation = []
        self._read_credit_active = False
        self._read_credit_entries = 0
        super()._load_from_state_dict(state, prefix, *args)

    def begin_read_credit(self):
        # One live training objective at a time, as in the continuing engine.
        # No unconsumed graph or receipt is carried across observation windows.
        self._read_participation = []
        self._read_credit_active = True
        self._read_credit_entries = 0

    def settle_read_credit(self):
        from scripts.ib_local.navigation_read_credit import settle_read_credit
        return settle_read_credit(self)

    def read(self, features, encoder, *, null_logit=None):
        return self.read_encoded(encoder(features), encoder, null_logit=null_logit)

    def read_encoded(self, latent, encoder, *, null_logit=None, eligible=None):
        """Same finite read, accepting an already encoded (possibly predicted) state.

        Does not commit an observation or touch the memory clock. The caller
        owns the distinction between imagined and actually observed states.
        """
        q = F.normalize(latent, dim=-1)
        keys = F.normalize(encoder(self.source), dim=-1)
        if latent.ndim != 2 or latent.shape[-1] != keys.shape[-1]:
            raise ValueError('expected [queries, latent_width] encoded states')
        occupied = self.occupied if eligible is None else self.occupied & eligible
        scores = (q @ keys.T).masked_fill(~occupied, -1e4)
        from scripts.ib_local.experience_consolidation import consolidates_reads
        if consolidates_reads(self):
            from scripts.ib_local.experience_consolidation import select_read_slots
            selected, slots = select_read_slots(self, scores)
        else:
            selected, slots = scores.topk(self.topk, dim=-1)
        valid = occupied[slots]
        from scripts.ib_local.navigation_read_credit import read_logits
        logits = read_logits(self, selected, slots, valid)
        if null_logit is None:
            # Exact legacy path: all assigned mass goes to occupied records.
            weights = logits.softmax(-1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            empty = null_logit(q)
            if empty.shape != (*logits.shape[:-1],1):
                raise ValueError('one finite null logit per query required')
            combined = torch.cat((logits.masked_fill(~valid,-torch.inf),empty),dim=-1)
            # The final candidate has exactly zero value. Its mass must NOT
            # be renormalized back into the occupied historical records.
            weights = combined.softmax(-1)[...,:-1] * valid
        # The injected value recalls the experienced action-state; predictors
        # can use the observed successor independently through long_value.
        action = (weights[..., None] * encoder(self.action)[slots]).sum(-2)
        successor = (weights[..., None] * encoder(self.long_successor)[slots]).sum(-2)
        return TransitionRead(action, successor, slots, weights)

    @torch.no_grad()
    def commit(self, features, encoder, read=None):
        """Advance actual byte time and pending contexts, without episode writes."""
        if features.ndim != 2 or features.shape[1] != self.width or not 0 < len(features) <= self.long:
            raise ValueError('bounded observed feature chunk required')
        next_clock = self.clock + len(features)
        if int(next_clock) < int(self.clock):
            raise OverflowError('event clock exhausted')
        if read is None:
            current_view = torch.zeros_like(features)
        else:
            current_view = (read.weights[..., None] * self.long_successor[read.slots]).sum(-2)
            touched = read.slots[read.weights > 0]
            if touched.numel():
                self.last_touch.index_fill_(0, touched.unique(), next_clock)
        previous = self.pending[-self.pending_count:] if self.pending_count else self.pending[:0]
        previous_view = self.pending_view[-self.pending_count:] if self.pending_count else self.pending_view[:0]
        tail = torch.cat((previous, features.detach()), 0)[-self.long:].clone()
        tail_view = torch.cat((previous_view, current_view.detach()), 0)[-self.long:].clone()
        self.pending.zero_(); self.pending[-len(tail):].copy_(tail)
        self.pending_view.zero_(); self.pending_view[-len(tail):].copy_(tail_view)
        self.pending_valid.zero_(); self.pending_valid[-len(tail):] = True
        self.pending_count = len(tail)
        self.history[self.history_cursor].copy_(features[-1])
        self.history_clock[self.history_cursor] = next_clock
        self.history_valid[self.history_cursor] = True
        self.history_cursor = (self.history_cursor + 1) % self.history_size
        self.history_count = min(self.history_count + 1, self.history_size)
        self.clock.copy_(next_clock)

    @torch.no_grad()
    def commit_episode(self, source, action, short, long, encoder, *, progress=None, source_time=None,
                       intervening=None, intervention_count=0):
        """Call once only after the actual successors and optional probe exist."""
        inputs = (source, action, short, long)
        if any(x.shape != (self.width,) or not torch.isfinite(x).all() for x in inputs):
            raise ValueError('each observed feature must be a finite width vector')
        if source_time is None or not 0 <= source_time or source_time + self.long > int(self.clock):
            raise ValueError('source and long successor must both be observed before commit')
        if progress is not None and not torch.isfinite(torch.as_tensor(progress)).all():
            raise ValueError('nonfinite progress')
        if intervention_count < 0 or (intervention_count > 0 and intervening is None):
            raise ValueError('intervening action history must be recorded')
        if intervening is None:
            intervening = torch.zeros_like(action)
        if intervening.shape != (self.width,) or not torch.isfinite(intervening).all():
            raise ValueError('invalid intervening action state')
        hit = False
        slot = 0
        if self.mode == 'merge' and bool(self.occupied.any()):
            encoded = [F.normalize(encoder(x[None]), dim=-1) for x in inputs]
            stores = (self.source, self.action, self.successor, self.long_successor)
            similarity = torch.stack([(encoded[i] * F.normalize(encoder(v), dim=-1)).sum(-1)
                                      for i, v in enumerate(stores)]).amin(0)
            similarity.masked_fill_(self.intervention_count != intervention_count, -2)
            if intervention_count:
                last = F.normalize(encoder(intervening[None]), dim=-1)
                similarity = torch.minimum(similarity, (last * F.normalize(encoder(self.intervening), dim=-1)).sum(-1))
            similarity.masked_fill_(~self.occupied, -2)
            best, index = similarity.max(0)
            slot = int(index)
            hit = bool(best >= self.threshold)
        archive = getattr(self, '_causal_archive', None)
        # Preserve the old version before either overwrite or approximate
        # merge. The archive is optional and does not alter legacy commits.
        # Select the victim first, before mutating any episode field.
        if not hit:
            if self.consolidation_enabled:
                from scripts.ib_local.experience_consolidation import replacement_slot
                slot = replacement_slot(self)
            elif self.mode == 'fifo':
                slot = int(self.cursor)
            else:
                priority = torch.where(self.occupied, self.last_touch, -1)
                slot = int(priority.argmin())
            if archive is not None:
                archive.before_slot_write(slot)
            if self.mode == 'fifo' and not self.consolidation_enabled:
                self.cursor.copy_((self.cursor + 1) % self.capacity)
            self.evicted_total.add_(int(self.occupied[slot]))
            self.admitted_total.add_(1)
            self.count[slot] = 0
            self.birth[slot] = source_time
            self.last_touch[slot] = self.clock
            self.progress_valid[slot] = False
            self.read_credit_mean[slot] = 0
            self.read_credit_count[slot] = 0
            self.read_credit_first[slot] = 0
            self.read_credit_second[slot] = 0
            self.read_credit_steps[slot] = 0
        else:
            if archive is not None:
                archive.before_slot_write(slot)
            self.merged_total.add_(1)
        # Even a merge changes the represented experience. An outstanding
        # receipt cannot silently be assigned to this new mixture.
        self.retention_utility_first[slot] = 0
        self.retention_utility_second[slot] = 0
        self.retention_utility_steps[slot] = 0
        self.read_credit_version[slot].add_(1)
        n = int(self.count[slot])
        for store, value in zip((self.source, self.action, self.successor, self.long_successor), inputs):
            store[slot].copy_((store[slot] * n + value.detach()) / (n + 1))
        self.intervening[slot].copy_((self.intervening[slot] * n + intervening.detach()) / (n + 1))
        self.intervention_count[slot] = intervention_count
        # A progress value must not be averaged with missing observations.
        # Conservative choice: omit the merged label if either is unavailable.
        valid = progress is not None and (n == 0 or bool(self.progress_valid[slot]))
        if valid:
            self.progress[slot] = (self.progress[slot] * n + float(progress)) / (n + 1)
        else:
            self.progress[slot] = 0
        self.progress_valid[slot] = valid
        self.count[slot] = n + 1
        self.occupied[slot] = True
        self.last_write[slot] = self.clock
        self.episodes.add_(1)
        return slot, hit


class LatentNavigationV3(LatentNavigationV2):
    """Existing GDN language body with one event-addressed experience table.

    This connects retrieval to the inherited GDN injection and provides an
    action-conditioned successor objective. The real event loop, endogenous
    decision timing, intent retention and progress-modulated learning are not
    connected merely by constructing this class.
    """
    def __init__(self, body, latent_dim=64, capacity=4096, short=32, long=512,
                 topk=8, seed=17, max_chunk=512, threshold=.7, mode='merge', history_size=256):
        super().__init__(body, latent_dim, capacity, short, long, topk, seed,
                         max_chunk, threshold, mode, history_size)
        self.memory = ExperienceMemory(body.base.d, capacity, short, long, topk,
                                       threshold, .2, mode, history_size)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2)
            self.action_decisions = nn.Linear(3 * latent_dim, 2)
        # A trainable, nonzero initial connection lets the shared GDN state
        # respond when an actual memory is present. The number is an explicit
        # initialization, not a proven optimal gain or a health criterion.
        with torch.no_grad():
            self.read_gain.fill_(.05)
        self.to(device=next(body.parameters()).device, dtype=next(body.parameters()).dtype)

    def plan_from_state(self, source):
        """Recalled action-state and predicted outcome from the SAME source."""
        if source.ndim != 2 or source.shape[-1] != self.memory.width:
            raise ValueError('expected [samples,width] source states')
        read = self.memory.read(source, self.encoder)
        condition = torch.cat((self.encoder(source), read.value), -1)
        history = (read.weights[..., None] * self.encoder(self.memory.intervening)[read.slots]).sum(-2)
        valid = self.memory.progress_valid[read.slots].to(read.weights.dtype)
        known_weight = (read.weights * valid).sum(-1)
        progress = ((read.weights * valid * self.memory.progress[read.slots]).sum(-1) /
                    known_weight.clamp_min(1e-12))
        # Index 0 is a start decision while listening; index 1 is an end
        # decision while speaking. The engine must sample at actual causal
        # event times and train the resulting likelihood; no schedule is set.
        decision_feature = torch.cat((condition, history), -1)
        return dict(action=read.value, decision_feature=decision_feature,
                    decision_logits=self.action_decisions(decision_feature),
                    intervening=history,
                    short=self.predictor(condition),
                    long=self.long_predictor(condition), progress=progress,
                    progress_weight=known_weight, slots=read.slots, weights=read.weights)

    def observed_outcome_loss(self, source, action, short, long):
        """Prediction loss on actual observed transitions; caller controls time."""
        if not (source.shape == action.shape == short.shape == long.shape):
            raise ValueError('observed source, action and outcomes must align')
        condition = torch.cat((self.encoder(source), self.encoder(action)), -1)
        return (F.mse_loss(self.predictor(condition), self.encoder(short)) +
                F.mse_loss(self.long_predictor(condition), self.encoder(long)))
