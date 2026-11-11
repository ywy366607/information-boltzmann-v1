"""Retrieve old local GDN computations, not just their cached vector values.

An experience keeps a bounded input slice and its HELD incoming recurrent state
on CPU. Reading it re-executes the original block with current parameters. The
current language error can consequently update old gates/projections through
that read. This is a held-boundary partial derivative, not full-history BPTT.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.latent_navigation_v3 import ExperienceMemory
from scripts.ib_local.navigation_read_credit import configure_read_credit
from scripts.ib_local.pretrained_gdn import RecallInlet, new_recurrent_cache


def _seal(value):
    if value is None:
        return None
    return value.detach().cpu().clone()


def copy_tree(value, device='cpu'):
    """A bounded, detached observation; never retain an earlier autograd graph."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: copy_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(copy_tree(item, device) for item in value)
    if isinstance(value, list):
        return [copy_tree(item, device) for item in value]
    return value


def tree_nbytes(value, seen=None):
    """Count shared tensor storage once, including a shared joint program."""
    seen=set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        identity=(str(value.device),value.untyped_storage().data_ptr())
        if identity in seen:
            return 0
        seen.add(identity)
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(tree_nbytes(item,seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tree_nbytes(item,seen) for item in value)
    if is_dataclass(value):
        return sum(tree_nbytes(getattr(value,field.name),seen) for field in fields(value))
    return 0


@dataclass
class JointExperience:
    """Real incoming vectors and a joint held boundary; no weights or optimizer.

    Re-encoding these inputs without recursive historical reads is a NEW
    conditional program, not a claim to reproduce all historical computation.
    Two layer-specific entries can reference this same immutable archive.
    """
    incoming_features: torch.Tensor
    boundary: dict
    parameter_version: int


def execute_joint_query(model,archive,query,*,detach_history=False):
    """Reencode one archived input window, then ask a current vector query.

    The same model owns all24 transitions; only its existing trainable native
    layers receive parameter derivatives. A held control detaches the old final
    states but keeps every current query derivative and the same forward values.
    Short autograd queries need the existing FLA chunk-derivative bridge.
    """
    if archive.incoming_features.shape[1]!=128 or not 1<=query.shape[1]<=128:
        raise ValueError('one archived128 window and a current query<=128 required')
    cache=new_recurrent_cache(archive.boundary['seen_tokens'])
    cache.states=copy_tree(archive.boundary['states'],query.device)
    old=model.model(inputs_embeds=archive.incoming_features.to(query.device),
                    past_key_values=cache,use_cache=True)
    cache=old.past_key_values
    if detach_history:
        from scripts.ib_local.pretrained_gdn import clone_cache
        cache=clone_cache(cache)
    return model(inputs_embeds=query,past_key_values=cache,use_cache=True)


@dataclass
class LocalExperience:
    inputs: torch.Tensor
    boundary: dict | None
    version: int
    continuation: torch.Tensor | None = None
    joint: JointExperience | None = None

    @property
    def complete_inputs(self):
        """Actual bounded local inputs; legacy response remains at index64."""
        tail=getattr(self,'continuation',None)
        return self.inputs if tail is None else torch.cat((self.inputs,tail),dim=1)


class LayerExperienceRecall(nn.Module):
    """Existing episode/read-credit rules plus a per-record executable capsule.

    Each selected layer has disjoint coordinates and a share of one explicit
    total capacity. Records are never averaged: two recurrent boundaries are
    not equivalent to an averaged nonlinear computation. Optional consolidation
    reuses actual read credit to select and retain complete capsules.
    """
    def __init__(self, width, *, capacity=8, latent_dim=64, topk=2,
                 replay_tokens=65, device='cpu', consolidation=False,
                 consolidation_rule='preference', read_interval=1):
        super().__init__()
        if replay_tokens <= 64:
            raise ValueError('pinned FLA needs >64 tokens for differentiable chunk replay')
        self.replay_tokens = replay_tokens
        self.capture_full_window = False
        self.reactivation_mode = 'legacy_endpoint'
        self.memory = ExperienceMemory(width, capacity=capacity, short=32, long=63,
                                       topk=topk, mode='fifo', history_size=16)
        self.memory.consolidation_enabled = consolidation
        if consolidation_rule not in ('preference', 'evidence', 'retention'):
            raise ValueError('unknown bounded consolidation rule')
        self.memory.consolidation_rule = consolidation_rule
        self.encoder = nn.Linear(width, latent_dim)  # address query only
        self.value_map = nn.Linear(width, latent_dim)
        self.value_map.load_state_dict(self.encoder.state_dict())
        self.inlet = RecallInlet(width, latent_dim, device=device)
        # One virtual zero-valued candidate, not a historical experience slot.
        # Its key/bias learn from the same language loss as the real read.
        self.null_key = nn.Parameter(torch.full((latent_dim,),latent_dim**-.5))
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.null_read_enabled = False
        self.recall_placement = 'input'
        self.validate_read_interval(read_interval)
        self.read_interval = read_interval
        self.suppressed_record = None
        self.capsules = [None] * capacity
        self.addresses = [None] * capacity
        self.position_addresses = [None] * capacity
        self.current = None
        self.replayed = []
        self.replayed_by_slot = {}
        self.last_read = None
        self._replaying = False
        self.detach_replay = False
        self.detach_replay_older_than = None
        self.query_clock_offsets = None
        self.suppress_older_than = None  # observation-only selected-value cut
        # Optional checked program over (read-weight micro-units, age tokens).
        # Off for every existing checkpoint unless explicitly installed.
        self.causal_relation = None
        self.causal_receipt = None

    @property
    def storage_nbytes(self):
        return self.storage_nbytes_shared(set())

    def storage_nbytes_shared(self,seen):
        return self.memory.nbytes + sum(tree_nbytes(x,seen) for x in self.capsules if x is not None)

    def get_extra_state(self):
        return dict(replay_tokens=self.replay_tokens, capsules=self.capsules,
                    addresses=self.addresses, position_addresses=self.position_addresses,
                    capture_full_window=self.capture_full_window,
                    reactivation_mode=self.reactivation_mode,
                    detach_replay=self.detach_replay,null_read_enabled=self.null_read_enabled,
                    recall_placement=self.recall_placement, read_interval=self.read_interval,
                    detach_replay_older_than=self.detach_replay_older_than,
                    causal_relation=self.causal_relation)

    def install_causal_relation(self, relation):
        """Install checked data at an event boundary, never a prior receipt fit."""
        from scripts.ib_local.causal_relation_runtime import CheckedRelation
        if self.current is not None or self.memory._read_credit_active:
            raise RuntimeError('finish current observation before installing a relation')
        if not isinstance(relation, CheckedRelation) or relation.inputs != 2:
            raise ValueError('a checked two-input causal relation is required')
        self.causal_relation = dict(code=[dict(op=i.op,arg=i.arg) for i in relation.code],
                                    digest=relation.digest, count=0, feedback_sum=0.)

    def _relation_weights(self, read, ages):
        """Apply a program to information available at read time only.

        Its micro-unit result is a proposed change to the read multiplier.
        This candidate mapping is explicitly bounded to [0,2]. The leaf's
        post-loss gradient measures local sensitivity, not finite removal.
        """
        if self.causal_relation is None:
            return read.weights
        from scripts.ib_local.executable_causal_relation import parse_candidate, execute
        code = parse_candidate(self.causal_relation['code'], 2)
        weights = read.weights.detach().cpu()
        ages = ages.detach().cpu()
        changes = []
        for weight, age in zip(weights.flatten().tolist(), ages.flatten().tolist()):
            x = [round(float(weight)*1_000_000), max(0, min(int(age), 10**9))]
            changes.append(max(-1., min(1., execute(code,x)/1_000_000)))
        values = torch.tensor(changes, dtype=read.weights.dtype, device=read.weights.device).reshape_as(read.weights)
        multiplier = (1 + values).detach()
        if torch.is_grad_enabled() and self.memory._read_credit_active:
            multiplier.requires_grad_(True)
            self.causal_receipt = (multiplier, values.detach(), read.slots.detach().clone(),
                                   self.memory.read_credit_version[read.slots].detach().clone(),
                                   self.memory.occupied[read.slots].detach().clone())
        return read.weights * multiplier

    def settle_causal_relation(self):
        """Settle only the matching read after its actual loss has arrived."""
        receipt, self.causal_receipt = self.causal_receipt, None
        if receipt is None:
            return dict(records=0)
        multiplier, predictions, slots, versions, valid = receipt
        if not torch.equal(self.memory.read_credit_version[slots], versions):
            raise RuntimeError('relation receipt no longer matches episode versions')
        if multiplier.grad is None:
            return dict(records=0)
        target = -multiplier.grad.detach()
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError('nonfinite causal-relation feedback')
        # Proposed change and loss derivative have different meanings; do not
        # claim their numerical difference is a prediction error.
        self.causal_relation['count'] += int(valid.sum())
        self.causal_relation['feedback_sum'] += float(target[valid].sum())
        return dict(records=int(valid.sum()), proposed_change=predictions.cpu().tolist(),
                    feedback=target.cpu().tolist(), valid=valid.cpu().tolist(),
                    semantics='negative local loss derivative by read multiplier')

    @staticmethod
    def validate_read_interval(interval):
        if interval is not None and (type(interval) is not int or interval < 1):
            raise ValueError('read interval must be a positive token count or None for legacy single read')

    def query_positions(self, length, first=0):
        """Reading cadence is independent of optimizer/BPTT window length.

        The first permitted position always queries; a positive interval then
        refreshes within this observation. None exactly preserves old snapshots.
        Queries use preceding-layer causal activations, never future positions.
        """
        if type(first) is not int or not 0 <= first < length:
            raise ValueError('read position must be an observed input position')
        return ([first] if self.read_interval is None else
                list(range(first, length, self.read_interval)))

    def _load_from_state_dict(self,state,prefix,*args):
        # Old individuals keep their old read and do not inherit null weights
        # left over from a previous temporary branch in the same process.
        if prefix+'null_key' not in state:
            state[prefix+'null_key'] = torch.full_like(self.null_key,self.null_key.numel()**-.5)
        if prefix+'null_bias' not in state:
            state[prefix+'null_bias'] = torch.zeros_like(self.null_bias)
        # A snapshot from the shared-encoder readout starts the value map at
        # that same encoder, then the two maps learn apart.
        for name in ('weight', 'bias'):
            key, source = prefix+'value_map.'+name, prefix+'encoder.'+name
            if key not in state and source in state:
                state[key] = state[source].detach().clone()
        super()._load_from_state_dict(state,prefix,*args)

    def set_extra_state(self, state):
        if state['replay_tokens'] != self.replay_tokens or len(state['capsules']) != len(self.capsules):
            raise ValueError('incompatible local replay capacity')
        # A restored learner replaces slots on commit. Its slot list must not
        # alias the source checkpoint used to initialize a second branch.
        self.capsules = list(state['capsules'])
        blank = [None] * len(self.capsules)
        self.addresses = [_seal(item) for item in state.get('addresses', blank)]
        self.position_addresses = [_seal(item) for item in state.get('position_addresses', blank)]
        self.detach_replay = state.get('detach_replay', False)
        self.capture_full_window = state.get('capture_full_window', False)
        self.reactivation_mode = state.get('reactivation_mode', 'legacy_endpoint')
        if self.reactivation_mode not in ('legacy_endpoint', 'complete_attention'):
            raise ValueError('unknown local reactivation mode')
        self.detach_replay_older_than = state.get('detach_replay_older_than')
        self.query_clock_offsets = None
        self.null_read_enabled = state.get('null_read_enabled',False)
        self.recall_placement = state.get('recall_placement','input')
        if self.recall_placement not in ('input','output'):
            raise ValueError('unknown recall placement')
        self.read_interval = state.get('read_interval', None)
        self.validate_read_interval(self.read_interval)
        self.current, self.replayed, self.last_read = None, [], None
        self.replayed_by_slot = {}
        self.suppressed_record = None
        self.causal_relation = state.get('causal_relation')
        self.causal_receipt = None

    @torch.no_grad()
    def enable_null_read(self):
        if self.current is not None or self.memory._read_credit_active:
            raise RuntimeError('finish the observation before enabling null read')
        if not self.null_read_enabled:
            occupied = self.memory.occupied
            # Initialize in the body's existing address coordinates, without
            # a target, external utility score, or new scale hyperparameter.
            encoded = (self.sealed_addresses()[occupied] if bool(occupied.any())
                       else self.encoder.bias[None])
            center = F.normalize(encoded,dim=-1).mean(0)
            if float(center.norm()) > 1e-12:
                self.null_key.copy_(F.normalize(center,dim=0))
            else:
                self.null_key.fill_(self.null_key.numel()**-.5)
            self.null_bias.zero_()
        self.null_read_enabled = True

    def null_logit(self,normalized_query):
        return ((normalized_query*F.normalize(self.null_key,dim=0)).sum(-1,keepdim=True)
                /self.memory.temperature+self.null_bias)

    def remember_address(self, slot, source):
        """Fix one slot's key from the address encoder at write time."""
        self.addresses[slot] = F.normalize(self.encoder(source.float()), dim=-1).detach()

    def sealed_addresses(self):
        """Stored write-time keys. A slot with none uses the current encoder for this read only."""
        latent = self.encoder.out_features
        rows = []
        with torch.no_grad():
            for index in range(self.memory.capacity):
                saved = self.addresses[index]
                if saved is None:
                    if bool(self.memory.occupied[index]):
                        saved = F.normalize(self.encoder(self.memory.source[index].float()), dim=-1)
                    else:
                        saved = self.encoder.weight.new_zeros(latent)
                rows.append(saved.detach())
        return torch.stack(rows)

    def sealed_position_addresses(self, slot, inputs):
        saved = self.position_addresses[slot]
        if saved is None or saved.shape[0] != inputs.shape[0]:
            with torch.no_grad():
                saved = F.normalize(self.encoder(inputs.float()), dim=-1).detach()
        return saved

    def _encode_value(self, response):
        return self.value_map(response.float().cpu())

    def read(self, query):
        from scripts.ib_local.experience_consolidation import consolidates_reads
        from scripts.ib_local.latent_navigation_v2 import TransitionRead
        from scripts.ib_local.navigation_read_credit import read_logits
        memory = self.memory
        q = F.normalize(self.encoder(query.float().cpu()), dim=-1)
        scores = (q @ self.sealed_addresses().T).masked_fill(~memory.occupied, -1e4)
        if consolidates_reads(memory):
            from scripts.ib_local.experience_consolidation import select_read_slots
            selected, slots = select_read_slots(memory, scores)
        else:
            selected, slots = scores.topk(memory.topk, dim=-1)
        valid = memory.occupied[slots]
        logits = read_logits(memory, selected, slots, valid)
        if not self.null_read_enabled:
            weights = logits.softmax(-1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            empty = self.null_logit(q)
            if empty.shape != (*logits.shape[:-1], 1):
                raise ValueError('one finite null logit per query required')
            combined = torch.cat((logits.masked_fill(~valid, -torch.inf), empty), dim=-1)
            weights = combined.softmax(-1)[..., :-1] * valid
        unused = torch.zeros(query.shape[0], self.encoder.out_features)
        return TransitionRead(unused, unused, slots, weights)

    def begin(self, optimizer_group, *, query_limit=128):
        if self.current is not None:
            raise RuntimeError('commit or discard the previous bounded observation first')
        configure_read_credit(self.memory, 'adam', optimizer_group)
        self.memory.begin_read_credit()
        # One finite observation's receipts, independent of reservoir capacity.
        if type(query_limit) is not int or query_limit < 1:
            raise ValueError('positive bounded query limit required')
        self.memory._read_credit_limit = max(
            3*max(self.memory.capacity,self.memory.long)*self.memory.topk,
            query_limit*self.memory.topk)
        self.replayed, self.last_read = [], None
        self.replayed_by_slot = {}

    def retrieve(self, query, replay):
        """Use existing address/participation credit; recompute selected values."""
        if self.reactivation_mode == 'complete_attention':
            return self._retrieve_complete(query, replay)
        read = self.read(query)
        self.last_read = read
        values, held_values, lookup = [], [], {}
        # A capsule is a held local computation at one parameter version.
        # All current queries can share it; gradients sum at its response.
        slots = read.slots.flatten().tolist()
        unique = list(dict.fromkeys(slots))
        for slot in unique:
            lookup[slot] = len(values)
            capsule = self.capsules[slot]
            if not bool(self.memory.occupied[slot]):
                values.append(query.new_zeros(1, self.encoder.out_features).float().cpu())
                held_values.append(values[-1])
                continue
            if capsule is None or capsule.version != int(self.memory.read_credit_version[slot]):
                raise RuntimeError('replay capsule and episode identity diverged')
            response = replay(capsule)[:, -1]
            if self.detach_replay:
                # Identical forward values. Only the route into the original
                # block's old computation is cut; current encoder stays live.
                response = response.detach()
            self.replayed.append(response)
            self.replayed_by_slot[slot] = response
            values.append(self._encode_value(response))
            if self.detach_replay_older_than is not None:
                held_values.append(self._encode_value(response.detach()))
        indices = torch.tensor([lookup[slot] for slot in slots],dtype=torch.long,
                               device=read.slots.device).reshape(read.slots.shape)
        selected_values = torch.cat(values, dim=0)[indices]
        if self.detach_replay_older_than is not None:
            offsets = (torch.zeros(query.shape[0], dtype=torch.long)
                       if self.query_clock_offsets is None else self.query_clock_offsets)
            if len(offsets) != query.shape[0]:
                raise ValueError('one event offset per actual query required')
            ages = (self.memory.clock + offsets.to(read.slots.device)[:, None]
                    - self.memory.birth[read.slots])
            older = ages > self.detach_replay_older_than
            held_selected = torch.cat(held_values, dim=0)[indices]
            selected_values = torch.where(older[..., None], held_selected, selected_values)
        offsets = (torch.zeros(query.shape[0], dtype=torch.long)
                   if self.query_clock_offsets is None else self.query_clock_offsets)
        ages = self.memory.clock + offsets.to(read.slots.device)[:, None] - self.memory.birth[read.slots]
        weights = self._relation_weights(read, ages)
        if self.suppress_older_than is not None:
            ages = self.memory.clock-self.memory.birth[read.slots]
            # Hold other contributions fixed rather than renormalizing them.
            weights = weights*(ages <= self.suppress_older_than)
        if self.suppressed_record is not None:
            slot, version = self.suppressed_record
            if int(self.memory.read_credit_version[slot]) != version:
                raise RuntimeError('counterfactual record identity changed')
            weights = weights*(read.slots != slot)
        return (selected_values*weights[..., None]).sum(1)

    def _retrieve_complete(self, query, replay):
        """Query all positions of a selected, actually observed local program.

        Slot selection/null mass are unchanged. Within a selected capsule the
        existing encoder supplies input keys and reexecuted output values; no
        new weights or historical graph. The callback caches only the old
        sequence, never this current-query-dependent read. Legacy65 capsules
        use their actual65 inputs; missing participation is never fabricated.
        """
        read = self.read(query)
        self.last_read = read
        slots = read.slots.flatten().tolist()
        unique = list(dict.fromkeys(slots))
        q = F.normalize(self.encoder(query.float().cpu()), dim=-1)
        values, held_values, ends = [], [], []
        lookup = {}
        for slot in unique:
            lookup[slot] = len(values)
            capsule = self.capsules[slot]
            if not bool(self.memory.occupied[slot]):
                value = q.new_zeros(q.shape)
                values.append(value)
                held_values.append(value)
                ends.append(int(self.memory.birth[slot]))
                continue
            if capsule is None or capsule.version != int(self.memory.read_credit_version[slot]):
                raise RuntimeError('replay capsule and episode identity diverged')
            inputs = capsule.complete_inputs
            response = replay(capsule)
            if (inputs.ndim != 3 or response.ndim != 3 or inputs.shape != response.shape
                    or inputs.shape[0] != 1 or not 65 <= inputs.shape[1] <= 128):
                raise ValueError('complete reactivation requires all actual65..128 input/output positions')
            if self.detach_replay:
                response = response.detach()
            self.replayed.append(response)
            self.replayed_by_slot[slot] = response
            keys = self.sealed_position_addresses(slot, inputs[0])
            position_weights = (q @ keys.T / self.memory.temperature).softmax(-1)
            value = position_weights @ self._encode_value(response[0])
            values.append(value)
            if self.detach_replay_older_than is not None:
                held_values.append(position_weights @ self._encode_value(response[0].detach()))
            # birth is the legacy key at index64. Only call the entire replay
            # old when its LATEST actual input is older than the cutoff.
            ends.append(int(self.memory.birth[slot]) - (self.replay_tokens-1) + inputs.shape[1]-1)
        indices = torch.tensor([lookup[slot] for slot in slots], device=read.slots.device).reshape(read.slots.shape)
        rows = torch.arange(len(query), device=indices.device)[:, None]
        selected = torch.stack(values, dim=1)[rows, indices]
        end_times = torch.tensor(ends, device=indices.device)[indices]
        offsets = (torch.zeros(len(query), dtype=torch.long, device=indices.device)
                   if self.query_clock_offsets is None else self.query_clock_offsets.to(indices.device))
        if len(offsets) != len(query):
            raise ValueError('one event offset per actual query required')
        ages = self.memory.clock + offsets[:, None] - end_times
        if self.detach_replay_older_than is not None:
            held = torch.stack(held_values, dim=1)[rows, indices]
            selected = torch.where((ages > self.detach_replay_older_than)[..., None], held, selected)
        weights = self._relation_weights(read, ages)
        if self.suppress_older_than is not None:
            weights = weights * (ages <= self.suppress_older_than)
        if self.suppressed_record is not None:
            slot, version = self.suppressed_record
            if int(self.memory.read_credit_version[slot]) != version:
                raise RuntimeError('counterfactual record identity changed')
            weights = weights * (read.slots != slot)
        return (selected * weights[..., None]).sum(1)

    @contextmanager
    def attach(self, model, *, layer, read_position=0, capture=True, clock_offset=0):
        """Temporary reads; only captured observations may become episodes.

        Read-only autoregressive inference accepts single tokens. Training and
        actual observation still require the full local episode (>=128 tokens).
        """
        block = model.model.layers[layer]

        def before(module, positional, keyword):
            if self._replaying:
                return positional, keyword
            hidden = positional[0] if positional else keyword['hidden_states']
            minimum = self.replay_tokens + self.memory.long if capture else 1
            if hidden.shape[0] != 1 or hidden.shape[1] < minimum:
                raise ValueError('one continuous individual and a >=128-token observation required')
            positions = self.query_positions(hidden.shape[1],read_position)
            cache = keyword.get('past_key_values')
            boundary = (copy_tree(cache.states[layer]) if capture and cache is not None
                        and len(cache) > layer else None)

            def replay(capsule):
                previous = None
                if capsule.boundary is not None:
                    previous = new_recurrent_cache()
                    previous.states = [{} for _ in range(layer)] + [copy_tree(capsule.boundary, hidden.device)]
                self._replaying = True
                try:
                    inputs = (capsule.complete_inputs if self.reactivation_mode == 'complete_attention'
                              else capsule.inputs)
                    return block(inputs.to(hidden.device),
                                 past_key_values=previous, use_cache=False)[0]
                finally:
                    self._replaying = False

            self.query_clock_offsets = torch.tensor(positions, dtype=torch.long) + clock_offset
            recalled = self.retrieve(hidden[0, positions], replay)
            projected = self.inlet.projection(recalled.to(hidden.device)).to(hidden.dtype)
            if len(positions) == 1:
                injection = projected[:, None]  # exact legacy broadcast
            else:
                lengths = torch.tensor([b-a for a,b in zip(positions,positions[1:]+[hidden.shape[1]])],
                                       device=hidden.device)
                # A later query replaces the previous recall from its own
                # position onward; it cannot inject backwards into the prefix.
                injection = torch.repeat_interleave(projected,lengths,dim=0)[None]
            # A query at the first answer-prediction input may use preceding
            # prompt context. Never send that later query back into the prefix.
            revised = (torch.cat((hidden[:, :read_position],
                                 hidden[:, read_position:] + injection), dim=1)
                       if self.recall_placement == 'input' else hidden)
            self.current = dict(inputs=revised, source=hidden, boundary=boundary, capture=capture,
                                injection=injection, query_positions=positions)
            if positional:
                return (revised, *positional[1:]), keyword
            keyword = dict(keyword, hidden_states=revised)
            return positional, keyword

        def after(module, positional, output):
            if not self._replaying:
                response = output[0]
                if self.recall_placement == 'output':
                    # Previous-layer activity queries an old local mapping;
                    # recalled responses enter the signal sent to the next
                    # layer, not the native block's current recurrent update.
                    response = torch.cat((response[:, :read_position],
                        response[:, read_position:] + self.current['injection']),dim=1)
                    output = (response,*output[1:])
                self.current['response'] = response
                return output

        handles = [block.register_forward_pre_hook(before, with_kwargs=True),
                   block.register_forward_hook(after)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def commit(self):
        """Only after scoring/backward: one episode per observed chunk/layer."""
        if self.current is None:
            raise RuntimeError('no actual observation to commit')
        if not self.current.get('capture', True):
            raise RuntimeError('read-only inference cannot commit an experience')
        m, current = self.memory, self.current
        source_index = self.replay_tokens-1
        source = current['source'][0].float().cpu()
        response = current['response'][0].float().cpu()
        start = int(m.clock)
        m.clock.add_(len(source))  # token events here, not old byte-clock units
        if self.last_read is not None:
            touched = self.last_read.slots[self.last_read.weights > 0].unique()
            m.last_touch[touched] = m.clock
        slot, _ = m.commit_episode(source[source_index], response[source_index],
            response[source_index+m.short], response[source_index+m.long], self.encoder,
            source_time=start+source_index)
        self.capsules[slot] = LocalExperience(
            copy_tree(current['inputs'][:, :self.replay_tokens]), current['boundary'],
            int(m.read_credit_version[slot]),
            copy_tree(current['inputs'][:, self.replay_tokens:self.replay_tokens+m.long])
                if self.capture_full_window else None)
        self.remember_address(slot, source[source_index])
        self.position_addresses[slot] = F.normalize(
            self.encoder(self.capsules[slot].complete_inputs[0].float()), dim=-1).detach()
        m.history[m.history_cursor].copy_(source[-1])
        m.history_valid[m.history_cursor] = True
        m.history_clock[m.history_cursor] = m.clock
        m.history_cursor = (m.history_cursor+1) % m.history_size
        m.history_count = min(m.history_count+1, m.history_size)
        self.current, self.replayed, self.last_read = None, [], None
        self.replayed_by_slot = {}
        return slot
