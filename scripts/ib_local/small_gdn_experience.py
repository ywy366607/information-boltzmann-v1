"""Bounded native participation replay for the continuing four-layer body.

Reuse the pretrained branch's addressing, null choice and credit retention.
Only the native execution adapter differs: small GDN boundaries are packed
S/short-convolution tensors, and replay includes the layer's trainable MLP.
Held replay cuts the old native derivative, never the current read or encoder.
"""
from contextlib import ExitStack, contextmanager
from types import MethodType

import torch
from torch import nn

from scripts.ib_local.pretrained_gdn_experience import (
    LayerExperienceRecall, LocalExperience, copy_tree,
)
from scripts.ib_local.latent_navigation_v2 import TransitionRead
from scripts.ib_local.navigation_read_credit import read_logits
from torch.nn import functional as F


class SmallLayerExperience(LayerExperienceRecall):
    @contextmanager
    def attach_native(self, block, *, capture):
        def before(module, args):
            hidden, boundary = args[:2]
            if hidden.shape[0] != 1:
                raise ValueError('one continuous individual required')
            positions = self.query_positions(hidden.shape[1])

            def replay(capsule):
                # Bypass module-level hooks: in particular, the live global
                # navigation recall must not enter a historical computation.
                return block.forward(capsule.inputs.to(hidden.device),
                                     capsule.boundary.to(hidden.device))[0]

            recalled = self.retrieve(hidden[0, positions], replay)
            projected = self.inlet.projection(recalled.to(hidden.device)).to(hidden.dtype)
            lengths = torch.tensor([b-a for a,b in zip(
                positions, positions[1:]+[hidden.shape[1]])], device=hidden.device)
            injection = torch.repeat_interleave(projected, lengths, dim=0)[None]
            revised = hidden + injection
            self.current = dict(source=hidden, inputs=revised,
                boundary=copy_tree(boundary) if capture else None,
                capture=capture, query_positions=positions)
            return (revised, boundary, *args[2:])

        def after(module, args, output):
            # Other existing body hooks can add to the input after ours. Store
            # the actual input consumed by the native block, not an earlier view.
            self.current['inputs'] = args[0]
            self.current['response'] = output[0]

        handles = [block.register_forward_pre_hook(before), block.register_forward_hook(after)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def discard(self):
        self.current, self.replayed, self.last_read = None, [], None
        self.replayed_by_slot = {}
        self.memory._read_participation = []
        self.memory._read_credit_active = False
        self.memory._read_credit_entries = 0

    @torch.no_grad()
    def commit(self):
        if self.current is None or not self.current['capture']:
            raise RuntimeError('no captured native observation')
        length = self.current['source'].shape[1]
        if length >= self.replay_tokens + self.memory.long:
            return super().commit()
        # A partial last window has no observed long successor for a new
        # capsule. It still advances real time; never pad it with a fake target.
        self.memory.clock.add_(length)
        self.discard()
        return None


class SmallReplayBank(nn.Module):
    def __init__(self, layers, width, device, *, capacity=16, latent_dim=64,
                 topk=2, replay_tokens=128, read_interval=1, schema=2,
                 native_output_projection=False):
        super().__init__()
        if schema not in (1, 2, 3, 4, 5):
            raise ValueError('unknown small native replay schema')
        if schema in (3, 4, 5) and read_interval != 1:
            raise ValueError('predictive responsibility requires per-position queries')
        if native_output_projection and schema != 5:
            raise ValueError('native output projection requires direct attention schema5')
        self.schema = schema
        self.config = dict(capacity=capacity, latent_dim=latent_dim, topk=topk,
                           replay_tokens=replay_tokens, read_interval=read_interval, schema=schema)
        options = {k: v for k, v in self.config.items() if k != 'schema'}
        if native_output_projection:
            self.config['native_output_projection'] = True
            options['native_output_projection'] = True
        cls = (SmallLayerExperience if schema == 1 else
               DirectAttentionLayerExperience if schema == 5 else
               DualViewLayerExperience if schema == 4 else ContextualLayerExperience)
        self.memories = nn.ModuleList([cls(width, device=device,
            consolidation=True, consolidation_rule='retention', **options)
            for _ in range(layers)])
        for memory in self.memories:
            memory.enable_null_read()
            memory.predictive_responsibility = schema in (3, 4)
        self.capturing = False
        self.enabled = True
        self.forecast = None

    def set_held(self, held):
        for memory in self.memories:
            memory.detach_replay = bool(held)

    @contextmanager
    def forward_context(self, layers):
        if not self.enabled:
            yield
            return
        with ExitStack() as stack:
            for layer, memory in zip(layers, self.memories):
                if memory.current is not None:
                    raise RuntimeError('settle or discard the previous observation')
                stack.enter_context(memory.attach_native(layer, capture=self.capturing))
            try:
                yield
            finally:
                if not self.capturing:
                    for memory in self.memories:
                        memory.discard()

    @contextmanager
    def observation(self, optimizer_group, length):
        if self.capturing:
            raise RuntimeError('nested training observation')
        for memory in self.memories:
            memory.begin(optimizer_group, query_limit=length)
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False
            for memory in self.memories:
                memory.discard()

    def settle_and_commit(self):
        # Call after backward/optimizer, while the real captured inputs and
        # responses still refer to the executed pre-update observation.
        return [dict(credit=m.memory.settle_read_credit(), slot=m.commit())
                for m in self.memories]

    def diagnostics(self, credit_window=128):
        result = []
        for i, wrapper in enumerate(self.memories):
            m, read = wrapper.memory, wrapper.last_read
            valid = m.occupied[read.slots]
            positions = torch.tensor(wrapper.current['query_positions'])[:, None]
            ages = (m.clock + positions - m.birth[read.slots])[valid]
            weights = read.weights.detach()[valid]
            item = dict(layer=i+1, occupied=int(m.occupied.sum()),
                replayed=len(wrapper.replayed_by_slot),
                max_age_bytes=int(ages.max()) if ages.numel() else 0,
                credit_window_bytes=credit_window,
                mass_older_than_window=float(weights[ages > credit_window].sum()/len(positions)),
                null_mass=float((1-read.weights.sum(-1)).detach().mean()),
                storage_bytes=wrapper.storage_nbytes)
            if hasattr(wrapper, 'view_logit'):
                item['transition_prior'] = float(wrapper.view_logit.detach().sigmoid())
                pr = wrapper.position_read
                if pr is not None:
                    for view in ('state', 'transition'):
                        selected = [j for j, v in enumerate(pr['prediction_views']) if v == view]
                        item[view+'_mass'] = float(pr['prediction_weights'][:, selected].detach().sum(-1).mean())
            if isinstance(wrapper, DirectAttentionLayerExperience):
                item['native_response_gain'] = float(wrapper.inlet.projection.gain.detach())
                if wrapper.inlet.projection.output_weight is not None:
                    item['native_output_weight_norm'] = float(wrapper.inlet.projection.output_weight.detach().norm())
            result.append(item)
        return result


def configure_small_replay(engine, config=None):
    """Explicit zero-output migration, also reconstructed before strict restore."""
    if hasattr(engine.model, 'local_replay'):
        raise ValueError('native replay already configured')
    model = engine.model
    # Migration does not advance the continuing individual's random streams.
    devices = [engine.device.index or 0] if engine.device.type == 'cuda' else []
    options = dict(config or {})
    if config is not None:
        # Existing complete individuals restore their original behavior.
        # Schema2 is an explicit new migration from the accepted source.
        options.setdefault('schema', 1)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(71113)
        bank = SmallReplayBank(len(model.base.layers), model.base.d,
                               engine.device, **options)
    model.local_replay = bank
    original = model.predict_observed

    def observed(self, ids, state, table_logp):
        self.local_replay.forecast = None
        with self.local_replay.forward_context(self.base.layers):
            result=original(ids, state, table_logp)
            if self.local_replay.schema in (3, 4) and self.local_replay.enabled:
                from scripts.ib_local.predictive_responsibility import native_predictive_mixture
                logits=native_predictive_mixture(self,self.local_replay,ids,state,table_logp,result[0])
                return (logits,*result[1:])
            return result

    model.predict_observed = MethodType(observed, model)
    parameters = list(bank.parameters())
    engine.optimizer.add_param_group(dict(params=parameters,
                                         lr=engine.optimizer.param_groups[0]['lr']))
    engine.params.extend(parameters)
    engine.groups = (*engine.groups, parameters)
    return bank


def enable_native_output_projection(engine):
    """Neutral explicit migration: g*r becomes g*r + W*r with W initially0.

    Reorder the existing bank optimizer group to the module traversal order so
    a strict checkpoint reconstruction preserves every old Adam state. New W
    receives the present derivative even when the inherited scalar gain is0.
    """
    bank = engine.model.local_replay
    if bank.schema != 5 or bank.capturing or any(m.current is not None for m in bank.memories):
        raise ValueError('migrate settled direct-attention bank only')
    if bank.config.get('native_output_projection', False):
        return []
    old_ids = {id(p) for p in bank.parameters()}
    groups = [g for g in engine.optimizer.param_groups
              if {id(p) for p in g['params']} == old_ids]
    if len(groups) != 1:
        raise ValueError('native bank must own one complete optimizer group')
    added = [m.inlet.projection.enable_output_projection(engine.model.base.d)
             for m in bank.memories]
    parameters = list(bank.parameters())
    groups[0]['params'] = parameters
    engine.params[:] = [p for p in engine.params if id(p) not in old_ids]+parameters
    engine.groups = tuple(parameters if {id(p) for p in group} == old_ids else group
                          for group in engine.groups)
    bank.config['native_output_projection'] = True
    return added


class ContextualLayerExperience(SmallLayerExperience):
    """Whole observed computations, addressed by causal native activity.

    All positions in the finite reservoir compete with one null candidate.
    Per-record mass has a length-normalized prior. Native responses are
    reexecuted once in length-batched groups, reused for all current queries.
    No target, token identity, task marker or future successor is used to admit
    a local computation. Existing retention/read-credit statistics are reused.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recall_placement = 'output'

    def begin(self, optimizer_group, *, query_limit=128):
        super().begin(optimizer_group, query_limit=query_limit)
        self.memory._read_credit_limit = max(self.memory._read_credit_limit,
                                             query_limit*self.memory.capacity)

    def encode_query(self, query):
        return F.normalize(self.encoder(query.float().cpu()), dim=-1)

    @contextmanager
    def attach_native(self, block, *, capture):
        def before(module, args):
            x, boundary = args[:2]
            if x.shape[0] != 1 or (capture and x.shape[1] > self.replay_tokens):
                raise ValueError('one bounded actual observation required')
            self.current = dict(inputs=x, boundary=copy_tree(boundary) if capture else None,
                                capture=capture, query_positions=self.query_positions(x.shape[1]))

        def after(module, args, output):
            native = output[0]
            positions = self.current['query_positions']
            self.current['inputs'] = args[0]
            self.current['source'] = native
            self.current['response'] = native
            recalled = self.retrieve_context(native[0, positions], block)
            projected = self.inlet.projection(recalled.to(native.device)).to(native.dtype)
            lengths = torch.tensor([b-a for a,b in zip(
                positions, positions[1:]+[native.shape[1]])],device=native.device)
            injection = torch.repeat_interleave(projected,lengths,dim=0)[None]
            return (native+injection, *output[1:])

        handles = [block.register_forward_pre_hook(before),block.register_forward_hook(after)]
        try:
            yield
        finally:
            for h in handles: h.remove()

    def retrieve_context(self, query, block):
        m = self.memory
        slots = torch.nonzero(m.occupied).flatten()
        q = self.encode_query(query)
        if not len(slots):
            # Keep an explicit empty read with stable [queries,0] receipts.
            empty = q.new_empty(len(query),0)
            self.last_read = TransitionRead(None,None,
                torch.empty(len(query),0,dtype=torch.long),empty)
            return q*0
        groups = {}
        for slot in slots.tolist():
            c = self.capsules[slot]
            if c is None or c.version != int(m.read_credit_version[slot]):
                raise RuntimeError('replay capsule and record identity diverged')
            groups.setdefault(c.inputs.shape[1],[]).append(slot)
        for length, members in groups.items():
            x = torch.cat([self.capsules[s].inputs for s in members]).to(query.device)
            boundary = torch.cat([self.capsules[s].boundary for s in members]).to(query.device)
            responses = block.forward(x,boundary)[0]
            for j,slot in enumerate(members):
                r = responses[j:j+1]
                if self.detach_replay: r = r.detach()
                self.replayed.append(r)
                self.replayed_by_slot[slot] = r
        responses = [self.replayed_by_slot[s] for s in slots.tolist()]
        encoded = [self.encoder(r[0].float().cpu()) for r in responses]
        keys = F.normalize(torch.cat(encoded),dim=-1)
        values = torch.cat(encoded)
        counts = [len(e) for e in encoded]
        record_ids = torch.repeat_interleave(torch.arange(len(slots)),torch.tensor(counts))
        log_length = torch.repeat_interleave(torch.tensor(counts,dtype=q.dtype).log(),torch.tensor(counts))
        record_slots = slots[None].expand(len(query),-1)
        bias = read_logits(m,q.new_zeros(len(query),len(slots)),record_slots,
                           torch.ones_like(record_slots,dtype=torch.bool))
        if hasattr(self, 'view_logit'):
            self._dual_query, self._dual_encoded, self._dual_bias = q, encoded, bias
        scores = q@keys.T/m.temperature - log_length[None] + bias[:,record_ids]
        joint_weights = torch.cat((scores,self.null_logit(q)),dim=-1).softmax(-1)
        weights = joint_weights[:,:-1]
        if self.suppress_older_than is not None:
            allowed = (m.clock-m.birth[slots] <= self.suppress_older_than)
            weights = weights*allowed[record_ids][None]
        if self.suppressed_record is not None:
            slot,version = self.suppressed_record
            if int(m.read_credit_version[slot]) != version:
                raise RuntimeError('removed native record identity changed')
            weights = weights*(slots[record_ids]!=slot)[None]
        # Segment reduction groups all positions of each record without CUDA
        # atomics; the metadata stays on CPU with existing preference updates.
        mass = torch.stack([part.sum(-1) for part in weights.split(counts,dim=-1)],dim=-1)
        self.last_read = TransitionRead(None,None,record_slots,mass)
        self.position_read = dict(weights=weights,record_ids=record_ids,slots=slots,
                                  lengths=counts,null_weight=1-mass.sum(-1))
        if getattr(self,'predictive_responsibility',False):
            # Preserve each record's prediction until the outcome arrives.
            # Within-record values are conditioned on the SAME current query.
            self.position_read['conditional_values']=torch.stack([
                score.softmax(-1)@value for score,value in
                zip(scores.split(counts,dim=-1),values.split(counts,dim=0))])
        return weights@values

    @torch.no_grad()
    def commit(self):
        """Admit one observed computation, with no fabricated future fields."""
        if self.current is None or not self.current['capture']:
            raise RuntimeError('no captured native observation')
        from scripts.ib_local.experience_consolidation import replacement_slot
        m,c = self.memory,self.current
        length = c['inputs'].shape[1]
        if not 0 < length <= self.replay_tokens:
            raise ValueError('native record exceeds its finite span')
        start = int(m.clock)
        m.clock.add_(length)
        slot = replacement_slot(m)
        m.evicted_total.add_(int(m.occupied[slot]));m.admitted_total.add_(1)
        for name in ('read_credit_mean','read_credit_count','read_credit_first',
                     'read_credit_second','read_credit_steps','retention_utility_first',
                     'retention_utility_second','retention_utility_steps'):
            getattr(m,name)[slot] = 0
        m.read_credit_version[slot].add_(1)
        # These are observed native activities, never claimed future targets.
        m.source[slot].copy_(c['source'][0,-1].float().cpu())
        m.action[slot].copy_(m.source[slot])
        m.successor[slot].zero_();m.long_successor[slot].zero_()
        m.intervening[slot].zero_();m.intervention_count[slot]=0
        m.progress[slot]=0;m.progress_valid[slot]=False
        m.count[slot]=1;m.birth[slot]=start;m.last_write[slot]=m.clock
        m.last_touch[slot]=m.clock;m.occupied[slot]=True;m.episodes.add_(1)
        self.capsules[slot]=LocalExperience(copy_tree(c['inputs']),c['boundary'],
                                          int(m.read_credit_version[slot]))
        m.history[m.history_cursor].copy_(m.source[slot])
        m.history_valid[m.history_cursor]=True;m.history_clock[m.history_cursor]=m.clock
        m.history_cursor=(m.history_cursor+1)%m.history_size
        m.history_count=min(m.history_count+1,m.history_size)
        self.discard()
        return slot

    def discard(self):
        super().discard()
        self.position_read = None


class DualViewLayerExperience(ContextualLayerExperience):
    """One retained computation, two associations; no second history page.

    State keys/values use the same observed position. Transition keys use j
    and values use the actually observed j+1, confined to this record. One
    learned prior per layer divides the record's existing prior mass between
    views. Their predictions remain separate until the boundary likelihood.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view_logit = nn.Parameter(torch.zeros(()))
        self.position_read = None

    def association_values(self, slot, encoded):
        return encoded

    def retrieve_context(self, query, block):
        # Reuse the original native replay, encoder, addressing and versioned
        # read-credit registration exactly once per record/query.
        self.position_read = None
        state_value = super().retrieve_context(query, block)
        if self.position_read is None:
            return state_value
        m = self.memory
        slots = self.position_read['slots']
        q = self._dual_query
        # Reuse the same live preference leaves for both views; register only
        # one receipt per record/query, then settle their combined derivative.
        bias = self._dual_bias
        encoded = self._dual_encoded
        parts, values, counts, record_ids, positions = [], [], [], [], []
        prediction_views, prediction_slots = [], []
        for record, (slot, e) in enumerate(zip(slots.tolist(), encoded)):
            length = len(e)
            native_values = self.association_values(slot, e)
            views = [('state', e, native_values, torch.arange(length))]
            if length > 1:
                views.append(('transition', e[:-1], native_values[1:], torch.arange(1, length)))
            for view, key, value, value_positions in views:
                if length == 1:
                    prior = q.new_zeros(())
                else:
                    prior = F.logsigmoid(self.view_logit if view == 'transition' else -self.view_logit)
                score = (q@F.normalize(key, dim=-1).T/m.temperature
                         - q.new_tensor(len(key)).log() + bias[:, record:record+1] + prior)
                parts.append(score); values.append(value); counts.append(len(key))
                record_ids.extend([record]*len(key)); positions.append(value_positions)
                prediction_views.append(view); prediction_slots.append(slot)
        scores = torch.cat(parts, dim=-1)
        all_values = torch.cat(values)
        record_ids = torch.tensor(record_ids, dtype=torch.long)
        weights = torch.cat((scores, self.null_logit(q)), dim=-1).softmax(-1)[:, :-1]
        if self.suppress_older_than is not None:
            allowed = m.clock-m.birth[slots] <= self.suppress_older_than
            weights = weights*allowed[record_ids][None]
        if self.suppressed_record is not None:
            slot, version = self.suppressed_record
            if int(m.read_credit_version[slot]) != version:
                raise RuntimeError('removed native record identity changed')
            weights = weights*(slots[record_ids] != slot)[None]
        view_mass = torch.stack([w.sum(-1) for w in weights.split(counts, dim=-1)], dim=-1)
        owner = torch.tensor([slots.tolist().index(s) for s in prediction_slots])
        mass = torch.stack([view_mass[:, owner == j].sum(-1) for j in range(len(slots))], dim=-1)
        self.last_read = TransitionRead(None, None, slots[None].expand(len(query), -1), mass)
        self.position_read = dict(weights=weights, record_ids=record_ids, slots=slots,
            lengths=[2*len(e)-1 for e in encoded], value_positions=torch.cat(positions),
            null_weight=1-mass.sum(-1), prediction_weights=view_mass,
            prediction_slots=prediction_slots, prediction_views=prediction_views)
        if self.predictive_responsibility:
            self.position_read['prediction_values'] = torch.stack([
                s.softmax(-1)@v for s, v in zip(parts, values)])
        self._dual_query = self._dual_encoded = self._dual_bias = None
        return weights.to(all_values.device)@all_values

    def discard(self):
        super().discard()
        self._dual_query = self._dual_encoded = self._dual_bias = None


class NativeResponseGain(nn.Module):
    """A zero-output migration into the existing native residual space."""
    def __init__(self, device, *, width=None, learned_output=False):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros((), device=device))
        self.register_parameter('output_weight', None)
        if learned_output:
            self.enable_output_projection(width)

    def enable_output_projection(self, width):
        if self.output_weight is not None:
            raise ValueError('native output projection already enabled')
        self.output_weight = nn.Parameter(self.gain.new_zeros(width, width))
        return self.output_weight

    def forward(self, response):
        scalar = self.gain*response
        return scalar if self.output_weight is None else scalar+F.linear(response, self.output_weight)


class DirectAttentionLayerExperience(DualViewLayerExperience):
    """Bounded soft cross-attention, then ONE actual downstream execution.

    Q/K are separate learned addresses; V is the current-parameter replay of
    the full old native GDN+MLP response, already in this layer's output space.
    No vocabulary experts or separate conditional suffix states are created.
    """
    def __init__(self, *args, native_output_projection=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Copy initial coordinates, then learn the query independently. This
        # replaces (rather than adds to) the former latent-to-native matrix.
        import copy
        self.query_encoder = copy.deepcopy(self.encoder)
        width = args[0] if args else kwargs['width']
        self.inlet.projection = NativeResponseGain(kwargs['device'], width=width,
            learned_output=native_output_projection)

    def encode_query(self, query):
        return F.normalize(self.query_encoder(query.float().cpu()), dim=-1)

    def association_values(self, slot, encoded):
        return self.replayed_by_slot[slot][0].float()

    def retrieve_context(self, query, block):
        value = super().retrieve_context(query, block)
        return query*0 if self.position_read is None else value
