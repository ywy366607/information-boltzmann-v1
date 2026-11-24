"""One retrieved field co-evolves with causal language through every layer.

Facts are immutable versioned anchors. The extra recurrent coordinates are
contextual interpretations, not new observations or copies of an archive.
Every layer uses its actual shared language operators plus one shared compact
memory FFN. Work is bounded by slots*rank, independently of lifetime length.
"""
from __future__ import annotations

import math
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.causal_workspace_scan import causal_scan
from scripts.ib_local.quadratic_revision import diagonal_gain


class PastPresentField(nn.Module):
    def __init__(self, width, latent, slots=8, rank=32):
        super().__init__()
        if not 1 <= rank <= width or slots < 1:
            raise ValueError('invalid bounded co-evolution workspace')
        self.slots, self.rank = slots, rank
        self.key = nn.Linear(latent, rank, bias=False)
        self.bind = nn.Linear(4*latent, rank, bias=False)
        self.memory_norm = nn.RMSNorm(rank)
        self.memory_gate = nn.Linear(2*rank, 1)
        self.memory_ffn = nn.Sequential(nn.Linear(rank, 2*rank, bias=False), nn.SiLU(),
                                        nn.Linear(2*rank, rank, bias=False))
        self.read_gate = nn.Linear(rank, 1)
        self.null = nn.Linear(rank, 1)
        # Nonzero ordinary initialization: no zero multiplier blocks learning.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.register_buffer('anchors', torch.zeros(4, slots, width))
        self.register_buffer('slots_id', torch.zeros(slots, dtype=torch.long))
        self.register_buffer('versions', torch.full((slots,), -1, dtype=torch.long))
        self.register_buffer('valid', torch.zeros(slots, dtype=torch.bool))
        self.register_buffer('preference', torch.zeros(slots))
        # Explicit scalar permits read-only no-field interventions, including
        # CUDA graphs, without changing workset identity or other parameters.
        self.register_buffer('enabled', torch.ones(()))

    @property
    def state_size(self):
        return self.slots*self.rank

    def prepare(self, owner, batch, length):
        cached = getattr(self, '_decode_projection_cache', None) if not torch.is_grad_enabled() else None
        if cached is None:
            encoded = owner.encoder(self.anchors)
            keys = F.normalize(self.key(encoded[0]), dim=-1)
            values = self.bind(encoded.permute(1, 0, 2).flatten(1)).tanh()
        else:
            keys, values = cached
        values = values*self.valid[:, None]*self.enabled
        # One shared receipt for the entire layer/time computation: its
        # derivative is the aggregate responsibility of this frozen workset.
        bias = self.preference.clone()
        memory = owner.memory
        if torch.is_grad_enabled() and memory._read_credit_active and memory.read_credit_enabled:
            from scripts.ib_local.navigation_read_credit import ReadParticipation
            bias.requires_grad_()
            memory._read_participation.append(ReadParticipation(
                bias, self.slots_id.clone(), self.versions.clone(), self.valid.clone()))
            memory._read_credit_entries += self.slots
        incoming = values[None, None].expand(batch, length, -1, -1)
        from scripts.ib_local.native_replay_batch import for_model
        panel = for_model(owner)
        if panel is not None:
            incoming = incoming*panel.valid[:, None, :, None]
        return incoming, keys, bias

    def exchange(self, layer, hidden, incoming, previous, keys, bias):
        """A causal depth/time cell, before the ordinary language block.

        Affine recurrence coefficients depend on current language and the
        previous-depth, same-time field; never on a future token. Convex time
        updates bound the contextual field while preserving differentiability.
        """
        from scripts.ib_local.native_replay_batch import for_field
        panel = for_field(self)
        valid = self.valid[None, None] if panel is None else panel.valid[:, None]
        if (panel is None and hidden.is_cuda and not torch.is_grad_enabled() and
                getattr(self, '_decode_fused_exchange', False)):
            from scripts.ib_local.coev_decode_fused import exchange
            fused = exchange(self, layer, hidden, incoming, previous, keys, bias)
            if fused is not None:
                return fused
        r = self.rank
        h = layer.norm(hidden)
        q = F.linear(h, layer.q_proj.weight[:r])
        write = F.linear(h, layer.v_proj.weight[:r]).tanh()
        normed = self.memory_norm(incoming)
        shared_key = F.linear(normed, layer.k_proj.weight[:r, :r])
        score = (F.normalize(q, dim=-1)[:, :, None] * (
            keys[None, None]+F.normalize(shared_key, dim=-1))).sum(-1)/math.sqrt(r)
        score = score+bias[None, None]
        score = score.masked_fill(~valid, -1e4)
        weights = torch.cat((score, self.null(q)), -1).softmax(-1)[..., :-1]
        weights = weights*valid*self.enabled
        # Shared FFN restricted to the bounded workspace subspace. It uses
        # the actual language matrices, rather than a private memory backbone.
        shared = 0.
        if layer.mlp is not None:
            n = min(2*r, layer.mlp['gate'].weight.shape[0])
            shared = F.linear(F.silu(F.linear(normed, layer.mlp['gate'].weight[:n, :r])) *
                F.linear(normed, layer.mlp['up'].weight[:n, :r]), layer.mlp['down'].weight[:r, :n])
        proposal = (incoming+shared+self.memory_ffn(normed)+
                    weights[..., None]*write[:, :, None]).tanh()
        query = q[:, :, None].expand_as(incoming)
        beta = diagonal_gain(self.memory_gate(torch.cat((query, normed), -1)))
        beta = beta*valid[..., None]*self.enabled
        workspace = causal_scan((1-beta).expand_as(proposal), beta*proposal,
                                previous.reshape(hidden.shape[0], self.slots, r))
        recalled = (weights[..., None]*workspace).sum(-2)
        delta = F.linear(recalled, layer.o_proj.weight[:, :r]) * self.read_gate(q).sigmoid()
        if getattr(self, '_decode_no_diagnostics', False) and not torch.is_grad_enabled():
            return hidden+delta, workspace, {}
        return hidden+delta, workspace, dict(
            workspace_write=beta.detach().mean(), workspace_read_mass=weights.detach().sum(-1).mean(),
            workspace_residual_rms=delta.detach().square().mean().sqrt())


@torch.no_grad()
def refresh_workset(engine):
    """Select once at a real event boundary; never mutate facts during a draft.

    Stable (slot,version) identity transports recurrent interpretations. A
    recycled address cannot inherit the previous occupant's working state.
    """
    owner = engine.model
    field, memory = owner.body.coevolution, owner.memory
    if memory.pending_count:
        query = memory.pending[-1:]
    else:
        query = memory.source[:1]*0
    read = memory.read(query, owner.encoder)
    slots = read.slots[0, :field.slots]
    versions = memory.read_credit_version[slots]
    valid = memory.occupied[slots]
    same = ((slots[:, None] == field.slots_id[None]) &
            (versions[:, None] == field.versions[None]) & field.valid[None] & valid[:, None])
    # Stable identities, not similarity, are used for state transport.
    index = same.to(torch.int64).argmax(-1)
    retained = same.any(-1)
    physical = owner.base.layers[0].state_size
    old = engine.recurrent[..., physical:].reshape(
        len(owner.base.layers), engine.recurrent.shape[1], field.slots, field.rank)
    work = old[:, :, index]*retained[None, None, :, None]
    engine.recurrent = torch.cat((engine.recurrent[..., :physical], work.flatten(2)), -1)
    field.slots_id.copy_(slots); field.versions.copy_(versions); field.valid.copy_(valid)
    for i, name in enumerate(('source', 'action', 'successor', 'long_successor')):
        field.anchors[i].copy_(getattr(memory, name)[slots])
    from scripts.ib_local.experience_consolidation import bounded_preference, consolidates_reads
    preference = bounded_preference(memory) if consolidates_reads(memory) else memory.read_credit_mean.tanh()
    field.preference.copy_(preference[slots])
    return dict(valid=int(valid.sum()), retained=int(retained.sum()),
                slots=slots.cpu().tolist(), versions=versions.cpu().tolist())


def _step(self, parent_step, *args, **kwargs):
    receipt = refresh_workset(self)
    # The wrapper is installed once, above the existing complete engine.
    result = parent_step(self, *args, **kwargs)
    field = self.model.body.coevolution
    result['past_present'] = dict(receipt, layers=len(self.model.base.layers), rank=field.rank,
        state_bytes=self.recurrent[..., -field.state_size:].numel()*self.recurrent.element_size(),
        workspace_rms=float(self.recurrent[..., -field.state_size:].square().mean().sqrt()))
    return result


def configure_coevolution(engine, spec, *, initialize=True):
    if spec.get('schema') != 1 or spec.get('layers') != 'all':
        raise ValueError('unknown all-layer causal co-evolution contract')
    owner, body = engine.model, engine.model.body
    if hasattr(body, 'coevolution'):
        raise ValueError('co-evolution already attached')
    if not all(l.conv_kernel and l.state_size == owner.base.layers[0].state_size
               for l in owner.base.layers):
        raise ValueError('packed co-evolution requires homogeneous convolutional GDN states')
    old_parameters = list(engine.params)
    w = owner.encoder.weight
    with torch.random.fork_rng(devices=[w.device] if w.is_cuda else []):
        torch.manual_seed(spec.get('seed', 151132))
        field = PastPresentField(owner.memory.width, owner.encoder.out_features,
            spec['slots'], spec['rank']).to(device=w.device, dtype=w.dtype)
    body.coevolution = field
    body._coev_owner = weakref.ref(owner)
    extra = list(field.parameters())
    engine.optimizer.add_param_group(dict(params=extra, lr=engine.optimizer.param_groups[0]['lr']))
    engine.params = list(owner.parameters())
    engine.groups = (*engine.groups, extra)
    if initialize:
        old_size = owner.base.layers[0].state_size
        old_shape = tuple(engine.recurrent.shape)
        def transport(v):
            if torch.is_tensor(v) and tuple(v.shape) == old_shape:
                return torch.cat((v, v.new_zeros(*v.shape[:-1], field.state_size)), -1)
            if isinstance(v, (tuple, list)) and len(v) == len(old_parameters) and all(
                    x is None or torch.is_tensor(x) for x in v):
                saved = {id(p): x for p, x in zip(old_parameters, v)}
                return type(v)(saved.get(id(p)) for p in engine.params)
            if isinstance(v, dict):
                return {k: transport(x) for k, x in v.items()}
            if isinstance(v, (tuple, list)):
                return type(v)(transport(x) for x in v)
            return v
        for name, value in list(vars(engine).items()):
            if name not in ('model', 'optimizer', 'params', 'groups', 'decision', 'world', 'table'):
                setattr(engine, name, transport(value))
        assert engine.recurrent.shape[-1] == old_size+field.state_size
    parent = type(engine)
    def step(self, *args, **kwargs):
        return _step(self, parent.step, *args, **kwargs)
    engine.__class__ = type('PastPresentEngine', (parent,), {'step': step})
    engine.past_present_schema = 1
    from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
    engine.decision = ScopedDecisionGraph(owner, capture=w.is_cuda)
    from scripts.ib_local.navigation_body_growth import validate_pending_participation
    validate_pending_participation(engine)
    return dict(added_parameters=sum(p.numel() for p in extra),
                total_parameters=sum(p.numel() for p in owner.parameters()),
                layers=len(owner.base.layers), slots=field.slots, rank=field.rank,
                physical_state_preserved=True, old_optimizer_preserved=True,
                state_bytes=len(owner.base.layers)*field.state_size*w.element_size())
