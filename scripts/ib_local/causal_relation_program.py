"""Typed, bounded conditional mechanisms shared by real and imagined navigation.

The neural generators belong to the existing body. A relation stores a stable
identity, a condition and versioned references to real evidence, not a private
predictor. All local mechanisms use the same conjugate linear-residual update
in the current encoder coordinates. Posterior hypotheses are alternative
operators for ONE bound relation, not alternative neighbouring episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import torch
from torch import nn
from torch.nn import functional as F


class Port(str, Enum):
    CONDITION = 'condition'
    ACTION = 'action'
    MOTHER = 'mother'
    DECISION = 'decision'
    VALUE = 'value'


@dataclass(frozen=True)
class Operation:
    name: str
    inputs: tuple[Port, ...]
    output: Port


TRANSITION = Operation('body_transition', (Port.CONDITION, Port.ACTION), Port.DECISION)
MOTHER = Operation('mother_response', (Port.CONDITION, Port.ACTION), Port.MOTHER)
TRANSPORT = Operation('ambient_transport', (Port.CONDITION, Port.ACTION, Port.MOTHER), Port.DECISION)
VALUE = Operation('existing_continuation', (Port.DECISION,), Port.VALUE)


@dataclass(frozen=True)
class Variable:
    identity: str
    port: Port


@dataclass(frozen=True)
class Call:
    operation: Operation
    arguments: tuple[Variable, ...]
    result: Variable


@dataclass(frozen=True)
class PortValue:
    tensor: torch.Tensor
    metadata: dict


def execute(calls, bindings, generators):
    """Typed string-diagram execution with explicit shared variable identity.

    The same variable can be constrained by two mechanisms. Both predictions
    are retained; an actually bound value takes precedence for continuation.
    A missing binding is a real open role, not a zero vector or a silent act.
    """
    values, trace = dict(bindings), []
    for call in calls:
        if tuple(v.port for v in call.arguments) != call.operation.inputs or call.result.port != call.operation.output:
            raise ValueError('mechanism composition has incompatible role types')
        if any(v not in values for v in call.arguments):
            raise ValueError('unbound premise in relation program')
        prediction = generators[call.operation.name](*(values[v] for v in call.arguments))
        if not isinstance(prediction, PortValue):
            raise TypeError('shared generator must return a typed port value')
        trace.append((call, prediction))
        values.setdefault(call.result, prediction)
    return values, trace


def check_program(operations, available):
    """Check the actual data dependencies of serial/parallel composition."""
    known = set(available)
    for op in operations:
        if not set(op.inputs) <= known:
            raise ValueError(f'{op.name} requires {set(op.inputs) - known}')
        known.add(op.output)
    return known


@dataclass(frozen=True)
class Evidence:
    source: torch.Tensor
    action: torch.Tensor
    outcome: torch.Tensor
    kind: Port
    observed: bool
    slot: int = -1
    stamp: int = -1
    encoded: bool = False
    mother: torch.Tensor | None = None
    fact_id: str = ''


@dataclass(frozen=True)
class RelationState:
    identity: int
    generation: int
    version: int
    index: int
    slots: torch.Tensor
    stamps: torch.Tensor
    valid: torch.Tensor
    extra: tuple[Evidence, ...] = ()
    release: int = -1
    program_weights: torch.Tensor | None = None
    revision_plan: dict | None = None
    scope: dict | None = None
    binding: Variable | None = None
    constraints: tuple[Evidence, ...] = ()
    organization: RelationState | None = None
    variable_uses: tuple = ()
    evidence_bound: bool = False
    retention_use: object = None
    retention_trace: dict | None = None


def revise(state: RelationState, evidence: Evidence, *, model=None) -> RelationState:
    """The same U for an observed or hypothetical labelled transition.

    This functional update changes no live bank. Committing references is a
    separate observed-only operation, after the real experience is stored.
    Each bounded preview adds at most one observation per output port.
    """
    if evidence.kind not in (Port.MOTHER, Port.DECISION):
        raise ValueError('only a typed observed outcome can revise a mechanism')
    if any(e.kind == evidence.kind for e in state.extra):
        raise ValueError('one extra evidence item per typed port and preview')
    if (evidence.source.ndim != 2 or evidence.source.shape != evidence.action.shape or
            evidence.outcome.ndim != 2 or len(evidence.outcome) != len(evidence.source) or
            not all(bool(torch.isfinite(t).all()) for t in
                    (evidence.source, evidence.action, evidence.outcome))):
        raise ValueError('finite matched role-bound evidence required')
    if state.binding is not None:
        if model is None:
            raise ValueError('shared binding revision requires the current body coordinates')
        from scripts.ib_local.shared_unknown_bindings import retain_constraints
        from scripts.ib_local.relation_retention_use import revise_raw
        kept,use,trace = revise_raw(model,state,evidence)
        organization = getattr(state,'organization',None)
        if getattr(model,'joint_relation_update_schema',None)==1 and organization is not None:
            organization = revise(organization,evidence,model=model)
        return replace(state, constraints=kept, extra=state.extra + (evidence,),organization=organization,
                       retention_use=use,retention_trace=trace)
    if getattr(model,'joint_relation_update_revision',1)>=3 and state.evidence_bound:
        from scripts.ib_local.shared_unknown_bindings import retain_constraints
        from scripts.ib_local.relation_retention_use import revise_raw
        kept,use,trace = revise_raw(model,state,evidence)
        updated=replace(state,constraints=kept,extra=state.extra+(evidence,),
                        retention_use=use,retention_trace=trace)
        if state.program_weights is not None:
            from scripts.ib_local.generative_relations import organize
            # The whole U, including pooled support compression, selects with
            # pre-event use. `updated` already carries the successor moments.
            if getattr(model,'relation_retention_use_schema',2)>=2:
                updated=organize(model,updated,evidence,selection_state=state)
            else:
                updated=organize(model,updated,evidence)  # exact legacy replay
        return updated
    valid = state.valid.clone()
    key = lambda e: (e.observed, e.slot, e.stamp)
    extra_events = {key(e) for e in state.extra}
    if key(evidence) not in extra_events and int(valid.sum()) + len(extra_events) >= len(valid):
        if state.release < 0 or not bool(valid[state.release]):
            raise ValueError('a full relation requires an evidence-supported release choice')
        valid[state.release] = False
    updated = replace(state, valid=valid, extra=state.extra + (evidence,))
    if model is not None and state.program_weights is not None:
        from scripts.ib_local.generative_relations import organize
        updated = organize(model, updated, evidence)
    return updated


def roles(source, action):
    """Factorized TPR: fixed condition/action role axes, continuous fillers."""
    if source.shape != action.shape or source.ndim != 2:
        raise ValueError('matched condition/action batches required')
    return torch.stack((source, action - source), -2)


def role_features(source, action, mother=None):
    bound = roles(source, action)
    if mother is not None:
        if mother.shape != source.shape:
            raise ValueError('mother and condition must use the same latent space')
        bound = torch.cat((bound, (mother - source)[:, None]), -2)
    return F.normalize(bound.flatten(1), dim=-1)


def relation_features(model, source, action, mother=None, *, legacy=False):
    if not legacy and hasattr(model, 'relation_basis'):
        return model.relation_basis(source, action, mother)
    return role_features(source, action, mother)


def posterior(x, y, query, ridge, *, noise_prior=None):
    """Normal-inverse-gamma residual mechanism, solved in evidence space.

    Per output: B|s2 ~ N(0,s2/ridge I), s2 ~ InvGamma(1,1).
    The unit prior is in the body's SIGReg latent coordinates. The inherited
    ridge remains explicit. All variance estimates below use PRIOR evidence;
    the outcome to be scored is never used to set its own noise floor.
    """
    ridge_scalar = (getattr(ridge, '_cached_float', None) if torch.is_tensor(ridge) else ridge)
    if ridge_scalar is None:
        ridge_scalar = float(ridge.detach())
    if ridge_scalar <= 0 or x.ndim != 2 or y.ndim != 2 or query.ndim != 2:
        raise ValueError('positive precision and 2D tensors required')
    if len(x) != len(y) or x.shape[1] != query.shape[1]:
        raise ValueError('incompatible role-bound evidence')
    from scripts.ib_local.quadratic_revision import evidence_correction
    kernel, coefficient, mean, projected, factor, gram = evidence_correction(
        x, y, query, ridge, return_factor=True, return_gram=True)
    leverage = ((query.square().sum(-1) - (query @ x.T * projected.T).sum(-1)) / ridge).clamp_min(0)
    alpha = 1. + len(x) / 2.
    residual_sum = (y.square().sum(0) - (y * (gram @ coefficient)).sum(0))
    if noise_prior is None:
        beta = (1. + .5 * residual_sum).clamp_min(torch.finfo(y.dtype).eps)
        noise = (beta / alpha).detach()
    else:
        if noise_prior.shape != y.shape[1:] or (
                not getattr(noise_prior, '_verified_positive', False) and not bool((noise_prior > 0).all())):
            raise ValueError('one positive shared prior scale per observed output required')
        beta = noise_prior + .5 * residual_sum.detach().clamp_min(0)
        noise = beta / alpha
    # X A^-1 X^T = K0 (K0+ridge I)^-1, as used above.
    scale2 = (1. + leverage[:, None]) * noise
    return dict(mean=mean, scale2=scale2, noise=noise, leverage=leverage,
                df=2*alpha, kernel=kernel, x=x, y=y, coefficient=coefficient, ridge=ridge,
                factor=factor)


def log_score(distribution, target):
    """Prequential Student-t score in the forecast's own latent coordinates."""
    if 'scale_matrix' in distribution:
        from scripts.ib_local.joint_successor import score
        return score(distribution,target)
    df = torch.as_tensor(distribution['df'],device=target.device,dtype=target.dtype)
    variance = distribution['scale2'].clamp_min(torch.finfo(target.dtype).eps)
    error = (target - distribution['mean']).square() / variance
    return (torch.lgamma((df+1)/2) - torch.lgamma(df/2) - .5*torch.log(df*math.pi*variance)
            - (df+1)/2*torch.log1p(error/df)).mean(-1)


def mechanism_hypotheses(distribution, query, directions, observation_directions):
    """Shared posterior operator draws, conditional on the fitted noise scale.

    A prior draw and a fictitious observation-noise draw are conditioned using
    the same dual solve. All actions therefore see the SAME operator draw.
    These are conditional Gaussian quadrature hypotheses, not calibrated
    discrete world explanations. Their likelihood is checked on real replies.
    """
    x, kernel = distribution['x'], distribution['kernel']
    prior = directions[:, :query.shape[1]] / math.sqrt(distribution['ridge'])
    if 'fact_identities' in distribution:
        from scripts.ib_local.identity_preview import observation_noise
        eps = observation_noise(distribution['fact_identities'], observation_directions)
    else:
        eps = observation_directions[:, :len(x)]
    latent = torch.einsum('nf,hfd->hnd', x, prior) + eps
    factor = distribution.get('factor')
    solved = (torch.linalg.solve(kernel[None], latent) if factor is None else
              torch.linalg.lu_solve(factor[0][None], factor[1][None], latent))
    conditional = prior - torch.einsum('fn,hnd->hfd', x.T, solved)
    deviation = torch.einsum('bf,hfd->bhd', query, conditional) * distribution['noise'].sqrt()
    return distribution['mean'][:, None] + deviation


def hypothesis_evidence(means, noise, observed):
    """Posterior weights for the same relation hypotheses under an outcome."""
    variance = noise.clamp_min(torch.finfo(means.dtype).eps)
    logits = -.5*((observed[..., None, :] - means).square()/variance).sum(-1)
    return logits.log_softmax(-1)


def response_hypotheses(model, distribution):
    """Shared world-response draws for every depth of finite-U imagination."""
    from scripts.ib_local import joint_response
    bank = model.causal_relations
    if joint_response.enabled(model):
        return joint_response.hypotheses(distribution, bank.directions, bank.observation_directions)
    centers = mechanism_hypotheses(distribution, distribution['query'],
                                  bank.directions, bank.observation_directions)
    points = centers+distribution['noise'].sqrt()*bank.observation_directions[:, -1]
    return centers, points, None


class RelationBank(nn.Module):
    """Fixed-size relation identities, fact references, and quadrature state."""
    def __init__(self, width, latent, capacity, evidence_budget, hypotheses, device, dtype):
        super().__init__()
        self.capacity, self.evidence_budget = capacity, evidence_budget
        self.register_buffer('condition', torch.zeros(capacity, width, device=device, dtype=dtype))
        self.register_buffer('identity', torch.zeros(capacity, device=device, dtype=torch.long))
        self.register_buffer('generation', torch.zeros_like(self.identity))
        self.register_buffer('version', torch.zeros_like(self.identity))
        self.register_buffer('last_use', torch.zeros_like(self.identity))
        self.register_buffer('next_identity', torch.ones((), device=device, dtype=torch.long))
        self.register_buffer('slots', torch.zeros(capacity, evidence_budget, device=device, dtype=torch.long))
        self.register_buffer('stamps', torch.zeros_like(self.slots))
        self.register_buffer('valid', torch.zeros_like(self.slots, dtype=torch.bool))
        # Saved, antithetic draws leave actor/world RNG streams unchanged.
        with torch.random.fork_rng(devices=[device] if torch.device(device).type == 'cuda' else []):
            torch.manual_seed(11010)
            half = (hypotheses+1)//2
            w = torch.randn(half, 3*latent, latent, device=device, dtype=dtype)
            e = torch.randn(half, evidence_budget+2, latent, device=device, dtype=dtype)
        self.register_buffer('directions', torch.cat((w, -w), 0)[:hypotheses])
        self.register_buffer('observation_directions', torch.cat((e, -e), 0)[:hypotheses])

    def binding_address(self, model, source):
        from scripts.ib_local.preview_runtime import read, tensor_key
        live = self.identity > 0
        if hasattr(self, 'applicable'):
            live = live & self.applicable(source)
        keys = read(model, 'preview_relation_keys',
            (tensor_key(self.condition), tuple(tensor_key(p) for p in model.encoder.parameters())),
            lambda: F.normalize(model.encoder(self.condition), dim=-1), anchors=(self,))
        score = (F.normalize(model.encoder(source), dim=-1) @ keys.T)[0]
        score = score.masked_fill(~live, -2)
        value, index = score.max(0)
        return bool(live.any()) and float(value) >= model.memory.threshold, int(index), value

    def binding_address_batch(self, model, sources):
        from scripts.ib_local.preview_runtime import read, tensor_key
        live = (self.identity > 0)[None]
        if hasattr(self, 'applicable_batch'):
            live = live & self.applicable_batch(sources)
        elif hasattr(self, 'applicable'):
            live = torch.stack([live[0] & self.applicable(sources[i:i+1]) for i in range(len(sources))])
        else:
            live = live.expand(len(sources), -1)
        keys = read(model, 'preview_relation_keys',
            (tensor_key(self.condition), tuple(tensor_key(p) for p in model.encoder.parameters())),
            lambda: F.normalize(model.encoder(self.condition), dim=-1), anchors=(self,))
        scores = F.normalize(model.encoder(sources), dim=-1) @ keys.T
        scores = scores.masked_fill(~live, -2)
        values, indices = scores.max(-1)
        any_live = live.any(-1)
        matched_t = any_live & (values >= model.memory.threshold)
        matched_list = matched_t.cpu().tolist()
        idx_list = indices.cpu().tolist()
        return [(bool(m), int(i), values[k]) for k, (m, i) in enumerate(zip(matched_list, idx_list))]

    @torch.no_grad()
    def bind(self, model, source):
        m, enc = model.memory, model.encoder
        matched, i, _ = self.binding_address(model, source)
        if not matched:
            free = torch.nonzero(self.identity == 0).flatten()
            if hasattr(self, 'applicable'):
                archive = getattr(model, '_causal_archive', None)
                protected = getattr(self, '_protected_identities', set())
                if archive is not None:
                    protected = protected | archive._protected()[1]
                candidates = [j for j in range(self.capacity) if int(self.identity[j]) not in protected]
                if not len(free) and not candidates:
                    raise RuntimeError('all relation slots are pinned by unfinished causal events')
                i = int(free[0]) if len(free) else min(candidates, key=lambda j: int(self.last_use[j]))
                if archive is not None and int(self.identity[i]) > 0:
                    archive.sync()
            else:
                i = int(free[0]) if len(free) else int(self.last_use.argmin())
            if int(self.next_identity) == torch.iinfo(torch.long).max:
                raise OverflowError('relation identity exhausted')
            self.condition[i].copy_(source[0])
            self.identity[i] = self.next_identity
            self.next_identity.add_(1); self.generation[i].add_(1); self.version[i] = 0
            # Existing real evidence supplies the initial mechanism. Selection
            # supplies applicability, not competing-hypothesis probabilities.
            eligible = m.occupied & (m.count == 1) & m.mother_valid & (m.mother_stamp == m.last_write)
            read = m.read_encoded(enc(source), enc, eligible=eligible)
            slots = read.slots[0, :self.evidence_budget]
            self.slots[i].copy_(slots); self.stamps[i].copy_(m.last_write[slots])
            self.valid[i].copy_(eligible[slots])
            if hasattr(self,'support_episode_ids'):
                archive=model._causal_archive
                identities=[archive.hot[str(int(slot))]['id'] for slot in slots if bool(eligible[slot])]
                from scripts.ib_local.relation_evidence import install_episode_ids
                install_episode_ids(self,model,i,identities)
        self.last_use[i] = m.clock
        self.valid[i] &= m.occupied[self.slots[i]] & (m.count[self.slots[i]] == 1)
        self.valid[i] &= (m.last_write[self.slots[i]] == self.stamps[i])
        self.valid[i] &= m.mother_valid[self.slots[i]] & (m.mother_stamp[self.slots[i]] == self.stamps[i])
        state = self.state(i)
        state = replace(state, release=release_choice(model, state))
        if getattr(model,'joint_relation_update_schema',None)==1:
            from scripts.ib_local.joint_relation_update import relation_variables
            state = relation_variables(model,state)
        return state

    def state(self, index):
        return RelationState(int(self.identity[index]), int(self.generation[index]),
            int(self.version[index]), index, self.slots[index].clone(),
            self.stamps[index].clone(), self.valid[index].clone())

    @torch.no_grad()
    def commit(self, model, state, evidence):
        if not evidence.observed or evidence.encoded:
            raise ValueError('imagined/encoded outcomes cannot become world facts')
        i, m = state.index, model.memory
        if (int(self.identity[i]) != state.identity or int(self.generation[i]) != state.generation
                or int(self.version[i]) != state.version):
            return False
        s = evidence.slot
        if (s < 0 or not bool(m.occupied[s]) or int(m.count[s]) != 1 or
                int(m.last_write[s]) != evidence.stamp or not bool(m.mother_valid[s]) or
                int(m.mother_stamp[s]) != evidence.stamp):
            return False
        valid = state.valid & m.occupied[state.slots] & (m.count[state.slots] == 1)
        valid &= m.last_write[state.slots] == state.stamps
        duplicate = valid & (self.slots[i] == s) & (self.stamps[i] == evidence.stamp)
        if not bool(duplicate.any()):
            # Real commit and imaginary revision use precisely the same
            # fixed-budget U, including which old evidence is released.
            valid = revise(replace(state, valid=valid), evidence).valid
            free = torch.nonzero(~valid).flatten()
            j = int(free[0])
            self.slots[i,j] = s; self.stamps[i,j] = evidence.stamp; valid[j] = True
        self.valid[i].copy_(valid)
        self.version[i].add_(1)
        self.last_use[i] = m.clock
        return True


def real_evidence(model, state, kind, exclude=-1):
    m = model.memory
    valid = state.valid & m.occupied[state.slots] & (m.count[state.slots] == 1)
    valid &= (m.last_write[state.slots] == state.stamps) & (state.slots != exclude)
    valid &= m.mother_valid[state.slots] & (m.mother_stamp[state.slots] == state.stamps)
    if kind == Port.DECISION:
        valid &= m.internal_valid[state.slots] & (m.internal_stamp[state.slots] == state.stamps)
    elif kind != Port.MOTHER:
        raise ValueError('unknown outcome port')
    return state.slots[valid]


@torch.no_grad()
def release_choice(model, state):
    """D-optimal release on actual bound condition/action/outcome distinctions.

    The least-leverage old observation contributes the least log determinant
    of supported role geometry. Age alone does not decide retention. The
    choice is frozen in the execution receipt, so real/shadow U agree.
    """
    from scripts.ib_local.preview_runtime import read, tensor_key
    m, enc = model.memory, model.encoder
    def _compute():
        if not bool(state.valid.all()):
            return -1
        x = relation_features(model, enc(m.source[state.slots]), enc(m.action[state.slots]),
                          enc(m.mother_next[state.slots]))
        x_cpu = x.detach().cpu()
        gram = x_cpu @ x_cpu.T
        from scripts.ib_local.learnable_precision_scales import effective_ridge
        precision = effective_ridge(model, getattr(model, 'causal_relation_ridge', .1))
        prec_f = getattr(precision, '_cached_float', None) if torch.is_tensor(precision) else float(precision)
        if prec_f is None:
            prec_f = float(precision.detach())
        eye = torch.eye(len(x_cpu), dtype=x_cpu.dtype)
        hat = gram @ torch.linalg.solve(gram + prec_f * eye, eye)
        return int(hat.diag().argmin())
    key = (tensor_key(state.slots), tensor_key(state.valid),
           tensor_key(m.source), tensor_key(m.action), tensor_key(m.mother_next))
    return read(model, 'preview_release_choice', key, _compute, anchors=(m, state.slots, state.valid))


def _bound_evidence_design(model, facts, kind, transport, legacy):
    """Same ordered facts, batched shared encoders/operators, no cached graph."""
    if not torch.is_grad_enabled():
        from scripts.ib_local.preview_runtime import _scope, tensor_key
        active = _scope.get()
        if active is not None and active[0] is model:
            cache = active[1]
            key = ('bound_evidence_design', kind, transport, legacy,
                   tuple((tensor_key(e.source), tensor_key(e.action), tensor_key(e.outcome),
                          e.encoded, None if not transport else tensor_key(e.mother)) for e in facts))
            entry = cache.get(key)
            if entry is not None:
                return entry[0]
            res = _bound_evidence_design_compute(model, facts, kind, transport, legacy)
            cache[key] = (res, tuple(facts))
            return res
    return _bound_evidence_design_compute(model, facts, kind, transport, legacy)


def _bound_evidence_design_compute(model, facts, kind, transport, legacy):
    enc = model.encoder
    es = enc(torch.cat([e.source for e in facts]))
    ea = enc(torch.cat([e.action for e in facts]))

    def outcomes(name):
        raw = [getattr(e, name) for e in facts if not e.encoded]
        converted = [] if not raw else list(enc(torch.cat(raw)).split([len(v) for v in raw]))
        cursor, pieces = 0, []
        for e in facts:
            if e.encoded:
                pieces.append(getattr(e, name))
            else:
                pieces.append(converted[cursor]); cursor += 1
        return torch.cat(pieces)

    ey = outcomes('outcome')
    if transport:
        em = outcomes('mother')
        return relation_features(model, es, ea, em, legacy=legacy), ey-em
    ex = relation_features(model, es, ea, legacy=legacy)
    prior = model.long_predictor(torch.cat((es, ea), -1))
    if kind == Port.MOTHER:
        prior = prior+model.mother_from_boundary(prior)
    return ex, ey-prior


def mechanism(model, state, source, action, *, kind=Port.MOTHER, mother=None,
              ridge=.1, momentum=None, exclude=-1, exclude_fact='', legacy=False):
    """Execute a role-bound body mechanism from its real and shadow evidence."""
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    ridge = effective_ridge(model, ridge)
    enc, m = model.encoder, model.memory
    canonical=getattr(model,'joint_relation_update_revision',1)>=3
    linked = getattr(model,'joint_relation_update_schema',None)==1 and bool(state.variable_uses)
    slots = (real_evidence(model, state, kind, exclude=exclude) if not canonical and state.binding is None and not linked
             else torch.empty(0, device=m.source.device, dtype=torch.long))
    s, a = enc(source), enc(action)
    transport = mother is not None
    if transport and kind != Port.DECISION:
        raise ValueError('only mother-to-decision transport consumes a mother port')
    if transport:
        query = relation_features(model, s, a, mother, legacy=legacy)
        base = mother
    else:
        query = relation_features(model, s, a, legacy=legacy)
        base = model.long_predictor(torch.cat((s, a), -1))
        if momentum is not None:
            base = base + model.momentum_to_boundary(momentum)
        if kind == Port.MOTHER:
            base = base + model.mother_from_boundary(base)
    if canonical:
        # Canonical evidence is supplied by identified constraints below.
        # Its legacy hot-slot prefix is exactly empty; avoid running shared
        # neural operators on zero rows before concatenating the real facts.
        x = query[:0]
        y = base[:0]
    else:
        old_s, old_a = enc(m.source[slots]), enc(m.action[slots])
        if transport:
            old_mother = enc(m.mother_next[slots])
            x = relation_features(model, old_s, old_a, old_mother, legacy=legacy)
            y = enc(m.internal_next[slots]) - old_mother
        else:
            x = relation_features(model, old_s, old_a, legacy=legacy)
            old_base = model.long_predictor(torch.cat((old_s, old_a), -1))
            if kind == Port.MOTHER:
                old_base = old_base + model.mother_from_boundary(old_base)
            target = m.mother_next[slots] if kind == Port.MOTHER else m.internal_next[slots]
            y = enc(target) - old_base
    shared_count = 0
    if canonical:
        from scripts.ib_local.relation_evidence import mechanism_evidence
        facts=mechanism_evidence(model,state,kind,exclude=exclude,exclude_fact=exclude_fact)
    else:facts=state.constraints+state.extra
    accepted = []
    for evidence in facts:
        if evidence.kind != kind or (transport and evidence.mother is None):
            continue
        if canonical:
            from scripts.ib_local.relation_evidence import excludes_evidence
            if excludes_evidence(model,evidence,exclude=exclude,exclude_fact=exclude_fact):continue
        else:
            if exclude_fact and evidence.fact_id == exclude_fact:continue
            if exclude >= 0 and evidence.slot == exclude:continue
        accepted.append(evidence)
    mech_cache = None
    mech_key = None
    if (canonical and not torch.is_grad_enabled() and not transport and kind == Port.DECISION):
        from scripts.ib_local.preview_runtime import _scope, tensor_key
        active = _scope.get()
        if active is not None and active[0] is model:
            mech_cache = active[1]
            r_id = getattr(ridge, '_cached_float', None) if torch.is_tensor(ridge) else ridge
            mech_key = ('preview_direct_mechanism', legacy,
                        tuple((tensor_key(e.source), tensor_key(e.action), tensor_key(e.outcome), e.encoded)
                              for e in accepted),
                        tensor_key(source), tensor_key(action),
                        None if momentum is None else tensor_key(momentum), r_id)
            cached_m = mech_cache.get(mech_key)
            if cached_m is not None:
                return cached_m[0]
    if accepted:
        ex, residual = _bound_evidence_design(model, accepted, kind, transport, legacy)
        x, y = torch.cat((x, ex)), torch.cat((y, residual))
        shared_count = len(accepted)
    noise_prior = (model.relation_basis.noise(mother=kind==Port.MOTHER, transport=transport)
                   if hasattr(model,'relation_basis') and not legacy else None)
    result = posterior(x, y, query, ridge, noise_prior=noise_prior)
    result['mean'] = base + result['mean']
    result.update(base=base, query=query, slots=slots, bound_evidence=shared_count,
                  basis_schema=0 if legacy or not hasattr(model,'relation_basis') else 1)
    if canonical and getattr(model, 'identity_preview_revision', 0) >= 1:
        from scripts.ib_local.identity_preview import fact_identity
        assert len(slots) == 0 and len(accepted) == len(x)
        result['fact_identities'] = tuple(fact_identity(e) for e in accepted)
    if mech_cache is not None and mech_key is not None:
        mech_cache[mech_key] = (result, (source, action, momentum, tuple(accepted)))
    return result


def _compose_fixed(model, state, source, action, *, ridge, reward_valid, momentum=None,
            mother=None, skip_value=False, skip_mother=False):
    """Mother effect -> actual ambient transport -> next action/value.

    This graph is evaluated, not merely recorded as a diagram. Independent
    candidate actions form the parallel axis. Receiver identity is the typed
    mother port; the subject is the continuously carried body.
    """
    from scripts.ib_local.navigation_internal_rollout import continuation
    condition = (state.binding if state.binding is not None and state.binding.port == Port.CONDITION
                 else Variable('condition',Port.CONDITION))
    act = (state.binding if state.binding is not None and state.binding.port == Port.ACTION
           else Variable('action',Port.ACTION))
    unknown = (state.binding if state.binding is not None and state.binding.port == Port.MOTHER
               else Variable(f'{state.identity}:mother',Port.MOTHER))
    next_state = Variable(f'{state.identity}:decision',Port.DECISION)
    gain = Variable(f'{state.identity}:value',Port.VALUE)
    if skip_mother and mother is not None:
        response = None
        decision = mechanism(model, state, source, action, kind=Port.DECISION, mother=mother, ridge=ridge)
        direct = mechanism(model, state, source, action, kind=Port.DECISION, ridge=ridge, momentum=momentum)
        if skip_value:
            value, packet = None, None
        else:
            value, packet = continuation(model, decision['mean'], reward_valid=reward_valid)
        return dict(mother=response, decision=decision, direct_decision=direct,
                    overlap_residual=decision['mean']-direct['mean'],
                    value=value, continuation=packet)
    calls = [Call(MOTHER,(condition,act),unknown),
             Call(TRANSPORT,(condition,act,unknown),next_state),
             Call(TRANSITION,(condition,act),next_state)]
    if not skip_value:
        calls.append(Call(VALUE,(next_state,),gain))
    def predict_mother(s,a):
        d=mechanism(model,state,s.tensor,a.tensor,kind=Port.MOTHER,ridge=ridge,momentum=momentum)
        return PortValue(d['mean'],d)
    def transport(s,a,m):
        d=mechanism(model,state,s.tensor,a.tensor,kind=Port.DECISION,mother=m.tensor,ridge=ridge)
        return PortValue(d['mean'],d)
    def direct_prediction(s,a):
        d=mechanism(model,state,s.tensor,a.tensor,kind=Port.DECISION,ridge=ridge,momentum=momentum)
        return PortValue(d['mean'],d)
    def value_prediction(z):
        value,packet=continuation(model,z.tensor,reward_valid=reward_valid)
        return PortValue(value,packet)
    bindings={condition:PortValue(source,{}),act:PortValue(action,{})}
    if mother is not None:
        bindings[unknown]=PortValue(mother,{'source':'bound observation or imagined response'})
    values,trace=execute(tuple(calls),bindings,{MOTHER.name:predict_mother,TRANSPORT.name:transport,
        TRANSITION.name:direct_prediction,VALUE.name:value_prediction})
    response,decision,direct=(trace[i][1].metadata for i in range(3))
    if skip_value:
        value, packet = None, None
    else:
        value,packet=values[gain].tensor,values[gain].metadata
    return dict(mother=response, decision=decision, direct_decision=direct,
                overlap_residual=decision['mean']-direct['mean'],
                value=value, continuation=packet)


def compose(model, state, source, action, *, ridge, reward_valid, momentum=None, mother=None,
            skip_mother=False):
    engine=getattr(model,'_epistemic_value_engine',None)
    carries_momentum=getattr(engine,'epistemic_value_schema',0)>=2
    if carries_momentum and momentum is None:
        from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
        momentum=decision_momentum_read(model,engine.bfss_momenta,source)
    skip_val = (state.program_weights is not None and
                getattr(model, 'joint_relation_update_schema', None) == 1)
    path = _compose_fixed(model, state, source, action, ridge=ridge,
        reward_valid=reward_valid, momentum=momentum, mother=mother,
        skip_value=skip_val, skip_mother=skip_mother)
    if state.program_weights is not None:
        from scripts.ib_local.generative_relations import execute_programs
        path = execute_programs(model, state, path, reward_valid=reward_valid)
    if carries_momentum:
        path['prediction_momentum']=momentum
    return path


@torch.no_grad()
def imagine(model, state, source, actions, *, ridge, reward_valid, momentum=None):
    """Functional U branches with read reuse scoped to this exact preview."""
    from scripts.ib_local.preview_runtime import preview_scope
    with preview_scope(model):
        return _imagine(model, state, source, actions, ridge=ridge,
                        reward_valid=reward_valid, momentum=momentum)


def _imagine(model, state, source, actions, *, ridge, reward_valid, momentum=None):
    """Execute the same graph after the same U under finite response hypotheses.

    The information quantity is conditional on the local residual model and
    quadrature. It is logged, not added as a hand-written question reward.
    """
    bank = model.causal_relations
    current = compose(model, state, source, actions, ridge=ridge,
                      reward_valid=reward_valid, momentum=momentum)
    momentum=current.get('prediction_momentum',momentum)
    dist = current['mother']
    hypotheses, response_points, response_geometry = response_hypotheses(model, dist)
    h_count = hypotheses.shape[1]
    after, information, shifts = [], [], []
    epistemic = []
    successor_routes=[]
    for b in range(len(source)):
        futures, kls, state_shifts = [], [], []
        epistemic_futures = []
        routes=[]
        # One noise quadrature per mechanism hypothesis, shared across actions.
        for h in range(h_count):
            y = response_points[b:b+1, h]
            evidence = Evidence(source[b:b+1], actions[b:b+1], y, Port.MOTHER,
                                observed=False, encoded=True)
            updated = revise(state, evidence, model=model)
            path = compose(model, updated, source[b:b+1], actions[b:b+1], ridge=ridge,
                           reward_valid=reward_valid, momentum=(None if momentum is None else momentum[b:b+1]),
                           mother=(y if state.program_weights is not None else None),
                           skip_mother=(state.program_weights is not None))
            futures.append(path['value'][0])
            if hasattr(model, '_epistemic_value_engine'):
                from scripts.ib_local.epistemic_value_reference import branch_value
                epistemic_futures.append(branch_value(model, updated, path['decision']['mean'],
                    momentum=None if momentum is None else momentum[b:b+1],
                    source=source[b:b+1],action=actions[b:b+1],mother=y,
                    decision_distribution=path['decision'])[0])
                if getattr(model._epistemic_value_engine,'epistemic_value_schema',0)>=5:
                    from copy import deepcopy
                    routes.append(deepcopy(model._epistemic_value_engine.successor_distribution_last))
            state_shifts.append((path['decision']['mean'][0]-current['decision']['mean'][b]).norm())
            if response_geometry is None:
                posterior_log = hypothesis_evidence(hypotheses[b:b+1], dist['noise'], y)
            else:
                from scripts.ib_local.joint_response import evidence as response_evidence
                posterior_log = response_evidence(hypotheses[b:b+1], response_geometry, y)
            kls.append((posterior_log.exp()*(posterior_log+math.log(h_count))).sum())
        after.append(torch.stack(futures).mean())
        information.append(torch.stack(kls).mean())
        shifts.append(torch.stack(state_shifts).mean())
        if epistemic_futures:
            epistemic.append(torch.stack(epistemic_futures).mean())
        if routes:successor_routes.append(routes)
    after = torch.stack(after)
    # Exact coefficient information CONDITIONAL on a noise value in this
    # Gaussian residual model. Avoid the log(H) ceiling of a finite operator
    # quadrature being mistaken for the information of the full posterior.
    conditional_information = .5*dist['mean'].shape[-1]*torch.log1p(dist['leverage'])
    result = dict(path=current, hypotheses=hypotheses, expected_after_value=after,
                future_delta=after-current['value'], information=conditional_information,
                quadrature_information=torch.stack(information),
                shadow_state_shift=torch.stack(shifts))
    if epistemic:
        result['epistemic_after_value'] = torch.stack(epistemic)
    if successor_routes:result['successor_routes']=successor_routes
    return result


def infer_role(model, state, *, open_role, condition, action, candidates=None,
               observed=None, ridge=.1):
    """Finite conditional inference on a common mechanism and typed role axes.

    Forward consequence queries and inverse condition/action queries use the
    same mechanism. Actual candidate production belongs to the existing body;
    this function cannot create actions or introduce an external goal.
    """
    if open_role == Port.MOTHER:
        return mechanism(model, state, condition, action, kind=Port.MOTHER, ridge=ridge)
    if open_role not in (Port.ACTION, Port.CONDITION) or candidates is None or observed is None:
        raise ValueError('inverse binding requires typed candidates and an observed/desired outcome')
    source = candidates if open_role == Port.CONDITION else condition.expand_as(candidates)
    act = candidates if open_role == Port.ACTION else action.expand_as(candidates)
    distribution = mechanism(model, state, source, act, kind=Port.MOTHER, ridge=ridge)
    scores = log_score(distribution, observed.expand_as(distribution['mean']))
    return dict(distribution=distribution, log_weights=scores.log_softmax(0), candidates=candidates)
