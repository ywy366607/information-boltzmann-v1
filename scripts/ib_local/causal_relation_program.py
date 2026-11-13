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


def revise(state: RelationState, evidence: Evidence) -> RelationState:
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
    valid = state.valid.clone()
    key = lambda e: (e.observed, e.slot, e.stamp)
    extra_events = {key(e) for e in state.extra}
    if key(evidence) not in extra_events and int(valid.sum()) + len(extra_events) >= len(valid):
        if state.release < 0 or not bool(valid[state.release]):
            raise ValueError('a full relation requires an evidence-supported release choice')
        valid[state.release] = False
    return replace(state, valid=valid, extra=state.extra + (evidence,))


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


def posterior(x, y, query, ridge):
    """Normal-inverse-gamma residual mechanism, solved in evidence space.

    Per output: B|s2 ~ N(0,s2/ridge I), s2 ~ InvGamma(1,1).
    The unit prior is in the body's SIGReg latent coordinates. The inherited
    ridge remains explicit. All variance estimates below use PRIOR evidence;
    the outcome to be scored is never used to set its own noise floor.
    """
    if ridge <= 0 or x.ndim != 2 or y.ndim != 2 or query.ndim != 2:
        raise ValueError('positive precision and 2D tensors required')
    if len(x) != len(y) or x.shape[1] != query.shape[1]:
        raise ValueError('incompatible role-bound evidence')
    kernel = x @ x.T + ridge * torch.eye(len(x), device=x.device, dtype=x.dtype)
    coefficient = torch.linalg.solve(kernel, y)
    mean = (query @ x.T) @ coefficient
    projected = torch.linalg.solve(kernel, x @ query.T)
    leverage = ((query.square().sum(-1) - (query @ x.T * projected.T).sum(-1)) / ridge).clamp_min(0)
    alpha = 1. + len(x) / 2.
    beta = (1. + .5 * (y.square().sum(0) - (y * (x @ x.T @ coefficient)).sum(0))).clamp_min(
        torch.finfo(y.dtype).eps)
    # X A^-1 X^T = K0 (K0+ridge I)^-1, as used above.
    noise = (beta / alpha).detach()
    scale2 = (1. + leverage[:, None]) * noise
    return dict(mean=mean, scale2=scale2, noise=noise, leverage=leverage,
                df=2*alpha, kernel=kernel, x=x, y=y, coefficient=coefficient, ridge=ridge)


def log_score(distribution, target):
    """Prequential Student-t score in the forecast's own latent coordinates."""
    df = target.new_tensor(distribution['df'])
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
    eps = observation_directions[:, :len(x)]
    latent = torch.einsum('nf,hfd->hnd', x, prior) + eps
    solved = torch.linalg.solve(kernel[None], latent)
    conditional = prior - torch.einsum('fn,hnd->hfd', x.T, solved)
    deviation = torch.einsum('bf,hfd->bhd', query, conditional) * distribution['noise'].sqrt()
    return distribution['mean'][:, None] + deviation


def hypothesis_evidence(means, noise, observed):
    """Posterior weights for the same relation hypotheses under an outcome."""
    variance = noise.clamp_min(torch.finfo(means.dtype).eps)
    logits = -.5*((observed[..., None, :] - means).square()/variance).sum(-1)
    return logits.log_softmax(-1)


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

    @torch.no_grad()
    def bind(self, model, source):
        m, enc = model.memory, model.encoder
        live = self.identity > 0
        score = (F.normalize(enc(source), dim=-1) @ F.normalize(enc(self.condition), dim=-1).T)[0]
        score.masked_fill_(~live, -2)
        value, index = score.max(0)
        i = int(index)
        if not bool(live.any()) or float(value) < m.threshold:
            free = torch.nonzero(~live).flatten()
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
        self.last_use[i] = m.clock
        self.valid[i] &= m.occupied[self.slots[i]] & (m.count[self.slots[i]] == 1)
        self.valid[i] &= (m.last_write[self.slots[i]] == self.stamps[i])
        self.valid[i] &= m.mother_valid[self.slots[i]] & (m.mother_stamp[self.slots[i]] == self.stamps[i])
        state = self.state(i)
        return replace(state, release=release_choice(model, state))

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
    if not bool(state.valid.all()):
        return -1
    m, enc = model.memory, model.encoder
    x = role_features(enc(m.source[state.slots]), enc(m.action[state.slots]),
                      enc(m.mother_next[state.slots]))
    gram = x @ x.T
    precision = getattr(model, 'causal_relation_ridge', .1)
    hat = gram @ torch.linalg.solve(gram + precision*torch.eye(len(x), device=x.device, dtype=x.dtype),
                                    torch.eye(len(x), device=x.device, dtype=x.dtype))
    return int(hat.diag().argmin())


def mechanism(model, state, source, action, *, kind=Port.MOTHER, mother=None,
              ridge=.1, momentum=None, exclude=-1):
    """Execute a role-bound body mechanism from its real and shadow evidence."""
    enc, m = model.encoder, model.memory
    slots = real_evidence(model, state, kind, exclude=exclude)
    s, a = enc(source), enc(action)
    old_s, old_a = enc(m.source[slots]), enc(m.action[slots])
    transport = mother is not None
    if transport and kind != Port.DECISION:
        raise ValueError('only mother-to-decision transport consumes a mother port')
    if transport:
        old_mother = enc(m.mother_next[slots])
        x = role_features(old_s, old_a, old_mother)
        y = enc(m.internal_next[slots]) - old_mother
        query = role_features(s, a, mother)
        base = mother
    else:
        x, query = role_features(old_s, old_a), role_features(s, a)
        old_base = model.long_predictor(torch.cat((old_s, old_a), -1))
        base = model.long_predictor(torch.cat((s, a), -1))
        if momentum is not None:
            base = base + model.momentum_to_boundary(momentum)
        target = m.mother_next[slots] if kind == Port.MOTHER else m.internal_next[slots]
        if kind == Port.MOTHER:
            old_base = old_base + model.mother_from_boundary(old_base)
            base = base + model.mother_from_boundary(base)
        y = enc(target) - old_base
    for evidence in state.extra:
        if evidence.kind != kind or (transport and evidence.mother is None):
            continue
        es, ea = enc(evidence.source), enc(evidence.action)
        ey = evidence.outcome if evidence.encoded else enc(evidence.outcome)
        if transport:
            em = evidence.mother if evidence.encoded else enc(evidence.mother)
            ex, residual = role_features(es, ea, em), ey-em
        else:
            ex = role_features(es, ea)
            prior = model.long_predictor(torch.cat((es, ea), -1))
            if kind == Port.MOTHER:
                prior = prior + model.mother_from_boundary(prior)
            residual = ey-prior
        x, y = torch.cat((x, ex)), torch.cat((y, residual))
    result = posterior(x, y, query, ridge)
    result['mean'] = base + result['mean']
    result.update(base=base, query=query, slots=slots)
    return result


def compose(model, state, source, action, *, ridge, reward_valid, momentum=None,
            mother=None):
    """Mother effect -> actual ambient transport -> next action/value.

    This graph is evaluated, not merely recorded as a diagram. Independent
    candidate actions form the parallel axis. Receiver identity is the typed
    mother port; the subject is the continuously carried body.
    """
    from scripts.ib_local.navigation_internal_rollout import continuation
    condition = Variable('condition',Port.CONDITION)
    act = Variable('action',Port.ACTION)
    unknown = Variable(f'{state.identity}:mother',Port.MOTHER)
    next_state = Variable(f'{state.identity}:decision',Port.DECISION)
    gain = Variable(f'{state.identity}:value',Port.VALUE)
    calls = (Call(MOTHER,(condition,act),unknown),
             Call(TRANSPORT,(condition,act,unknown),next_state),
             Call(TRANSITION,(condition,act),next_state),
             Call(VALUE,(next_state,),gain))
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
    values,trace=execute(calls,bindings,{MOTHER.name:predict_mother,TRANSPORT.name:transport,
        TRANSITION.name:direct_prediction,VALUE.name:value_prediction})
    response,decision,direct=(trace[i][1].metadata for i in range(3))
    value,packet=values[gain].tensor,values[gain].metadata
    return dict(mother=response, decision=decision, direct_decision=direct,
                overlap_residual=decision['mean']-direct['mean'],
                value=value, continuation=packet)


@torch.no_grad()
def imagine(model, state, source, actions, *, ridge, reward_valid, momentum=None):
    """Execute the same graph after the same U under finite response hypotheses.

    The information quantity is conditional on the local residual model and
    quadrature. It is logged, not added as a hand-written question reward.
    """
    bank = model.causal_relations
    current = compose(model, state, source, actions, ridge=ridge,
                      reward_valid=reward_valid, momentum=momentum)
    dist = current['mother']
    hypotheses = mechanism_hypotheses(dist, dist['query'], bank.directions,
                                     bank.observation_directions)
    h_count = hypotheses.shape[1]
    after, information, shifts = [], [], []
    for b in range(len(source)):
        futures, kls, state_shifts = [], [], []
        # One noise quadrature per mechanism hypothesis, shared across actions.
        for h in range(h_count):
            y = hypotheses[b:b+1, h] + dist['noise'].sqrt()*bank.observation_directions[h, -1]
            evidence = Evidence(source[b:b+1], actions[b:b+1], y, Port.MOTHER,
                                observed=False, encoded=True)
            updated = revise(state, evidence)
            path = compose(model, updated, source[b:b+1], actions[b:b+1], ridge=ridge,
                           reward_valid=reward_valid, momentum=(None if momentum is None else momentum[b:b+1]))
            futures.append(path['value'][0])
            state_shifts.append((path['decision']['mean'][0]-current['decision']['mean'][b]).norm())
            posterior_log = hypothesis_evidence(hypotheses[b:b+1], dist['noise'], y)
            kls.append((posterior_log.exp()*(posterior_log+math.log(h_count))).sum())
        after.append(torch.stack(futures).mean())
        information.append(torch.stack(kls).mean())
        shifts.append(torch.stack(state_shifts).mean())
    after = torch.stack(after)
    # Exact coefficient information CONDITIONAL on a noise value in this
    # Gaussian residual model. Avoid the log(H) ceiling of a finite operator
    # quadrature being mistaken for the information of the full posterior.
    conditional_information = .5*dist['mean'].shape[-1]*torch.log1p(dist['leverage'])
    return dict(path=current, hypotheses=hypotheses, expected_after_value=after,
                future_delta=after-current['value'], information=conditional_information,
                quadrature_information=torch.stack(information),
                shadow_state_shift=torch.stack(shifts))


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
