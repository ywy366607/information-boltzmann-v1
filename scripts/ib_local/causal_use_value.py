"""Versioned prospective evidence value on the continuing body's real ports.

All information values are nats per latent coordinate. A query's own outcome
is excluded from its fit. Queries and coordinates are frozen before the world
call; real response labels arrive later. The existing policy eligibility and
TD machinery consume the resulting scalar with a distinct settlement version.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch.nn import functional as F

from scripts.ib_local.causal_relation_program import Port, mechanism, posterior, log_score


KINDS = {'B': 'coefficient_information_v1', 'C': 'targeted_use_v1',
         'C2': 'finite_targeted_use_v2'}


def active(engine):
    return getattr(engine, 'causal_use_arm', 'A') in KINDS


@torch.no_grad()
def configure(engine, arm):
    if arm not in ('A', 'shadow', *KINDS):
        raise ValueError(arm)
    if getattr(engine, 'causal_use_schema', None) == 1:
        if engine.causal_use_arm != arm:
            raise ValueError('restore a common birth boundary to change experimental arm')
        if active(engine):
            object.__setattr__(engine.model, '_causal_use_value_evidence', engine.causal_use_value_evidence)
        if arm == 'C2':
            # This model-side mode is not a tensor in the saved state_dict.
            engine.model.identity_preview_revision = 1
        return
    if getattr(engine.model, 'joint_relation_update_revision', 0) < 4:
        raise ValueError('requires canonical, qualified, shared relation evidence')
    engine.causal_use_schema = 1
    engine.causal_use_arm = arm
    if arm == 'C2':
        engine.model.identity_preview_revision = 1
    engine.causal_use_clock = 0
    engine.causal_use_calibration = dict(count=0, cross=0., second=0.)
    engine.causal_use_pending = None
    if active(engine):
        from scripts.ib_local.navigation_direct_read import gain_state
        from scripts.ib_local.language_navigation_kanli_v4p import ActionEvidence
        engine.causal_use_value_evidence = ActionEvidence(engine.wood_decay).to(engine.device)
        # The module is engine-owned persistent calibration, like audit_unpooled.
        # Its model-side reference is runtime-only, restored by configure().
        object.__setattr__(engine.model, '_causal_use_value_evidence', engine.causal_use_value_evidence)
        # Old labels and old pending actor keep their original contract. The
        # inherited TD code pays them once without cross-objective bootstrap.
        engine.causal_use_migration = dict(old_kind=engine.settlement_kind,
            baseline=engine.baseline, critic_error=engine.critic_error,
            calibration={k: copy.deepcopy(getattr(engine, k)) for k in
                         ('read_gain_state', 'internal_gain_state', 'shadow_gain_state')})
        engine.settlement_kind = KINDS[arm]
        engine.baseline = 0.
        engine.critic_error = None
        engine.adv_ms = None
        for key in ('read_gain_state', 'internal_gain_state', 'shadow_gain_state'):
            setattr(engine, key, gain_state(engine.wood_decay))


def information(x, query, candidates, ridge):
    """Conditional Gaussian mutual information / dimension, all candidate pairs.

    x: evidence x features; query: questions x features; candidates: actions x
    features. The noise scale cancels in the conditional correlation.
    """
    if ridge <= 0:
        raise ValueError('positive posterior precision required')
    dtype = candidates.dtype
    x, query, candidates = (t.double() for t in (x, query, candidates))
    a = x.T @ x + ridge * torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    solved = torch.linalg.solve(a, torch.cat((query, candidates)).T).T
    sq, sc = solved[:len(query)], solved[len(query):]
    lq = (query * sq).sum(-1)
    lc = (candidates * sc).sum(-1)
    rho2 = (query @ sc.T).square() / ((1+lq[:, None]) * (1+lc[None]))
    if not bool(torch.isfinite(rho2).all()) or bool((rho2 >= 1).any()):
        raise FloatingPointError('invalid predictive correlation')
    return (.5*torch.log1p(lc)).to(dtype), (-.5*torch.log1p(-rho2)).to(dtype)


@torch.no_grad()
def queries(engine, state):
    """Same bounded actual scope; identity ordering is presentation invariant."""
    from scripts.ib_local.relation_evidence import mechanism_evidence
    facts = [e for e in mechanism_evidence(engine.model, state, Port.MOTHER)
             if e.observed and not e.encoded and e.fact_id]
    unique = {e.fact_id: e for e in facts}
    return [unique[k] for k in sorted(unique)[:engine.model.memory.topk]]


@torch.no_grad()
def evaluate(engine, state, source, actions, distribution=None):
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    model = engine.model
    ridge = effective_ridge(model, engine.relation_ridge)
    d = distribution or mechanism(model, state, source, actions, ridge=ridge)
    coefficient, _ = information(d['x'], d['query'][:0], d['query'], ridge)
    targets = []
    for fact in queries(engine, state):
        q = mechanism(model, state, fact.source, fact.action,
                      exclude_fact=fact.fact_id, ridge=ridge)
        _, value = information(q['x'], q['query'], d['query'], ridge)
        targets.append(value[0])
    targeted = torch.stack(targets).mean(0) if targets else coefficient * 0
    return dict(coefficient=coefficient, targeted=targeted, queries=len(targets))


def calibrated(engine, value):
    state = engine.causal_use_calibration
    # Prior is the explicit model expectation. Real outcomes replace that
    # assumption with the existing bounded slope-calibration rule.
    alpha = 1. if state['count'] < 2 else max(0., min(1.,
        state['cross']/state['second'] if state['second'] > 0 else 0.))
    return value * alpha


@torch.no_grad()
def utility(engine, state, source, actions, forecast):
    """Wood units (10 * raw use), shared by binding, drafts and final choice."""
    old = forecast.get('expected_after_value', forecast.get('value'))
    if not active(engine):
        return old
    path = forecast.get('path', forecast)
    if engine.causal_use_arm == 'C2':
        from scripts.ib_local.finite_use_value import expected
        raw = calibrated(engine, expected(engine, state, source, actions,
            path['mother'], queries(engine, state)))
    else:
        values = evaluate(engine, state, source, actions, path.get('mother'))
        raw = values['coefficient'] if engine.causal_use_arm == 'B' else calibrated(engine, values['targeted'])
    # continuation() reads only compatible reward versions. Existing old
    # labels remain in memory and cannot masquerade as the new use score.
    future = path['value']
    if getattr(engine, 'epistemic_value_schema', 0) in (1,2,3,4,5,6):
        if 'epistemic_after_value' in forecast:
            future = forecast['epistemic_after_value']
        else:
            from scripts.ib_local.epistemic_value_reference import branch_value
            future = branch_value(engine.model, state, path['decision']['mean'],
                                  momentum=path.get('prediction_momentum'),source=source,action=actions,
                                  mother=path['mother']['mean'],decision_distribution=path['decision'])
        engine.epistemic_value_last = dict(schema=engine.epistemic_value_schema,
            local_expected=(10*raw).tolist(), old_future=path['value'].tolist(),
            contextual_future=future.tolist(),
            old_total=(10*raw+engine.gamma*path['value']).tolist(),
            contextual_total=(10*raw+engine.gamma*future).tolist(),
            state='branch expected body boundary plus revised finite R and next Q',
            gamma=engine.gamma)
        if engine.epistemic_value_schema>=3:
            engine.epistemic_value_last['last_hypothesis_decision_trace']=getattr(engine,'prospective_decision_last',[])
        if engine.epistemic_value_schema>=4:
            engine.epistemic_value_last['successor_distribution']=copy.deepcopy(engine.successor_distribution_last)
        if engine.epistemic_value_schema>=5:
            engine.epistemic_value_last['successor_routes']=copy.deepcopy(forecast.get('successor_routes',[]))
    return 10*raw + engine.gamma*future


def freeze(distribution):
    return {k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in distribution.items() if k in ('mean', 'scale2', 'df')}


@torch.no_grad()
def begin(engine, record):
    if getattr(engine, 'causal_use_schema', None) != 1:
        return
    state = engine._program_question_state
    model = engine.model
    from scripts.ib_local.learnable_precision_scales import effective_ridge
    ridge = effective_ridge(model, engine.relation_ridge)
    d = engine._program_forecasts['path']['mother']
    selected = record['choice']
    xnew = d['query'][selected:selected+1].detach().clone()
    base = d['base'][selected:selected+1].detach().clone()
    coefficient, _ = information(d['x'], xnew[:0], xnew, ridge)
    entries = []
    for fact in queries(engine, state):
        q = mechanism(model, state, fact.source, fact.action,
                      exclude_fact=fact.fact_id, ridge=ridge)
        _, info = information(q['x'], q['query'], xnew, ridge)
        entries.append(dict(identity=fact.fact_id, stamp=fact.stamp,
            x=q['x'].detach().clone(), y=q['y'].detach().clone(),
            query=q['query'].detach().clone(), base=q['base'].detach().clone(),
            target=model.encoder(fact.outcome).detach().clone(), before=freeze(q),
            information=float(info[0, 0])))
    encoder = model.encoder.weight.detach().clone()
    bias = model.encoder.bias
    roles = engine._program_question_roles
    engine.causal_use_pending = dict(schema=1, arm=engine.causal_use_arm,
        clock=engine.causal_use_clock, prediction_event=int(engine.autonomous_event_clock),
        prediction_update=int(engine.model.memory.clock),
        unknown=roles['unknown_identity'], unknown_version=roles.get('unknown_version'),
        relation=[state.identity, state.generation, state.version], choice=selected,
        coordinate_hash=hashlib.sha256(encoder.cpu().numpy().tobytes()).hexdigest(),
        encoder=encoder, encoder_bias=None if bias is None else bias.detach().clone(),
        incoming_x=xnew, incoming_base=base, coefficient=float(coefficient[0]),
        noise=model.relation_basis.noise(mother=True).detach().clone(),
        ridge=float(ridge), queries=entries)
    if active(engine):
        s = model.encoder(engine._internal_source.reshape(1, -1))
        a = model.encoder(engine._internal_action[selected:selected+1])
        baseline = engine.causal_use_value_evidence.bases()[selected].detach().clone()
        from scripts.ib_local.causal_value_learning import prediction
        raw = prediction(model, s, a-s, torch.tensor([selected], device=s.device))[0]
        engine.causal_use_pending.update(gain_base=baseline, gain_raw=raw)
    if engine.causal_use_arm == 'C2':
        from scripts.ib_local.finite_use_value import begin as begin_finite
        begin_finite(engine, record, d, queries(engine, state))
    if getattr(engine, 'native_amortization', {}).get('planner') == 'action_batch':
        from scripts.ib_local.latent_action_planner import capture_world
        capture_world(engine, record)
    from scripts.ib_local.causal_use_transport import begin as begin_transport
    begin_transport(engine)


@torch.no_grad()
def settle(engine, text_gain, *, slot=None, merged=None):
    if getattr(engine, 'causal_use_schema', None) != 1:
        return text_gain, None
    receipt = engine.causal_use_pending
    if receipt is None:
        raise RuntimeError('use settlement requires a pre-world receipt')
    target = engine._mother_boundary_target
    comparisons = []
    finite_report = None
    if engine.causal_use_arm == 'C2':
        if target is None and slot is None:
            comparisons, finite_report = [], dict(committed=False, reason='no external observation')
        elif slot is None or merged is None:
            raise ValueError('finite U settlement requires actual episode allocation')
        else:
            from scripts.ib_local.finite_use_value import observed as observed_finite
            comparisons, finite_report = observed_finite(engine, receipt, slot, merged)
    elif target is not None:
        incoming = F.linear(target.reshape(1, -1), receipt['encoder'], receipt['encoder_bias'])
        residual = incoming-receipt['incoming_base']
        for q in receipt['queries']:
            # Held-out query leaves one free evidence slot. This is precisely
            # the conditional residual posterior used by the body's U, using
            # the pre-action coordinates on both sides of the comparison.
            after = posterior(torch.cat((q['x'], receipt['incoming_x'])),
                torch.cat((q['y'], residual)), q['query'], receipt['ridge'],
                noise_prior=receipt['noise'])
            after['mean'] = after['mean'] + q['base']
            before_score = float(log_score(q['before'], q['target']).mean())
            after_score = float(log_score(after, q['target']).mean())
            comparisons.append(dict(query=q['identity'], query_stamp=q['stamp'],
                before=before_score, after=after_score, contribution=after_score-before_score,
                expected=q['information']))
    observed = sum(q['contribution'] for q in comparisons)/len(comparisons) if comparisons else 0.
    expected = sum(q['expected'] for q in comparisons)/len(comparisons) if comparisons else 0.
    if not all(math.isfinite(x) for x in (observed, expected, text_gain)):
        raise FloatingPointError('nonfinite real use settlement')
    arm = engine.causal_use_arm
    result = (receipt['coefficient'] if target is not None else 0.) if arm == 'B' else observed if arm in ('C', 'C2') else text_gain
    if active(engine) and (slot is not None or target is not None):
        gain_target = receipt['gain_base'].clone()
        gain_target[3] = 10*result
        engine.causal_use_value_evidence.observe(receipt['choice'], gain_target,
                                               receipt['gain_base'], receipt['gain_raw'])
    calibration = engine.causal_use_calibration
    alpha_before = float(calibrated(engine, 1.))
    if comparisons:
        decay = engine.wood_decay
        calibration['cross'] = decay*calibration['cross']+(1-decay)*observed*expected
        calibration['second'] = decay*calibration['second']+(1-decay)*expected**2
        calibration['count'] += 1
    report = {k: receipt[k] for k in ('schema', 'arm', 'clock', 'prediction_event',
        'prediction_update', 'unknown', 'unknown_version', 'relation', 'choice', 'coordinate_hash', 'coefficient')}
    report.update(settlement_event=int(engine.autonomous_event_clock),
        settlement_update=int(engine.model.memory.clock), text_gain=text_gain,
        observed_use=observed, expected_use=expected, calibration_before=alpha_before,
        settlement=result, comparisons=comparisons, target_available=target is not None,
        reward_version=engine.settlement_kind, units='nats_per_latent_coordinate',
        eligibility_delivery='existing prev.actor, applied at next real world update')
    if finite_report is not None:
        report['finite_U'] = finite_report
    if getattr(engine, 'epistemic_value_schema', 0) in (1,2,3,4,5,6):
        report['epistemic_value'] = copy.deepcopy(engine.epistemic_value_last)
    engine.causal_use_clock += 1
    engine.causal_use_pending = None
    engine.causal_use_last = report
    if getattr(engine, 'native_amortization', {}).get('schema') == 1:
        from scripts.ib_local.native_amortized_decision import settle as settle_native_value
        report['native_amortization'] = settle_native_value(engine, 10*result,
            committed=bool(finite_report and finite_report['committed']))
    return result, report
