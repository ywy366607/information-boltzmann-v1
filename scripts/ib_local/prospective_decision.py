"""Finite decision programs at a predicted boundary, using the original body.

The latent transition is retained. Its missing raw coordinates are predicted
from actual decision-boundary transitions with the existing conditional
posterior, then constrained to that latent forecast. Scope tests, typed query
proposals, binding weights, draft selection and actor use their native rules.
Spoken programs are represented by observed utterance endpoint displacements;
their future execution is a model forecast and is scored by the same finite U.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch
from torch.nn import functional as F

from scripts.ib_local.causal_relation_program import (
    Port, RelationBank, RelationState, compose, posterior, relation_features,
    release_choice,
)


_terminal = ContextVar('native_decision_terminal', default=False)


@contextmanager
def terminal():
    token = _terminal.set(True)
    try:
        yield
    finally:
        _terminal.reset(token)


def at_terminal():
    return _terminal.get()


def materialize_branch(state):
    """Advance shadow fact names and their use clock together, without a write."""
    from scripts.ib_local.relation_evidence import evidence_key
    renames={}
    facts=[]
    for j,e in enumerate(state.constraints+state.extra):
        if not e.observed and not e.fact_id:
            name=f'shadow:carried:{state.identity}:{j}:{e.kind.value}'
            renames[evidence_key(e)]=name
            e=replace(e,fact_id=name)
        facts.append(e)
    use=state.retention_use
    if use is not None and renames:
        use=replace(use,last_event=renames.get(use.last_event,use.last_event),
            moments=tuple((renames.get(row[0],row[0]),*row[1:]) for row in use.moments))
    return replace(state,constraints=tuple(facts),extra=(),retention_use=use,evidence_bound=True,
        organization=None if state.organization is None else materialize_branch(state.organization))


def _right_inverse(model):
    from scripts.ib_local.preview_runtime import read, tensor_key
    weight = model.encoder.weight
    def inverse():
        w = weight.double()
        return torch.linalg.solve(w @ w.T, w).T.to(weight)
    return read(model, 'decision_right_inverse', tensor_key(weight), inverse, anchors=(weight,))


def lift_boundary(model, anchor, latent):
    return anchor + (latent-model.encoder(anchor)) @ _right_inverse(model).T


def _raw_anchor(engine, state, source, action, mother):
    """Original relation features fit actual raw successor displacements."""
    from scripts.ib_local.relation_evidence import mechanism_evidence
    model = engine.model
    facts = [e for e in mechanism_evidence(model, state, Port.DECISION)
             if e.observed and not e.encoded and e.mother is not None]
    if facts:
        s = torch.cat([e.source for e in facts])
        a = torch.cat([e.action for e in facts])
        y = torch.cat([e.outcome for e in facts])
        m = torch.cat([model.encoder(e.mother) for e in facts])
        x = relation_features(model, model.encoder(s), model.encoder(a), m)
        q = relation_features(model, model.encoder(source), model.encoder(action), mother)
        displacement = posterior(x, y-s, q, engine.relation_ridge)['mean']
        raw = source+displacement
    else:
        raw = source.clone()
    return raw, len(facts)


def prepare_boundary(engine, state, source, action, mother):
    # All quadrature points for this action share the fitted raw anchor.
    raw, _ = _raw_anchor(engine, state, source, action, mother)
    encoded = engine.model.encoder(raw)
    right = _right_inverse(engine.model).T
    return lambda latent: raw + (latent-encoded) @ right


def boundary(engine, state, source, action, mother, latent):
    model = engine.model
    raw, support = _raw_anchor(engine, state, source, action, mother)
    value = lift_boundary(model, raw, latent)
    return value, dict(raw_support=support,
        latent_residual=float((model.encoder(value)-latent).abs().max()),
        raw_displacement=float((value-source).norm()))


def preview_bind(model, source, *, overlay=None):
    """The native address and scope rule, without allocating a live relation."""
    bank, m = model.causal_relations, model.memory
    matched, i, similarity = bank.binding_address(model, source)
    if matched:
        from scripts.ib_local.preview_runtime import read

        def bound_snapshot():
            snapshot = (bank.preview_state(i) if hasattr(bank, 'preview_state')
                        else RelationBank.state(bank, i))
            valid = snapshot.valid & m.occupied[snapshot.slots] & (m.count[snapshot.slots] == 1)
            valid &= (m.last_write[snapshot.slots] == snapshot.stamps) & m.mother_valid[snapshot.slots]
            valid &= m.mother_stamp[snapshot.slots] == snapshot.stamps
            snapshot = replace(snapshot, valid=valid)
            if getattr(model, 'joint_relation_update_schema', None) == 1:
                from scripts.ib_local.joint_relation_update import relation_variables
                snapshot = relation_variables(model, snapshot)
            return snapshot

        # Within one functional imagination, bank/memory/parameters are fixed.
        # Only this source-independent snapshot is shared. Each hypothetical
        # overlay and its finite revision remain separate below.
        state = read(model, 'native_bound_snapshot', (id(bank), i), bound_snapshot,
                     anchors=(bank, m))
        if overlay is not None and (state.identity, state.generation) == (overlay.identity, overlay.generation):
            state = overlay.organization if overlay.organization is not None else overlay
            state = replace(state, binding=None, organization=None)
    else:
        eligible = m.occupied & (m.count == 1) & m.mother_valid & (m.mother_stamp == m.last_write)
        slots = m.read_encoded(model.encoder(source), model.encoder, eligible=eligible).slots[0, :bank.evidence_budget]
        # A fresh shadow address owns no persisted identity or source buffer.
        state = RelationState(-1, 0, 0, -1, slots.clone(), m.last_write[slots].clone(), eligible[slots].clone())
        from scripts.ib_local.relation_evidence import mechanism_evidence
        facts = tuple(e for kind in (Port.MOTHER, Port.DECISION)
                      for e in mechanism_evidence(model, state, kind))
        state = replace(state, constraints=facts, evidence_bound=True)
        if hasattr(bank,'program_scores'):
            from scripts.ib_local.generative_relations import PROGRAMS
            state=replace(state,program_weights=bank.program_scores.new_full(
                (len(PROGRAMS),),1/len(PROGRAMS)))
    # Materialize branch facts so the next U has its own one-event extra slot.
    if state.extra:
        state = materialize_branch(state)
    state = replace(state, release=release_choice(model, state))
    return state, dict(matched=matched, relation=state.identity, generation=state.generation,
                       similarity=float(similarity))


def preview_bind_batch(model, sources, *, overlay=None):
    """Exact batched counterpart of `[preview_bind(model, s, overlay=overlay)[0] for s in sources]`."""
    if len(sources) <= 1:
        return [preview_bind(model, sources[i:i+1], overlay=overlay)[0] for i in range(len(sources))]
    bank, m = model.causal_relations, model.memory
    from scripts.ib_local.preview_runtime import read
    addrs = bank.binding_address_batch(model, sources)
    results = []
    for k, (matched, i, _) in enumerate(addrs):
        if matched:
            def bound_snapshot(i=i):
                snapshot = (bank.preview_state(i) if hasattr(bank, 'preview_state')
                            else RelationBank.state(bank, i))
                valid = snapshot.valid & m.occupied[snapshot.slots] & (m.count[snapshot.slots] == 1)
                valid &= (m.last_write[snapshot.slots] == snapshot.stamps) & m.mother_valid[snapshot.slots]
                valid &= m.mother_stamp[snapshot.slots] == snapshot.stamps
                snapshot = replace(snapshot, valid=valid)
                if getattr(model, 'joint_relation_update_schema', None) == 1:
                    from scripts.ib_local.joint_relation_update import relation_variables
                    snapshot = relation_variables(model, snapshot)
                return snapshot
            state = read(model, 'native_bound_snapshot', (id(bank), i), bound_snapshot,
                         anchors=(bank, m))
            if overlay is not None and (state.identity, state.generation) == (overlay.identity, overlay.generation):
                state = overlay.organization if overlay.organization is not None else overlay
                state = replace(state, binding=None, organization=None)
        else:
            eligible = m.occupied & (m.count == 1) & m.mother_valid & (m.mother_stamp == m.last_write)
            slots = m.read_encoded(model.encoder(sources[k:k+1]), model.encoder, eligible=eligible).slots[0, :bank.evidence_budget]
            state = RelationState(-1, 0, 0, -1, slots.clone(), m.last_write[slots].clone(), eligible[slots].clone())
            from scripts.ib_local.relation_evidence import mechanism_evidence
            facts = tuple(e for kind in (Port.MOTHER, Port.DECISION)
                          for e in mechanism_evidence(model, state, kind))
            state = replace(state, constraints=facts, evidence_bound=True)
            if hasattr(bank, 'program_scores'):
                from scripts.ib_local.generative_relations import PROGRAMS
                state = replace(state, program_weights=bank.program_scores.new_full(
                    (len(PROGRAMS),), 1/len(PROGRAMS)))
        if state.extra:
            state = materialize_branch(state)
        state = replace(state, release=release_choice(model, state))
        results.append(state)
    return results


def program_actions(engine, source, proposal):
    """Predict bounded word-program endpoints, retaining their real provenance."""
    model, m = engine.model, engine.model.memory
    z = model.encoder(source)
    from scripts.ib_local.prospective_self_execution import typed_filler
    filler = typed_filler(model,source,proposal)
    query = z+model.question_reactivation(filler)
    eligible = m.occupied & (m.count == 1) & (m.word_count > 0)
    read = m.read_encoded(query, model.encoder, eligible=eligible)
    slots = read.slots[0][eligible[read.slots[0]]][:engine.joint_proposal_budget]
    if not len(slots):
        return source[:0], [], None
    actions = source+m.action[slots]-m.source[slots]
    if getattr(model, '_expression_speech_only', False):
        return actions, slots.tolist(), None
    # Reuse availability follows the SAME current read/provenance as the actor.
    current = m.read_encoded(z, model.encoder)
    reusable = (m.word_count[current.slots[0]] > 0) & (m.progress[current.slots[0]] > 0)
    reusable &= engine.reward_label_valid()[current.slots[0]] & m.occupied[current.slots[0]]
    weights = current.weights[0]*reusable
    reuse = None
    if bool(weights.sum() > 0):
        slot = current.slots[0, weights.argmax()]
        reuse = source+m.action[slot:slot+1]-m.source[slot:slot+1]
    return actions, slots.tolist(), reuse


@torch.no_grad()
def value(engine, state, source, action, mother, latent, momentum):
    """Next address -> question -> word programs -> native actor -> finite U."""
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local.causal_use_value import calibrated
    from scripts.ib_local.navigation_generative_cycle import question_choices
    from scripts.ib_local.navigation_persistent_question import retained_receipt
    from scripts.ib_local.conditional_actor_reference import encoded_view, evaluate
    from scripts.ib_local.navigation_gain_units import wood_choice_logits
    model = engine.model
    results, details = [], []
    with terminal():
        for b in range(len(latent)):
            raw, dynamics = boundary(engine, state, source[b:b+1], action[b:b+1],
                                     mother[b:b+1], latent[b:b+1])
            selected, address = preview_bind(model, raw, overlay=state)
            carried = materialize_branch(state)
            from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
            next_momentum = decision_momentum_read(model, engine.bfss_momenta, raw)
            def transform(proposal, bound):
                old = proposal['question']
                if old is None:
                    bound = replace(bound, constraints=selected.constraints,
                                    extra=selected.extra, evidence_bound=True)
                elif state.binding is not None and old['identity'] == state.binding.identity:
                    bound = replace(bound, constraints=carried.constraints,
                                    extra=(), evidence_bound=True, retention_use=carried.retention_use)
                return bound
            def utility(qs, s, a):
                context = ev.from_body(engine, qs, (), momentum=next_momentum)
                context = replace(context, queries=context.facts)
                following = compose(model, qs, s, a, ridge=engine.relation_ridge,
                    reward_valid=engine.reward_label_valid(wood=True),
                    momentum=next_momentum.expand(len(a), -1))
                local = 10*float(calibrated(engine, 1.))*ev.expected(model, context,
                    model.encoder(s), model.encoder(a), ridge=engine.relation_ridge,
                    distribution=following['mother'])
                return local+engine.gamma*following['value']
            choices = question_choices(engine, selected, raw[0],
                retained_packet=retained_receipt(model, raw), state_transform=transform, utility=utility)
            p_question = choices['logits'].softmax(0)
            question_values, cases = [], []
            for proposal in choices['proposals']:
                qs = proposal['state']
                # The scoring scope includes branch facts already known here.
                drafts, slots, reuse = program_actions(engine, raw, proposal)
                nominal_drafts, nominal_reuse = drafts, reuse
                acting_source=raw
                acting_momentum=next_momentum
                if getattr(engine,'epistemic_word_execution','historical')=='native_anchor':
                    from scripts.ib_local.prospective_self_execution import execute
                    acting_source,drafts,reuse=execute(engine,raw,proposal,slots)
                    acting_momentum=decision_momentum_read(model,engine.bfss_momenta,acting_source)
                context = ev.from_body(engine, qs, (), momentum=acting_momentum)
                context = replace(context, queries=context.facts)
                if not len(drafts):
                    drafts = acting_source
                endpoints = torch.cat((acting_source, drafts, acting_source if reuse is None else reuse))
                z = model.encoder(acting_source)
                path = compose(model, qs, acting_source.expand_as(endpoints), endpoints,
                    ridge=engine.relation_ridge, reward_valid=engine.reward_label_valid(wood=True),
                    momentum=acting_momentum.expand(len(endpoints), -1))
                q = 10*float(calibrated(engine, 1.))*ev.expected(model, context, z,
                    model.encoder(endpoints), ridge=engine.relation_ridge, distribution=path['mother'])
                # Original two-program execution supplies the terminal boundary.
                total = q+engine.gamma*path['value']
                if getattr(engine, 'native_amortization', {}).get('planner') == 'action_batch':
                    from scripts.ib_local.latent_action_planner import teach
                    if not len(nominal_drafts):
                        nominal_drafts = raw
                    nominal = torch.cat((raw, nominal_drafts, raw if nominal_reuse is None else nominal_reuse))
                    options = [0]+[1]*len(drafts)+[2]
                    keep = len(options)-int(getattr(model, '_expression_speech_only', False))
                    teach(engine, qs, raw, proposal, nominal[:keep], options[:keep],
                          acting_source, endpoints[:keep], q[:keep], total[:keep])
                pd = wood_choice_logits(engine, total[1:-1]).softmax(0)
                view = encoded_view(model, z, engine.intero(), reward_valid=engine.reward_label_valid())
                if (getattr(engine, 'epistemic_value_schema', 0) >= 6 or
                    getattr(engine, 'native_amortization', {}).get('planner') == 'action_batch'):
                    indices = torch.stack((torch.zeros(len(drafts), device=q.device, dtype=torch.long),
                        torch.arange(1, len(drafts)+1, device=q.device),
                        torch.full((len(drafts),), len(endpoints)-1, device=q.device, dtype=torch.long)), -1)
                    expanded = replace(view, habit=view.habit.expand(len(drafts), -1),
                        available=view.available.expand(len(drafts), -1))
                    actor = evaluate(expanded, total[indices], precision=engine.precision(),
                                     gain_units_schema=engine.gain_units_schema)
                    leaf_values = (actor['probability']*q[indices]).sum(-1)
                else:
                    leaf_values = []
                    for d in range(len(drafts)):
                        indices = torch.tensor([0, d+1, len(endpoints)-1], device=q.device)
                        actor = evaluate(view, total[indices][None], precision=engine.precision(),
                                         gain_units_schema=engine.gain_units_schema)
                        leaf_values.append((actor['probability'][0]*q[indices]).sum())
                    leaf_values = torch.stack(leaf_values)
                question_values.append((pd*leaf_values).sum())
                cases.append(dict(role=proposal['role'].value,
                    identity=qs.binding.identity if qs.binding is not None else '',
                    scope=[f.identity for f in context.facts], programs=slots,
                    program_stamps=[int(model.memory.last_write[s]) for s in slots],
                    proposal_probability=pd.tolist(), actor_values=leaf_values.tolist(),
                    intention_source_shift=float((acting_source-raw).norm()),
                    word_execution=getattr(engine,'epistemic_word_execution','historical')))
            qv = torch.stack(question_values)
            results.append((p_question*qv).sum())
            details.append(dict(dynamics=dynamics, address=address,
                question_probability=p_question.tolist(), question_values=qv.tolist(), cases=cases,
                model='raw boundary posterior + native typed questions + transported observed word programs',
                terminal='one explicit Q/program/actor step; original learned terminal value; current intero persistence',
                word_endpoint_model='observed action displacement transported to predicted raw boundary',
                reuse_model='highest native reuse weight; transported endpoint'))
    engine.prospective_decision_last = details
    return torch.stack(results)
