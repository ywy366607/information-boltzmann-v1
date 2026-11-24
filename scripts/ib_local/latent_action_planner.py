"""Batch conditional action planning, learned from native and real transitions.

The student predicts post-intention state, language-program endpoint and
immediate finite-U. The existing world model supplies the terminal value.
Only real settlement changes its bounded bank. Production shares two native
teachers across one real decision; the explicit per-call residual mode also
provides a full-support original-expectation reference.
"""
import hashlib
import math

import torch
from torch.nn import functional as F


def enabled(engine):
    return getattr(engine, 'native_amortization', {}).get('planner') == 'action_batch'


def configure(engine):
    selection = engine.native_amortization.get('teacher_selection', 'novelty_diversity')
    if selection not in ('novelty_diversity', 'competing_actions'):
        raise ValueError('unknown native teacher selection')
    if engine.native_amortization.get('root_native_correction') not in (None, 'observe', 'apply'):
        raise ValueError('unknown root native correction mode')
    if not hasattr(engine, 'latent_action_bank'):
        engine.latent_action_bank = dict(records=[], pending={}, revision=0)
    from scripts.ib_local.native_amortized_decision import restore_host_bank
    restore_host_bank(engine.latent_action_bank)
    engine._latent_action_fit = None
    engine._latent_teacher_queue = {}
    engine._latent_teacher_spent = 0
    engine._latent_teacher_checks = []
    engine._latent_world_receipt = None
    engine._competing_action_roots = None
    engine._latent_teacher_selection = None
    engine._root_native_correction_last = None


def _context_packet(engine, state, pre, proposal):
    from scripts.ib_local.causal_relation_program import Port
    from scripts.ib_local.prospective_decision import lift_boundary
    from scripts.ib_local.relation_evidence import mechanism_evidence
    model = engine.model
    blocks = [pre, pre, proposal['source'], proposal['action'],
              lift_boundary(model, pre, proposal['target'])]
    anchor = getattr(engine, '_internal_boundary', None)
    blocks.append(torch.zeros_like(pre) if anchor is None else anchor.reshape_as(pre))
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    from scripts.ib_local.preview_runtime import read, tensor_key
    p = read(model, 'latent_action_pre_momentum', tensor_key(pre),
             lambda: decision_momentum_read(model, engine.bfss_momenta, pre), anchors=(pre,))
    blocks.append(lift_boundary(model, torch.zeros_like(pre), model.momentum_to_boundary(p)))
    for kind in (Port.MOTHER, Port.DECISION):
        facts = mechanism_evidence(model, state, kind)
        if facts:
            s = torch.cat([f.source for f in facts])
            a = torch.cat([f.action for f in facts])
            y = torch.cat([lift_boundary(model, f.source, f.outcome) if f.encoded
                           else f.outcome for f in facts])
            blocks.extend((s.mean(0, keepdim=True), (a-s).mean(0, keepdim=True),
                           (y-s).mean(0, keepdim=True)))
        else:
            blocks.extend([torch.zeros_like(pre)]*3)
    role = (Port.MOTHER, Port.ACTION, Port.CONDITION).index(proposal['role'])
    scalars = [float(role == i) for i in range(3)]+[0.]*3
    scalars += [engine.gamma, math.log1p(engine.max_utterance), len(state.valid)]
    scalars += engine.intero().detach().reshape(-1).tolist()
    scalars += torch.log1p(engine.recurrent.detach().float().square().mean(-1).sqrt()).reshape(-1).tolist()
    binding = state.binding
    identity = '' if binding is None else binding.identity
    scalars += [(b-127.5)/127.5 for b in hashlib.sha256(identity.encode()).digest()[:8]]
    return dict(raw=torch.cat(blocks).detach().cpu(), scalars=scalars,
                role=proposal['role'].value, relation=(state.identity, state.generation, state.version),
                unknown=identity)


@torch.no_grad()
def packet(engine, state, pre, proposal, nominal, option):
    from scripts.ib_local.preview_runtime import read, tensor_key
    anchors = (state, pre, proposal['source'], proposal['action'], proposal['target'])
    key = (id(state), tuple(tensor_key(t) for t in anchors[1:]), proposal['role'].value)
    template = read(engine.model, 'latent_action_context', key,
                    lambda: _context_packet(engine, state, pre, proposal), anchors=anchors)
    raw = template['raw'].clone()
    raw[1:2] = nominal.detach().cpu()
    scalars = list(template['scalars'])
    scalars[3:6] = [float(option == i) for i in range(3)]
    return dict(template, raw=raw, scalars=scalars, option=int(option))


def features(engine, packets):
    raw = torch.stack([p['raw'] for p in packets]).to(engine.model.encoder.weight)
    z = engine.model.encoder(raw)
    direction = F.normalize(z, dim=-1).flatten(1)/math.sqrt(z.shape[1])
    scalars = z.new_tensor([p['scalars'] for p in packets])
    magnitude = torch.log1p(z.square().mean(-1).sqrt())
    return torch.cat((direction, F.normalize(torch.cat((magnitude, scalars), -1), dim=-1)), -1)/math.sqrt(2)


def gain_readout(model, source, delta, options):
    """The SAME evidence-calibrated gain read as original continuation()."""
    from scripts.ib_local.causal_value_learning import prediction
    evidence = model._causal_use_value_evidence
    base = evidence.bases()[options]
    raw = prediction(model, source, delta, options)
    return (base+evidence.confidence()[options]*(raw-base))[:, 3]


def prior(engine, packets):
    model = engine.model
    pre = torch.cat([p['raw'][:1] for p in packets]).to(model.encoder.weight)
    action = torch.cat([p['raw'][1:2] for p in packets]).to(pre)
    z, a = model.encoder(pre), model.encoder(action)
    option = torch.tensor([p['option'] for p in packets], device=pre.device)
    q = gain_readout(model, z, a-z, option)
    return torch.cat((torch.zeros_like(pre), action-pre, q[:, None]), -1)


@torch.no_grad()
def predict(engine, packets):
    from scripts.ib_local.native_amortized_decision import _kernel
    bank = engine.latent_action_bank
    key = (bank['revision'], int(engine.model.memory.clock),
           tuple(p._version for p in engine.model.parameters()))
    cache = engine._latent_action_fit
    if cache is None or cache['key'] != key:
        records = bank['records']
        if records:
            inputs = [r['packet'] for r in records]
            x = features(engine, inputs)
            base = prior(engine, inputs)
            target = torch.stack([r['target'] for r in records]).to(base)
            counts = {kind: sum(r['kind'] == kind for r in records) for kind in ('native', 'world')}
            sources = sum(n > 0 for n in counts.values())
            scale = base.new_tensor([len(records)/(sources*counts[r['kind']]) for r in records]).sqrt()
            gram = _kernel(x, x)*scale[:, None]*scale[None]
            factor = torch.linalg.cholesky(gram+engine.relation_ridge*torch.eye(len(x), device=x.device))
            coef = torch.cholesky_solve((target-base)*scale[:, None], factor)
            cache = dict(key=key, x=x, scale=scale, factor=factor, coef=coef)
        else:
            cache = dict(key=key, x=None)
        engine._latent_action_fit = cache
    base = prior(engine, packets)
    query = features(engine, packets)
    if cache['x'] is None:
        return base, torch.ones(len(packets), device=base.device)
    cross = _kernel(query, cache['x'])*cache['scale'][None]
    got = base+cross@cache['coef']
    projected = torch.linalg.solve_triangular(cache['factor'], cross.T, upper=False)
    novelty = ((1+query.square().sum(-1)).square()-projected.square().sum(0)).clamp_min(0)
    return got, novelty/(1+query.square().sum(-1)).square()


def retain(engine, p, target, kind):
    bank = engine.latent_action_bank
    key = hashlib.sha256(p['raw'].numpy().tobytes()+repr(p['scalars']).encode()+kind.encode()).hexdigest()
    bank['pending'][key] = dict(packet=p, target=target.detach().cpu(), kind=kind,
                                key=key, revision=bank['revision'])
    cap = engine.model.causal_relations.evidence_budget**2
    if len(bank['pending']) > cap:
        native = [k for k, r in bank['pending'].items() if r['kind'] == 'native']
        del bank['pending'][max(native or bank['pending'])]


@torch.no_grad()
def teach(engine, state, pre, proposal, nominal, options, post, endpoints, q, total):
    for j, option in enumerate(options):
        p = packet(engine, state, pre, proposal, nominal[j:j+1], option)
        target = torch.cat(((post-pre)[0], (endpoints[j:j+1]-pre)[0], q[j:j+1]))
        retain(engine, p, target, 'native')


@torch.no_grad()
def plan(engine, states, sources, *, carried=None):
    """All successor/question/program candidates share ONE student evaluation."""
    from scripts.ib_local.causal_relation_program import compose
    from scripts.ib_local.navigation_generative_cycle import question_choices
    from scripts.ib_local.navigation_persistent_question import retained_receipt
    from scripts.ib_local.prospective_decision import program_actions, materialize_branch
    from scripts.ib_local.conditional_actor_reference import encoded_view, evaluate
    from scripts.ib_local.navigation_gain_units import wood_choice_logits
    model = engine.model
    packs, groups, menus = [], [], []
    for i, (state, pre) in enumerate(zip(states, sources)):
        def transform(proposal, bound):
            from dataclasses import replace
            old = proposal['question']
            if old is None:
                return replace(bound, constraints=state.constraints, extra=state.extra, evidence_bound=True)
            if carried is not None and carried.binding is not None and old['identity'] == carried.binding.identity:
                remembered = materialize_branch(carried)
                return replace(bound, constraints=remembered.constraints, extra=(), evidence_bound=True,
                               retention_use=remembered.retention_use)
            return bound
        def utility(qs, s, a):
            z, az = model.encoder(s), model.encoder(a)
            options = torch.ones(len(a), device=s.device, dtype=torch.long)
            return gain_readout(model, z.expand_as(az), az-z, options)
        menu = question_choices(engine, state, pre[0], utility=utility,
                                retained_packet=retained_receipt(model, pre), state_transform=transform)
        menus.append(menu)
        for proposal in menu['proposals']:
            nominal, slots, reuse = program_actions(engine, pre, proposal)
            if not len(nominal):
                nominal = pre
            endpoints = torch.cat((pre, nominal, pre if reuse is None else reuse))
            options = [0]+[1]*len(nominal)+[2]
            begin = len(packs)
            packs.extend(packet(engine, proposal['state'], pre, proposal, a[None], option)
                         for a, option in zip(endpoints, options))
            groups.append(dict(node=i, begin=begin, end=len(packs), pre=pre,
                               proposal=proposal, slots=slots))
    estimated, novelty = predict(engine, packs)
    width = sources[0].shape[-1]
    values, group_traces = [[] for _ in sources], [[] for _ in sources]
    # The post-intention source differs across questions, but all P reads use
    # the SAME current parameters and momenta. The batch axis preserves each
    # group's identity and its original mean across word alternatives.
    posts = torch.cat([group['pre']+estimated[group['begin']:group['end'], :width].mean(0, keepdim=True)
                       for group in groups])
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    momentum_reads = decision_momentum_read(model, engine.bfss_momenta, posts)
    for group_index, group in enumerate(groups):
        rows = estimated[group['begin']:group['end']]
        pre = group['pre']
        # The post-intention port is shared across action alternatives.
        post = posts[group_index:group_index+1]
        q = rows[:, -1]
        actions = pre+rows[:, width:2*width]
        p = momentum_reads[group_index:group_index+1]
        path = compose(model, group['proposal']['state'], post.expand_as(actions), actions,
            ridge=engine.relation_ridge, reward_valid=engine.reward_label_valid(wood=True),
            momentum=p.expand(len(actions), -1))
        total = q+engine.gamma*path['value']
        p_draft = wood_choice_logits(engine, total[1:-1]).softmax(0)
        view = encoded_view(model, model.encoder(post), engine.intero(), reward_valid=engine.reward_label_valid())
        count = len(rows)-2
        indices = torch.stack((torch.zeros(count, device=q.device, dtype=torch.long),
                               torch.arange(1, count+1, device=q.device),
                               torch.full((count,), len(rows)-1, device=q.device, dtype=torch.long)), -1)
        from dataclasses import replace
        view = replace(view, habit=view.habit.expand(count, -1), available=view.available.expand(count, -1))
        actor = evaluate(view, total[indices], precision=engine.precision(), gain_units_schema=engine.gain_units_schema)
        leaf = (actor['probability']*q[indices]).sum(-1)
        values[group['node']].append((p_draft*leaf).sum())
        group_traces[group['node']].append(dict(role=group['proposal']['role'].value,
            unknown=group['proposal']['state'].binding.identity if group['proposal']['state'].binding else '',
            programs=group['slots'], proposal_probability=p_draft.tolist(),
            predicted_local_use=q.tolist(), predicted_total=total.tolist(),
            novelty=novelty[group['begin']:group['end']].tolist()))
    answer = torch.stack([(menus[i]['logits'].softmax(0)*torch.stack(v)).sum()
                          for i, v in enumerate(values)])
    return answer, group_traces, len(packs)


@torch.no_grad()
def value(engine, state, source, action, mother, distribution, momentum):
    from scripts.ib_local.joint_successor import quadrature
    from scripts.ib_local.prospective_decision import prepare_boundary, preview_bind_batch, value as native_value
    from scripts.ib_local import native_amortized_decision as amort
    project = prepare_boundary(engine, state, source, action, mother)
    points, weights, geometry = quadrature(engine, state, source, action, mother, distribution, project=project)
    raws_all = project(points[0])
    sources = [raws_all[j:j+1] for j in range(len(raws_all))]
    states = preview_bind_batch(engine.model, raws_all, overlay=state)
    from scripts.ib_local import persistent_rhythm
    if persistent_rhythm.enabled(engine):
        predicted, cases, candidate_count = persistent_rhythm.read_plans(
            engine, states, sources, carried=state)
    else:
        predicted, cases, candidate_count = plan(engine, states, sources, carried=state)
    k = len(sources)
    if engine.native_amortization.get('correction') == 'decision_budget':
        queue = engine._latent_teacher_queue
        from scripts.ib_local import competing_action_teachers as competing
        from scripts.ib_local.root_native_correction import mode as root_mode
        track_root = competing.enabled(engine) or root_mode(engine) is not None
        root_key = competing.action_key(source, action) if track_root else None
        roots = getattr(engine, '_competing_action_roots', None)
        if not roots or root_key not in roots:
            root_key = None
        root_mass = None
        if root_key is not None:
            root_mass = engine._root_native_mass[root_key]
            root_mass['calls'] += 1
        call_mass = {}
        prefix = b''.join(x.detach().cpu().numpy().tobytes() for x in (source, action, mother))
        point_rows = points[0].cpu().numpy()
        masses = weights.tolist()
        for j, raw in enumerate(sources):
            key = hashlib.sha256(prefix + point_rows[j].tobytes()).hexdigest()
            call_mass[key] = call_mass.get(key, 0.) + masses[j]
            if root_mass is not None:
                root_mass['weights'][key] = root_mass['weights'].get(key, 0.) + masses[j]
            novelties = [v for group in cases[j] for v in group['novelty']]
            priority = masses[j]*sum(novelties)/max(1,len(novelties))
            # Keep an evidence-sized teacher menu through all root candidates.
            if (competing.enabled(engine) and roots or key in queue or
                    len(queue)<engine.model.causal_relations.evidence_budget or
                    priority>min(v['priority'] for v in queue.values())):
                item = dict(priority=priority, key=key, state=state, root_key=root_key,
                    mass=call_mass[key],
                    source=source.detach().clone(), action=action.detach().clone(),
                    mother=mother.detach().clone(), latent=points[:,j].detach().clone(),
                    momentum=momentum.detach().clone(), raw=raw.detach().clone(),
                    selected=states[j], prediction=float(predicted[j]),
                    context=None if getattr(engine.model,'_question_context',None) is None else engine.model._question_context.detach().clone())
                if competing.enabled(engine) and roots:
                    competing.retain(queue, item, engine.model.causal_relations.evidence_budget)
                else:
                    queue[key] = item
                    if len(queue)>engine.model.causal_relations.evidence_budget:
                        del queue[min(queue,key=lambda key:(queue[key]['priority'],key))]
        stats = engine.native_value_bank.setdefault('epoch_stats',dict(calls=0,nodes=0,native=0))
        stats['calls'] += 1; stats['nodes'] += k
        answer = (weights*predicted).sum()
        engine.successor_distribution_last = dict(schema=5,budget=k,branches=k,
            weights=weights.tolist(),values=[predicted.tolist()],expected=[float(answer)],
            df=float(distribution['df']),scale_mean=float(distribution['scale2'].mean()),joint=geometry,
            native_indices=[],native_values=[],correction_weights=[],native_count=0,allocated_draws=0,
            proposal_weights=weights.tolist(),bank_revision=engine.latent_action_bank['revision'],
            planned_candidates=candidate_count,action_plans=cases,
            semantics='batched latent candidate expectation; two native teachers shared by the entire real decision')
        return answer.reshape(1)
    warm = bool(engine.latent_action_bank['records'])
    if warm and k > 1:
        # Content-dependent proposal; original mass supplies full support.
        # Softmax uses the same learned precision and gain units as the actor.
        from scripts.ib_local.navigation_gain_units import wood_choice_logits
        learned = (weights.log()+wood_choice_logits(engine, predicted)).softmax(0)
        proposal = (weights+learned)/2
        # Canonical identity order gives the same private draw after permutation.
        order = sorted(range(k), key=lambda i: hashlib.sha256(sources[i].cpu().numpy().tobytes()).digest())
        signature = source.detach().cpu().numpy().tobytes()+action.detach().cpu().numpy().tobytes()
        signature += mother.detach().cpu().numpy().tobytes()+str(engine.latent_action_bank['revision']).encode()
        seed = int.from_bytes(hashlib.sha256(signature).digest()[:8], 'little') % (2**63-1)
        generator = torch.Generator().manual_seed(seed ^ engine.native_amortization['seed'])
        # Resolve the largest original mass deterministically. Only the
        # complementary residual is sampled: variance is concentrated in its
        # smaller remaining mass, while the native work stays at two nodes.
        anchor = max(order, key=lambda i: float(weights[i]))
        tail_order = [i for i in order if i != anchor]
        tail_mass = proposal[tail_order].sum()
        if float(tail_mass) <= 0:
            index = weights.new_tensor([anchor], dtype=torch.long)
            correction = weights[index]
            sampled = []
        else:
            conditional = proposal[tail_order]/tail_mass
            draw = int(torch.multinomial(conditional.cpu(), 1, generator=generator))
            index = weights.new_tensor([anchor, tail_order[draw]], dtype=torch.long)
            correction = torch.stack((weights[anchor], weights[tail_order[draw]]/conditional[draw]))
            sampled = [tail_order[draw]]
    else:
        proposal = weights
        index = torch.arange(k, device=source.device)
        correction = weights
        anchor = None
        sampled = []
    def expand(t):
        return t.expand(len(index), *t.shape[1:])
    native = native_value(engine, state, expand(source), expand(action), expand(mother),
                          points[0, index], expand(momentum))
    if warm:
        answer = (weights*predicted).sum()+(correction*(native-predicted[index])).sum()
    else:
        answer = native[0]+(weights*(native-native[0])).sum()
    for j, target in zip(index.tolist(), native.tolist()):
        p = amort.recipe(engine, state, sources[j], selected=states[j])
        amort._retain_pending(engine, p, target, predicted[j], predicted[j], 'native')
    stats = engine.native_value_bank.setdefault('epoch_stats', dict(calls=0, nodes=0, native=0))
    stats['calls'] += 1; stats['nodes'] += k; stats['native'] += len(index)
    engine.successor_distribution_last = dict(schema=4, budget=k, branches=k,
        weights=weights.tolist(), values=[predicted.tolist()], expected=[float(answer)],
        df=float(distribution['df']), scale_mean=float(distribution['scale2'].mean()), joint=geometry,
        native_indices=index.tolist(), native_values=native.tolist(), correction_weights=correction.tolist(),
        native_count=len(index), allocated_draws=len(index), exact_index=anchor,
        sampled_indices=sampled,
        proposal_weights=proposal.tolist(), bank_revision=engine.latent_action_bank['revision'],
        planned_candidates=candidate_count, action_plans=cases,
        semantics='batched learned action plans; full-support original native residual correction')
    return answer.reshape(1)


@torch.no_grad()
def flush_teachers(engine, *, decision_values=None):
    """Run the shared teacher budget after candidate comparison, before action."""
    from scripts.ib_local.preview_runtime import preview_scope
    # Teacher labels accumulate only in pending; fitted records, parameters,
    # S/P, the real archive and the fast table remain unchanged throughout.
    with preview_scope(engine.model):
        return _flush_teachers(engine, decision_values=decision_values)


def _flush_teachers(engine, *, decision_values=None):
    if not enabled(engine) or engine.native_amortization.get('correction') != 'decision_budget':
        return
    from scripts.ib_local.prospective_decision import value as native_value
    from scripts.ib_local import native_amortized_decision as amort
    from scripts.ib_local import phase_coupled_rhythm
    if not phase_coupled_rhythm.allow_teacher_flush(engine):
        from scripts.ib_local.root_native_correction import correct
        engine._latent_teacher_queue = {}
        return correct(engine, decision_values, engine._latent_teacher_checks)
    if engine.native_amortization.get('review_cadence') == 'internal_step':
        # This is the released action's own receipt, not the internal steps
        # already accumulating for the next action during language learning.
        if not engine.autonomous_action_receipt['internal_steps']:
            from scripts.ib_local.root_native_correction import correct
            engine._latent_teacher_queue = {}
            return correct(engine, decision_values, engine._latent_teacher_checks)
    menu = sorted(engine._latent_teacher_queue.values(),key=lambda r:(-r['priority'],r['key']))
    # One high-uncertainty candidate plus a different supported candidate.
    selected = menu[:1]
    if len(menu)>1:
        first = menu[0]['raw']
        selected += [max(menu[1:],key=lambda r:float((r['raw']-first).square().mean()))]
    from scripts.ib_local import competing_action_teachers as competing
    if competing.enabled(engine):
        selected, engine._latent_teacher_selection = competing.select(engine, menu, decision_values)
    remaining = max(0,2-engine._latent_teacher_spent)
    from scripts.ib_local import root_latent_planning
    if root_latent_planning.enabled(engine):
        remaining = root_latent_planning.teacher_budget(engine, menu, decision_values, remaining)
    checks_before = len(engine._latent_teacher_checks)
    old_context = getattr(engine.model,'_question_context',None)
    try:
        for item in selected[:remaining]:
            engine.model._question_context = item['context']
            actual = native_value(engine,item['state'],item['source'],item['action'],item['mother'],
                                  item['latent'],item['momentum'])[0]
            p = amort.recipe(engine,item['state'],item['raw'],selected=item['selected'])
            amort._retain_pending(engine,p,actual,item['prediction'],item['prediction'],'native')
            from scripts.ib_local.competing_action_teachers import root_node_mass
            mass, response_calls = root_node_mass(engine, item.get('root_key'), item['key'], item['mass'])
            engine._latent_teacher_checks.append(dict(key=item['key'],predicted=item['prediction'],
                native=float(actual),priority=item['priority'],error=float(actual)-item['prediction'],
                root_key=item.get('root_key'), mass=mass, local_successor_mass=item['mass'],
                response_calls=response_calls))
            engine._latent_teacher_spent += 1
            engine.native_value_bank['epoch_stats']['native'] += 1
    finally:
        engine.model._question_context = old_context
        engine._latent_teacher_queue = {}

    if root_latent_planning.enabled(engine):
        root_latent_planning.observe_teacher_checks(engine, engine._latent_teacher_checks[checks_before:])
    from scripts.ib_local.root_native_correction import correct
    return correct(engine, decision_values, engine._latent_teacher_checks)


@torch.no_grad()
def capture_world(engine, record):
    """Freeze all predictive conditions at the ORIGINAL pre-world U boundary."""
    from scripts.ib_local.causal_relation_program import Port
    captured = getattr(engine, '_native_actual_packet', None)
    if captured is None:
        raise RuntimeError('action planner requires a pre-question physical boundary')
    pre = captured[0]['raw'][:1].to(engine.model.encoder.weight)
    proposal = dict(engine._generative_query)
    proposal['role'] = Port(proposal['role'])
    action = engine._internal_action[int(record['choice'])].reshape_as(pre)
    p = packet(engine, engine._program_question_state, pre, proposal, action, int(record['choice']))
    prediction, _ = predict(engine, [p])
    engine._latent_world_receipt = dict(packet=p, pre=pre.detach().cpu(),
                                       predicted=float(prediction[0,-1]))


@torch.no_grad()
def settle(engine, actual, *, committed):
    bank = engine.latent_action_bank
    captured = engine._latent_world_receipt
    world = False
    if committed and captured is not None:
        pre = captured['pre'].to(engine.model.encoder.weight)
        post = engine._wood_source.reshape_as(pre)
        action = engine._wood_action.reshape_as(pre)
        p = captured['packet']
        q = pre.new_tensor([actual])
        target = torch.cat(((post-pre)[0], (action-pre)[0], q))
        retain(engine, p, target, 'world'); world = True
    cap = engine.model.causal_relations.evidence_budget**2
    joined = {r['key']:r for r in [*bank['records'], *bank['pending'].values()]
              if r['revision'] >= bank['revision']-cap}
    observed = sorted((r for r in joined.values() if r['kind']=='world'), key=lambda r:r['key'])[:cap//2]
    teacher = sorted((r for r in joined.values() if r['kind']=='native'), key=lambda r:r['key'])[:cap-len(observed)]
    bank['records'] = observed+teacher
    bank['pending'] = {}; bank['revision'] += 1
    engine._latent_action_fit = None
    checks = engine._latent_teacher_checks
    spent = engine._latent_teacher_spent
    selection = engine._latent_teacher_selection
    engine._latent_teacher_selection = None
    engine._latent_teacher_checks = []; engine._latent_teacher_spent = 0
    engine._latent_teacher_queue = {}
    engine._latent_world_receipt = None
    return dict(revision=bank['revision'], records=len(bank['records']), capacity=cap,
                world_labels=len(observed), native_labels=len(teacher), new_world_label=world,
                teacher_count=spent,teacher_checks=checks,
                teacher_selection=selection,
                world_prediction=None if captured is None else captured['predicted'],
                world_target=float(actual) if committed else None,
                input_boundary='frozen in original pre-world finite-U begin',
                learned_outputs='post-intention boundary, language endpoint, finite-U; original world-model continuation',
                proposal='content-conditioned softmax from predicted candidate total; real/native settlement learning')
