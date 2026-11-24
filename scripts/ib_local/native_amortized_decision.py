"""Conditional learned native-decision readout with exact sparse correction.

Original schema5 nodes, addresses, actor and finite-U remain the reference.
The finite shared predictor consumes raw boundary/evidence/question/P/budget
recipes in CURRENT encoder coordinates. Teacher labels are explicitly imagined
native-port values; world labels are the actual arrived 10*finite-U. Its bounded
bank changes only at real settlement, after all candidate choices are finished.
No new neural backbone, historical optimizer graph or global RNG is retained.
"""
import hashlib
import math

import torch
from torch.nn import functional as F


def enabled(engine):
    return getattr(engine, 'native_amortization', {}).get('schema') == 1


def restore_host_bank(bank):
    """Restore the stored raw/label bank to its canonical CPU placement.

    The generic engine restorer transports nested tensors to the model's
    device. These banks deliberately keep historical payload on the host;
    current features are uploaded together only when fitting or predicting.
    Newly captured records and restored ones must share that storage contract.
    """
    for record in [*bank['records'], *bank['pending'].values()]:
        record['packet']['raw'] = record['packet']['raw'].detach().cpu()
        if isinstance(record['target'], torch.Tensor):
            record['target'] = record['target'].detach().cpu()


def configure(engine, config):
    if config.get('schema') != 1 or getattr(engine, 'epistemic_value_schema', 0) != 5:
        raise ValueError('conditional amortization requires original schema5')
    engine.native_amortization = dict(config)
    if not hasattr(engine, 'native_value_bank'):
        engine.native_value_bank = dict(records=[], pending={}, revision=0, quality=[])
    restore_host_bank(engine.native_value_bank)
    engine._native_fit_cache = None
    if config.get('planner') == 'action_batch':
        from scripts.ib_local.latent_action_planner import configure as configure_planner
        configure_planner(engine)


def _hash_features(identity):
    raw = hashlib.sha256(identity.encode()).digest()[:8]
    return [(v-127.5)/127.5 for v in raw]


def _role_blocks(model, st, port, zero_row):
    from scripts.ib_local.relation_evidence import mechanism_evidence
    from scripts.ib_local.prospective_decision import lift_boundary
    from scripts.ib_local.preview_runtime import read
    if st is None:
        return [zero_row] * 4, 0
    def _compute():
        facts = mechanism_evidence(model, st, port)
        if facts:
            s = torch.cat([f.source for f in facts])
            a = torch.cat([f.action for f in facts])
            y = torch.cat([lift_boundary(model, f.source, f.outcome) if f.encoded
                           else f.outcome for f in facts])
            return [(s.mean(0, keepdim=True), (a-s).mean(0, keepdim=True),
                     (y-s).mean(0, keepdim=True), ((a-s)*(y-s)).mean(0, keepdim=True)), len(facts)]
        return [(zero_row, zero_row, zero_row, zero_row), 0]
    blocks4, count = read(model, 'nad_recipe_role_blocks', (id(st), port), _compute, anchors=(st,))
    return list(blocks4), count


def _common_recipe_scalars(engine):
    from scripts.ib_local.preview_runtime import read, tensor_key
    model = engine.model
    key = (tensor_key(engine.recurrent), int(model.memory.clock))
    def _compute():
        intero = engine.intero().detach().reshape(-1).tolist()
        rec_rms = torch.log1p(engine.recurrent.detach().float().square().mean(-1).sqrt()).reshape(-1).tolist()
        return intero + rec_rms
    return read(model, 'nad_recipe_common_scalars', key, _compute, anchors=(engine.recurrent,))


def _lifted_intention(model, zero_row):
    intention = getattr(model, '_question_context', None)
    if intention is None:
        return zero_row
    from scripts.ib_local.prospective_decision import lift_boundary
    from scripts.ib_local.preview_runtime import read, tensor_key
    return read(model, 'nad_recipe_intention', tensor_key(intention),
                lambda: lift_boundary(model, zero_row, intention.reshape(1, -1)), anchors=(intention,))


@torch.no_grad()
def recipe(engine, state, raw, *, selected=None, momentum=None):
    """Canonical raw payload, re-encoded after future parameter updates."""
    from scripts.ib_local.causal_relation_program import Port
    from scripts.ib_local.prospective_decision import lift_boundary, preview_bind
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    model = engine.model
    if selected is None:
        selected, _ = preview_bind(model, raw, overlay=state)
    if momentum is None:
        momentum = decision_momentum_read(model, engine.bfss_momenta, raw)
    zero_row = torch.zeros_like(raw)
    pm = model.momentum_to_boundary(momentum)
    blocks = [raw, lift_boundary(model, zero_row, pm)]
    # Native word execution uses the CURRENT body anchor as well as the
    # predicted raw boundary. Its retrieved action portfolio is also input.
    anchor = getattr(engine, '_internal_boundary', None)
    blocks.append(zero_row if anchor is None else anchor.reshape_as(raw))
    blocks.append(_lifted_intention(model, zero_row))
    memory = model.memory
    read_out = memory.read_encoded(model.encoder(raw), model.encoder)
    slots, probabilities = read_out.slots[0], read_out.weights[0, :, None]
    blocks.extend(((probabilities*memory.source[slots]).sum(0, keepdim=True),
                   (probabilities*(memory.action[slots]-memory.source[slots])).sum(0, keepdim=True),
                   (probabilities*memory.intervening[slots]).sum(0, keepdim=True)))
    roles = []
    denom = max(1, len(state.valid))
    for st in (state, selected, state.organization):
        for port in (Port.MOTHER, Port.DECISION):
            rblocks, count = _role_blocks(model, st, port, zero_row)
            blocks.extend(rblocks)
            roles.append(count/denom)
    for st in (state, selected):
        binding = st.binding
        tag = [0.]*4
        tag[3 if binding is None else (Port.MOTHER, Port.ACTION, Port.CONDITION).index(binding.port)] = 1.
        roles.extend(tag+_hash_features('none' if binding is None else binding.identity))
    weights = state.program_weights
    roles.extend([.5, .5] if weights is None else weights.detach().reshape(-1).tolist())
    use = getattr(state, 'retention_use', None)
    moments = () if use is None else use.moments
    if getattr(model, 'retention_scope_prior_mode', None) == 'relation' and use is not None:
        from scripts.ib_local.relation_retention_use import effective_rows
        moments = tuple((identity, *row) for identity, row in effective_rows(use).items())
    # Retention's already accumulated evidence participates in the predictor.
    roles.extend([sum(float(r[j]) for r in moments)/max(1, len(moments)) for j in range(1, 4)])
    roles.extend([math.log1p(engine.max_utterance), len(state.valid), engine.gamma])
    roles.extend(_common_recipe_scalars(engine))
    wc_prog = torch.stack(((probabilities[:, 0]*memory.word_count[slots]).sum(),
                           (probabilities[:, 0]*memory.progress[slots]).sum())).cpu().tolist()
    roles.extend([float(wc_prog[0]), float(wc_prog[1])])
    return dict(raw=torch.cat(blocks).detach().cpu(), scalars=roles,
        relation=(state.identity, state.generation, state.version),
        unknown=None if state.binding is None else state.binding.identity)


@torch.no_grad()
def recipes_batched(engine, states, sources, *, carried=None):
    """Exact batched counterpart of `[recipe(engine, carried or st, src, selected=st) ...]`."""
    k = len(sources)
    if k <= 1:
        return [recipe(engine, carried or st, src, selected=st) for st, src in zip(states, sources)]
    from scripts.ib_local.causal_relation_program import Port
    from scripts.ib_local.prospective_decision import lift_boundary
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    model, memory = engine.model, engine.model.memory
    raws = torch.cat(sources, 0)
    zero_row = torch.zeros_like(sources[0])
    momenta = decision_momentum_read(model, engine.bfss_momenta, raws)
    pm_lifted = lift_boundary(model, torch.zeros_like(raws), model.momentum_to_boundary(momenta))
    anchor = getattr(engine, '_internal_boundary', None)
    anchor_row = zero_row if anchor is None else anchor.reshape_as(zero_row)
    intent_row = _lifted_intention(model, zero_row)
    read_out = memory.read_encoded(model.encoder(raws), model.encoder)
    slots_all, probs_all = read_out.slots, read_out.weights
    p3 = probs_all[..., None]
    mem_src = (p3 * memory.source[slots_all]).sum(1)
    mem_dact = (p3 * (memory.action[slots_all] - memory.source[slots_all])).sum(1)
    mem_int = (p3 * memory.intervening[slots_all]).sum(1)
    wc_all = (probs_all * memory.word_count[slots_all]).sum(1)
    pr_all = (probs_all * memory.progress[slots_all]).sum(1)
    wc_pr_list = torch.stack((wc_all, pr_all), -1).cpu().tolist()
    common_tail = _common_recipe_scalars(engine)
    raw_batches = []
    scalars_list = []
    meta_list = []
    for i, (selected, raw) in enumerate(zip(states, sources)):
        state = carried or selected
        blocks = [raw, pm_lifted[i:i+1], anchor_row, intent_row,
                  mem_src[i:i+1], mem_dact[i:i+1], mem_int[i:i+1]]
        roles = []
        denom = max(1, len(state.valid))
        for st in (state, selected, state.organization):
            for port in (Port.MOTHER, Port.DECISION):
                rblocks, count = _role_blocks(model, st, port, zero_row)
                blocks.extend(rblocks)
                roles.append(count/denom)
        for st in (state, selected):
            binding = st.binding
            tag = [0.]*4
            tag[3 if binding is None else (Port.MOTHER, Port.ACTION, Port.CONDITION).index(binding.port)] = 1.
            roles.extend(tag+_hash_features('none' if binding is None else binding.identity))
        weights = state.program_weights
        roles.extend([.5, .5] if weights is None else weights.detach().reshape(-1).tolist())
        use = getattr(state, 'retention_use', None)
        moments = () if use is None else use.moments
        if getattr(model, 'retention_scope_prior_mode', None) == 'relation' and use is not None:
            from scripts.ib_local.relation_retention_use import effective_rows
            moments = tuple((identity, *row) for identity, row in effective_rows(use).items())
        roles.extend([sum(float(r[j]) for r in moments)/max(1, len(moments)) for j in range(1, 4)])
        roles.extend([math.log1p(engine.max_utterance), len(state.valid), engine.gamma])
        roles.extend(common_tail)
        roles.extend([float(wc_pr_list[i][0]), float(wc_pr_list[i][1])])
        raw_batches.append(torch.cat(blocks))
        scalars_list.append(roles)
        meta_list.append(((state.identity, state.generation, state.version),
                          None if state.binding is None else state.binding.identity))
    raw_cpu = torch.stack(raw_batches, 0).detach().cpu()
    return [dict(raw=raw_cpu[i].contiguous(), scalars=scalars_list[i],
                 relation=meta_list[i][0], unknown=meta_list[i][1])
            for i in range(k)]


def features(engine, packets):
    raw = torch.stack([p['raw'] for p in packets]).to(engine.model.encoder.weight)
    z = engine.model.encoder(raw)
    # All role blocks remain distinct, including explicit zero/missing blocks.
    directions = F.normalize(z, dim=-1).flatten(1)/math.sqrt(z.shape[1])
    scales = torch.log1p(z.square().mean(-1).sqrt())
    scalars = z.new_tensor([p['scalars'] for p in packets])
    return torch.cat((directions, F.normalize(torch.cat((scales, scalars), -1), dim=-1)), -1)/math.sqrt(2)


def _prior(engine, packets):
    from scripts.ib_local.navigation_internal_rollout import continuation
    raw = torch.cat([p['raw'][:1] for p in packets]).to(engine.model.encoder.weight)
    return continuation(engine.model, engine.model.encoder(raw),
                        reward_valid=engine.reward_label_valid(wood=True))[0]


def _kernel(x, y):
    return (1+x@y.T).square()


@torch.no_grad()
def predict(engine, packets):
    bank = engine.native_value_bank
    key = (bank['revision'], int(engine.model.memory.clock),
           tuple(p._version for p in engine.model.parameters()))
    cache = engine._native_fit_cache
    if cache is None or cache['key'] != key:
        records = bank['records']
        if records:
            inputs = [r['packet'] for r in records]
            x = features(engine, inputs)
            prior = _prior(engine, inputs)
            y = x.new_tensor([r['target'] for r in records])-prior
            # Balance arrived-world and imagined-native evidence as distinct
            # sources; imagined forecasts never enter the real causal archive.
            observed = sum(r['kind'] == 'world' for r in records)
            count = {k: sum(r['kind'] == k for r in records) for k in ('world', 'native')}
            scale = x.new_tensor([len(records)/(len([n for n in count.values() if n])*count[r['kind']])
                                  for r in records]).sqrt()
            gram = _kernel(x, x)*scale[:, None]*scale[None]
            factor = torch.linalg.cholesky(gram+engine.relation_ridge*torch.eye(len(x), device=x.device))
            coefficient = torch.cholesky_solve((y*scale)[:, None], factor)
            cache = dict(key=key, x=x, scale=scale, factor=factor, coefficient=coefficient,
                         observed=observed)
        else:
            cache = dict(key=key, x=None)
        engine._native_fit_cache = cache
    base = _prior(engine, packets)
    query = features(engine, packets)
    if cache['x'] is None:
        return base, base, torch.ones_like(base)
    cross = _kernel(query, cache['x'])*cache['scale'][None]
    estimate = base+(cross@cache['coefficient'])[:, 0]
    solved = torch.linalg.solve_triangular(cache['factor'], cross.T, upper=False)
    diagonal = (1+query.square().sum(-1)).square()
    novelty = ((diagonal-solved.square().sum(0))/diagonal).clamp(0, 1)
    return estimate, base, novelty


def _retain_pending(engine, packet, target, prediction, prior, kind):
    bank = engine.native_value_bank
    key = hashlib.sha256(packet['raw'].numpy().tobytes()+repr(packet['scalars']).encode()+kind.encode()).hexdigest()
    bank['pending'][key] = dict(packet=packet, target=float(target), predicted=float(prediction),
                               prior=float(prior), kind=kind, key=key, revision=bank['revision'])
    cap = engine.model.causal_relations.evidence_budget**2
    if len(bank['pending']) > cap:
        # Stable content priorities: a bounded, order-independent teacher set.
        teacher_keys = [k for k, r in bank['pending'].items() if r['kind']=='native']
        del bank['pending'][max(teacher_keys or bank['pending'])]


@torch.no_grad()
def value(engine, state, source, action, mother, distribution, momentum):
    if engine.native_amortization.get('planner') == 'action_batch':
        from scripts.ib_local.latent_action_planner import value as planned_value
        return planned_value(engine, state, source, action, mother, distribution, momentum)
    from scripts.ib_local.joint_successor import quadrature
    from scripts.ib_local.prospective_decision import boundary, value as native_value
    points, weights, geometry = quadrature(engine, state, source, action, mother, distribution)
    packets = []
    for point in points[0]:
        raw, _ = boundary(engine, state, source, action, mother, point[None])
        packets.append(recipe(engine, state, raw))
    predicted, prior, novelty = predict(engine, packets)
    quality = engine.native_value_bank['quality']
    ratio = (sum(r['error2'] for r in quality)/max(sum(r['prior_error2'] for r in quality), 1e-20)
             if quality else 1.)
    k = len(packets)
    n = min(k, max(1, math.ceil(k*max(ratio, float((weights*novelty).sum())))))
    if n == k:
        index = torch.arange(k, device=source.device)
        correction_weights = weights
    else:
        generator = torch.Generator().manual_seed(engine.native_amortization['seed'])
        draw = torch.multinomial(weights.cpu(), n, replacement=True, generator=generator)
        unique, counts = draw.unique(return_counts=True)
        index = unique.to(source.device)
        correction_weights = counts.to(weights)/n
    def expand(tensor):
        return tensor.expand(len(index), *tensor.shape[1:])
    native = native_value(engine, state, expand(source), expand(action), expand(mother),
                          points[0, index], expand(momentum))
    if n == k:
        # Exact original centering also preserves the original full-budget
        # floating execution, before the learned predictor earns a smaller cost.
        answer = native[0]+(weights*(native-native[0])).sum()
    else:
        answer = predicted[0]+(weights*(predicted-predicted[0])).sum()
        answer = answer+(correction_weights*(native-predicted[index])).sum()
    for j, result in zip(index.tolist(), native.tolist()):
        _retain_pending(engine, packets[j], result, predicted[j], prior[j], 'native')
    stats = engine.native_value_bank.setdefault('epoch_stats', dict(calls=0, nodes=0, native=0))
    stats['calls'] += 1
    stats['nodes'] += k
    stats['native'] += len(index)
    engine.successor_distribution_last = dict(schema=3, budget=k, branches=k,
        weights=weights.tolist(), values=[predicted.tolist()], expected=[float(answer)],
        df=float(distribution['df']), scale_mean=float(distribution['scale2'].mean()), joint=geometry,
        native_indices=index.tolist(), native_values=native.tolist(), correction_weights=correction_weights.tolist(),
        native_count=len(index), allocated_draws=n, prequential_error_ratio=ratio,
        novelty=novelty.tolist(), bank_revision=engine.native_value_bank['revision'],
        semantics='same schema5 nodes; frozen conditional predictor plus original native residuals')
    return answer.reshape(1)


@torch.no_grad()
def capture_actual(engine, state, source):
    if engine.native_amortization.get('correction') == 'decision_budget':
        # Internal autonomous steps may have advanced S/P since those previews.
        # Only the current actual question/body boundary supplies teachers.
        engine._latent_teacher_queue = {}
    packet = recipe(engine, state, source, selected=state)
    predicted, prior, _ = predict(engine, [packet])
    engine._native_actual_packet = (packet, float(predicted[0]), float(prior[0]))


@torch.no_grad()
def settle(engine, actual, *, committed):
    planner_report = None
    if engine.native_amortization.get('planner') == 'action_batch':
        from scripts.ib_local.latent_action_planner import settle as settle_planner
        planner_report = settle_planner(engine, actual, committed=committed)
    bank = engine.native_value_bank
    actual_packet = getattr(engine, '_native_actual_packet', None)
    if actual_packet is not None and committed:
        packet, predicted, prior = actual_packet
        _retain_pending(engine, packet, actual, predicted, prior, 'world')
    cap = engine.model.causal_relations.evidence_budget**2
    new = list(bank['pending'].values())
    joined = {r['key']: r for r in [*bank['records'], *new]
              if r['revision'] >= bank['revision']-cap}
    world = sorted((r for r in joined.values() if r['kind']=='world'), key=lambda r:r['key'])[:cap//2]
    teacher = sorted((r for r in joined.values() if r['kind']=='native'), key=lambda r:r['key'])[:cap-len(world)]
    bank['records'] = world+teacher
    bank['quality'] = [dict(error2=(r['target']-r['predicted'])**2,
                            prior_error2=(r['target']-r['prior'])**2)
                       for r in new if r['kind']=='native']
    bank['pending'] = {}
    bank['revision'] += 1
    engine._native_fit_cache = None
    engine._native_actual_packet = None
    stats = bank.pop('epoch_stats', dict(calls=0, nodes=0, native=0))
    return dict(revision=bank['revision'], world_labels=len(world), native_labels=len(teacher),
                new_world_label=actual_packet is not None and committed,
                capacity=cap, records=len(bank['records']), units='10*finite_U',
                calls=stats['calls'], proposed_nodes=stats['nodes'], native_nodes=stats['native'],
                real_prediction=None if actual_packet is None else actual_packet[1],
                real_target=float(actual) if committed else None,
                real_prior=None if actual_packet is None else actual_packet[2],
                labels='arrived world and imagined native remain explicitly tagged',
                **({'action_planner': planner_report} if planner_report is not None else {}))
