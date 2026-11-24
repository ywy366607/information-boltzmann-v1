"""Read-only four-corner transport at an existing native teacher call.

The observer consumes the teacher's already computed post state, endpoints,
local q and total. It adds compose reads, not teachers or world interactions.
Install only inside an explicit measurement scope in one execution process.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from itertools import combinations
from pathlib import Path
import time

import torch


def _list(x):
    return x.detach().cpu().tolist()


@torch.no_grad()
def measure(engine, state, pre, proposal, nominal, options, post, endpoints, q, total,
            *, option_calibration=False, capsule_path=None):
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local.causal_relation_program import compose
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    from scripts.ib_local.conditional_actor_reference import encoded_view, evaluate
    from scripts.ib_local.navigation_gain_units import wood_choice_logits

    width = pre.shape[-1]
    packs = [planner.packet(engine, state, pre, proposal, a[None], option)
             for a, option in zip(nominal, options)]
    # plan() includes the unavailable reuse row in its common-state average.
    # In speech-only mode program_actions returns reuse=None, hence nominal=pre.
    if 2 not in options:
        if not getattr(engine.model, '_expression_speech_only', False):
            raise ValueError('teacher menu must include the native reuse row')
        packs.append(planner.packet(engine, state, pre, proposal, pre, 2))
    predicted, _ = planner.predict(engine, packs)
    student_post = pre+predicted[:, :width].mean(0, keepdim=True)
    rows = predicted[:len(options)]
    student_endpoints, student_q = pre+rows[:, width:2*width], rows[:, -1]
    prior_q = planner.prior(engine, packs)[:len(options), -1]
    from scripts.ib_local.causal_value_learning import prediction as gain_prediction
    option_tensor = torch.tensor(options, device=pre.device)
    pre_z, nominal_z = engine.model.encoder(pre), engine.model.encoder(nominal)
    raw_gain_q = gain_prediction(engine.model, pre_z.expand_as(nominal_z),
                                nominal_z-pre_z, option_tensor)[:, 3]
    gain_evidence = engine.model._causal_use_value_evidence
    base_q = gain_evidence.bases()[option_tensor, 3]
    confidence_q = gain_evidence.confidence()[option_tensor, 3]
    prior_accounting = base_q+confidence_q*(raw_gain_q-base_q)
    query = planner.features(engine, packs)[:len(options)]
    cache = engine._latent_action_fit
    source_parts = {}
    if cache['x'] is not None:
        from scripts.ib_local.native_amortized_decision import _kernel
        cross = _kernel(query, cache['x'])*cache['scale'][None]
        bank = engine.latent_action_bank['records']
        for kind in ('native', 'world'):
            keep = torch.tensor([r['kind'] == kind for r in bank], device=query.device)
            source_parts[kind] = (cross[:, keep]@cache['coef'][keep, -1:])[:, 0]
    else:
        source_parts = {kind: torch.zeros_like(prior_q) for kind in ('native', 'world')}
    prediction_accounting = prior_q+source_parts['native']+source_parts['world']

    def following(s, a):
        p = decision_momentum_read(engine.model, engine.bfss_momenta, s)
        return compose(engine.model, state, s.expand_as(a), a,
            ridge=engine.relation_ridge, reward_valid=engine.reward_label_valid(wood=True),
            momentum=p.expand(len(a), -1))['value']

    ss = following(student_post, student_endpoints)
    st = following(student_post, endpoints)
    ts = following(post, student_endpoints)
    tt = following(post, endpoints)
    rebuilt = q+engine.gamma*tt
    anchor_error = float((rebuilt-total).abs().max())
    tolerance = 2e-4*(1+float(total.abs().max()))
    if anchor_error > tolerance:
        raise RuntimeError(f'teacher total reconstruction differs: {anchor_error}')
    q_error = student_q-q
    endpoint_error = engine.gamma*(ss-st)
    state_error = engine.gamma*(st-tt)
    error = student_q+engine.gamma*ss-rebuilt
    interaction = engine.gamma*(ss-st-ts+tt)
    pairs = []
    for a, b in combinations(range(len(options)), 2):
        parts = [float(x[a]-x[b]) for x in (q_error, endpoint_error, state_error)]
        actual = float(error[a]-error[b])
        pairs.append(dict(indices=[a, b], options=[options[a], options[b]],
            nominal_max_difference=float((nominal[a]-nominal[b]).abs().max()),
            teacher_endpoint_max_difference=float((endpoints[a]-endpoints[b]).abs().max()),
            student_local_q_difference=float(student_q[a]-student_q[b]),
            teacher_local_q_difference=float(q[a]-q[b]),
            prior_q_difference=float(prior_q[a]-prior_q[b]),
            native_bank_q_difference=float(source_parts['native'][a]-source_parts['native'][b]),
            world_bank_q_difference=float(source_parts['world'][a]-source_parts['world'][b]),
            teacher_total_difference=float(rebuilt[a]-rebuilt[b]),
            student_total_difference=float(student_q[a]-student_q[b]+engine.gamma*(ss[a]-ss[b])),
            q_error=parts[0], endpoint_error=parts[1], common_state_error=parts[2],
            total_error=actual, reconstruction_error=abs(actual-sum(parts)),
            mixed_interaction=float(interaction[a]-interaction[b])))

    silence = options.index(0)
    reuse = options.index(2) if 2 in options else silence
    drafts = [j for j, option in enumerate(options) if option == 1]
    indices = torch.tensor([[silence, j, reuse] for j in drafts], device=pre.device)

    def policy(s, local_q, full):
        view = encoded_view(engine.model, engine.model.encoder(s), engine.intero(),
                            reward_valid=engine.reward_label_valid())
        expanded = replace(view, habit=view.habit.expand(len(drafts), -1),
                           available=view.available.expand(len(drafts), -1))
        actor = evaluate(expanded, full[indices], precision=engine.precision(),
                         gain_units_schema=engine.gain_units_schema)
        weights = wood_choice_logits(engine, full[drafts]).softmax(0)
        expected = (weights*(actor['probability']*local_q[indices]).sum(-1)).sum()
        return dict(actor_probability=_list(actor['probability']),
                    draft_probability=_list(weights), expected_local_use=float(expected),
                    teacher_local_use_under_policy=float((weights*(
                        actor['probability']*q[indices]).sum(-1)).sum()))

    policies = dict(
        student=policy(student_post, student_q, student_q+engine.gamma*ss),
        q_corrected=policy(student_post, q, q+engine.gamma*ss),
        endpoint_corrected=policy(student_post, student_q, student_q+engine.gamma*st),
        state_corrected=policy(post, student_q, student_q+engine.gamma*ts),
        teacher=policy(post, q, rebuilt))
    calibration = None
    if option_calibration:
        from scripts.option_basis_calibration import predict_q
        if pre.is_cuda:
            torch.cuda.synchronize(pre.device)
        started = time.perf_counter()
        calibrated, diagnostics, capsule = predict_q(engine, packs)
        if pre.is_cuda:
            torch.cuda.synchronize(pre.device)
        diagnostics['elapsed_ms'] = 1000*(time.perf_counter()-started)
        calibrated = calibrated[:len(options)]
        policies['option_calibrated'] = policy(
            student_post, calibrated, calibrated+engine.gamma*ss)
        calibration = dict(**diagnostics, q=_list(calibrated),
            original_q_mse=float((student_q-q).square().mean()),
            calibrated_q_mse=float((calibrated-q).square().mean()))
        if capsule is not None and capsule_path is not None:
            capsule.update(teacher_q=q.cpu(), ss=ss.cpu(), teacher_total=total.cpu(),
                kinds=[r['kind'] for r in engine.latent_action_bank['records']])
            Path(capsule_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(capsule, capsule_path)
            calibration['capsule'] = str(capsule_path)
    identity = hashlib.sha256(pre.detach().cpu().numpy().tobytes()+
        nominal.detach().cpu().numpy().tobytes()+repr(options).encode()+
        repr((state.identity, state.generation, state.version)).encode()).hexdigest()
    return dict(schema=1, identity=identity, bank_revision=engine.latent_action_bank['revision'],
        role=proposal['role'].value, relation=[state.identity, state.generation, state.version],
        unknown=None if state.binding is None else state.binding.identity,
        options=list(options), gamma=engine.gamma, teacher_anchor_max_error=anchor_error,
        vectors=dict(student_q=_list(student_q), teacher_q=_list(q), prior_q=_list(prior_q),
                     raw_gain_q=_list(raw_gain_q),
                     native_bank_q=_list(source_parts['native']), world_bank_q=_list(source_parts['world']),
                     ss=_list(ss), st=_list(st), ts=_list(ts), tt=_list(tt)),
        common_state_l2_error=float((student_post-post).norm()),
        endpoint_l2_errors=_list((student_endpoints-endpoints).norm(dim=-1)),
        score_reconstruction_max_error=float((error-q_error-endpoint_error-state_error).abs().max()),
        bank_prediction_reconstruction_max_error=float((student_q-prediction_accounting).abs().max()),
        prior_reconstruction_max_error=float((prior_q-prior_accounting).abs().max()),
        gain_calibration=dict(seen=_list(gain_evidence.seen),
            base_q=_list(gain_evidence.bases()[:, 3]), confidence_q=_list(gain_evidence.confidence()[:, 3]),
            base_error_q=_list(gain_evidence.base_error[:, 3]),
            raw_error_q=_list(gain_evidence.raw_error[:, 3])),
        pairs=pairs, policies=policies, option_calibration=calibration,
        extra_world_calls=0, extra_teacher_calls=0,
        target='same finite native q plus gamma times original compose continuation')


@contextmanager
def observe(engine, *, max_cases=2, option_calibration=False, capsule_dir=None):
    """Scope an observer around existing teacher execution; restore on exit."""
    from scripts.ib_local import latent_action_planner as planner
    original = planner.teach
    records = []

    def observed(e, *args, **kwargs):
        if e is engine and len(records) < max_cases:
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state_all() if args[1].is_cuda else None
            revision = engine.latent_action_bank['revision']
            capsule = None if capsule_dir is None else Path(capsule_dir)/f'case_{len(records)}.pt'
            records.append(measure(e, *args, **kwargs,
                                   option_calibration=option_calibration, capsule_path=capsule))
            if not torch.equal(rng, torch.get_rng_state()) or revision != engine.latent_action_bank['revision']:
                raise RuntimeError('read-only action observer changed RNG or bank revision')
            if cuda_rng is not None and any(not torch.equal(a, b) for a, b in
                    zip(cuda_rng, torch.cuda.get_rng_state_all())):
                raise RuntimeError('read-only action observer changed CUDA RNG')
        return original(e, *args, **kwargs)

    planner.teach = observed
    try:
        yield records
    finally:
        planner.teach = original
