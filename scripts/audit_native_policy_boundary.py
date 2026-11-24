"""Inspect the actual conditional actor in a private complete-body continuation.

No world step or optimizer update. Historical feelings are used only for
same-boundary input interventions, never as known future state.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--momentum-repair',action='store_true')
    args = p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local.epistemic_value_reference import from_body, continuation
    from scripts.ib_local.causal_use_value import queries, calibrated
    torch.set_num_threads(1)
    before = sha256_file(args.checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(args.checkpoint),
        '--output', str(args.checkpoint.parent), '--resume-output','--cycles', '1', '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(args.work/'table'), '--causal-storage', str(args.work/'causal')])
    loaded = prepare_continuation(opts,resume_output=True)
    e, model = loaded.engine, loaded.engine.model
    try:
        scheduling = []
        for _ in range(64):
            if e.autonomous_ready:
                break
            row = dispatch(e)
            assert row['autonomous_event']['kind'] != 'external'
            scheduling.append(row['autonomous_event'])
        else:
            raise RuntimeError('no natural release within audit cap; no world step forced')
        model._causal_archive.prepare()
        e._identified_event_fact = None
        # Actual external dispatch draws these directions before step/_utter.
        torch.randn(model.encoder.out_features, 32, device=e.device)
        capture = {}
        plan, choice_offset = e.plan, e._choice_offset

        def tracked_plan(h):
            result = plan(h)
            capture.update(source=h.detach().clone(), habit=result[0].detach().clone(),
                           plan=result[1], reusable=result[2].detach().clone(),
                           intero=e.intero().detach().clone())
            return result

        def tracked_offset(values, available):
            result = choice_offset(values, available)
            capture.update(values=values.detach().clone(), available=available.detach().clone(),
                           precision=e.precision(), offset=result.detach().clone())
            return result

        e.plan, e._choice_offset = tracked_plan, tracked_offset
        try:
            words, record = e._utter()
        finally:
            del e.plan, e._choice_offset
        with torch.no_grad():
            habit = capture['habit'][0]
            logits = habit+capture['offset']
            probability = logits.softmax(-1)
            recorded = logits.new_tensor(record['choice_logits'])
            torch.testing.assert_close(logits, recorded, rtol=0, atol=0)
            packet = capture['plan']
            m = model.memory
            reusable = capture['reusable']
            reuse_weight = (packet['weights']*reusable).sum(-1, keepdim=True)
            prefix = torch.cat((packet['decision_feature'], packet['action'], reuse_weight), -1)

            def actor(feeling):
                raw = model.kanli.choice(torch.cat((prefix, feeling), -1))
                raw = torch.cat((raw[..., :2], raw[..., 2:].masked_fill(reuse_weight <= 0, -1e4)), -1)
                return raw[0], (raw[0]+capture['offset']).softmax(-1)

            reconstructed, _ = actor(capture['intero'])
            torch.testing.assert_close(reconstructed, habit, rtol=0, atol=0)
            q = logits.new_tensor(e.epistemic_value_last['local_expected'])
            actual_value = float((probability*q).sum())
            occupied = torch.nonzero(m.occupied & (m.count == 1)).flatten()
            interventions = []
            for slot in occupied.tolist():
                feeling = m.feeling[slot:slot+1]
                _, prob = actor(feeling)
                interventions.append(dict(slot=slot, stamp=int(m.last_write[slot]),
                    feeling=feeling[0].tolist(), probability=prob.tolist(),
                    tv_from_current=float((prob-probability).abs().sum()/2),
                    expected_same_q=float((prob*q).sum())))
            interventions.sort(key=lambda r:r['tv_from_current'], reverse=True)
            state = e._program_question_state
            context = from_body(e, state, queries(e, state))
            z = model.encoder(capture['source'])
            recall_value, detail = continuation(model, context, z,
                reward_valid=e.reward_label_valid(wood=True), ridge=e.relation_ridge,
                calibration=float(calibrated(e, 1.)))
            d = detail[0]
            mass = probability.new_zeros(3)
            for slot, weight in zip(d['slots'], d.get('weights', [])):
                mass[int(m.wood_option[slot])] += weight
            labels_value = float((mass*q).sum()) if d['slots'] else None
            # The actual valuation keeps all three typed ports. Without reuse,
            # the raw draft list has two entries and the unavailable third port
            # has a silent placeholder in _internal_action, exactly as in _value.
            candidate_actions = e._internal_action
            candidate_comparison = []
            for slot in d['slots']:
                option = int(m.wood_option[slot])
                recalled_action = z[0]+model.encoder(m.action[slot])-model.encoder(m.source[slot])
                candidate_comparison.append(dict(slot=slot, option=option,
                    latent_difference_to_actual_option=float((recalled_action-model.encoder(candidate_actions[option])).norm())))
            from scripts.ib_local.conditional_actor_reference import (
                encoded_view,evaluate,candidate_policy_decomposition)
            view=encoded_view(model,z,capture['intero'],reward_valid=e.reward_label_valid())
            reconstructed=evaluate(view,capture['values'][None]/10,
                precision=capture['precision'],gain_units_schema=e.gain_units_schema)
            logit_error=float((reconstructed['logits'][0]-logits).abs().max())
            torch.testing.assert_close(reconstructed['logits'][0],logits,rtol=2e-6,atol=3e-6)
            decomposition=candidate_policy_decomposition(
                probability.new_tensor(d.get('weights',[])),probability.new_tensor(d['gains']),
                torch.tensor([int(m.wood_option[s]) for s in d['slots']],device=e.device,dtype=torch.long),probability,q)
            # Same current body/actions/policy, cross the actually persisted old
            # question with the newly selected question. This is a retrospective
            # controlled value calculation, with no future label supplied.
            from scripts.ib_local.shared_unknown_bindings import _row,_with_constraints
            from scripts.ib_local import epistemic_value_reference as ev
            from scripts.ib_local.causal_relation_program import mechanism
            from scripts.ib_local.finite_use_value import expected as raw_expected
            previous=e.internal_pending['causal_program']['state']
            old_question=_with_constraints(model._causal_archive,previous,
                _row(model._causal_archive,previous.binding.identity))
            old_context=from_body(e,old_question,queries(e,old_question))
            new_context=from_body(e,state,queries(e,state))
            current_actions=model.encoder(candidate_actions)
            crossed={}
            raw_states=dict(old=old_question,new=state)
            raw_queries=dict(old=queries(e,old_question),new=queries(e,state))
            current_source=capture['source'].expand_as(candidate_actions)
            momentum=e._momentum_decision_read[None].expand(len(candidate_actions),-1)
            for rname,rc in (('old',old_context),('new',new_context)):
                dist=mechanism(model,raw_states[rname],current_source,candidate_actions,
                    ridge=e.relation_ridge,momentum=momentum)
                for qname,qc in (('old',old_context),('new',new_context)):
                    ctx=replace(rc,queries=qc.queries)
                    values=10*float(calibrated(e,1.))*raw_expected(e,raw_states[rname],
                        current_source,candidate_actions,dist,raw_queries[qname])
                    encoded=10*float(calibrated(e,1.))*ev.expected(model,ctx,z,current_actions,
                        ridge=e.relation_ridge,distribution=dist)
                    parity=float((values-encoded).abs().max())
                    assert parity<3e-5
                    crossed[rname+'_'+qname]=dict(q=values.tolist(),value=float((probability*values).sum()),
                                                raw_encoded_error=parity)
            oo,no,on,nn=(crossed[k]['value'] for k in ('old_old','new_old','old_new','new_new'))
            candidate_condition={}
            if d['slots']:
                slots=torch.tensor(d['slots'],device=e.device,dtype=torch.long)
                recalled_source=capture['source'].expand(len(slots),-1)
                recalled_actions=recalled_source+m.action[slots]-m.source[slots]
                matched_gains={}
                for name,p_read in (('without_momentum',None),('with_momentum',
                        e._momentum_decision_read[None].expand(len(slots),-1))):
                    distribution=mechanism(model,state,recalled_source,recalled_actions,
                        ridge=e.relation_ridge,momentum=p_read)
                    matched_gains[name]=10*float(calibrated(e,1.))*raw_expected(e,state,
                        recalled_source,recalled_actions,distribution,queries(e,state))
                original=probability.new_tensor(d['gains'])
                error=float((matched_gains['without_momentum']-original).abs().max())
                assert error<3e-5
                weights=probability.new_tensor(d['weights'])
                options=m.wood_option[slots]
                momentum_term=(weights*(original-matched_gains['with_momentum'])).sum()
                endpoint_term=(weights*(matched_gains['with_momentum']-q[options])).sum()
                candidate_condition=dict(momentum_condition_term=float(momentum_term),
                    physical_endpoint_term=float(endpoint_term),
                    policy_term=float(decomposition['policy_term']),
                    original_gain_reconstruction_error=error,
                    momentum_conditioned_gains=matched_gains['with_momentum'].tolist(),
                    total_reconstruction_error=float((momentum_term+endpoint_term+
                        decomposition['policy_term']-(decomposition['recall_value']-
                        decomposition['actual_policy_value'])).abs()),
                    meaning='same R/Q, parameters and recalled raw actions; only the original P-conditioned mother prediction changes before comparing actual endpoints')
            learning_scope=dict(old_unknown=old_question.binding.identity,
                current_unknown=state.binding.identity,
                old_support=[f.identity for f in old_context.facts],
                current_support=[f.identity for f in new_context.facts],
                crossed=crossed,retained_state_effect=no-oo,query_effect=on-oo,
                interaction=nn-no-on+oo,total=nn-oo,
                current_raw_encoded_q_error=float((q-q.new_tensor(crossed['new_new']['q'])).abs().max()),
                meaning='common current encoder/parameters/momentum, actual current actions and actor; old state from persisted previous question, without an arriving current response')
            assert learning_scope['current_raw_encoded_q_error']<3e-5
            repair=None
            if args.momentum_repair:
                from scripts.ib_local.causal_relation_program import imagine
                from scripts.ib_local.causal_use_value import utility
                original_schema=e.epistemic_value_schema
                changes=model._causal_archive.db.total_changes
                trials={}
                try:
                    for schema in (1,2):
                        ev.configure(e,schema=schema)
                        f=imagine(model,state,current_source,candidate_actions,ridge=e.relation_ridge,
                            reward_valid=e.reward_label_valid(wood=True),momentum=momentum)
                        total=utility(e,state,current_source,candidate_actions,f)
                        policy=evaluate(view,total[None],precision=capture['precision'],
                                        gain_units_schema=e.gain_units_schema)
                        trials[schema]=dict(forecast=f,total=total,policy=policy)
                finally:ev.configure(e,schema=original_schema)
                a0,a1=trials[1],trials[2]
                initial_error=float((a0['forecast']['path']['mother']['mean']-
                                     a1['forecast']['path']['mother']['mean']).abs().max())
                assert initial_error==0
                assert model._causal_archive.db.total_changes==changes
                repair=dict(old_future=a0['forecast']['epistemic_after_value'].tolist(),
                    carried_P_future=a1['forecast']['epistemic_after_value'].tolist(),
                    old_total=a0['total'].tolist(),carried_P_total=a1['total'].tolist(),
                    old_probability=a0['policy']['probability'][0].tolist(),
                    carried_P_probability=a1['policy']['probability'][0].tolist(),
                    policy_tv=float((a0['policy']['probability']-a1['policy']['probability']).abs().sum()/2),
                    same_initial_response_mean_error=initial_error,
                    old_actor_replay_error=float((a0['policy']['probability'][0]-probability).abs().max()),
                    preview_SQL_changes=model._causal_archive.db.total_changes-changes,
                    new_parameters=0,world_calls=0,optimizer_updates=0,
                    P_forecast='known projected current P carried for one prospective step; no later P observation')
            out = dict(source=str(args.checkpoint), source_sha256=before, cycle=loaded.cycle,
                source_unchanged=before == sha256_file(args.checkpoint), optimizer_updates=0, world_calls=0,
                scheduling=scheduling, scope='Actual actor conditional on its sampled drafts; not the marginal over future drafts.',
                event_credit_mode=e.event_credit_mode, settlement=e.settlement_kind,
                relation=dict(identity=state.identity, generation=state.generation, version=state.version),
                q=[f.identity for f in context.queries], actual_source=z[0].tolist(),
                actor=dict(habit=habit.tolist(), offset=capture['offset'].tolist(),
                    precision=capture['precision'], intero=capture['intero'][0].tolist(),
                    available=capture['available'].tolist(), logits=logits.tolist(),
                    probability=probability.tolist(), logits_reconstruction_max_error=float((logits-recorded).abs().max()),
                    same_q=q.tolist(), expected_same_q=actual_value, choice=record['choice'],
                    draft_words=[p['words'].hex() for p in e._joint_proposals],
                    proposal_selection=e._joint_selection, utterance_hex=words.hex()),
                encoded_actor=dict(logits_max_error=logit_error,
                    probability_max_error=float((reconstructed['probability'][0]-probability).abs().max()),
                    body_values=capture['values'].tolist(),
                    actual_endpoint_expected_use=float((reconstructed['probability'][0]*q).sum()),
                    meaning='same conditional actor, same current intero and actual candidate utilities; no inverse encoder or recurrent snapshot'),
                candidate_policy_decomposition={k:(v.tolist() if v.ndim else float(v))
                    for k,v in decomposition.items()},
                learning_scope_cross=learning_scope,
                candidate_condition_decomposition=candidate_condition,
                momentum_repair=repair,
                recalled=dict(value=float(recall_value[0]), detail=d,
                    option_mass=mass.tolist(),
                    expected_on_actual_option_q=labels_value,
                    same_q_policy_difference=None if labels_value is None else labels_value-actual_value,
                    meaning='Push recalled weights to option labels only. Its physical candidates differ; not an equality of candidate actions.',
                    candidate_comparison=candidate_comparison),
                intero_interventions=dict(count=len(interventions), largest=interventions[:2],
                    meaning='Only habit intero input changes, using actual stored feelings; z/K/q/candidates/precision fixed. Not future observations.'),
                missing_future_inputs=['future intero and precision/calibration',
                    'future proposal payloads and endpoint states',
                    'future eligible reuse set after event storage/learning',
                    'future applicability and query binding after state/parameter learning'])
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps(dict(cycle=loaded.cycle, source_unchanged=out['source_unchanged'],
            actor=out['actor']['probability'], recall_option_mass=mass.tolist(),
            policy_difference=out['recalled']['same_q_policy_difference'],
            max_intero_tv=interventions[0]['tv_from_current'] if interventions else 0.,
            output=str(args.output))))
    finally:
        model._causal_archive.close()


if __name__ == '__main__':
    main()
