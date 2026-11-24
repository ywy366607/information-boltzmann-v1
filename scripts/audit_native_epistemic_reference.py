"""Bind the encoded learning-state evaluator to one actual complete organism."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.causal_use_value import queries, calibrated
    from scripts.ib_local.causal_relation_program import mechanism, compose
    from scripts.ib_local.finite_use_value import expected, score_scope
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    from scripts.ib_local.navigation_internal_rollout import continuation
    from scripts.run_native_relation_mode_use import sha
    torch.set_num_threads(1)
    start = time.perf_counter()
    source_hash = sha(args.checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(args.checkpoint),
        '--output', str(args.work/'body'), '--cycles', '1', '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(args.work/'table'), '--causal-storage', str(args.work/'causal')])
    loaded = prepare_continuation(opts)
    e, model = loaded.engine, loaded.engine.model
    archive = model._causal_archive
    try:
        archive.db.execute('PRAGMA query_only=ON')
        changes = archive.db.total_changes
        with torch.no_grad():
            state = e._program_question_state
            # Last actual decision scope is preserved in the complete engine;
            # use it explicitly instead of allocating a new relation.
            src = e._internal_source.reshape(1,-1)
            actions = e._internal_action
            sources = src.expand_as(actions)
            facts = queries(e, state)
            context = ev.from_body(e, state, facts)
            assert len(context.facts) >= 2 and context.queries
            s, a = model.encoder(sources), model.encoder(actions)
            d = mechanism(model, state, sources, actions, ridge=e.relation_ridge)
            rd = ev.response(model, context, s, a, ridge=e.relation_ridge)
            native = expected(e, state, sources, actions, d, facts)
            ref = ev.expected(model, context, s, a, ridge=e.relation_ridge)
            torch.testing.assert_close(ref,native,rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(ev.scores(model, context,ridge=e.relation_ridge),
                                       score_scope(e,state,facts),rtol=3e-5,atol=3e-6)
            p_read = decision_momentum_read(model,e.bfss_momenta,src).expand(len(actions),-1)
            path = compose(model,state,sources,actions,ridge=e.relation_ridge,
                           reward_valid=e.reward_label_valid(wood=True),momentum=p_read)
            z = path['decision']['mean']
            old_value, old_packet = continuation(model,z,reward_valid=e.reward_label_valid(wood=True))
            alpha = float(calibrated(e,1.))
            value, detail = ev.continuation(model,context,z,
                reward_valid=e.reward_label_valid(wood=True),ridge=e.relation_ridge,calibration=alpha)
            # Same z, same candidate action prior, same Q. Change only actual
            # support membership by a fixed lexicographic identity rule.
            victim = min(context.facts,key=lambda f:f.identity)
            reduced = replace(context,facts=tuple(f for f in context.facts if f.identity != victim.identity))
            reduced_value, reduced_detail = ev.continuation(model,reduced,z,
                reward_valid=e.reward_label_valid(wood=True),ridge=e.relation_ridge,calibration=alpha)
            assert [r['slots'] for r in detail] == [r['slots'] for r in reduced_detail]
            # A branch's predicted response updates its bounded knowledge;
            # the next query reads that revised context at the SAME z.
            incoming = ev.LatentFact('event:-1:-1','mother:0:event:-1:-1',s[:1],a[:1],d['mean'][:1])
            revised = ev.revise(model,context,incoming,ridge=e.relation_ridge)
            revised_value, revised_detail = ev.continuation(model,revised,z[:1],
                reward_valid=e.reward_label_valid(wood=True),ridge=e.relation_ridge,calibration=alpha)
            reordered = replace(context,facts=context.facts[::-1])
            reordered = replace(reordered,facts=tuple(sorted(reordered.facts,key=lambda f:f.phase_identity)))
            reordered_value,_ = ev.continuation(model,reordered,z,
                reward_valid=e.reward_label_valid(wood=True),ridge=e.relation_ridge,calibration=alpha)
            torch.testing.assert_close(reordered_value,value,rtol=0,atol=0)
            result = dict(cycle=loaded.cycle, source=str(args.checkpoint), source_sha256=source_hash,
                source_unchanged=sha(args.checkpoint)==source_hash,
                context_source='last actual decision R/Q and actions; current learned operators',
                relation=[state.identity,state.generation,state.version], queries=[q.fact_id for q in facts],
                facts=[f.identity for f in context.facts], removed=victim.identity,
                native_expected=native.tolist(),encoded_expected=ref.tolist(),
                parity_max_error=float((native-ref).abs().max()),
                posterior_max_error=float((d['mean']-rd['mean']).abs().max()),
                calibration=alpha, old_continuation=old_value.tolist(),
                contextual_continuation=value.tolist(), removed_context_continuation=reduced_value.tolist(),
                same_z_context_effect=(reduced_value-value).tolist(),
                revised_context_continuation=revised_value.tolist(),
                same_z_revision_effect=(revised_value-value[:1]).tolist(),
                candidates=detail,removed_candidates=reduced_detail,revised_candidates=revised_detail,
                hypothetical_fact_count=len(revised.facts), bounded_budget=context.budget,
                permutation_max_error=float((reordered_value-value).abs().max()),
                sql_changes=archive.db.total_changes-changes,optimizer_updates=0,world_calls=0,
                scope='frozen actual-body read experiment; next expected use retains branch learning state',
                seconds=time.perf_counter()-start)
        assert result['source_unchanged'] and result['sql_changes']==0
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({k:v for k,v in result.items() if k not in
            ('candidates','removed_candidates','revised_candidates')},ensure_ascii=False))
    finally:
        archive.close()


if __name__ == '__main__':
    main()
