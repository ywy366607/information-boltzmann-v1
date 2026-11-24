"""Retrospective substitutions on one real body's prospective decision.

The later response and pre-intention boundary come from the completed native
trajectory. They are diagnostic inputs only: no optimizer/world call and no
change to the actual choice. Each case reuses the original typed U, question
selector, native self executor, actor and utility. Future labels never enter
the production planner.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time


def archived_vectors(path, episode):
    import numpy as np
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        schema = json.loads(db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0])
        payload = db.execute('SELECT payload FROM episodes WHERE id=?', (episode,)).fetchone()[0]
        result, offset = {}, 0
        for name in sorted(schema):
            spec = schema[name]
            count = int(np.prod(spec['shape'])) if spec['shape'] else 1
            dtype = np.dtype(spec['dtype'])
            result[name] = np.frombuffer(payload, dtype=dtype, count=count,
                                         offset=offset).copy().reshape(spec['shape'])
            offset += count*dtype.itemsize
        assert offset == len(payload)
        assert bool(result['internal_valid']) and bool(result['mother_valid'])
        return result
    finally:
        db.close()


def measure_cases(engine, later, episode):
    """Same retrospective substitutions at the REAL _utter return boundary."""
    import torch
    from scripts.ib_local import prospective_decision as pd
    from scripts.ib_local.causal_relation_program import Evidence, Port, revise, compose
    from scripts.ib_local.preview_runtime import preview_scope
    model = engine.model
    source = engine._internal_source.reshape(1, -1)
    selected = engine._audit_actual_choice
    action = engine._internal_action[selected:selected+1]
    tensor = lambda name: torch.as_tensor(later[name],device=engine.device,dtype=source.dtype).reshape(1,-1)
    source_error = float((source-tensor('source')).abs().max())
    action_error = float((action-tensor('action')).abs().max())
    if max(source_error,action_error)>3e-5:
        raise RuntimeError(f'archived actual ports differ: source={source_error}, action={action_error}')
    state = engine._program_question_state
    momentum = engine._momentum_decision_read[None]
    actual_boundary = tensor('internal_next')
    actual_y = model.encoder(tensor('mother_next'))
    mean_y = engine._program_forecasts['path']['mother']['mean'][selected:selected+1]
    sql = model._causal_archive.db.total_changes
    cpu,gpu = torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
    bank = {n:t.clone() for n,t in model.causal_relations.named_buffers()}
    original,old_trace = pd.boundary,getattr(engine,'prospective_decision_last',None)
    rows = []
    try:
        with torch.no_grad(),preview_scope(model):
            for name,y,coordinate,identified in (
                ('mean_response_predicted_boundary',mean_y,'predicted',False),
                ('actual_response_predicted_boundary',actual_y,'predicted',False),
                ('actual_response_actual_latent',actual_y,'actual_latent',False),
                ('actual_response_actual_raw',actual_y,'actual_raw',False),
                ('actual_response_actual_raw_identified',actual_y,'actual_raw',True)):
                updated = revise(state,Evidence(source,action,y,Port.MOTHER,False,encoded=True,
                    fact_id=f'episode:{episode}' if identified else ''),model=model)
                path = compose(model,updated,source,action,ridge=engine.relation_ridge,
                    reward_valid=engine.reward_label_valid(wood=True),momentum=momentum,mother=y)
                predicted,info = original(engine,updated,source,action,y,path['decision']['mean'])
                fixed = (pd.lift_boundary(model,predicted,model.encoder(actual_boundary))
                         if coordinate=='actual_latent' else actual_boundary if coordinate=='actual_raw' else None)
                pd.boundary = original if fixed is None else lambda *a,**kw: (fixed,dict(actual_boundary_substitution=True,latent_residual=0.))
                try:
                    value = pd.value(engine,updated,source,action,y,path['decision']['mean'],momentum)
                finally:
                    pd.boundary = original
                rows.append(dict(case=name,value=float(value[0]),
                    raw_MSE=float((predicted-actual_boundary).square().mean()),
                    latent_MSE=float((path['decision']['mean']-model.encoder(actual_boundary)).square().mean()),
                    details=engine.prospective_decision_last))
    finally:
        pd.boundary=original
        engine.prospective_decision_last=old_trace
    checks=dict(sql_read_only=model._causal_archive.db.total_changes==sql,
        cpu_rng=torch.equal(cpu,torch.get_rng_state()),gpu_rng=torch.equal(gpu,torch.cuda.get_rng_state()),
        bank_unchanged=all(torch.equal(t,bank[n]) for n,t in model.causal_relations.named_buffers()))
    assert all(checks.values()),checks
    return dict(source_max_error=source_error,action_max_error=action_error,cases=rows,checks=checks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local import prospective_decision as pd
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local.causal_relation_program import Evidence, Port, revise, compose
    from scripts.ib_local.preview_runtime import preview_scope
    torch.set_num_threads(1)
    start = time.perf_counter()
    a.work.mkdir(parents=True, exist_ok=False)
    def status(**kw):
        (a.work/'status.json').write_text(json.dumps(kw, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    status(status='loading')
    digest = sha256_file(a.checkpoint)
    events = [json.loads(line) for line in (a.reference/'events.jsonl').read_text(encoding='utf8').splitlines()]
    events = [row for row in events if 'causal_use' in row]
    first, following = events[:2]
    episode = int(first['causal_program']['committed_episode_identity'].split(':')[1])
    archives = list((a.reference/'body').glob('causal_archive_*.sqlite'))
    if len(archives) != 1:
        raise ValueError('require the completed reference archive identity')
    later = archived_vectors(archives[0], episode)
    opts = continuation_parser().parse_args(['--checkpoint', str(a.checkpoint),
        '--output', str(a.checkpoint.parent), '--resume-output', '--cycles', '1',
        '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(a.work/'table'), '--causal-storage', str(a.work/'causal')])
    loaded = prepare_continuation(opts, resume_output=True)
    e, model = loaded.engine, loaded.engine.model
    try:
        ev.configure(e, schema=3, word_execution='native_anchor')
        for _ in range(64):
            if e.autonomous_ready:
                break
            assert dispatch(e)['autonomous_event']['kind'] != 'external'
        else:
            raise RuntimeError('native release unavailable')
        # Match the complete outer step before observing the utterance. This
        # transports identity/version-matched coevolution workspaces once.
        if hasattr(model.body,'coevolution'):
            from scripts.ib_local.past_present_coevolution import refresh_workset
            refresh_workset(e)
        model._causal_archive.prepare()
        e._identified_event_fact = None
        torch.randn(model.encoder.out_features, 32, device=e.device)
        status(status='actual decision')
        words, record = e._utter()
        checks = dict(same_choice=record['choice']==first['causal_use']['choice'],
                      same_utterance=words.hex()==first['utterance_hex'],
                      same_current_unknown=e._program_question_roles['unknown_identity']==first['causal_use']['unknown'])
        source = e._internal_source.reshape(1, -1)
        action = e._internal_action[record['choice']:record['choice']+1]
        state = e._program_question_state
        momentum = e._momentum_decision_read[None]
        tensor = lambda name: torch.as_tensor(later[name], device=e.device, dtype=source.dtype).reshape(1, -1)
        actual_mother, actual_boundary = tensor('mother_next'), tensor('internal_next')
        actual_y = model.encoder(actual_mother)
        mean_y = e._program_forecasts['path']['mother']['mean'][record['choice']:record['choice']+1]
        source_error = float((source-tensor('source')).abs().max())
        action_error = float((action-tensor('action')).abs().max())
        checks['same_archived_source'] = source_error<3e-5
        checks['same_archived_action'] = action_error<3e-5
        if not checks['same_archived_source'] or not checks['same_archived_action']:
            raise RuntimeError(f'archived actual ports differ: source={source_error}, action={action_error}')
        sql = model._causal_archive.db.total_changes
        cpu, gpu = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        bank = {name: value.clone() for name, value in model.causal_relations.named_buffers()}
        read_boundary = pd.boundary
        rows = []
        with torch.no_grad(), preview_scope(model):
            for name, y, coordinate, identified in (
                ('mean_response_predicted_boundary', mean_y, 'predicted', False),
                ('actual_response_predicted_boundary', actual_y, 'predicted', False),
                ('actual_response_actual_latent', actual_y, 'actual_latent', False),
                ('actual_response_actual_raw', actual_y, 'actual_raw', False),
                ('actual_response_actual_raw_identified', actual_y, 'actual_raw', True)):
                status(status='substitutions', case=name)
                incoming = Evidence(source, action, y, Port.MOTHER, False, encoded=True,
                                    fact_id=f'episode:{episode}' if identified else '')
                updated = revise(state, incoming, model=model)
                path = compose(model, updated, source, action, ridge=e.relation_ridge,
                    reward_valid=e.reward_label_valid(wood=True), momentum=momentum, mother=y)
                predicted_raw, dynamics = read_boundary(e, updated, source, action, y, path['decision']['mean'])
                if coordinate == 'actual_latent':
                    fixed = pd.lift_boundary(model, predicted_raw, model.encoder(actual_boundary))
                elif coordinate == 'actual_raw':
                    fixed = actual_boundary
                else:
                    fixed = None
                def override(*args, **kw):
                    return fixed, dict(actual_boundary_substitution=True, latent_residual=0.)
                pd.boundary = read_boundary if fixed is None else override
                begin = time.perf_counter()
                try:
                    value = pd.value(e, updated, source, action, y, path['decision']['mean'], momentum)
                finally:
                    pd.boundary = read_boundary
                details = e.prospective_decision_last
                rows.append(dict(case=name, value=float(value[0]), seconds=time.perf_counter()-begin,
                    raw_MSE=float((predicted_raw-actual_boundary).square().mean()),
                    latent_MSE=float((path['decision']['mean']-model.encoder(actual_boundary)).square().mean()),
                    details=details))
                print(json.dumps({k:rows[-1][k] for k in ('case','value','raw_MSE','latent_MSE','seconds')}),flush=True)
        checks.update(source_unchanged=sha256_file(a.checkpoint)==digest,
            sql_read_only=model._causal_archive.db.total_changes==sql,
            cpu_rng=torch.equal(cpu, torch.get_rng_state()), gpu_rng=torch.equal(gpu, torch.cuda.get_rng_state()),
            bank_unchanged=all(torch.equal(value,bank[name]) for name,value in model.causal_relations.named_buffers()))
        f = first['causal_use']['epistemic_value']['contextual_future'][record['choice']]
        nf = following['causal_use']['epistemic_value']['local_expected']
        actual_value = sum(x*y for x,y in zip(following['choice_probability'], nf))
        chain = [f]+[r['value'] for r in rows]+[actual_value]
        differences = [x-y for x,y in zip(chain,chain[1:])]
        out = dict(valid=all(checks.values()), checks=checks, source_cycle=loaded.cycle,
            source_sha256=digest, reference=str(a.reference), source_episode=episode,
            source_max_error=source_error,action_max_error=action_error,
            original_integrated_value=f, actual_next_policy_value=actual_value,
            actual_next_unknown=following['causal_use']['unknown'], cases=rows,
            ordered_value_chain=chain, signed_differences=differences,
            reconstruction_residual=f-actual_value-sum(differences),
            interpretation='hindsight inputs are substituted in the old complete body; last difference includes later learning, physical state, scope and actual generation',
            optimizer_updates=0, world_calls=0, seconds=time.perf_counter()-start)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        status(status='completed' if out['valid'] else 'failed_checks', output=str(a.output),checks=checks)
        assert out['valid'],checks
    finally:
        pd.boundary = read_boundary if 'read_boundary' in locals() else pd.boundary
        model._causal_archive.close()


if __name__ == '__main__':
    main()
