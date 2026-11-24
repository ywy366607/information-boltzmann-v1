"""Compare existing program valuations on a complete private C2 boundary.

No optimizer/world step; candidate displacements follow the existing
_condition_mother_question rule. Production compose/continuation execute as-is.
"""
import argparse
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
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    status = a.work/'status.json'
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    write(status, dict(status='loading', pid=os.getpid()))
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.causal_relation_program import compose, Port
    from scripts.ib_local.relation_evidence import mechanism_evidence
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    from scripts.ib_local import navigation_internal_rollout as rollout
    from scripts.ib_local.causal_use_value import utility
    from scripts.run_native_relation_mode_use import sha
    torch.set_num_threads(1)
    started = time.perf_counter()
    original_sha = sha(a.checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(a.checkpoint),
        '--output', str(a.work/'body'), '--cycles', '1', '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(a.work/'table'), '--causal-storage', str(a.work/'causal')])
    loaded = prepare_continuation(opts)
    engine = loaded.engine
    assert engine.causal_use_arm == 'C2'
    write(status, dict(status='measuring', cycle=loaded.cycle))
    original = rollout.continuation
    observed = []
    def capture(*args, **kwargs):
        value, packet = original(*args, **kwargs)
        observed.append((args[1].detach().clone(), value.detach().clone(), packet))
        return value, packet
    rollout.continuation = capture
    try:
        with torch.no_grad():
            source = engine._read_source()
            state = engine.model.causal_relations.bind(engine.model, source)
            facts = mechanism_evidence(engine.model, state, Port.MOTHER)
            actions = torch.cat((source, *[source+f.action-f.source for f in facts]))[:engine.model.memory.topk]
            sources = source.expand(len(actions), -1)
            momentum = decision_momentum_read(engine.model, engine.bfss_momenta, source).expand(len(actions), -1)
            path = compose(engine.model, state, sources, actions, ridge=engine.relation_ridge,
                reward_valid=engine.reward_label_valid(wood=True), momentum=momentum)
            assert len(observed) == 3, 'expected mediated/direct/coordinated production calls'
            vm, vd, vc = [r[1] for r in observed]
            w = path['program_weights']
            assert torch.allclose(w.sum(), w.new_tensor(1.))
            assert torch.equal(vc, path['value'])
            assert torch.equal(observed[0][0], path['program_predictions'][0]['mean'])
            assert torch.equal(observed[1][0], path['program_predictions'][1]['mean'])
            total = utility(engine, state, sources, actions, path)
            mixed_v = w[0]*vm+w[1]*vd
            shift = engine.gamma*(mixed_v-vc)
            mixed_total = total+shift
            rows = []
            for i in range(len(actions)):
                rows.append(dict(candidate=i, origin='current' if i==0 else facts[i-1].fact_id,
                    mediated_value=float(vm[i]), direct_value=float(vd[i]),
                    coordinated_value=float(vc[i]), mixed_value=float(mixed_v[i]),
                    immediate_term=float(total[i]-engine.gamma*vc[i]),
                    coordinated_total=float(total[i]), mixed_total=float(mixed_total[i]),
                    shift=float(shift[i]), slots=[r[2]['slots'][i].tolist() for r in observed],
                    read_weights=[r[2]['weights'][i].tolist() for r in observed]))
            result = dict(status='measured', checkpoint=str(a.checkpoint), checkpoint_sha256=original_sha,
                cycle=loaded.cycle, relation=[state.identity,state.generation,state.version],
                candidate_scope='current condition plus actual recalled action displacements, native topk',
                weights=w.tolist(), gamma=engine.gamma, candidates=rows,
                coordinated_order=total.argsort(descending=True).tolist(),
                mixed_order=mixed_total.argsort(descending=True).tolist(),
                best_changed=int(total.argmax())!=int(mixed_total.argmax()),
                max_value_shift=float(shift.abs().max()),
                value_confidence=engine.causal_use_value_evidence.confidence().tolist(),
                calibration=engine.causal_use_calibration,
                source_unchanged=sha(a.checkpoint)==original_sha,
                updates=0, world_calls=0, seconds=time.perf_counter()-started)
    finally:
        rollout.continuation = original
        engine.model._causal_archive.close()
    assert result['source_unchanged']
    write(a.output, result)
    write(status, dict(status='completed', output=str(a.output), seconds=result['seconds']))
    print(json.dumps({k:result[k] for k in ('cycle','best_changed','max_value_shift','seconds')}))


if __name__ == '__main__':
    main()
