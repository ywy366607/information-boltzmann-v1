"""Read the actual C2 reward-head training/read contracts on stored episodes."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, default=0)
    args = p.parse_args()
    import torch
    from torch.nn import functional as F
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.run_native_relation_mode_use import sha
    torch.set_num_threads(1)
    birth_hash = sha(args.checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(args.checkpoint),
        '--output', str(args.work/'body'), '--cycles', '1', '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(args.work/'table'), '--causal-storage', str(args.work/'causal')])
    loaded = prepare_continuation(opts)
    e, m = loaded.engine, loaded.engine.model.memory
    try:
        with torch.no_grad():
            slots = torch.nonzero(m.occupied & (m.count == 1) &
                (m.wood_target_count == 1) & (m.wood_option >= 0) &
                e.reward_label_valid(wood=True)).flatten()
            assert len(slots)
            options = m.wood_option[slots]
            src, act = e.model.encoder(m.source[slots]), e.model.encoder(m.action[slots])
            x = torch.cat((src, act-src, F.one_hot(options, 3).to(src)), -1)
            head = e.model.kanli.outcome
            hidden = head[0](x)
            f = m.causal_evidence[slots]*m.causal_evidence_valid[slots, None]
            plain = head(x)[:, 3]
            contextual = head[2](head[1](hidden+e.model.kanli.causal_input(f)))[:, 3]
            old_base = m.wood_baseline[slots, 3]
            new_base = e.causal_use_value_evidence.bases()[options, 3]
            target = m.wood_target[slots, 3]
            training = old_base+contextual
            reading = new_base+plain
            difference = new_base-old_base+plain-contextual
            torch.testing.assert_close(reading-training, difference)
            # Exact minimizer in the old training output coordinate, applied
            # to these REAL labels/baselines/features, leaves this read error.
            old_fit_read_error = difference
            rows = [dict(slot=int(slots[i]), option=int(options[i]), target=float(target[i]),
                training_base=float(old_base[i]), reading_base=float(new_base[i]),
                plain=float(plain[i]), contextual=float(contextual[i]),
                training_prediction=float(training[i]), reading_prediction=float(reading[i]),
                contract_gap=float(difference[i])) for i in range(len(slots))]
            ev = e.causal_use_value_evidence
            result = dict(cycle=loaded.cycle, samples=len(rows), rows=rows,
                evidence={k:getattr(ev,k).tolist() for k in
                          ('seen','mean','blind','base_error','raw_error')},
                confidence=ev.confidence().tolist(),
                training_mse=float((training-target).square().mean()),
                reading_raw_mse=float((reading-target).square().mean()),
                baseline_mse=float((new_base-target).square().mean()),
                train_read_gap_mse=float(old_fit_read_error.square().mean()),
                base_gap_rms=float((new_base-old_base).square().mean().sqrt()),
                inlet_gap_rms=float((plain-contextual).square().mean().sqrt()),
                source_sha256=birth_hash, source_unchanged=sha(args.checkpoint)==birth_hash,
                optimizer_updates=0, world_calls=0)
            # Actual final-row features, not generated examples. Compare equal
            # parameter-displacement norms along old and shared-loss gradients.
            phi = torch.cat((head[1](hidden), torch.ones_like(target[:, None])), -1).double()
            old_phi = torch.cat((head[1](hidden+e.model.kanli.causal_input(f)),
                                 torch.ones_like(target[:, None])), -1).double()
            theta = torch.cat((head[2].weight[3], head[2].bias[3:4])).double()
            y, b, ob = target.double(), new_base.double(), old_base.double()
            error = b+phi@theta-y
            grad = 2*phi.T@error/len(y)
            old_grad = 2*old_phi.T@(ob+old_phi@theta-y)/len(y)
            rate = grad.square().sum()/(2*(phi@grad).square().mean())
            step = rate*grad
            old_step = old_grad*step.norm()/old_grad.norm()
            result['actual_last_row_step'] = dict(
                shared_before=float(error.square().mean()),
                shared_after=float((b+phi@(theta-step)-y).square().mean()),
                old_direction_after=float((b+phi@(theta-old_step)-y).square().mean()),
                gradient_cosine=float(F.cosine_similarity(grad[None], old_grad[None])[0]),
                matched_step_norm=float(step.norm()),
                scope='one analytic line-search step in cloned actual last row; complete body unchanged')
            from scripts.ib_local.causal_value_learning import prediction
            shared = prediction(e.model, src, act-src, options)[:, 3]
            torch.testing.assert_close(shared, reading)
            result['shared_function_read_max_error'] = float((shared-reading).abs().max())
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','evidence')}))
        if args.cycles:
            import gc
            import time
            from scripts.ib_local.autonomous_cycle import dispatch
            from scripts.native_policy_activation_storage import activation_storage, policy_storage
            from scripts.run_native_causal_use_value import fixed_queries, evaluate
            from scripts.ib_local.mother_checkpoint import save
            material = fixed_queries(e)
            evaluation_before = evaluate(e, material)
            old_policy, old_advance, old_objective = e._policy_grad, e._advance, e.model.objective
            hooks = [None]
            def policy(*a, **kw):
                gc.collect(); torch.cuda.empty_cache()
                with policy_storage(e.params, args.work/'policy_activations'):
                    return old_policy(*a, **kw)
            def advance(*a, **kw):
                gc.collect(); torch.cuda.empty_cache()
                return old_advance(*a, **kw)
            def objective(*a, **kw):
                with hooks[0]():
                    return old_objective(*a, **kw)
            e._policy_grad, e._advance, e.model.objective = policy, advance, objective
            rows = []
            start = time.perf_counter()
            status = args.work/'status.json'
            def status_write(**kw):
                status.write_text(json.dumps(kw), encoding='utf8')
            status_write(status='running', events=0, birth=loaded.cycle)
            try:
                with (args.work/'events.jsonl').open('w', encoding='utf8') as out:
                    for _ in range(max(512, args.cycles*64)):
                        with activation_storage(e.params, args.work/'world_activations') as (_, ctx):
                            hooks[0] = ctx
                            row = dispatch(e)
                        hooks[0] = None
                        out.write(json.dumps(row, ensure_ascii=False)+'\n'); out.flush()
                        if row['autonomous_event']['kind'] != 'external':
                            continue
                        assert row.get('world', {}).get('provider_calls', 0) == 0
                        assert abs(row['causal_gain_learning']['current_target']-10*row['metal_g']) < 1e-5
                        receipt = row['causal_use']['finite_U']
                        assert not receipt['committed'] or receipt['verified']
                        rows.append(row)
                        status_write(status='running', events=len(rows),
                            seconds=time.perf_counter()-start, gain=row['metal_g'],
                            replay=row['causal_gain_learning']['replay_samples'])
                        if len(rows) == args.cycles:
                            break
                    else:
                        raise RuntimeError('external event budget not reached')
            finally:
                del e._policy_grad, e._advance, e.model.objective
            after = evaluate(e, material)
            loaded.config['causal_gain_learning'] = dict(schema=1, adopted_at_cycle=loaded.cycle)
            digest = save(e, args.work/'body', loaded.cycle+len(rows), loaded.config)
            result['whole_loop'] = dict(events=len(rows), updates=sum(r['updates'] for r in rows),
                seconds=time.perf_counter()-start, before=evaluation_before, after=after,
                gains=[r['metal_g'] for r in rows],
                gain_learning=[r['causal_gain_learning'] for r in rows],
                confidence_after=e.causal_use_value_evidence.confidence().tolist(),
                checkpoint_sha256=digest, provider_calls=0,
                source_unchanged=sha(args.checkpoint)==birth_hash)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
            status_write(status='completed', events=len(rows), output=str(args.output))
    finally:
        e.model._causal_archive.close()


if __name__ == '__main__':
    main()
