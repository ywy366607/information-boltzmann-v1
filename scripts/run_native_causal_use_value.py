"""Serial A/B/C complete continuations from the immutable 2401 organism."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.run_native_relation_mode_use import ROOT, sha, write


def fixed_queries(engine):
    from scripts.ib_local.causal_use_value import queries
    bank = engine.model.causal_relations
    result = []
    indices = sorted((i for i in range(bank.capacity) if int(bank.identity[i]) > 0),
                     key=lambda i: int(bank.identity[i]))[:8]
    for i in indices:
        state = bank.state(i)
        facts = tuple(queries(engine, state))
        state = replace(state, constraints=facts, evidence_bound=True,
                        extra=(), organization=None, variable_uses=())
        for fact in facts:
            result.append((state, fact))
    return result


def evaluate(engine, material):
    import torch
    from scripts.ib_local.causal_relation_program import mechanism, log_score
    scores = []
    with torch.no_grad():
        for state, fact in material:
            d = mechanism(engine.model, state, fact.source, fact.action,
                          exclude_fact=fact.fact_id, ridge=engine.relation_ridge)
            target = engine.model.encoder(fact.outcome)
            scores.append(dict(relation=state.identity, query=fact.fact_id,
                score=float(log_score(d, target).mean()),
                mse=float((target-d['mean']).square().mean())))
    return dict(queries=scores, mean_score=sum(r['score'] for r in scores)/max(len(scores), 1),
                mean_mse=sum(r['mse'] for r in scores)/max(len(scores), 1))


def worker(args):
    import gc
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local.causal_use_value import configure
    from scripts.ib_local.mother_checkpoint import save
    from scripts.audit_spectrum_runtime import carried
    torch.set_num_threads(1)
    plan = json.loads((args.work/'plan.json').read_text(encoding='utf8'))
    folder = args.work/args.arm
    body = folder/'body'
    folder.mkdir(exist_ok=True); body.mkdir(exist_ok=True)
    status = folder/'status.json'
    write(status, dict(status='loading', pid=os.getpid(), arm=args.arm))
    opts = continuation_parser().parse_args(['--checkpoint', str(body/'last.pt.gz'),
        '--output', str(body), '--resume-output', '--cycles', str(args.cycles),
        '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(folder/'table'), '--causal-storage', str(folder/'causal')])
    loaded = prepare_continuation(opts, resume_output=True)
    engine = loaded.engine
    loaded.life_card = None
    assert loaded.cycle == 2401
    before = carried(engine)
    write(folder/'before.json', before)
    if plan.get('contact_momentum'):
        reference = json.loads((Path(plan['baseline_work'])/'A/result.json').read_text(encoding='utf8'))
        assert before == reference['before'], 'contact comparison must share complete original boundary'
        from scripts.ib_local.contact_momentum import configure as configure_contact
        configure_contact(engine)
        loaded.config['contact_momentum'] = dict(schema=1, adopted_at_cycle=2401,
            normalized_momentum=True, gate='beta_logit+offset+alignment*cosine',
            eligibility='persistent local S/P participation, real delayed read payment')
    engine.model.identity_preview_revision = 1
    material = fixed_queries(engine)
    assert material, 'actual held-out-query material required'
    evaluation = [(0, evaluate(engine, material))]
    # A records the same prospective receipts but keeps its original action
    # policy and g. All three arms retain the same overhead and measurements.
    configure(engine, 'shadow' if args.arm == 'A' else args.arm)
    engine.causal_use_transport_audit = bool(plan.get('use_transport_audit', False))
    loaded.config['causal_use_value'] = dict(schema=1, arm=engine.causal_use_arm,
        migrated_at_cycle=2401, main_body_unchanged=True)
    loaded.config['common_settlement'] = engine.settlement_kind
    original_policy = engine._policy_grad
    activation_bytes = []

    def policy(*a, **kw):
        from scripts.native_policy_activation_storage import policy_storage
        gc.collect(); torch.cuda.empty_cache()
        with policy_storage(engine.params, folder/'activations') as usage:
            result = original_policy(*a, **kw)
        activation_bytes.append(usage.get('file_high_water_bytes', 0))
        return result

    engine._policy_grad = policy
    original_advance = engine._advance

    def advance(*a, **kw):
        # Same release point as the already admitted observer's update_before.
        gc.collect(); torch.cuda.empty_cache()
        return original_advance(*a, **kw)

    engine._advance = advance
    original_objective = engine.model.objective
    world_hooks = [None]

    def objective(*a, **kw):
        # Scope only differentiable language learning. The autonomous S/P
        # integrator uses torch.func and must run outside saved-tensor hooks.
        with world_hooks[0]():
            return original_objective(*a, **kw)

    engine.model.objective = objective
    updates = [0]
    original_step = engine.optimizer.step

    def optimizer_step(*a, **kw):
        result = original_step(*a, **kw)
        updates[0] += 1
        return result

    engine.optimizer.step = optimizer_step
    started = time.perf_counter()
    external, byte_count = 0, 0
    rows = []
    reference_path = args.work/'replay_reference.json'
    reference = []
    if reference_path.exists():
        old_root = Path(json.loads(reference_path.read_text(encoding='utf8'))['work'])
        old_events = old_root/args.arm/'events.jsonl'
        if old_events.exists():
            reference = [r for line in old_events.read_text(encoding='utf8').splitlines()
                         if (r := json.loads(line))['autonomous_event']['kind'] == 'external']
    try:
        with (folder/'events.jsonl').open('w', encoding='utf8') as stream:
            for poll in range(max(512, args.cycles*64)):
                tick = time.perf_counter()
                from scripts.native_policy_activation_storage import activation_storage
                contact = getattr(engine.model.head_coupling, 'contact', None)
                gate_before = contact.phi.detach().clone() if contact is not None else None
                with activation_storage(engine.params, folder/'world_activations') as (usage, hooks):
                    world_hooks[0] = hooks
                    row = dispatch(engine)
                world_hooks[0] = None
                if row['autonomous_event']['kind'] == 'external':
                    from scripts.ib_local.causal_use_transport import finish as finish_transport
                    finish_transport(engine, row)
                if contact is not None:
                    row['contact_momentum'] = contact.diagnostics()
                    row['contact_momentum']['parameter_step_norm'] = float((contact.phi.detach()-gate_before).norm())
                activation_bytes.append(usage.get('file_high_water_bytes', 0))
                row['runtime_seconds'] = time.perf_counter()-tick
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
                if row['autonomous_event']['kind'] != 'external':
                    continue
                external += 1
                byte_count += row['external_bytes']
                rows.append(row)
                if external <= len(reference):
                    old = reference[external-1]
                    keys = ('choice', 'metal_g', 'response_nll', 'choice_probability')
                    differences = {k: [old.get(k), row.get(k)] for k in keys
                                   if old.get(k) != row.get(k)}
                    write(folder/'replay_check.json', dict(checked=external,
                        reference_events=len(reference), exact=not differences,
                        differences=differences))
                    assert not differences, 'replay diverged; see replay_check.json'
                assert row['causal_use']['unknown'] == row['causal_program']['roles']['unknown_identity']
                assert row.get('world', {}).get('provider_calls', 0) == 0
                if external % 8 == 0 or external == args.cycles:
                    evaluation.append((external, evaluate(engine, material)))
                    write(folder/'evaluations.json', evaluation)
                write(status, dict(status='running', arm=args.arm, external_events=external,
                    cycle=2401+external, updates=updates[0], external_bytes=byte_count,
                    seconds=time.perf_counter()-started, last_event_seconds=row['runtime_seconds'],
                    last_use=row['causal_use']['observed_use'], last_text_gain=row['causal_use']['text_gain'],
                    gpu_peak_mib=torch.cuda.max_memory_allocated()/1048576))
                torch.cuda.empty_cache()
                if external == args.cycles:
                    break
            else:
                raise RuntimeError('bounded native poll budget exhausted; retained internal events in log')
    finally:
        del engine._policy_grad
        del engine._advance
        del engine.model.objective
        engine.optimizer.step = original_step
    # Complete causal archive, pending actor, weights, optimizer, S/P, world,
    # table, use receipts/calibration and RNG. Atomic publication replaces the
    # private hardlink; it never writes through the source anchor.
    loaded.config['identity_preview_revision'] = 1
    digest = save(engine, body, 2401+external, loaded.config)
    write(body/'config.json', loaded.config)
    after = carried(engine)
    result = dict(status='completed', arm=args.arm, cycles=external,
        updates=updates[0], external_bytes=byte_count, seconds=time.perf_counter()-started,
        source_sha=loaded.source_sha, before=before, after=after,
        evaluations=evaluation, receipts=rows, checkpoint_sha256=digest,
        peak_gpu_mib=torch.cuda.max_memory_allocated()/1048576,
        max_policy_disk_bytes=max(activation_bytes, default=0), provider_calls=0,
        fixed_query_scope='same preselected real old facts; target excluded from its fit; weights learn normally')
    write(folder/'result.json', result)
    write(status, {k: result[k] for k in ('status', 'arm', 'cycles', 'updates', 'seconds', 'external_bytes')})
    engine.model._causal_archive.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--previous', type=Path, default=ROOT/'scratch/native_identity_preview_bound_2401_20261117')
    p.add_argument('--cycles', type=int, default=32)
    p.add_argument('--arm', choices=('A', 'B', 'C', 'C2'))
    p.add_argument('--contact-momentum', action='store_true')
    p.add_argument('--arms', nargs='+', choices=('A', 'B', 'C', 'C2'))
    p.add_argument('--use-transport-audit', action='store_true')
    p.add_argument('--baseline-work', type=Path, default=Path('F:/mengli-causal-use-2401-v5'))
    args = p.parse_args()
    args.work = args.work.resolve()
    if args.arm:
        worker(args); return
    args.work.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.previous/'plan.json').read_text(encoding='utf8'))
    source = Path(previous['checkpoint']['path'])
    plan = dict(schema=1, checkpoint=previous['checkpoint'], archive=previous['archive'],
        cycles=args.cycles, arms=args.arms or (['A'] if args.contact_momentum else ['A', 'B', 'C']),
        use_transport_audit=args.use_transport_audit,
        contact_momentum=args.contact_momentum, baseline_work=str(args.baseline_work.resolve()),
        source_body=str(source.parent),
        budget=dict(external_bytes_per_cycle=960, full_optimizer_updates=True),
        sources={str(f.relative_to(ROOT)): sha(f) for f in [Path(__file__).resolve(),
            ROOT/'scripts/ib_local/causal_use_value.py', ROOT/'scripts/ib_local/world_learning_cycle.py',
            ROOT/'scripts/ib_local/causal_use_transport.py',
            ROOT/'scripts/ib_local/finite_use_value.py',
            ROOT/'scripts/ib_local/language_navigation_kanli_v4p.py',
            ROOT/'scripts/ib_local/navigation_settlement.py', ROOT/'scripts/ib_local/navigation_direct_read.py',
            ROOT/'scripts/ib_local/navigation_generative_cycle.py', ROOT/'scripts/ib_local/navigation_joint_cycle.py',
            ROOT/'scripts/ib_local/navigation_causal_program.py', ROOT/'scripts/ib_local/navigation_internal_rollout.py',
            ROOT/'scripts/ib_local/causal_archive.py',
            ROOT/'scripts/ib_local/adaptive_hamiltonian.py', ROOT/'scripts/ib_local/contact_momentum.py',
            ROOT/'scripts/ib_local/contact_momentum_fused.py', ROOT/'scripts/ib_local/bfss_gdn_capture.py',
            ROOT/'scripts/ib_local/navigation_bfss_dynamics.py', ROOT/'scripts/ib_local/run_causal_water_reply.py',
            ROOT/'scripts/native_policy_activation_storage.py']})
    plan_path = args.work/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding='utf8')) == plan
    else:
        write(plan_path, plan)
    anchor = args.work/'anchor.pt.gz'
    if not anchor.exists():
        shutil.copy2(source, anchor)
    assert sha(anchor) == plan['checkpoint']['sha256']
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native,garbage_collection_threshold:0.4'
    results = {}
    for arm in plan['arms']:
        folder = args.work/arm
        body = folder/'body'
        body.mkdir(parents=True, exist_ok=True)
        if not (body/'last.pt.gz').exists():
            os.link(anchor, body/'last.pt.gz')
            for name in ('last.metadata.json', 'config.json'):
                shutil.copy2(source.parent/name, body/name)
        if not (folder/'result.json').exists():
            with (folder/'stdout.log').open('w', encoding='utf8') as out, (folder/'stderr.log').open('w', encoding='utf8') as err:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_native_causal_use_value',
                    '--work', str(args.work), '--arm', arm, '--cycles', str(args.cycles)],
                    cwd=ROOT, env=env, stdout=out, stderr=err,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                write(args.work/'status.json', dict(status='running', arm=arm, pid=child.pid, completed=list(results)))
                code = child.wait()
            if code:
                write(args.work/'status.json', dict(status='failed', arm=arm, exit_code=code))
                raise RuntimeError(f'{arm}: {folder / "stderr.log"}')
        result = json.loads((folder/'result.json').read_text(encoding='utf8'))
        assert result['before'] == (results['A']['before'] if results else result['before'])
        results[arm] = result
    for key in ('checkpoint', 'archive'):
        assert sha(plan[key]['path']) == plan[key]['sha256']
    summary = dict(status='completed', main_hashes_preserved=True, plan=plan,
        arms={name: {k:r[k] for k in ('cycles', 'updates', 'external_bytes', 'seconds', 'evaluations',
                                    'peak_gpu_mib', 'provider_calls')} for name,r in results.items()})
    write(args.work/'summary.json', summary)
    name = ('cgb_finite_causal_use_2401_20261118.json' if 'C2' in plan['arms'] else
            'cgb_causal_use_transport_2401_20261118.json' if args.use_transport_audit else
            'cgb_contact_momentum_policy_2401_20261118.json' if args.contact_momentum
            else 'cgb_native_causal_use_policy_2401_20261117.json')
    write(ROOT/'results/published'/name, summary)
    write(args.work/'status.json', dict(status='completed', summary=str(args.work/'summary.json')))


if __name__ == '__main__':
    main()
