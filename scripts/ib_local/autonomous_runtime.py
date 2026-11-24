"""Durable mixed internal/world event owner for the same complete learner."""
from __future__ import annotations

import json
import time

import torch

from scripts.ib_local.autonomous_cycle import dispatch
from scripts.native_policy_activation_storage import event_storage
from scripts.ib_local.event_journal import EventJournal, complete_jsonl, execution_digest, rng_digest


def run(args, loaded, evaluation, root):
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.life_card import export_live
    from scripts.ib_local.continuation_schedule import validation_due
    from scripts.ib_local.native_navigation_body import record_birth_event
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.variable_expression import release_fallback_cache

    engine, config = loaded.engine, loaded.config
    cycle, completed, updates = loaded.cycle, 0, [0]
    original_step = engine.optimizer.step
    def counted_step(*a, **kw):
        result = original_step(*a, **kw)
        updates[0] += 1
        return result
    engine.optimizer.step = counted_step
    anchor_sha = loaded.checkpoint_sha or sha256_file(args.output/'last.pt.gz')
    # A new mixed clock does not reinterpret the old external-only journal.
    old_path = args.output/'events.pending.jsonl'
    if old_path.exists():
        old = complete_jsonl(old_path)
        if old and old[-1].get('cycle', old[0].get('base_cycle', 0)) > cycle:
            raise ValueError('finish old external journal recovery before autonomous migration')
    journal = EventJournal(args.output, cycle=engine.autonomous_event_clock,
        checkpoint_sha=anchor_sha, code_sha=execution_digest(root), config=config,
        filename='events.autonomous.pending.jsonl', config_key='autonomous_event_journal')
    history = args.output/'autonomous_events.jsonl'
    logged = complete_jsonl(history)
    last_logged = max((r['event'] for r in logged), default=0)
    del logged
    last_saved_event, last_saved_cycle = engine.autonomous_event_clock, cycle
    started = time.perf_counter()
    last_row = None
    pending_updates = 0
    pending_seconds = 0.
    # Storage quantum only: it neither forces external action nor prescribes
    # thought length. Continuing internally can span any number of snapshots.
    quiet_checkpoint_events = max(1, args.checkpoint_every*engine.joint_proposal_budget)
    status = dict(status='running', cycle=cycle, event=engine.autonomous_event_clock,
                  internal_ticks=engine.joint_internal_ticks)

    def publish():
        nonlocal last_saved_event, last_saved_cycle
        journal.bind(config, engine.autonomous_event_clock)
        sha = save(engine, args.output, cycle, config)
        journal.reset(engine.autonomous_event_clock, sha)
        last_saved_event, last_saved_cycle = engine.autonomous_event_clock, cycle
        if loaded.life_card is not None:
            export_live(engine, checkpoint=args.output/'last.pt.gz', config=config,
                cycle=cycle, row=last_row, output=loaded.life_card['path'],
                branch_id=loaded.life_branch, committed_sha=sha)
        return sha

    pending = iter(list(journal.pending))
    recovering = next(pending, None)
    with history.open('a', encoding='utf-8') as events, (args.output/'metrics.jsonl').open('a', encoding='utf-8') as metrics:
        metric_rows = complete_jsonl(args.output/'metrics.jsonl')
        last_metric = max((r.get('cycle', 0) for r in metric_rows), default=cycle)
        del metric_rows
        while completed < args.cycles:
            in_transaction = getattr(engine, '_rhythm_runtime', None) is not None
            if not in_transaction and recovering is None and args.stop_file is not None and args.stop_file.exists():
                break
            before = rng_digest()
            before_updates = updates[0]
            fallback_before = getattr(engine, '_batched_replay_fallbacks', 0)
            event_start = time.perf_counter()
            with event_storage(engine):
                row = dispatch(engine)
            elapsed = time.perf_counter()-event_start
            row['updates'] = updates[0]-before_updates
            external = row['autonomous_event']['kind'] == 'external'
            if external:
                row['updates'] += pending_updates
                elapsed += pending_seconds
                pending_updates, pending_seconds = 0, 0.
                cycle += 1
                completed += 1
                if config.get('native_body') is not None:
                    record_birth_event(engine, row)
                if config.get('body_growth') is not None:
                    engine.growth_optimizer_updates += row['updates']
                    engine.growth_external_bytes += row['external_bytes']
                last_row = row
            elif ((row['updates'] and row['autonomous_event']['kind'] != 'handoff')
                  or row['external_bytes'] or row['self_bytes']):
                raise RuntimeError('input-free event fabricated learning or boundary traffic')
            else:
                pending_updates += row['updates']
                if getattr(engine, '_rhythm_runtime', None) is not None:
                    pending_seconds += elapsed
            after = rng_digest()
            event = engine.autonomous_event_clock
            if recovering is not None:
                journal.verify(recovering, row, before, after)
                recovering = next(pending, None)
            else:
                journal.append(event, row, before, after)
            receipt = dict(row['autonomous_event'])
            receipt.update(cycle=cycle, seconds=elapsed,
                           internal_ticks=engine.joint_internal_ticks, updates=row['updates'])
            if event > last_logged:
                events.write(json.dumps(receipt, ensure_ascii=False)+'\n')
                events.flush()
                last_logged = event
            save_due = recovering is None and getattr(engine, '_rhythm_runtime', None) is None and (
                cycle-last_saved_cycle >= args.checkpoint_every or
                event-last_saved_event >= quiet_checkpoint_events or
                completed == args.cycles or
                (args.stop_file is not None and args.stop_file.exists()) or
                (external and evaluation is not None and validation_due(cycle, config['validation_schedule'])))
            save_start = time.perf_counter()
            if save_due:
                publish()
            status = dict(status='running', cycle=cycle, event=event,
                kind=row['autonomous_event']['kind'], internal_ticks=engine.joint_internal_ticks,
                ready=engine.autonomous_ready, complete_checkpoint_event=last_saved_event,
                complete_checkpoint_cycle=last_saved_cycle, pending_choices=engine.autonomous_choices,
                last_admission=engine.autonomous_last_admission)
            temporary = args.output/'autonomous_runtime.tmp.json'
            temporary.write_text(json.dumps(status, ensure_ascii=False)+'\n', encoding='utf-8')
            temporary.replace(args.output/'autonomous_runtime.json')
            if external:
                row['runtime'] = dict(event_seconds=elapsed, checkpoint_seconds=time.perf_counter()-save_start,
                    complete_checkpoint_cycle=last_saved_cycle, durable_event_cycle=cycle,
                    autonomous_event=event, external_bytes_per_event_second=row['external_bytes']/elapsed,
                    replay_fallback_cache_release=release_fallback_cache(engine, fallback_before),
                    cuda_allocated_mib=torch.cuda.memory_allocated()/2**20 if engine.device.type=='cuda' else None,
                    cuda_reserved_mib=torch.cuda.memory_reserved()/2**20 if engine.device.type=='cuda' else None,
                    cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if engine.device.type=='cuda' else None,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if engine.device.type=='cuda' else None)
                if evaluation is not None and validation_due(cycle, config['validation_schedule'], final=completed==args.cycles):
                    row['fixed_validation'] = evaluation()
                row.update(cycle=cycle, mode=args.mode, elapsed_seconds=time.perf_counter()-started)
                if cycle > last_metric:
                    metrics.write(json.dumps(row, ensure_ascii=False)+'\n')
                    metrics.flush()
                    last_metric = cycle
                print(json.dumps(dict(cycle=cycle, event=event, choice=row['choice'],
                    internal_ticks=engine.joint_internal_ticks, metal_g=row.get('metal_g'))), flush=True)
        if recovering is not None:
            raise ValueError('continuation budget ends before pending real events are recovered')
        if last_saved_event != engine.autonomous_event_clock:
            publish()
    status.update(status='complete' if completed == args.cycles else 'stopped')
    (args.output/'autonomous_runtime.json').write_text(json.dumps(status)+'\n', encoding='utf-8')
    (args.output/'completed.json').write_text(json.dumps(dict(status=status['status'], cycles=completed,
        requested_cycles=args.cycles, final_cycle=cycle, final_event=engine.autonomous_event_clock,
        internal_ticks=engine.joint_internal_ticks, source_sha256=loaded.source_sha), indent=2)+'\n', encoding='utf-8')
