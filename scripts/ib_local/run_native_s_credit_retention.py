"""Disposable actual-source native-S research branches; never advance the main life.

Explicit stages only: prepare, admission, run (one arm), analyze. The future is
new real OWT traffic, not invented feedback for the inherited pending answer.
No heldout reset, independent evaluation, subprocess scheduling or auto-training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WINDOW, PHASE_WINDOWS, RANK, SEED = 128, 32, 2, 11
PHASE_TOKENS = WINDOW * PHASE_WINDOWS
MAX_BYTES = 64 * 2**20
ARCHIVE_BUDGET = 6 * 2**30
ARMS = ('raw', 'cut', 'mismatch')
PHASES = ('A_first', 'B', 'A_return')
SOURCE_BYTES = 711394149
SOURCE_SHA256 = '514a45cbe1753c8f056ff75cae2b84121f07c2b5b2a70a6e9e24b9aa9eaca1b5'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(64*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def untimed(value):
    """Drop instrumentation only, never scientific metrics or clocks/cursors."""
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items()
                if not isinstance(k, str) or (k != 'cuda_allocator_before_backward' and
                not any(s in k.lower() for s in ('timing', 'elapsed', 'seconds')))}
    if isinstance(value, (list, tuple)):
        return [untimed(v) for v in value]
    return value


def distribution(values):
    values = list(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('expected nonempty finite NLL observations')
    return dict(min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), max=max(values), windows=len(values))


def summarize(windows):
    formal = [w for w in windows if w['phase'] != 'bridge']
    phases = {name: distribution(w['lived_preupdate_ce'] for w in formal
                                if w['phase'] == name) for name in PHASES}
    returned = [w['lived_preupdate_ce'] for w in formal if w['phase'] == 'A_return']
    first = [w['lived_preupdate_ce'] for w in formal if w['phase'] == 'A_first']
    return dict(phases=phases, return_first4=distribution(returned[:4]),
                return_later28=distribution(returned[4:]),
                return_last4=distribution(returned[-4:]),
                return_first4_minus_initial_first4=statistics.mean(returned[:4])-statistics.mean(first[:4]),
                later_minus_return_first4=statistics.mean(returned[4:])-statistics.mean(returned[:4]))


def phase_for_window(index):
    return 'bridge' if index == 0 else PHASES[(index-1)//PHASE_WINDOWS]


def traffic_segments(bridge_tokens):
    """Actual arrival intervals, including the real B token opening final partial."""
    if not 0 <= bridge_tokens < WINDOW:
        raise ValueError('invalid inherited partial input length')
    result, cursor = [], 0
    for phase, block, count in [('bridge', 'a', bridge_tokens),
                                ('A_first', 'a', PHASE_TOKENS),
                                ('B', 'b', PHASE_TOKENS),
                                ('A_return', 'a', PHASE_TOKENS),
                                ('B_continuation', 'b', 1)]:
        result.append(dict(phase=phase, block=block, start=cursor,
                           stop=cursor+count, token_start=PHASE_TOKENS if phase == 'bridge' else 0))
        cursor += count
    return result


def receive_slice(stream, tokens, segments, start, stop):
    """Keep phase/history labels truthful when a chunk crosses a real boundary."""
    for segment in segments:
        lo, hi = max(start, segment['start']), min(stop, segment['stop'])
        if lo < hi:
            stream.receive(tokens[lo:hi], phase=segment['phase'], teach=True)


def arrival_positions(segments, start, stop):
    positions = []
    for segment in segments:
        lo, hi = max(start, segment['start']), min(stop, segment['stop'])
        if lo < hi:
            positions.append(dict(phase=segment['phase'], block=segment['block'],
                raw_array_start=8_000_000 if segment['block'] == 'a' else 10_000_000,
                model_token_start=segment['token_start']+lo-segment['start'],
                model_token_stop=segment['token_start']+hi-segment['start']))
    return positions


def check_native(result, *, bridge=False, empty=False):
    native = result.get('native_s', {})
    if not bridge and (native.get('has_factual_trace') is not True or
                       native.get('teaching_available') is not True):
        raise RuntimeError('complete fresh native-S teaching unavailable; do not silently skip it')
    if (bridge or empty) and (native.get('trace_empty_before') is not True or
                             native.get('injected_gradient_norm') != 0):
        raise AssertionError('empty native-S eligibility must inject exactly zero')
    if 'persistent_bytes' not in native or 'estimated_temporary_bytes' not in native:
        raise RuntimeError('native-S explicit memory accounting missing')
    if native['persistent_bytes'] + native['estimated_temporary_bytes'] > MAX_BYTES:
        raise MemoryError('native-S explicit arrays exceed fixed 64 MiB budget')
    return native


def comparison_tree(value):
    """Adapt NumPy sidecar arrays to the existing exact-state comparator."""
    import numpy as np
    import torch
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {k: comparison_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [comparison_tree(v) for v in value]
    return value


def provenance(args):
    """Compare the eleven historical runtime files against their frozen manifest."""
    historical = read_json(args.runtime_manifest)
    if 'source_files' in historical:
        historical = historical['source_files']
    if len(historical) != 11:
        raise ValueError('--runtime-manifest must enumerate exactly eleven historical runtime files')
    files = {}
    for name, digest in historical.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / name.replace('\\', '/')
        if sha256(path) != digest:
            raise AssertionError(f'historical runtime changed: {path}')
        files[str(path.resolve())] = digest
    source_hash = sha256(args.source)
    if args.source.stat().st_size != SOURCE_BYTES or source_hash != SOURCE_SHA256:
        raise ValueError('not the actual 524-update pending-recovery source')
    return dict(source=str(args.source), source_sha256=source_hash,
                source_bytes=SOURCE_BYTES, historical_runtime_files=files,
                runtime_manifest_sha256=sha256(args.runtime_manifest))


def verify_preserved(args, report):
    same = sha256(args.source) == report['source_sha256']
    runtime = all(sha256(path) == digest for path, digest in report['historical_runtime_files'].items())
    execution = all(sha256(path) == digest for path, digest in report.get('execution_sources', {}).items())
    report.update(source_unchanged=same, historical_11_runtime_files_unchanged=runtime,
                  execution_sources_unchanged=execution)
    if not same or not runtime or not execution:
        raise AssertionError('immutable source or execution runtime changed')


def configuration(args):
    return dict(model=str(args.model), vendor=str(args.vendor), window=WINDOW,
                phase_windows=PHASE_WINDOWS, rank=RANK, max_explicit_sidecar_bytes=MAX_BYTES,
                seed=SEED, phases=list(PHASES), checkpoint_every_formal_updates=32,
                bounded_autotune=getattr(args, 'bounded_autotune', False),
                readonly_checkpoints=getattr(args, 'readonly_checkpoints', False),
                disable_joint_archive=getattr(args, 'disable_joint_archive', False),
                collect_window_garbage=getattr(args, 'collect_window_garbage', False),
                max_owned_archive_bytes=getattr(args, 'archive_budget_gib', 6) * 2**30,
                original_source_only=True, state_resets=0, independent_evaluation=False,
                future='new OWT researcher branch; inherited pending answer is archival only',
                scope='conditional approximate local native-S; not full BPTT/joint credit',
                interpretation='one fixed short seed/budget, not convergence or capacity evidence')


def prepare(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from scripts.ib_local.run_pretrained_gdn_cross_window import read_text
    from scripts.ib_local.prepare_pretrained_gdn_credit_stream import pack_stream

    destination = args.output / 'curriculum.pt'
    if destination.exists() or (args.output / 'curriculum.json').exists():
        raise FileExistsError('preserve the existing tokenized curriculum')
    if len(np.load(args.data / 'train.npy', mmap_mode='r')) < 10_000_000+196608:
        raise ValueError('local real train corpus does not cover the fixed raw slices')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    offset = read_json(args.data / 'tokenizer.json')['offset']
    manifest = read_json(args.data / 'manifest.json')
    blocks, hashes = {}, {}
    for name, start in [('a', 8_000_000), ('b', 10_000_000)]:
        text, hashes[name] = read_text(args.data / 'train.npy', offset, start, 196608)
        blocks[name] = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    packed = pack_stream(blocks['a'], blocks['b'], blocks['a'],
                         phase_tokens=PHASE_TOKENS, validation_tokens=0)
    metadata = dict(dataset='OpenWebText', dataset_revision=manifest['revision'], split='train',
        raw_unit='local encoded corpus array entries, not model tokens',
        raw_starts=dict(a=8_000_000, b=10_000_000), raw_length_each=196608,
        decoded_source_sha256=hashes, data_manifest_sha256=sha256(args.data / 'manifest.json'),
        data_tokenizer_sha256=sha256(args.data / 'tokenizer.json'),
        model_tokenizer_files={p.name: sha256(p) for p in args.model.glob('*token*') if p.is_file()},
        stream_sha256=hashlib.sha256(packed['stream'].numpy().tobytes()).hexdigest(),
        phase_tokens=PHASE_TOKENS, phase_windows=PHASE_WINDOWS,
        note='Not claimed unseen during foundation pretraining. A/B are corpus slices, '
             'not certified semantic domains; A is reused exactly; no heldout probes.')
    if len(blocks['a']) < PHASE_TOKENS + WINDOW:
        raise ValueError('A does not cover its disjoint real bridge tail')
    metadata['bridge_model_token_start'] = PHASE_TOKENS
    torch.save(dict(stream=packed['stream'], bridge_pool=blocks['a'][PHASE_TOKENS:PHASE_TOKENS+WINDOW],
                    metadata=metadata), destination)
    metadata['curriculum_file_sha256'] = sha256(destination)
    write_json(args.output / 'curriculum.json', metadata)


def load_curriculum(args):
    import torch
    path = args.output / 'curriculum.pt'
    metadata = read_json(args.output / 'curriculum.json')
    if sha256(path) != metadata['curriculum_file_sha256']:
        raise ValueError('prepared curriculum changed')
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data, metadata


def load_runtime(args):
    # Must precede imports which transitively load FLA. No cache on C: or E:.
    sys.path.insert(0, str(args.vendor))
    import torch
    from scripts.ib_local.triton_lazy_debug_ir import enable_lazy_debug_ir
    debug_ir = enable_lazy_debug_ir()
    resource_policy = None
    if getattr(args, 'bounded_autotune', False):
        from scripts.ib_local.triton_bounded_autotune import enable_bounded_autotune
        resource_policy = enable_bounded_autotune()
    from scripts.ib_local.native_s_learner import NativeSEligibilityLearner, NativeSEligibilityBackend
    from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream
    from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
    from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model, loading = load_streamed_pure_gdn(args.model)
    loading['compiler_debug_ir'] = debug_ir
    loading['autotune_resource_policy'] = resource_policy
    enable_streaming_convolution(model)
    return NativeSEligibilityLearner, NativeSEligibilityBackend, IrreversibleCreditStream, model, loading


def load_checkpoint(path, args):
    """Immutable source/candidate tensors; learner restoration copies live state."""
    if getattr(args, 'readonly_checkpoints', False):
        from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
        return load_readonly_checkpoint(path)
    import torch
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def start_branch(args, data, report):
    import torch
    cls, backend_cls, stream_cls, model, loading = load_runtime(args)
    owner = load_checkpoint(args.source, args)
    source = owner['learner']
    if (source['updates'], source['cache']['seen_tokens'], tuple(source['layers']), source['capacity']) != (524, 450757, (1, 12), 64):
        raise ValueError('actual source counters/layers/capacity differ')
    if len(owner['stream']['inputs']) != 69 or len(owner['stream']['targets']) != 68 or owner['next_case'] != 14:
        raise ValueError('actual source pending state differs')
    prefix = getattr(args, '_resume_prefix', None)
    if prefix is not None:
        from scripts.ib_local.native_s_resume import validate_resume_state
        owner = load_checkpoint(prefix['path'], args)
        validate_resume_state(owner, prefix)
        source = owner['learner']
    learner = cls(model, layers=source['layers'], capacity=source['capacity'])
    learner.load_state_dict(source)
    if prefix is None:
        learner.enable_native_s_eligibility(mode=args.arm or 'raw', rank=RANK, max_bytes=MAX_BYTES)
    if getattr(args, 'disable_joint_archive', False):
        from scripts.ib_local.native_s_storage import stop_future_joint_archives
        report['storage_migration'] = stop_future_joint_archives(learner)
    # Preserve mapped storage aliases and the original pending/history for the process lifetime.
    learner._snapshot_owner = owner
    backend = backend_cls(learner)
    stream = stream_cls(backend)
    stream.restore_latest(owner['stream'])
    bridge = prefix['prior']['bridge_tokens'] if prefix else WINDOW - len(stream.inputs)
    tokens = torch.cat((data['bridge_pool'][:bridge], data['stream'])).tolist()
    report.update(weight_loading=loading, source_updates=learner.updates,
                  source_seen=learner.cache._seen_tokens, source_events=stream.events,
                  source_windows=stream.windows, source_history_chain=stream.chain,
                  bridge_tokens=bridge, pending_answer_archival=True, next_case=owner['next_case'])
    if prefix:
        for key in ('source_updates', 'source_seen', 'source_events', 'source_windows', 'source_history_chain'):
            report[key] = prefix['prior'][key]
        report['resumed_from'] = dict(path=str(prefix['path']), sha256=prefix['sha256'],
            committed_formal_updates=prefix['formal'], actual_next_arrival=prefix['cursor'],
            unsaved_later_attempt_excluded=True)
    return learner, backend, stream, tokens, traffic_segments(bridge), owner


def snapshot(path, learner, backend, stream, owner, driver, *, archive_budget=ARCHIVE_BUDGET):
    import torch
    backend.close()
    payload = dict(learner=learner.state_dict(), stream=stream.state_dict(),
                   next_case=owner['next_case'], pending=owner.get('pending'),
                   branch_driver=driver, pending_answer_archival_only=True)
    temporary = path.with_suffix('.tmp.pt')
    if path.exists() or temporary.exists():
        raise FileExistsError(f'preserve existing candidate: {path}')
    # Includes other arms and admission files; never enumerate/delete external archives.
    sizes = [p.stat().st_size for p in path.parent.parent.rglob('*.pt')]
    used, estimated = sum(sizes), max([SOURCE_BYTES]+sizes)
    if used + estimated > archive_budget:
        raise OSError('owned candidate archives including transient write would exceed archive budget')
    if shutil.disk_usage(path.parent).free < estimated+256*2**20:
        raise OSError('insufficient output drive free space for the next candidate plus 256 MiB margin')
    torch.save(payload, temporary)
    if used + temporary.stat().st_size > archive_budget:
        raise OSError('candidate archive exceeds archive budget; temporary retained for diagnosis')
    temporary.replace(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def prune_owned(path, directory):
    """Only obsolete generation files created by this driver may be removed."""
    if path.parent.resolve() != directory.resolve() or not path.name.startswith(('checkpoint_', 'admission_')):
        raise ValueError('refuse cleanup outside owned candidate generations')
    entry = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path),
                 reason='obsolete disposable research candidate; source/mapped input never pruned')
    with (directory / 'checkpoint_pruned.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry)+'\n')
    path.unlink()


def resource_metrics():
    import torch
    return dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                cpu_accounting='native_s persistent_bytes and estimated_temporary_bytes are explicit '
                               'array bytes only; exclude model, optimizer, Python, BLAS, RSS and mmap pages')


def admission(args, data, report, directory):
    import torch
    from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same
    from scripts.ib_local.checkpoint_digest import state_digest
    learner, backend, stream, tokens, segments, owner = start_branch(args, data, report)
    windows = report['windows']
    stream.on_window = windows.append
    torch.cuda.reset_peak_memory_stats()
    try:
        # First new A token commits the existing 69+59-input historical window.
        cursor = report['bridge_tokens']+1
        receive_slice(stream, tokens, segments, 0, cursor)
        report['arrival_segments'] = segments
        report['admission_arrivals'] = [dict(start=0, stop=cursor,
            source_positions=arrival_positions(segments, 0, cursor),
            events=stream.events, seen=learner.cache._seen_tokens, updates=learner.updates,
            history_chain=stream.chain)]
        if len(windows) != 1:
            raise AssertionError('expected exactly one inherited bridge commit')
        check_native(windows[0], bridge=True)
        for _ in range(2):
            receive_slice(stream, tokens, segments, cursor, cursor+WINDOW)
            cursor += WINDOW
            report['admission_arrivals'].append(dict(start=cursor-WINDOW, stop=cursor,
                source_positions=arrival_positions(segments, cursor-WINDOW, cursor),
                events=stream.events, seen=learner.cache._seen_tokens, updates=learner.updates,
                history_chain=stream.chain))
        for index, record in enumerate(windows[1:]):
            check_native(record, empty=index == 0)
        candidate = directory / 'admission_before_third.pt'
        snapshot(candidate, learner, backend, stream, owner, dict(next_arrival=cursor),
                 archive_budget=report['configuration']['max_owned_archive_bytes'])
        # Both continuations explicitly start by restoring the same complete disk state.
        continuation_metrics = []
        continuation_digests = []
        for attempt in range(2):
            backend.close()
            disk = load_checkpoint(candidate, args)
            learner.load_state_dict(disk['learner'])  # Sidecar restores itself; do NOT re-enable.
            learner._candidate_owner = disk
            backend = type(backend)(learner)
            stream = type(stream)(backend)
            stream.restore_latest(disk['stream'])
            metrics = []
            stream.on_window = metrics.append
            receive_slice(stream, tokens, segments, cursor, cursor+WINDOW)
            if len(metrics) != 1:
                raise AssertionError('one third-window update expected')
            check_native(metrics[0])
            continuation_metrics.append(untimed(metrics))
            backend.close()
            continuation_digests.append(state_digest(dict(learner=learner.state_dict(),
                                                          stream=stream.state_dict())))
        if continuation_digests[0]['sha256'] != continuation_digests[1]['sha256']:
            raise AssertionError('full restored learner/stream/sidecar digest differs')
        report['complete_next_state_digests'] = continuation_digests
        assert_same(*comparison_tree(continuation_metrics), path='untimed_next_update_metrics')
        for index, record in enumerate(windows):
            record['phase'] = phase_for_window(index)
        report.update(status='admission_passed', complete_disk_next_update_equal=True,
                      empty_trace_zero_injection=True, actual_s_teaching_available=True,
                      third_window_metrics=continuation_metrics[0],
                      discarded_research_third_updates=2, retained_main_updates=0,
                      candidate_before_third=str(candidate), candidate_sha256=sha256(candidate),
                      candidate_bytes=candidate.stat().st_size,
                      third_arrivals=dict(start=cursor, stop=cursor+WINDOW,
                          source_positions=arrival_positions(segments, cursor, cursor+WINDOW),
                          end_seen=learner.cache._seen_tokens, end_events=stream.events,
                          end_updates=learner.updates, end_history_chain=stream.chain),
                      **resource_metrics())
        if getattr(args, 'disable_joint_archive', False):
            # Same complete disk start and same real next window. Re-enable the
            # old passive archive only in one discarded audit continuation.
            from scripts.ib_local.native_s_storage import learning_state_view
            policy_digests, policy_metrics, policy_storage = [], [], []
            for capture in (True, False):
                backend.close()
                disk = load_checkpoint(candidate, args)
                learner.load_state_dict(disk['learner'])
                learner._candidate_owner = disk
                learner.joint_capture_enabled = capture
                backend = type(backend)(learner)
                stream = type(stream)(backend)
                stream.restore_latest(disk['stream'])
                metrics = []
                stream.on_window = metrics.append
                receive_slice(stream, tokens, segments, cursor, cursor+WINDOW)
                backend.close()
                if getattr(args, 'collect_window_garbage', False):
                    gc.collect()
                policy_digests.append(state_digest(learning_state_view(
                    learner.state_dict(), stream.state_dict())))
                policy_metrics.append({k:untimed(v) for k,v in metrics[0].items()
                    if k not in ('experience_cpu_bytes', 'joint_boundary_bytes',
                                 'joint_input_bytes', 'joint_entries_share_one_archive')})
                policy_storage.append(learner.experience_storage_nbytes)
            assert_same(*comparison_tree(policy_metrics), path='passive_archive_policy_metrics')
            if policy_digests[0]['sha256'] != policy_digests[1]['sha256']:
                raise AssertionError('disabling passive archive changed actual learning state')
            report['passive_archive_learning_equivalent'] = True
            report['archive_policy_learning_digests'] = policy_digests
            report['archive_policy_experience_bytes_on_off'] = policy_storage
            report['discarded_archive_equivalence_updates'] = 2
        # Retain the full disk restart source and both complete terminal digests.
    finally:
        report['last_native_s_on_exit'] = dict(learner._last_native_s)
        report.update(resource_metrics())
        backend.close()


def run_arm(args, data, report, directory):
    passed = read_json(args.output / 'admission' / 'report.json')
    if passed['status'] != 'admission_passed':
        raise RuntimeError('admission_passed report required before a run')
    for key in ('source_sha256', 'historical_runtime_files', 'runtime_manifest_sha256',
                'configuration', 'data_sources', 'execution_sources'):
        if report[key] != passed[key]:
            raise ValueError(f'admission provenance/configuration mismatch: {key}')
    prefix = None
    if getattr(args, 'resume_run_dir', None) is not None:
        from scripts.ib_local.native_s_resume import resume_prefix
        prefix = resume_prefix(args.resume_run_dir, report)
        args._resume_prefix = prefix
    learner, backend, stream, tokens, segments, owner = start_branch(args, data, report)
    import torch
    torch.cuda.reset_peak_memory_stats()
    windows = report['windows']
    generations = []
    journal = (directory / 'windows.jsonl').open('x', encoding='utf-8')
    if prefix:
        windows.extend(prefix['windows'])
        for record in prefix['windows']:
            journal.write(json.dumps(record, allow_nan=False)+'\n')
        journal.flush()

    def window_done(result):
        index = len(windows)
        native = check_native(result, bridge=index == 0, empty=index == 1)
        # Preserve useful scalar diagnostics without repeated per-query tensors.
        record = {k: v for k, v in result.items() if k not in ('reads', 'credit', 'diagnostics')}
        record.update(phase=phase_for_window(index), formal_update=index,
                      native_s=native, phase_bridge=index in (0, 32, 64, 96))
        windows.append(record)
        journal.write(json.dumps(record, allow_nan=False)+'\n')
        journal.flush()

    stream.on_window = window_done
    try:
        cursor = prefix['cursor'] if prefix else report['bridge_tokens']+1
        if prefix is None:
            receive_slice(stream, tokens, segments, 0, cursor)
        arrival_log = (directory / 'arrivals.jsonl').open('x', encoding='utf-8')
        with arrival_log:
            if prefix:
                for arrival in prefix['arrivals']:
                    arrival_log.write(json.dumps(arrival)+'\n')
            else:
                arrival_log.write(json.dumps(dict(start=0, stop=cursor, phase='bridge_and_first_A_token',
                    source_positions=arrival_positions(segments, 0, cursor),
                    events=stream.events, seen=learner.cache._seen_tokens, history_chain=stream.chain))+'\n')
            for formal in range(prefix['formal']+1 if prefix else 1, 97):
                started = time.perf_counter()
                before = dict(events=stream.events, seen=learner.cache._seen_tokens,
                              updates=learner.updates, chain=stream.chain)
                receive_slice(stream, tokens, segments, cursor, cursor+WINDOW)
                cursor += WINDOW
                if len(windows) != formal+1 or learner.updates != report['source_updates']+formal+1:
                    raise AssertionError('formal/bridge update count diverged')
                if stream.events-report['source_events'] != cursor or learner.cache._seen_tokens-report['source_seen'] != cursor:
                    raise AssertionError('actual arrival/lifetime clock diverged')
                arrival = dict(start=cursor-WINDOW, stop=cursor, before=before,
                    source_positions=arrival_positions(segments, cursor-WINDOW, cursor),
                    events=stream.events, seen=learner.cache._seen_tokens, updates=learner.updates,
                    history_chain=stream.chain, seconds=time.perf_counter()-started,
                    pending_inputs=len(stream.inputs), pending_targets=len(stream.targets))
                arrival_log.write(json.dumps(arrival)+'\n')
                arrival_log.flush()
                if getattr(args, 'collect_window_garbage', False):
                    gc.collect()  # End-of-window Python lifetimes, never state/reset.
                if formal % 32 == 0:
                    current = directory / f'checkpoint_{formal:06d}.pt'
                    saved = snapshot(current, learner, backend, stream, owner,
                                     dict(next_arrival=cursor, formal_updates=formal, arm=args.arm),
                                     archive_budget=report['configuration']['max_owned_archive_bytes'])
                    report.setdefault('checkpoints', []).append(saved)
                    write_json(directory / 'checkpoints.json', report['checkpoints'])
                    generations.append(current)
                    write_json(directory / 'latest.json', dict(current=current.name,
                        sha256=saved['sha256'], bytes=saved['bytes'],
                        previous=generations[-2].name if len(generations) > 1 else None))
                    if len(generations) > 2:
                        prune_owned(generations.pop(0), directory)
                report.update(status='running', formal_updates=formal, actual_arrival_tokens=cursor,
                              end_updates=learner.updates, end_seen=learner.cache._seen_tokens,
                              end_events=stream.events, end_history_chain=stream.chain)
                write_json(directory / 'progress.json', {k: v for k, v in report.items() if k != 'windows'})
        if cursor != len(tokens) or len(stream.inputs) != 1 or stream.targets:
            raise AssertionError('final real B token must open a preserved one-input partial')
        report.update(status='complete', summary=summarize(windows),
                      phase_counts=dict(stream.phase_counts), arrival_segments=segments,
                      final_pending_inputs=1, final_pending_targets=0,
                      formal_updates=96, bridge_updates=1, main_source_updates=0,
                      step_curves=[dict(formal_update=w['formal_update'], phase=w['phase'],
                                       nll=w['lived_preupdate_ce']) for w in windows],
                      **resource_metrics())
    finally:
        backend.close()
        journal.close()


def analyze(args):
    destination = args.output / 'analysis.json'
    if destination.exists():
        raise FileExistsError('preserve existing aggregate report')
    reports = {arm: read_json(args.output / arm / 'report.json') for arm in ARMS}
    base = reports['raw']
    for arm, report in reports.items():
        if report['status'] != 'complete':
            raise RuntimeError(f'{arm} is not complete')
        for key in ('source_sha256', 'configuration', 'data_sources', 'execution_sources',
                    'end_events', 'end_history_chain', 'actual_arrival_tokens'):
            if report[key] != base[key]:
                raise ValueError(f'unmatched arms: {key}')
    differences = {}
    for control in ('cut', 'mismatch'):
        raw, other = base['summary'], reports[control]['summary']
        differences[control] = dict(
            return_first4_mean=raw['return_first4']['mean']-other['return_first4']['mean'],
            return_later28_mean=raw['return_later28']['mean']-other['return_later28']['mean'],
            phases={p: raw['phases'][p]['mean']-other['phases'][p]['mean'] for p in PHASES},
            window_nll=[a['nll']-b['nll'] for a, b in zip(base['step_curves'], reports[control]['step_curves'])])
    write_json(destination, dict(status='complete', summaries={a: r['summary'] for a, r in reports.items()},
        raw_minus_control=differences, source_sha256=base['source_sha256'],
        data_sources=base['data_sources'], seed=SEED,
        interpretation='Descriptive single short-seed prequential retention/relearning only; '
                       'negative NLL difference favors raw. No significance/convergence/capacity claim.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'admission', 'run', 'analyze'), required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--bounded-autotune', action='store_true',
                        help='use one existing pruned Triton configuration without benchmarking')
    parser.add_argument('--readonly-checkpoints', action='store_true',
                        help='map immutable checkpoint inputs read-only instead of copy-on-write')
    parser.add_argument('--disable-joint-archive', action='store_true',
                        help='stop future unused all-layer audit copies; preserve local capsules and eligibility')
    parser.add_argument('--collect-window-garbage', action='store_true',
                        help='collect unreachable Python objects at completed window boundaries')
    parser.add_argument('--resume-run-dir', type=Path,
                        help='raw only: continue the committed old research checkpoint, preserving prefix journals')
    parser.add_argument('--archive-budget-gib', type=int, choices=(6, 10), default=6,
                        help='total candidate disk archive limit, including transient checkpoint writes')
    parser.add_argument('--source', type=Path, default=ROOT / 'scratch/joint_next_pending_20261108/pending_recovery.pt')
    parser.add_argument('--runtime-manifest', type=Path,
                        default=ROOT / 'scratch/bounded_causal_tags_20261108/operator_source_manifest.json')
    parser.add_argument('--model', type=Path, default=Path('E:/cgb_gdn340_probe/model'))
    parser.add_argument('--vendor', type=Path, default=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'))
    parser.add_argument('--data', type=Path, default=ROOT / 'data/ib_owt_10000')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/cgb_native_s_credit_20261108')
    args = parser.parse_args()
    if (args.stage == 'run') != (args.arm is not None):
        parser.error('--arm is required only for --stage run')
    if args.resume_run_dir is not None and (args.stage != 'run' or args.arm != 'raw'):
        parser.error('--resume-run-dir applies only to --stage run --arm raw')
    if args.stage in ('admission', 'run') and args.runtime_manifest is None:
        parser.error('--runtime-manifest is required for admission/run')
    if args.output.resolve().drive.upper() not in ('D:', 'F:'):
        parser.error('research output must stay on the local D: or F: research drive')
    for name, suffix in [('ML_CACHE_ROOT', ''), ('HF_HOME', '/huggingface'),
                         ('TRANSFORMERS_CACHE', '/transformers'), ('MODELSCOPE_CACHE', '/modelscope'),
                         ('TORCH_HOME', '/torch'), ('TRITON_CACHE_DIR', '/triton'),
                         ('TORCHINDUCTOR_CACHE_DIR', '/inductor'), ('TEMP', '/tmp'), ('TMP', '/tmp')]:
        os.environ[name] = 'D:/ml_cache'+suffix
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'analyze':
        analyze(args)
        return
    if args.stage == 'prepare':
        prepare(args)
        return
    directory = args.output / (args.arm or 'admission')
    directory.mkdir(exist_ok=False)  # Never overwrite an earlier attempt, including failed evidence.
    report = dict(status='loading', stage=args.stage, arm=args.arm or 'raw', windows=[],
                  configuration=configuration(args), new_lean_coverage=0)
    started = time.perf_counter()
    try:
        report.update(provenance(args))
        if args.stage == 'admission' and not (args.output / 'curriculum.pt').exists():
            prepare(args)
        data, report['data_sources'] = load_curriculum(args)
        report['execution_sources'] = {str(path): sha256(path) for path in [
            Path(__file__), ROOT / 'scripts/ib_local/native_s_learner.py',
            ROOT / 'scripts/ib_local/native_s_eligibility.py',
            ROOT / 'scripts/ib_local/native_s_storage.py',
            ROOT / 'scripts/ib_local/native_s_resume.py',
            ROOT / 'scripts/ib_local/triton_lazy_debug_ir.py',
            ROOT / 'scripts/ib_local/triton_bounded_autotune.py',
            ROOT / 'scripts/ib_local/readonly_checkpoint.py',
            ROOT / 'scripts/ib_local/checkpoint_digest.py',
            ROOT / 'scripts/ib_local/bounded_causal_tag_reference.py',
            ROOT / 'scripts/ib_local/bounded_causal_tag_blocked.py']}
        write_json(directory / 'config.json', {k: v for k, v in report.items() if k != 'windows'})
        if args.stage == 'admission':
            admission(args, data, report, directory)
        else:
            run_arm(args, data, report, directory)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        try:
            if 'source_sha256' in report:
                verify_preserved(args, report)
        except BaseException as error:
            report.update(status='failed', preservation_error=repr(error))
            raise
        finally:
            write_json(directory / 'report.json', report)
        gc.collect()


if __name__ == '__main__':
    main()
