"""Matched continuous A/B/A language learning, separating recall and old credit.

Stages run sequentially on one GPU: prepare (shared warm start + next-update
restore), full (native replay gradients), held (same recall, stop old native
gradients). No navigation actor or world is replaced by this language study.
"""
from __future__ import annotations

import argparse
from dataclasses import is_dataclass, fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def assert_same(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(f'next-update restore mismatch: {path}')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'keys differ: {path}')
        for key in left:
            assert_same(left[key], right[key], f'{path}/{key}')
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f'length differs: {path}')
        for i, (x, y) in enumerate(zip(left, right)):
            assert_same(x, y, f'{path}/{i}')
    elif is_dataclass(left):
        for field in fields(left):
            assert_same(getattr(left, field.name), getattr(right, field.name), f'{path}/{field.name}')
    elif left != right:
        raise AssertionError(f'value differs: {path}')


def read_text(path, offset, start, length=196608):
    raw = np.load(path, mmap_mode='r')[start:start+length]
    data = (raw[raw >= offset]-offset).astype(np.uint8).tobytes()
    return data.decode('utf-8', errors='replace'), hashlib.sha256(data).hexdigest()


def prepare_curriculum(args, tokenizer):
    offset = json.loads((args.data/'tokenizer.json').read_text())['offset']
    blocks, hashes = {}, {}
    for name, split, start in [('a', 'train', 0), ('b', 'train', 2_000_000),
                                ('validation', 'validation', 0)]:
        text, hashes[name] = read_text(args.data/f'{split}.npy', offset, start)
        blocks[name] = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    warm = 4*args.window
    n = args.phase_steps*args.window
    if len(blocks['a']) < warm+n+args.validation_tokens+2 or len(blocks['b']) < n+args.validation_tokens+2:
        raise ValueError('real curriculum slices are too short')
    a, b = blocks['a'], blocks['b']
    stream = torch.cat((a[:warm+n], b[:n], a[warm:warm+n], a[warm+n:warm+n+1]))
    origin = torch.cat((torch.arange(warm+n), 1_000_000_000+torch.arange(n),
                        torch.arange(warm, warm+n), torch.tensor([warm+n])))
    seen, revisit = {int(origin[0])}, []
    for identifier in origin[1:].tolist():
        revisit.append(identifier in seen)
        seen.add(identifier)
    heldouts = dict(validation=blocks['validation'][:args.validation_tokens+1],
        a_unseen=a[warm+n+1:warm+n+args.validation_tokens+2],
        b_unseen=b[n+1:n+args.validation_tokens+2])
    manifest = json.loads((args.data/'manifest.json').read_text())
    return dict(stream=stream, target_revisit=torch.tensor(revisit), heldouts=heldouts, warm_tokens=warm,
        source_hashes=hashes, dataset_revision=manifest['revision'],
        stream_sha256=hashlib.sha256(stream.numpy().tobytes()).hexdigest(),
        phase_steps=args.phase_steps, window=args.window,
        phase_labels=['A_fresh', 'B_fresh', 'A_revisit'],
        note='A and B are disjoint OWT stream slices, not certified semantic domains; score both phase bridges')


def new_learner(args):
    torch.manual_seed(11)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    return PretrainedExperienceLearner(model, layers=(1, 12), capacity=args.capacity)


def evaluate_all(learner, curriculum):
    return {name: learner.evaluate(ids, window=curriculum['window'])
            for name, ids in curriculum['heldouts'].items()}


def prepare(args):
    from transformers import AutoTokenizer
    if (args.output/'bootstrap.pt').exists():
        raise FileExistsError('shared bootstrap already exists; do not overwrite provenance')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    curriculum = prepare_curriculum(args, tokenizer)
    torch.save(curriculum, args.output/'curriculum.pt')
    learner = new_learner(args)
    seq = curriculum['stream']
    checks = dict(status='started', warm_updates=4, discarded_next_update_checks=2)
    write_json(args.output/'checks.json', checks)
    for step in range(4):
        start = step*args.window
        record = learner.step(seq[start:start+args.window][None].cuda(),
            seq[start+1:start+args.window+1][None].cuda(), diagnose=step == 3)
        print('WARM', step+1, record['ce'], flush=True)
    state = learner.state_dict()
    torch.save(state, args.output/'bootstrap.pt')
    pos = curriculum['warm_tokens']
    x, target = seq[pos:pos+args.window][None].cuda(), seq[pos+1:pos+args.window+1][None].cuda()
    first = learner.step(x, target, diagnose=True)
    first_state = learner.state_dict()
    disk = torch.load(args.output/'bootstrap.pt', map_location='cpu', weights_only=False)
    learner.load_state_dict(disk)
    second = learner.step(x, target, diagnose=True)
    assert_same(first, second, 'next_update_metrics')
    assert_same(first_state, learner.state_dict())
    checks['complete_next_update_equal'] = True
    learner.load_state_dict(disk)
    with torch.no_grad():
        full, _ = learner.predict(x)
        learner.set_replay_gradient(True)
        held, _ = learner.predict(x)
    checks['same_start_full_held_logit_max'] = float((full-held).abs().max())
    if checks['same_start_full_held_logit_max'] != 0:
        raise AssertionError('gradient-only control changed forward values')
    checks.update(status='passed', curriculum_sha256=curriculum['stream_sha256'],
                  dataset_revision=curriculum['dataset_revision'])
    write_json(args.output/'checks.json', checks)
    print('PREPARE PASSED', json.dumps(checks), flush=True)


def run_arm(args):
    checks = json.loads((args.output/'checks.json').read_text())
    if checks['status'] != 'passed':
        raise RuntimeError('complete-state/gradient-control checks must pass before comparisons')
    curriculum = torch.load(args.output/'curriculum.pt', weights_only=False)
    learner = new_learner(args)
    arm_dir = args.output/args.stage
    arm_dir.mkdir(exist_ok=True)
    pointer = arm_dir/'latest.json'
    last = arm_dir/'last.pt'
    previous = arm_dir/'previous.pt'
    if pointer.exists():
        names = json.loads(pointer.read_text())
        last = arm_dir/names['current']
        previous = arm_dir/names['previous'] if names.get('previous') else None
    driver = dict(next_step=0, cumulative_nll=0., target_tokens=0,
                  phases={}, traffic={}, elapsed_seconds=0.)
    if last.exists():
        if not args.resume:
            raise FileExistsError('existing arm; use explicit --resume')
        payload = torch.load(last, map_location='cpu', weights_only=False, mmap=True)
        learner.load_state_dict(payload['learner'], detach_replay=args.stage == 'held')
        driver = payload['driver']
        # Keep the mapped checkpoint owner alive for the complete segment,
        # including all nested storages and optimizer aliases on Windows.
        learner._snapshot_owner = payload
    else:
        learner._snapshot_owner = torch.load(args.output/'bootstrap.pt', map_location='cpu', weights_only=False, mmap=True)
        learner.load_state_dict(learner._snapshot_owner, detach_replay=args.stage == 'held')
    torch.cuda.reset_peak_memory_stats()
    progress = dict(status='running', arm=args.stage, planned_updates=3*args.phase_steps,
        base_updates=4, base_tokens=curriculum['warm_tokens'], score_unit='nats/token',
        capacity=args.capacity, curriculum_sha256=curriculum['stream_sha256'])
    if driver['next_step'] == 0:
        initial = evaluate_all(learner, curriculum)
        write_json(arm_dir/'initial_validation.json', initial)
    seq, window = curriculum['stream'], curriculum['window']
    journal = arm_dir/'steps.jsonl'
    first_step = driver['next_step']
    mapped_source = last if last.exists() else args.output/'bootstrap.pt'

    def checkpoint():
        nonlocal last, previous
        payload = dict(learner=learner.state_dict(), driver=driver,
                       model=str(args.model), curriculum_sha256=curriculum['stream_sha256'])
        # Mapped files cannot be renamed while views are live on Windows. Use
        # generation names and a small atomic pointer; retain only current and
        # its immediate predecessor. Never delete the process's mapped source.
        current = arm_dir/f"checkpoint_{driver['next_step']:06d}.pt"
        temp = arm_dir/'next.tmp.pt'
        torch.save(payload, temp)
        temp.replace(current)
        write_json(pointer, dict(current=current.name, previous=last.name if last.exists() else None))
        if previous is not None and previous.exists() and previous not in (last, current, mapped_source):
            if previous.resolve().parent != arm_dir.resolve():
                raise ValueError('checkpoint retention target escaped the arm directory')
            previous.unlink()
        previous, last = (last if last.exists() else None), current
        del payload

    try:
        with journal.open('a', encoding='utf-8') as log:
            for step in range(driver['next_step'], 3*args.phase_steps):
                started = time.perf_counter()
                start = curriculum['warm_tokens']+step*window
                phase = curriculum['phase_labels'][step//args.phase_steps]
                x = seq[start:start+window][None].cuda()
                target = seq[start+1:start+window+1][None].cuda()
                record = learner.step(x, target, diagnose=(step % 32 == 0),
                    revisit_mask=curriculum['target_revisit'][start:start+window])
                elapsed = time.perf_counter()-started
                driver['next_step'] = step+1
                driver['elapsed_seconds'] += elapsed
                driver['cumulative_nll'] += record['nll_sum']
                driver['target_tokens'] += record['target_tokens']
                ph = driver['phases'].setdefault(phase, dict(nll_sum=0., tokens=0))
                ph['nll_sum'] += record['nll_sum']
                ph['tokens'] += record['target_tokens']
                for kind, values in record['traffic'].items():
                    counts = driver['traffic'].setdefault(kind, dict(nll_sum=0., tokens=0))
                    counts['nll_sum'] += values['nll_sum']
                    counts['tokens'] += values['tokens']
                record.update(step=step+1, phase=phase, seconds=elapsed,
                    curriculum_target_start=start+1,
                    phase_bridge=(step+1) % args.phase_steps == 0 and step+1 < 3*args.phase_steps)
                log.write(json.dumps(record)+'\n')
                log.flush()
                progress.update(driver, current_phase=phase,
                    cumulative_nats_per_token=driver['cumulative_nll']/driver['target_tokens'],
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                write_json(arm_dir/'progress.json', progress)
                if (step+1) % 16 == 0:
                    print(args.stage, step+1, phase, round(record['ce'], 5),
                          round(progress['cumulative_nats_per_token'], 5), flush=True)
                if (step+1) % 64 == 0 or step+1 == 3*args.phase_steps:
                    evaluation = dict(step=step+1, phase=phase, metrics=evaluate_all(learner, curriculum))
                    with (arm_dir/'validation.jsonl').open('a', encoding='utf-8') as out:
                        out.write(json.dumps(evaluation)+'\n')
                    checkpoint()
                    if args.max_updates and step+1 >= first_step+args.max_updates and step+1 < 3*args.phase_steps:
                        progress.update(status='checkpointed')
                        write_json(arm_dir/'progress.json', progress)
                        print(args.stage, 'CHECKPOINTED', step+1, flush=True)
                        return
        progress.update(status='complete', final_validation=evaluation['metrics'],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(arm_dir/'progress.json', progress)
        print(args.stage, 'COMPLETE', flush=True)
    except Exception as error:
        progress.update(status='failed', error=repr(error))
        write_json(arm_dir/'progress.json', progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'full', 'held'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_10000'))
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--phase-steps', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=512)
    parser.add_argument('--validation-tokens', type=int, default=1024)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int, default=0,
                        help='exit at a complete checkpoint after this many updates;0 means whole remaining arm')
    args = parser.parse_args()
    if args.window < 128 or args.phase_steps < 64 or args.phase_steps % 64:
        parser.error('window >=128 and phase steps a positive multiple of64')
    if args.capacity < 4*args.phase_steps or args.capacity % 2:
        parser.error('reserve both A/B episode sets at two layers; fixed capacity >=4*phase_steps')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare(args)
    else:
        run_arm(args)


if __name__ == '__main__':
    main()
