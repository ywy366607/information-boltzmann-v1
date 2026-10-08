"""OWT online learning of the existing 3D medium, with no BPTT window.

One-event autodiff tapes are discarded; physical belief and forward eligibility
survive updates and checkpoint continuation. Primary evaluation is live
test-then-learn prediction plus actual continuous A->B->A experience.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.evaluation import field_energy_statistics
from information_boltzmann.runtime.gpu_memory import total_gpu_memory
from information_boltzmann.runtime.online_credit import CapturedOnlineEvent, OnlinePlasticTrainer
from information_boltzmann.runtime.local_credit import CapturedLocalEvent, LocalPlasticTrainer
from information_boltzmann.runtime.lifelong_evaluation import LiveOnlineEvaluation
from information_boltzmann.runtime.medium_health import MediumHealthAuditor, conditional_state_response
from information_boltzmann.runtime.optimization import (
    make_medium_optimizer, optimizer_policy, load_medium_branch_weights)
from scripts.ib.train_plastic_conductance import (
    atomic_json, learning_rate, pack_belief, unpack_belief)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--tokens', type=int, default=128, help='Optimizer cadence, never a credit horizon')
    parser.add_argument('--rank', type=int, default=1, help='Independent unbiased compression factors')
    parser.add_argument('--credit', choices=('uoro', 'local-receptors'), default='uoro',
                        help='Global stochastic reference OR deterministic local receptor eligibility')
    parser.add_argument('--event-duration', type=float, required=True)
    parser.add_argument('--substeps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Prediction matrices only; physical/scaling parameters use zero decay')
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--decay', type=int, default=300)
    parser.add_argument('--validate-every', type=int, default=500)
    parser.add_argument('--change-tokens', type=int, default=384,
                        help='Fresh adaptation-stream observations per live context change')
    parser.add_argument('--measurement-block', type=int, default=32)
    parser.add_argument('--recovery-hold-blocks', type=int, default=2)
    parser.add_argument('--health-window', type=int, default=512,
                        help='Bounded fourth-pillar measurement history, not a credit horizon')
    parser.add_argument('--response-directions', type=int, default=2,
                        help='Full-physical-state JVP directions at live evaluation boundaries')
    parser.add_argument('--no-health', action='store_true', help='Explicit monitoring-off execution comparison')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=449)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--execution', default='graph', choices=('graph','eager'))
    parser.add_argument('--vram-limit-gib', type=float, default=4.0)
    parser.add_argument('--calibrate-only', action='store_true', help='Real-data execution check, no capability claim or validation')
    parser.add_argument('--resume', type=Path, help='Exact online continuation, including eligibility and RNG')
    parser.add_argument('--initialize-from', type=Path, help='Explicit new learning branch from physical/model state, born eligibility')
    args = parser.parse_args()
    if min(args.steps, args.tokens, args.rank, args.substeps, args.validate_every, args.save_every,
           args.change_tokens, args.measurement_block, args.recovery_hold_blocks, args.response_directions) < 1:
        parser.error('Positive budgets, rank and intervals required')
    if args.health_window < 2:
        parser.error('Health measurement window must contain at least two observations')
    if not math.isfinite(args.event_duration) or args.event_duration <= 0:
        parser.error('Finite positive physical event duration required')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Finite nonnegative weight decay required')
    if not 0 < args.min_lr <= args.lr or min(args.warmup, args.decay) < 0 or args.warmup+args.decay > args.steps:
        parser.error('Invalid WSD schedule')
    if not 0 < args.vram_limit_gib <= 4 or (args.resume and args.initialize_from):
        parser.error('Dedicated VRAM cap in (0,4]; choose resume OR initialization')
    if args.device == 'cpu' and args.execution == 'graph':
        parser.error('Use --execution eager for CPU')
    if (args.output/'config.json').exists() and not args.resume:
        parser.error('Existing run requires --resume; choose a new output otherwise')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.6)
    train = np.load(args.data_dir/'train.npy', mmap_mode='r')
    constructor = dict(vocab_size=50257, shape=(8,8,4), channels=128,
                       bath_type='conductance', port_scope='compact', write_exchange='contact_mode',
                       activity_adaptation=True, short_term_plasticity=True,
                       medium_execution='native', port_execution='native', pre_decoder_norm=True)
    saved = None
    if args.resume or args.initialize_from:
        saved = torch.load(args.resume or args.initialize_from, map_location='cpu', weights_only=False)
        constructor = dict(saved['config']['constructor'])
        constructor.update(medium_execution='native', port_execution='native')
        if args.resume:
            constructor.setdefault('pre_decoder_norm', False)
        else:
            constructor['pre_decoder_norm'] = True
    model = PlasticMediumPorts3D(**constructor).to(args.device).train()
    if saved:
        if args.resume:
            model.load_state_dict(saved['model'])
        else:
            load_medium_branch_weights(model, saved['model'])
    belief = unpack_belief(saved['belief'], args.device) if saved else model.initial_belief()
    if args.credit == 'local-receptors':
        if args.rank != 1:
            parser.error('--rank applies only to UORO')
        online = LocalPlasticTrainer(model, belief, event_duration=args.event_duration,
                                     substeps=args.substeps)
    else:
        online = OnlinePlasticTrainer(model, belief, event_duration=args.event_duration,
                                     substeps=args.substeps, rank=args.rank, seed=args.seed)
    optimizer = make_medium_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
        saved_state=saved['optimizer'] if args.resume else None)
    step, data_offset, adaptation_offset, best, best_step = 0, 0, 0, float('inf'), None
    if args.resume:
        if saved['config'].get('credit', 'uoro') != args.credit:
            raise ValueError('Resume must preserve the credit algorithm')
        for key, value in (('event_duration', args.event_duration),
                           ('substeps', args.substeps), ('tokens_per_update', args.tokens)):
            if saved['config'][key] != value:
                raise ValueError(f'Resume mismatch: {key}')
        if args.credit == 'uoro' and saved['config']['rank'] != args.rank:
            raise ValueError('Resume mismatch: rank')
        online_saved = dict(saved['online'])
        online_saved['belief'] = belief
        online.load_state_dict(online_saved)
        step, data_offset = saved['step'], saved['data_offset']
        adaptation_offset = saved.get('adaptation_offset', 0)
        if 'live_evaluation' in saved:
            if saved['config']['validation']['change_tokens'] != args.change_tokens:
                raise ValueError('Resume mismatch: live evaluation change_tokens')
            if (saved['config']['validation']['measurement_block'] != args.measurement_block
                    or saved['config']['validation']['hold_blocks'] != args.recovery_hold_blocks):
                raise ValueError('Resume mismatch: recovery measurement policy')
            best, best_step = saved['best_nll'], saved['best_step']
        torch.set_rng_state(saved['cpu_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'])
    elif args.initialize_from:
        data_offset = int(saved['data_offset'])
        if saved['config']['event_duration'] != args.event_duration or saved['config']['substeps'] != args.substeps:
            raise ValueError('Branch must preserve the checkpoint physical cadence')
    if data_offset+(args.steps-step)*args.tokens+1 > len(train):
        parser.error('Corpus exhausted; no silent stream wrapping')
    validation = None if args.calibrate_only else np.load(args.data_dir/'validation.npy', mmap_mode='r')
    probes = args.steps // args.validate_every - step // args.validate_every
    if args.steps > step and args.steps % args.validate_every:
        probes += 1
    if validation is not None and adaptation_offset + probes * args.change_tokens > len(validation):
        parser.error('Fresh adaptation stream exhausted; no silent replay or wrapping')
    files = [str(Path(__file__).relative_to(Path.cwd())),
             'information_boltzmann/runtime/online_credit.py',
             'information_boltzmann/runtime/local_credit.py',
             'information_boltzmann/runtime/lifelong_evaluation.py',
             'information_boltzmann/runtime/medium_health.py',
             'information_boltzmann/runtime/training.py',
             'information_boltzmann/runtime/optimization.py']
    files += ['information_boltzmann/core/'+name for name in (
        'plastic_medium.py','plastic_ports.py','conductance_response.py',
        'conduction_plasticity.py','short_term_plasticity.py','torus3d.py',
        'local_ports.py','readout_probes.py','mode_port.py')]
    config = {'constructor':constructor, 'parameters':sum(p.numel() for p in model.parameters()),
              'algorithm':('operator-local receptor e-prop' if args.credit == 'local-receptors'
                           else 'UORO / rank-compressed RTRL'),
              'credit':args.credit, 'rank':args.rank if args.credit == 'uoro' else None,
              'historical_credit_scope':('local receptor closing kinetics at frozen voltage/current inputs'
                                        if args.credit == 'local-receptors' else 'all persistent degrees of freedom'),
              'current_event_credit':'exact joint CE + W4 gradients for all model parameters',
              'execution':args.execution,
              'optimizer':'AdamW', 'training_groups':args.steps, 'tokens_per_update':args.tokens,
              'optimizer_groups':optimizer_policy(optimizer),
              'optimizer_update_policy':'Every tokens_per_update scored events, including live B and A replay; report actual count.',
              'event_duration':args.event_duration, 'substeps':args.substeps, 'seed':args.seed,
              'lr':args.lr,'min_lr':args.min_lr,'warmup':args.warmup,'decay':args.decay,
              'bptt_window':None, 'credit_horizon':None, 'eligibility_bytes':online.eligibility_bytes(),
              'state_policy':'Physical belief and all online sensitivity factors persist across updates; only one-event AD tape exists.',
              'initialize_from':str(args.initialize_from) if args.initialize_from else None,
              'eligibility_birth':'checkpoint continuation' if args.resume else 'zero sensitivity at branch birth',
              'training_start_offset':data_offset,'calibration_only':args.calibrate_only,
              'validation':{'protocol':'live_prequential_A_B_A_v1', 'reset':False,
                            'learning_active':True, 'change_tokens':args.change_tokens,
                            'measurement_block':args.measurement_block,
                            'hold_blocks':args.recovery_hold_blocks,
                            'B_role':'fresh online adaptation stream, not frozen held-out validation',
                            'domain_claim':'OWT context change; different semantic domains require labeled corpora',
                            'A_replay':'exact latest training group, scored separately from novel traffic',
                            'best_metric':'lifetime novel prequential NLL; not frozen validation NLL'},
              'health': {'enabled': not args.no_health, 'protocol': 'persistent_medium_health_v1',
                         'window_tokens': args.health_window, 'block_tokens': args.measurement_block,
                         'response_directions': args.response_directions,
                         'scope': 'Descriptive energy, spatial/covariance structure, risk trend, actual updates and conditional complete-state response; no new objective.'},
              'data':{'directory':str(args.data_dir),'train_tokens':len(train),
                      'manifest_sha256':hashlib.sha256((args.data_dir/'manifest.json').read_bytes()).hexdigest()},
              'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
              'stopping':'Requested budget, STOP, nonfinite values, or 4 GiB cap. Calibration is execution evidence only.'}
    atomic_json(args.output/'config.json', config)
    stop = False
    def request_stop(*unused):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    progress = {'status':'initializing','pid':os.getpid(),'step':step,'steps':args.steps}
    def publish(**values):
        progress.update(values,step=step,elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/'progress.json',progress)
    def memory():
        if args.device == 'cpu':
            return {'eligibility_mib':online.eligibility_bytes()/2**20}
        gpu = total_gpu_memory()
        if gpu['used_bytes'] >= min(gpu['total_bytes'],int(args.vram_limit_gib*2**30)):
            raise RuntimeError('Dedicated VRAM cap reached')
        return {'eligibility_mib':online.eligibility_bytes()/2**20,
                'allocated_mib':torch.cuda.memory_allocated()/2**20,
                'reserved_mib':torch.cuda.memory_reserved()/2**20,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'total_dedicated_mib':gpu['used_bytes']/2**20}
    def save(name):
        online_saved = online.state_dict()
        del online_saved['belief']
        payload = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'belief':pack_belief(online.belief),'online':online_saved,
                   'step':step,'data_offset':data_offset,'config':config,
                   'adaptation_offset':adaptation_offset,'live_evaluation':live.state_dict(),
                   'best_nll':best,'best_step':best_step,'cpu_rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state() if args.device == 'cuda' else None}
        temporary = args.output/f'{name}.{os.getpid()}.tmp'
        torch.save(payload,temporary)
        os.replace(temporary,args.output/name)
    publish(memory=memory())
    try:
        health = None if args.no_health else MediumHealthAuditor(
            model, window_tokens=args.health_window, block_tokens=args.measurement_block)
        if health is not None:
            online.health_capture = health.capture
        capture_type = CapturedLocalEvent if args.credit == 'local-receptors' else CapturedOnlineEvent
        captured = capture_type(online,loss_scale=1/args.tokens) if args.execution == 'graph' else None
        live = LiveOnlineEvaluation(online, optimizer, tokens_per_update=args.tokens,
                                     window_tokens=args.tokens, captured=captured, health=health)
        if args.resume and 'live_evaluation' in saved:
            live.load_state_dict(saved['live_evaluation'])
        publish(memory=memory())
        while step < args.steps:
            if stop or (args.output/'STOP').exists():
                break
            sequence = torch.from_numpy(np.array(train[data_offset:data_offset+args.tokens+1],
                                                dtype=np.int64)).to(args.device)
            lr = learning_rate(step+1,steps=args.steps,warmup=args.warmup,decay=args.decay,
                               peak=args.lr,floor=args.min_lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            if args.device == 'cuda':
                torch.cuda.synchronize()
            tick = time.perf_counter()
            rows = []
            for index in range(args.tokens):
                result = live.step(sequence[index:index+1], sequence[index+1:index+2])
                rows.append(result)
                if index == 0 or (index+1) % 16 == 0:
                    publish(status='training',events=online.events,memory=memory())
            if args.device == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            step += 1
            data_offset += args.tokens
            row = {key:float(torch.stack([r[key] for r in rows if key in r]).mean())
                   for key in set().union(*(r.keys() for r in rows))}
            row.update(step=step,events=online.events,data_offset=data_offset,lr=lr,
                       seconds_per_update=seconds,
                       seconds_per_token=seconds/args.tokens,memory=memory(),
                       physical_time=float(online.belief.medium.elapsed[0]),
                       **field_energy_statistics(online.belief.medium.field))
            if not args.calibrate_only and (step % args.validate_every == 0 or step == args.steps):
                publish(status='live_evaluation')
                response = (conditional_state_response(
                    model, online.belief, sequence[-1:].reshape(1), event_duration=args.event_duration,
                    substeps=args.substeps, directions=args.response_directions) if health is not None else None)
                b_targets = torch.from_numpy(np.array(
                    validation[adaptation_offset:adaptation_offset+args.change_tokens],
                    dtype=np.int64)).to(args.device)
                report = live.revisit_after_change(
                    sequence, [float(r['token_nll']) for r in rows], b_targets,
                    block_tokens=args.measurement_block, hold_blocks=args.recovery_hold_blocks)
                report.update(training_group=step, adaptation_stream_start=adaptation_offset,
                              adaptation_stream_end=adaptation_offset+args.change_tokens,
                              a_observation_start=data_offset-args.tokens,
                              a_last_target_position=data_offset,
                              a_sequence_sha256=hashlib.sha256(sequence.cpu().numpy().tobytes()).hexdigest(),
                              b_sequence_sha256=hashlib.sha256(b_targets.cpu().numpy().tobytes()).hexdigest())
                if response is not None:
                    report['conditional_state_response_before_change'] = response
                adaptation_offset += args.change_tokens
                with (args.output/'lifelong_evaluation.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(report,allow_nan=False)+'\n')
                row.update(lifelong_shock=report['shock'], lifelong_savings=report['savings'],
                           adaptation_offset=adaptation_offset)
                score = live.summary()['novel_prequential_nll']
                if score < best:
                    best,best_step = score,step
                    save('best_prequential.pt')
            row.update(events=online.events, prequential=live.summary(),memory=memory(),
                       physical_time=float(online.belief.medium.elapsed[0]),
                       **field_energy_statistics(online.belief.medium.field))
            if health is not None:
                row['health'] = health.summary()
            with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,allow_nan=False)+'\n')
            publish(status='training',**row)
            print(json.dumps(row,allow_nan=False),flush=True)
            if not args.calibrate_only and (step == 1 or step % args.save_every == 0):
                save('last.pt')
        if not args.calibrate_only:
            save('last.pt')
        publish(status='completed' if step == args.steps else 'stopped')
    except Exception as error:
        publish(status='failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
