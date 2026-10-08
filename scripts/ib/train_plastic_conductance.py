"""Joint OWT training of one persistent, spatially plastic conductance medium.

The event cadence is explicit model time, not a biological millisecond claim.
CUDA Graph captures each differentiable chunk; detach preserves the full belief.
Validation clones mature training state per local site, warms it, then scores.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from torch.nn import functional as F

from information_boltzmann.core.plastic_medium import MediumState
from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.core.temporal_probes import TemporalProbeState
from information_boltzmann.core.gradient_norms import stable_clip_grad_norm_
from information_boltzmann.evaluation import WarmSiteSpec, field_energy_statistics
from information_boltzmann.runtime.gpu_memory import total_gpu_memory, windows_gpu_memory
from information_boltzmann.runtime.execution_cache import configure_execution_cache
from information_boltzmann.runtime.optimization import make_medium_optimizer, optimizer_policy
from information_boltzmann.runtime.training import (
    CapturedPlasticChunk, belief_tensors, clone_belief)


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def pack_belief(belief):
    def cpu(tensor):
        return None if tensor is None else tensor.detach().cpu().clone()
    state = belief.medium
    return {'field': cpu(state.field), 'flux': [cpu(x) for x in state.flux],
            'elapsed': cpu(state.elapsed), 'conduction': cpu(state.conduction),
            'receptors': cpu(state.receptors), 'transmission': cpu(state.transmission),
            'precision': cpu(belief.precision),
            'temporal': None if belief.temporal is None else
                {key: cpu(value) for key, value in belief.temporal.state_dict().items()}}


def unpack_belief(saved, device):
    def tensor(value):
        return None if value is None else value.to(device).clone()
    return PlasticBelief(MediumState(
        tensor(saved['field']), tuple(tensor(x) for x in saved['flux']),
        tensor(saved['elapsed']), tensor(saved['conduction']), tensor(saved['receptors']),
        tensor(saved.get('transmission'))),
        tensor(saved['precision']), None if saved.get('temporal') is None else
        TemporalProbeState.from_state_dict({key: tensor(value)
                                            for key, value in saved['temporal'].items()}))


def learning_rate(step, *, steps, warmup, decay, peak, floor):
    if warmup and step <= warmup:
        return floor + (peak-floor)*step/warmup
    if decay and step > steps-decay:
        fraction = (step-(steps-decay))/decay
        return peak * (floor/peak)**fraction
    return peak


@torch.no_grad()
def evaluation_chunk(model, ids, targets, belief, *, event_duration, substeps):
    table = F.normalize(model.source.embedding.weight, dim=-1)
    prepared = model.medium.prepare_evolution()
    features = []
    for index in range(ids.shape[1]):
        belief, _ = model.assimilate(belief, ids[:, index], token_features=table,
                                     diagnostics=False, training_terms=False)
        belief, _ = model.advance(belief, event_duration, substeps=substeps,
                                  prepared=prepared, diagnostics=False)
        feature, _ = model.read(belief, decode=False, prepared=prepared)
        features.append(feature)
    logits = model.decode(torch.stack(features, 1))
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten()), belief


class EvaluationGraph:
    """No-backward graph; parameter values refresh on every replay."""

    @torch.no_grad()
    def __init__(self, model, ids, targets, belief, duration, substeps):
        self.ids, self.targets = ids.clone(), targets.clone()
        self.input = clone_belief(belief)
        self.duration = torch.tensor(duration, device=ids.device, dtype=torch.float64)
        def operation():
            return evaluation_chunk(model, self.ids, self.targets, self.input,
                                    event_duration=self.duration, substeps=substeps)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                operation()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.nll, self.output = operation()

    @torch.no_grad()
    def __call__(self, ids, targets, belief):
        if ids.shape != self.ids.shape or targets.shape != self.targets.shape:
            raise ValueError('Evaluation graph requires the captured chunk shape')
        self.ids.copy_(ids)
        self.targets.copy_(targets)
        for destination, source in zip(belief_tensors(self.input), belief_tensors(belief)):
            destination.copy_(source)
        self.graph.replay()
        return self.nll.clone(), clone_belief(self.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--chunk-tokens', type=int, default=8)
    parser.add_argument('--event-duration', type=float, required=True)
    parser.add_argument('--substeps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--decay', type=int, default=300)
    parser.add_argument('--validate-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--memory-every', type=int, default=10)
    parser.add_argument('--allocator-fraction', type=float, default=0.40)
    parser.add_argument('--vram-limit-gib', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=449)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--write-exchange', choices=('global', 'contact_mode'),
                        default='contact_mode')
    parser.add_argument('--port-scope', choices=('global', 'compact'), default='compact')
    parser.add_argument('--activity-adaptation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--short-term-plasticity', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--medium-execution', choices=('native', 'fused'), default='fused')
    parser.add_argument('--port-execution', choices=('native', 'fused'), default='fused')
    parser.add_argument('--compile-cache-dir', type=Path, default=Path('scratch/compiler_cache'))
    parser.add_argument('--write-port-radius', type=float, nargs=3)
    parser.add_argument('--read-port-radius', type=float, nargs=3)
    args = parser.parse_args()
    execution_cache = configure_execution_cache(args.compile_cache_dir)
    if min(args.steps, args.tokens, args.chunk_tokens, args.substeps,
           args.validate_every, args.save_every, args.memory_every) < 1:
        parser.error('Positive budgets and intervals required')
    if args.tokens % args.chunk_tokens or not math.isfinite(args.event_duration) or args.event_duration <= 0:
        parser.error('Tokens must divide into fixed chunks; finite positive duration required')
    if not 0 < args.min_lr <= args.lr or min(args.warmup, args.decay) < 0 or args.warmup+args.decay > args.steps:
        parser.error('Invalid WSD schedule')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Finite nonnegative weight decay required')
    if not 0 < args.allocator_fraction <= 1 or not 0 < args.vram_limit_gib <= 4:
        parser.error('Allocator fraction in (0,1]; dedicated limit in (0,4] GiB')
    if (args.output/'config.json').exists() and args.resume is None:
        parser.error('Existing run requires explicit --resume; choose a new output for scratch')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.allocator_fraction)
    train = np.load(args.data_dir/'train.npy', mmap_mode='r')
    validation = np.load(args.data_dir/'validation.npy', mmap_mode='r')
    if args.steps*args.tokens+1 > len(train):
        parser.error('Training budget exceeds corpus; no silent wrapping')
    spec = WarmSiteSpec()
    if spec.warm_in_tokens % args.chunk_tokens or spec.score_tokens % args.chunk_tokens:
        parser.error('Validation lengths must divide into fixed chunks')
    if max(spec.site_starts)+spec.required_tokens_per_site > len(validation):
        parser.error('Validation corpus too short')
    constructor = dict(vocab_size=50257, shape=(8,8,4), channels=128,
                       bath_type='conductance', write_exchange=args.write_exchange,
                       port_scope=args.port_scope, activity_adaptation=args.activity_adaptation,
                       short_term_plasticity=args.short_term_plasticity,
                       medium_execution=args.medium_execution, port_execution=args.port_execution,
                       pre_decoder_norm=True)
    saved = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if saved is not None:
        constructor['pre_decoder_norm'] = saved['config']['constructor'].get('pre_decoder_norm', False)
    if args.port_scope == 'compact':
        constructor.update(write_port_radius=args.write_port_radius, read_port_radius=args.read_port_radius)
    model = PlasticMediumPorts3D(**constructor).cuda().train()
    if args.port_scope == 'compact':
        constructor.update(write_port_radius=list(model.write_agent.local_ports.physical_radius),
                           read_port_radius=list(model.readout.physical_radius))
    optimizer = make_medium_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
        saved_state=saved['optimizer'] if saved is not None else None)
    belief = model.initial_belief()
    step, best, best_step = 0, float('inf'), None
    files = ['information_boltzmann/core/'+name for name in
             ('plastic_medium.py','plastic_ports.py','conductance_response.py','conduction_plasticity.py',
              'short_term_plasticity.py')]
    files.append('information_boltzmann/runtime/optimization.py')
    files += ['information_boltzmann/runtime/training.py', str(Path(__file__).relative_to(Path.cwd()))]
    files += ['information_boltzmann/runtime/execution_cache.py']
    files += ['information_boltzmann/core/torus3d.py', 'information_boltzmann/core/mode_port.py']
    files += ['information_boltzmann/core/local_ports.py', 'information_boltzmann/core/readout_probes.py']
    config = {'architecture':model.architecture, 'constructor':constructor,
              'parameters':sum(p.numel() for p in model.parameters()),
              'optimizer_updates':args.steps, 'tokens_per_update':args.tokens,
              'training_tokens':args.steps*args.tokens, 'bptt_chunk_tokens':args.chunk_tokens,
              'event_duration':args.event_duration, 'substeps':args.substeps,
              'physical_time_note':'Explicit nondimensional launch cadence, inherited from verified execution; not optimized biological or cognitive time.',
              'optimizer':'AdamW', 'optimizer_groups':optimizer_policy(optimizer),
              'lr':args.lr,'min_lr':args.min_lr,
              'warmup':args.warmup,'decay':args.decay,'seed':args.seed,
              'backend':'FP32 CUDA Graph forward/backward, all active components jointly trained',
              'execution_cache':execution_cache,
              'state_policy':'One birth initialization; field, three fluxes, time, conduction, receptors, transmission resources/utilization and precision persist across all chunks/updates.',
              'validation':{'protocol':'mature-state warm-local continuation', **asdict(spec),
                            'reset':False, 'start_state':'owned full training-belief clone per site; 256 context tokens before scoring'},
              'data':{'directory':str(args.data_dir),'train_tokens':len(train),'validation_tokens':len(validation),
                      'manifest_sha256':hashlib.sha256((args.data_dir/'manifest.json').read_bytes()).hexdigest()},
              'source_hashes':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files},
              'memory_policy':{'dedicated_limit_gib':args.vram_limit_gib,'allocator_fraction':args.allocator_fraction,'shared_growth':'recorded and permitted'},
              'stopping':'Requested update budget, explicit STOP file, nonfinite loss/gradient, or dedicated VRAM cap; budget alone does not establish convergence.'}
    if args.resume:
        # Historical checkpoints predate the mode-selective boundary. Preserve
        # their actual law; a changed law is a new branch, not exact continuation.
        saved['config']['constructor'].setdefault('write_exchange', 'global')
        saved['config']['constructor'].setdefault('port_scope', 'global')
        saved['config']['constructor'].setdefault('activity_adaptation', False)
        saved['config']['constructor'].setdefault('short_term_plasticity', False)
        saved['config']['constructor'].setdefault('medium_execution', 'native')
        saved['config']['constructor'].setdefault('port_execution', 'native')
        saved['config']['constructor'].setdefault('pre_decoder_norm', False)
        # Execution fusion changes arithmetic scheduling, not weights, state
        # or the physical law. It may be enabled on an existing checkpoint.
        config['resume_execution'] = {
            key: saved['config']['constructor'][key]
            for key in ('medium_execution', 'port_execution')}
        for key in config['resume_execution']:
            saved['config']['constructor'][key] = constructor[key]
        for key in ('constructor','tokens_per_update','bptt_chunk_tokens','event_duration','substeps'):
            if saved['config'][key] != config[key]:
                raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(saved['model'])
        belief = unpack_belief(saved['belief'], 'cuda')
        step, best, best_step = saved['step'], saved['best_nll'], saved['best_step']
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
    atomic_json(args.output/'config.json', config)
    stop_requested = False
    def request_stop(*unused):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.time()
    progress = {'status':'initializing','pid':os.getpid(),'step':step,'steps':args.steps,
                'parameters':config['parameters'],'best_validation_nll':None if not math.isfinite(best) else best}
    def publish(**values):
        progress.update(values, step=step, elapsed_seconds=time.time()-started)
        atomic_json(args.output/'progress.json', progress)
    def save(name):
        payload = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'belief':pack_belief(belief),'step':step,'data_offset':step*args.tokens,
                   'best_nll':best,'best_step':best_step,'config':config,
                   'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        temporary = args.output/f'{name}.{os.getpid()}.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, args.output/name)
    def memory_guard(stage):
        gpu = total_gpu_memory()
        memory = {'stage':stage,'total_dedicated_mib':gpu['used_bytes']/2**20,
                  'allocated_mib':torch.cuda.memory_allocated()/2**20,
                  'reserved_mib':torch.cuda.memory_reserved()/2**20,
                  'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        publish(memory=memory)
        if gpu['used_bytes'] >= min(gpu['total_bytes'], int(args.vram_limit_gib*2**30)):
            raise RuntimeError('Dedicated VRAM cap reached')
        return memory
    evaluator = None
    @torch.no_grad()
    def validate():
        nonlocal evaluator
        model.eval()
        per_site, stats = [], []
        try:
            for start in spec.site_starts:
                context = torch.from_numpy(np.array(validation[start:start+spec.required_tokens_per_site], dtype=np.int64)).cuda()[None]
                current = clone_belief(belief)
                if evaluator is None:
                    evaluator = EvaluationGraph(model, context[:,:args.chunk_tokens],
                                                context[:,1:args.chunk_tokens+1], current,
                                                args.event_duration, args.substeps)
                    memory_guard('evaluation_graph')
                losses = []
                for offset in range(0, spec.warm_in_tokens+spec.score_tokens, args.chunk_tokens):
                    nll, current = evaluator(context[:,offset:offset+args.chunk_tokens],
                                             context[:,offset+1:offset+args.chunk_tokens+1], current)
                    if offset >= spec.warm_in_tokens:
                        losses.append(nll)
                score = float(torch.stack(losses).mean())
                if not math.isfinite(score):
                    raise FloatingPointError('Nonfinite validation NLL')
                per_site.append({'start':start,'nll':score})
                stats.append(field_energy_statistics(current.medium.field))
            return {'validation_nll':sum(x['nll'] for x in per_site)/len(per_site),
                    'sites':per_site,'site_field_statistics':stats}
        finally:
            model.train()
    publish()
    print(json.dumps({'output':str(args.output),'pid':os.getpid(),'parameters':config['parameters'],
                      'event_duration':args.event_duration,'tokens_per_update':args.tokens,'steps':args.steps}), flush=True)
    try:
        memory_guard('before_capture')
        index = step*args.tokens
        sample = torch.from_numpy(np.array(train[index:index+args.chunk_tokens+1], dtype=np.int64)).cuda()[None]
        capture = CapturedPlasticChunk(model, sample[:,:-1], sample[:,1:], belief,
                                       event_duration=args.event_duration, substeps=args.substeps,
                                       loss_scale=args.chunk_tokens/args.tokens)
        torch.cuda.synchronize()
        memory_guard('after_training_capture')
        publish(status='training')
        while step < args.steps:
            if stop_requested or (args.output/'STOP').exists():
                break
            offset = step*args.tokens
            ids = torch.from_numpy(np.array(train[offset:offset+args.tokens+1], dtype=np.int64)).cuda()[None]
            lr = learning_rate(step+1, steps=args.steps, warmup=args.warmup, decay=args.decay, peak=args.lr, floor=args.min_lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            capture.zero_grad()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            losses, nlls = [], []
            for index in range(0,args.tokens,args.chunk_tokens):
                loss, next_belief, nll = capture.backward(ids[:,index:index+args.chunk_tokens],
                                                         ids[:,index+1:index+args.chunk_tokens+1], belief)
                losses.append(loss)
                nlls.append(nll)
                belief = next_belief.detach()
            mean_loss, mean_nll = torch.stack(losses).mean(), torch.stack(nlls).mean()
            if not bool(torch.isfinite(mean_loss)):
                raise FloatingPointError('Nonfinite training loss')
            norm = stable_clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            step += 1
            row = {'step':step,'tokens_seen':step*args.tokens,'train_nll':float(mean_nll),
                   'loss':float(mean_loss),'port_objective':float(mean_loss-mean_nll),
                   'gradient_norm_before_clip':float(norm),'lr':lr,'seconds_per_update':seconds,
                   'physical_time':float(belief.medium.elapsed[0]),
                   **field_energy_statistics(belief.medium.field)}
            if step == 1 or step % args.memory_every == 0:
                row['memory'] = memory_guard(f'update_{step}')
                if model.short_term_plasticity:
                    with torch.no_grad():
                        state = belief.medium
                        coefficients = model.medium.short_term_plasticity.coefficients(model.medium.material_field())
                        gain = model.medium.short_term_plasticity.transmission_gain(state.transmission, coefficients)
                        row['short_term_plasticity'] = {
                            'resource_mean': float(state.transmission[..., 0].mean()),
                            'utilization_mean': float(state.transmission[..., 1].mean()),
                            'gain_mean': float(gain.mean()), 'gain_min': float(gain.amin()),
                            'gain_max': float(gain.amax()),
                            'gain_spatial_variance': float(gain.var((1, 2, 3), unbiased=False).mean())}
                if model.port_scope == 'compact':
                    with torch.no_grad():
                        row['local_ports'] = {
                            key: value.cpu().tolist() for key, value in
                            model.port_snapshot(belief).items()
                            if key not in ('write_port_weights', 'read_port_weights')}
            if step % args.validate_every == 0 or step == args.steps:
                publish(status='validating')
                row.update(validate())
                if row['validation_nll'] < best:
                    best, best_step = row['validation_nll'], step
                    save('best.pt')
                publish(status='training',best_validation_nll=best,best_step=best_step)
            with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,allow_nan=False)+'\n')
            publish(**row)
            if step == 1 or step % args.save_every == 0:
                save('last.pt')
            if step == 1 or step % 10 == 0:
                print(f"step={step}/{args.steps} NLL={row['train_nll']:.5f} port={row['port_objective']:.5f} grad={float(norm):.4g} {seconds:.3f}s/update",flush=True)
        save('last.pt')
        publish(status='completed' if step == args.steps else 'stopped')
    except Exception as error:
        publish(status='failed',error=f'{type(error).__name__}: {error}')
        # Preserve the most recent successful checkpoint instead of overwriting
        # it with a potentially nonfinite or partially evolved state.
        raise
    finally:
        try:
            publish(process_memory=windows_gpu_memory())
        except Exception:
            pass


if __name__ == '__main__':
    main()
