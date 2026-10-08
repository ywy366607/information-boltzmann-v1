"""Continuous OWT training with a student or exact physical response backend.

One continuous real-input tick per token; endpoint quiet teacher queries are
separate counterfactual branches, counted separately. Shared task and student
train jointly; active evaluation preserves all state and optimizer cadence.
The exact backend differentiates live-weight quiet responses directly through
the next-token loss; activation checkpointing bounds retained branch memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_rtc_learning import FlyRTCLearner
from information_boltzmann.runtime.prequential_meter import FirstPassPredictionMeter
from information_boltzmann.runtime.lifelong_evaluation import recovery_summary, savings_summary
from scripts.ib.train_fly_bptt_stream import atomic_json, atomic_save

RESUME_CONFIG_FIELDS = (
    'd_model', 'observer_dim', 'sample_per_region', 'seed', 'window',
    'lambda_observer', 'learn_stp', 'read_norm_init', 'lr', 'lr_decoder',
    'lr_synapse', 'lr_sensory', 'horizon', 'query_radius', 'replay_capacity',
    'backend', 'read_horizon', 'no_checkpoint_branch',
)


def validate_resume_configuration(saved, args):
    for name in RESUME_CONFIG_FIELDS:
        if saved[name] != getattr(args, name):
            raise ValueError(f'Continuation configuration mismatch: {name}')


def execution_metadata(args):
    metadata = dict(execution_backend='eager streaming; CUDA Graph capture not implemented')
    if args.backend == 'exact':
        metadata.update(route='fly-causal-rtc-joint-v1',
            serial_rollouts=args.read_horizon, query_teacher_ticks_per_window=0,
            forecast_control='exact quiet responses at live weights, 0..read_horizon; no future tokens',
            dagger_scope='none: direct exact-response joint CE',
            graph_scope='live learned E/I edges and physical parameters every window',
            execution_backend='eager Triton physical kernels with activation checkpoint; no CUDA graph claim',
            gradient_scope='Actual BPTT window plus differentiable quiet-response branch',
            checkpoint_branch=not args.no_checkpoint_branch)
    return metadata


def check_cuda_checkpoint_parity(learner, targets):
    """Short real-data branch check; never update life, parameters or optimizer."""
    saved_horizon, saved_mode = learner.horizon, learner.checkpoint_branch
    saved_metrics = learner._read_metrics
    device = learner.state.h.device
    targets = torch.as_tensor(targets, device=device, dtype=torch.long)[None]
    inputs = torch.cat((targets.new_tensor([[learner.previous_token]]), targets[:, :-1]), dim=1)

    def run(mode):
        learner.horizon, learner.checkpoint_branch = 2, mode
        scores, state, features = learner.forward_window(inputs, targets)
        scores.mean().backward()
        result = dict(scores=scores.detach().cpu(), features=features.detach().cpu(),
            h=state.h.detach().cpu(), ring=tuple(p.detach().cpu() for p in state.ring),
            grads={name: p.grad.detach().cpu().clone() for name,p in learner.model.named_parameters()
                   if p.requires_grad and p.grad is not None})
        learner.model.zero_grad(set_to_none=True)
        return result

    try:
        plain, recomputed = run(False), run(True)
        forward_error = max(float((plain[k]-recomputed[k]).abs().max())
                            for k in ('scores','features','h'))
        pulses_match = all(torch.equal(a,b) for a,b in zip(plain['ring'],recomputed['ring']))
        if plain['grads'].keys() != recomputed['grads'].keys():
            raise AssertionError('CUDA checkpoint gradient parameter coverage differs')
        gradients = {}
        for name, first in plain['grads'].items():
            second = recomputed['grads'][name]
            difference = float(torch.linalg.vector_norm(first-second))
            norm = float(torch.linalg.vector_norm(first))
            gradients[name] = dict(norm=norm, difference_norm=difference,
                                   relative_difference=difference/max(norm,1e-12))
            if not torch.isfinite(second).all() or difference > 1e-6+2e-3*norm:
                raise AssertionError(f'CUDA checkpoint gradient mismatch: {name}')
        if forward_error > 2e-5 or not pulses_match:
            raise AssertionError('CUDA checkpoint forward or actual pulse mismatch')
        return dict(tokens=targets.numel(), horizon=2, forward_max_abs_error=forward_error,
                    actual_pulses_match=pulses_match, gradients=gradients,
                    parameter_updates=0, actual_clock_advanced=0,
                    claim='Short CUDA checkpoint numerical check; not full-horizon exactness or capability')
    finally:
        learner.horizon, learner.checkpoint_branch = saved_horizon, saved_mode
        learner._read_metrics = saved_metrics
        learner.model.zero_grad(set_to_none=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def reference_prior(train, vocab_size=50257):
    counts = np.ones(vocab_size, dtype=np.float64)
    for left in range(0, len(train), 1_000_000):
        counts += np.bincount(np.asarray(train[left:left+1_000_000], dtype=np.int64),
                              minlength=vocab_size)
    return -np.log(counts / counts.sum()), hashlib.sha256(counts.tobytes()).hexdigest()


def newborn_physical(model, device):
    h = torch.zeros(1, model.n_neurons, device=device)
    u = model.get_stp_params()[0].detach().expand_as(h).clone()
    baseline = torch.zeros(1, model.topographic_writer.n_total, device=device)
    return FlyPhysicalState(h, tuple(h.clone() for _ in range(4)),
        h.clone(), h.clone(), h.clone(), torch.ones_like(h), u, baseline,
        torch.zeros_like(h))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('results/fly_rtc_tick_dagger_100k'))
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    parser.add_argument('--pretrained-embedding', type=Path, default=Path('data/gpt2_model.safetensors'))
    parser.add_argument('--additional-tokens', type=int, default=100000)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--observer-dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--sample-per-region', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=14)
    parser.add_argument('--backend', choices=('student','exact'), default='student',
                        help='Approximate RTC student or differentiable exact physical response')
    parser.add_argument('--read-horizon', type=int,
                        help='Exact arm response budget;0 provides matched instantaneous control')
    parser.add_argument('--no-checkpoint-branch', action='store_true',
                        help='Exact arm only: retain full branch activations (diagnostics)')
    parser.add_argument('--query-radius', type=float, default=0.1)
    parser.add_argument('--replay-capacity', type=int, default=2)
    parser.add_argument('--read-norm-init', type=float, default=0.1)
    parser.add_argument('--lambda-observer', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-decoder', type=float)
    parser.add_argument('--lr-synapse', type=float, default=2e-4)
    parser.add_argument('--lr-sensory', type=float, default=2e-4)
    parser.add_argument('--learn-stp', action='store_true')
    parser.add_argument('--validate-every-tokens', type=int, default=5000)
    parser.add_argument('--eval-tokens', type=int, default=256)
    parser.add_argument('--replay-tokens', type=int, default=128)
    parser.add_argument('--log-every-tokens', type=int, default=128)
    parser.add_argument('--vram-limit-mib', type=float, default=3900)
    parser.add_argument('--calibrate', action='store_true',
                        help='Perform one real update, save full life, and exit')
    args = parser.parse_args(argv)
    if args.read_horizon is None:
        args.read_horizon = args.horizon
    if not 0 <= args.read_horizon <= args.horizon:
        parser.error('read-horizon must lie within0..horizon')
    if args.backend == 'exact':
        args.lambda_observer = 0.0
    for name in ('window', 'additional_tokens', 'observer_dim', 'sample_per_region',
                 'eval_tokens', 'replay_tokens', 'validate_every_tokens', 'log_every_tokens', 'horizon', 'replay_capacity'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if any(value % args.window for value in
           (args.additional_tokens, args.eval_tokens, args.replay_tokens)):
        parser.error('Training/evaluation token budgets must be multiples of window')
    if not args.calibrate and args.additional_tokens // args.window < 3000:
        parser.error('Formal run requires at least 3000 fresh optimizer windows')
    if args.read_norm_init <= 0 or args.lambda_observer < 0:
        parser.error('Positive read gain and nonnegative auxiliary weight required')
    if not 0 <= args.query_radius <= 1:
        parser.error('Query trust radius must be in [0, 1]')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this full-connectome real-data entry point')
    from information_boltzmann.runtime.gpu_memory import total_gpu_memory
    gpu_budget = total_gpu_memory()
    if gpu_budget['free_bytes'] / 2**20 < args.vram_limit_mib:
        raise MemoryError('The registered dedicated-memory budget is not available; '
                          'wait for the other GPU job to finish')
    # Import after CLI help so installation/interface inspection uses no GPU.
    from information_boltzmann.core.fly_rtc_student import TickResponseStudent
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    storage = args.checkpoint_dir or args.output
    storage.mkdir(parents=True, exist_ok=True)
    if (storage / 'last.pt').exists() and args.resume is None:
        raise FileExistsError('Existing individual requires explicit --resume')
    train = np.load(args.data / 'train.npy', mmap_mode='r')
    validation = np.load(args.data / 'validation.npy', mmap_mode='r')
    reference, counts_hash = reference_prior(train)
    reference_metadata = {
        'train': str((args.data / 'train.npy').resolve()), 'tokens': len(train),
        'vocab_size': 50257, 'smoothing': 'add one', 'frozen': True,
        'validation_labels_used_for_prior': False, 'counts_sha256': counts_hash}
    origin = {
        'seed': args.seed, 'graph_sha256': file_hash(args.graph),
        'train_sha256': file_hash(args.data / 'train.npy'),
        'validation_sha256': file_hash(args.data / 'validation.npy'),
        'embedding_sha256': file_hash(args.pretrained_embedding),
        'initialization': 'GPT-2 WTE input and decoder; train-only add-one bias; read gain 0.1',
        'source_sha256': {
            name: file_hash(ROOT / name) for name in (
                'information_boltzmann/core/fly_rtc_student.py',
                'information_boltzmann/core/fly_streaming_observer.py',
                'information_boltzmann/runtime/fly_rtc.py',
                'information_boltzmann/runtime/fly_rtc_flow.py',
                'information_boltzmann/core/fly_rtc_learning.py',
                'information_boltzmann/core/fly_causal_rtc_learning.py',
                'scripts/ib/train_fly_rtc_dagger.py',
                'information_boltzmann/core/fly_reservoir.py',
                'information_boltzmann/core/fly_bptt_learning.py',
                'information_boltzmann/core/triton_synapse.py')},
    }
    model = FlyReservoirLM(args.graph, vocab_size=50257, d_model=args.d_model,
        injection='topographic', read_surface='output', synapse_model='coba',
        use_alif=True, use_stp=True, decoder_bias=True,
        read_centering=False, use_read_gamma_trace=False,
        use_latent_predictor=False, use_graph_observer=False)
    model.dan_plastic_lr = 0.0
    if args.backend == 'exact':
        from information_boltzmann.core.fly_causal_rtc_learning import CausalHorizonReadout, FlyCausalRTCLearner
        model.causal_read = CausalHorizonReadout(args.d_model, horizon=args.horizon,
                                                key_dim=args.observer_dim)
    else:
        model.rtc_student = TickResponseStudent.from_model(
            model, latent_dim=args.observer_dim, sample_per_region=args.sample_per_region,
            seed=args.seed, horizon=args.horizon)
    model = model.to(device)
    from safetensors.torch import load_file
    with torch.no_grad():
        pretrained = load_file(str(args.pretrained_embedding))['wte.weight']
        if pretrained.shape != model.embedding.weight.shape:
            raise ValueError('Pretrained embedding dimension must match --d-model')
        model.embedding.weight.copy_(pretrained)
        model.decoder.weight.copy_(pretrained)
        model.decoder.bias.copy_(torch.from_numpy(-reference.astype(np.float32)).to(device))
        model.read_norm.weight.fill_(args.read_norm_init)
        del pretrained
    common = dict(lr=args.lr, lr_decoder=args.lr_decoder, lr_synapse=args.lr_synapse,
        lr_sensory=args.lr_sensory, learn_stp=args.learn_stp)
    if args.backend == 'exact':
        learner = FlyCausalRTCLearner(model, newborn_physical(model,device),
            horizon=args.read_horizon, checkpoint_branch=not args.no_checkpoint_branch, **common)
        response_module = model.causal_read
        checkpoint_format = 'fly-causal-rtc-joint-individual-v1'
    else:
        learner = FlyRTCLearner(model,newborn_physical(model,device), **common,
            lambda_jepa=args.lambda_observer,replay_capacity=args.replay_capacity,query_radius=args.query_radius)
        response_module = model.rtc_student
        checkpoint_format = 'fly-rtc-tick-dagger-individual-v1'
    learner.previous_token = int(train[0])
    cursor = val_cursor = trained = 0
    best = math.inf
    recent_a, recent_scores = [], []
    meter = FirstPassPredictionMeter(0, 0, reference_metadata)
    if args.resume is not None:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['format'] != checkpoint_format:
            raise ValueError('Resume requires this route’s complete checkpoint')
        for name in ('graph_sha256', 'train_sha256', 'validation_sha256', 'embedding_sha256'):
            if saved['origin'][name] != origin[name]:
                raise ValueError(f'Continuation provenance mismatch: {name}')
        if saved['origin']['source_sha256'] != origin['source_sha256']:
            raise ValueError('Continuation source changed; declare a separate experiment instead of silently replacing code')
        validate_resume_configuration(saved['config'], args)
        with torch.no_grad():
            named = dict(model.named_parameters())
            if set(named) != set(saved['model']):
                raise ValueError('Checkpoint parameter coverage mismatch')
            for name, parameter in named.items():
                parameter.copy_(saved['model'][name].to(device))
        response_module.load_state_dict(saved['student_module'])
        learner.restore_learning_state(saved['learner'])
        cursor, val_cursor, trained = (saved[key] for key in
                                     ('train_cursor', 'val_cursor', 'bptt_train_tokens'))
        recent_a, recent_scores = saved['recent_a'], saved['recent_scores']
        meter = FirstPassPredictionMeter.from_state_dict(saved['first_pass_meter'], reference_metadata)
        best, origin = saved['best_live_nll'], saved['origin']
        torch.set_rng_state(saved['rng_cpu'])
        torch.cuda.set_rng_state(saved['rng_cuda'])
        del saved
    start_trained, start = trained, time.perf_counter()
    target = trained + (args.window if args.calibrate else args.additional_tokens)
    if cursor + (target - trained) + 1 > len(train):
        raise ValueError('Insufficient fresh training data; stream cannot wrap')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(pid=os.getpid(), route='fly-rtc-tick-dagger-v1',
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        physical_ticks_per_token=1, gamma=False, serial_rollouts=args.horizon,
        query_teacher_ticks_per_window=args.horizon + 1,
        forecast_control='zero-drive future; observed drives supervised separately',
        dagger_scope='student-error-guided reencoded physical queries; approximate reduced state',
        graph_scope='initial learned-edge E/I-delay aggregate; fixed during this arm',
        runtime='timestamped handoff; trainer sequential; no parallel throughput claim',
        reference=reference_metadata, origin=origin,
        evaluation='active, pre-update, never reset; bridge targets included',
        integration='existing COBA-ALIF-STP; truncated BPTT window is learning scope',
        gpu_budget_at_start=gpu_budget,
        resource_dimensions='horizon/observer_dim/sample_per_region/replay_capacity/query_radius are declared computational and query budgets',
        budget_start_train_tokens=trained, additional_fresh_tokens=target-trained,
        target_train_tokens=target,
        stopping='fixed fresh-data budget; completion does not certify convergence')
    config.update(execution_metadata(args))
    if args.backend == 'exact':
        config.update(
            trainable_response_parameters=sum(p.numel() for p in model.causal_read.parameters()))
    atomic_json(args.output / 'config.json', config)
    (args.output / 'process.pid').write_text(str(os.getpid()), encoding='utf-8')
    if args.calibrate and args.backend == 'exact':
        atomic_json(args.output / 'cuda_checkpoint_parity.json',
                    check_cuda_checkpoint_parity(learner, train[cursor+1:cursor+3]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    next_eval = (trained // args.validate_every_tokens + 1) * args.validate_every_tokens
    last_metrics = {}

    def append(row, filename='metrics.jsonl'):
        with (args.output / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def progress(status):
        row = dict(status=status, bptt_train_tokens=trained, target_bptt_tokens=target,
            train_cursor=cursor, val_cursor=val_cursor, events=learner.events,
            bptt_optimizer_updates=learner.updates, physical_ticks=learner.physical_ticks,
            speed_tokens_per_sec=(trained-start_trained)/max(time.perf_counter()-start, 1e-6),
            best_live_nll=best if math.isfinite(best) else None,
            vram_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            vram_reserved_mib=torch.cuda.memory_reserved()/2**20,
            **meter.summary(), **last_metrics)
        atomic_json(args.output / 'progress.json', row)
        return row

    def checkpoint(is_best=False):
        payload = dict(format=checkpoint_format,
            model=dict(model.named_parameters()), student_module=response_module.state_dict(),
            learner=learner.state_dict(), config=config, origin=origin,
            train_cursor=cursor, val_cursor=val_cursor, bptt_train_tokens=trained,
            best_live_nll=best, recent_a=recent_a, recent_scores=recent_scores,
            first_pass_meter=meter.state_dict(), rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state())
        atomic_save(storage / 'last.pt', payload)
        if is_best:
            atomic_save(storage / 'best.pt', payload)

    def observe(values):
        nonlocal last_metrics
        values = np.asarray(values, dtype=np.int64)
        scores = []
        for left in range(0, len(values), args.window):
            chunk, last_metrics = learner.observe(values[left:left+args.window])
            scores.extend(chunk)
            if max(torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved())/2**20 > args.vram_limit_mib:
                raise MemoryError('Dedicated CUDA budget exceeded')
        return scores

    try:
        while trained < target:
            begin = time.perf_counter()
            values = np.asarray(train[cursor+1:cursor+args.window+1], dtype=np.int64)
            scores = observe(values)
            torch.cuda.synchronize()
            duration = time.perf_counter()-begin
            cursor += len(values)
            trained += len(values)
            meter.record(scores, reference[values])
            meter.verify_cursor(trained, cursor)
            recent_a = (recent_a + values.tolist())[-args.replay_tokens:]
            recent_scores = (recent_scores + scores)[-args.replay_tokens:]
            if trained == start_trained+args.window or trained % args.log_every_tokens == 0:
                row = progress('running')
                row.update(window_prequential_nll=float(np.mean(scores)), seconds_per_window=duration)
                append(row)
                print(f'STREAM {trained}/{target} | NLL {np.mean(scores):.4f} | {duration:.3f}s/{args.window}', flush=True)
            if args.calibrate:
                from information_boltzmann.runtime.gpu_memory import windows_gpu_memory
                atomic_json(args.output / 'calibration.json', dict(
                    window_tokens=len(values), pre_update_nll=float(np.mean(scores)),
                    fixed_unigram_nll=float(reference[values].mean()),
                    seconds_per_window=duration, tokens_per_sec=len(values)/duration,
                    cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    windows_memory=windows_gpu_memory(),
                    actual_ticks=learner.physical_ticks,
                    speculative_ticks=getattr(learner,'speculative_ticks',None),
                    optimizer_updates=learner.updates, metrics=last_metrics,
                    claim='One real-data numerical/resource calibration; no capability conclusion'))
                checkpoint()
                progress('calibrated')
                return
            if (args.output / 'STOP').exists():
                checkpoint()
                progress('paused')
                return
            if trained >= next_eval or trained == target:
                if val_cursor + args.eval_tokens > len(validation):
                    raise ValueError('Fresh validation stream exhausted')
                event_start, updates_start = learner.events, learner.updates
                fresh = np.asarray(validation[val_cursor:val_cursor+args.eval_tokens], dtype=np.int64)
                b_scores = observe(fresh)
                val_cursor += len(fresh)
                a_scores = observe(recent_a)
                live = float(np.mean(b_scores))
                improved = live < best
                best = min(best, live)
                # The first A token is a scored bridge with a different source
                # than A1; savings compare the remaining identical target pairs.
                report = dict(bptt_train_tokens=trained, fresh_validation_cursor=val_cursor,
                    event_interval=[event_start, learner.events],
                    update_interval=[updates_start, learner.updates],
                    live_prequential_nll=live, fixed_unigram_B_nll=float(reference[fresh].mean()),
                    gain_over_fixed_unigram=float(reference[fresh].mean())-live,
                    B_curve=b_scores, A1_curve=recent_scores, A2_replay_curve=a_scores,
                    A2_prequential_nll=float(np.mean(a_scores)),
                    fresh_events_between_encounters=len(fresh),
                    actual_intervening_events=len(fresh)+1,
                    matched_intervening_convention='includes scored return bridge before matched A2 pairs',
                    bridge_scoring='all scored; A opening bridge excluded from matched savings',
                    recovery=recovery_summary(b_scores, block_tokens=16, hold_blocks=2),
                    savings=savings_summary(recent_scores[1:], a_scores[1:],
                        intervening_events=len(fresh)+1, block_tokens=16, hold_blocks=2),
                    **meter.summary())
                append(report, 'lifelong_evaluation.jsonl')
                print(f'[ACTIVE EVAL {trained}] fresh B={live:.4f}; A revisit={np.mean(a_scores):.4f}', flush=True)
                checkpoint(improved)
                progress('running')
                next_eval += args.validate_every_tokens
        checkpoint()
        progress('completed')
    except BaseException as error:
        row = progress('failed')
        row['error'] = repr(error)
        atomic_json(args.output / 'progress.json', row)
        raise


if __name__ == '__main__':
    main()
