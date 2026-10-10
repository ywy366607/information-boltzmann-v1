"""Continuous real-OWT COBA learning with 32-event truncated BPTT.

Resume physical life from the archived online learner or this branch. Active
validation uses the same learner and scores each target before its update.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState, FlyBPTTLearner, FlyBPTTGraph, STP_PARAMETER_NAMES,
)
from information_boltzmann.runtime.lifelong_evaluation import recovery_summary, savings_summary
from information_boltzmann.runtime.prequential_meter import FirstPassPredictionMeter


def atomic_json(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.25 * min(attempt + 1, 8))
    raise PermissionError(f"Could not replace {path}")


_CHECKPOINT_DIR = None


def atomic_save(path, payload):
    if _CHECKPOINT_DIR is not None:
        path = Path(_CHECKPOINT_DIR) / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
    # The 1.1 GB checkpoint write attracts transient Windows file locks
    # (indexer/antivirus corrupting the tmp mid-write AND locking the fresh
    # file); retry the WHOLE save with fresh names until one lands cleanly.
    last_error = None
    for attempt in range(5):
        temporary = path.with_suffix(f'.pt.tmp{attempt}')
        try:
            torch.save(payload, temporary)
            for retry in range(40):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError as error:
                    last_error = error
                    time.sleep(0.5 * min(retry + 1, 8))
            break
        except (RuntimeError, OSError) as error:
            last_error = error
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(2.0 * (attempt + 1))
    raise type(last_error)(f"atomic_save failed after retries: {last_error}")


def resolve_learning_modes(args, previous_config, previous_learner):
    """Omitted flags inherit the continuing individual's algorithm."""
    if getattr(args, 'detach_reset', None) is None:
        args.detach_reset = bool(previous_config.get('detach_reset', False))
    if getattr(args, 'surrogate_mode', None) is None:
        args.surrogate_mode = previous_config.get('surrogate_mode', 'absolute')
    if getattr(args, 'transmission_mode', None) is None:
        args.transmission_mode = previous_config.get('transmission_mode', 'atomic')
    if args.read_centering is None:
        args.read_centering = bool(previous_learner.get('read_centering',
                                  previous_config.get('read_centering', False)))
    if args.dan_plastic_lr is None:
        args.dan_plastic_lr = float(previous_learner.get('dan_plastic_lr',
                                   previous_config.get('dan_plastic_lr', 0.0)))
    if args.dan_plastic_lr < 0:
        raise ValueError('DAN plasticity learning rate must be nonnegative')
    if getattr(args, 'use_read_gamma_trace', None) is None:
        args.use_read_gamma_trace = bool(previous_learner.get('use_read_gamma_trace',
                                         previous_config.get('use_read_gamma_trace', False)))
    if getattr(args, 'init_read_gamma', None) is None:
        saved_gamma = previous_learner.get('init_read_gamma', previous_config.get('init_read_gamma', 'anatomical'))
        try:
            args.init_read_gamma = float(saved_gamma)
        except (ValueError, TypeError):
            args.init_read_gamma = saved_gamma
    if getattr(args, 'use_latent_predictor', None) is None:
        args.use_latent_predictor = bool(previous_learner.get('use_latent_predictor',
                                         previous_config.get('use_latent_predictor', False)))
    if getattr(args, 'use_graph_observer', None) is None:
        args.use_graph_observer = bool(previous_learner.get('use_graph_observer',
                                       previous_config.get('use_graph_observer', False)))
    if getattr(args, 'obs_init_gamma', None) is None:
        args.obs_init_gamma = float(previous_learner.get('obs_init_gamma',
                                     previous_config.get('obs_init_gamma', 0.0)))
    if getattr(args, 'ctm_loss', None) is None:
        args.ctm_loss = bool(previous_learner.get('use_ctm_loss',
                             previous_config.get('ctm_loss', True)))
    if getattr(args, 'lambda_mcr2', None) is None:
        args.lambda_mcr2 = float(previous_learner.get('lambda_mcr2',
                                 previous_config.get('lambda_mcr2', 0.0)))
    if getattr(args, 'eps_mcr2', None) is None:
        args.eps_mcr2 = float(previous_learner.get('eps_mcr2',
                              previous_config.get('eps_mcr2', 0.5)))
    if getattr(args, 'adaptive_admission', None) is None:
        args.adaptive_admission = bool(previous_learner.get('adaptive_admission',
                                       previous_config.get('adaptive_admission', False)))
    if getattr(args, 'min_settle_ticks', None) is None:
        args.min_settle_ticks = int(previous_learner.get('min_settle_ticks',
                                    previous_config.get('min_settle_ticks', 3)))
    if getattr(args, 'flux_baseline', None) is None:
        args.flux_baseline = float(previous_learner.get('flux_baseline',
                                   previous_config.get('flux_baseline', 0.048)))


def disable_alif(model):
    """beta = clamp(exp(log_beta), 1e-4, 2): log_beta = -20 sits below the clamp, so beta = 1e-4 and its gradient is 0."""
    if not getattr(model, 'use_alif', False):
        raise ValueError('This model has no ALIF to disable')
    with torch.no_grad():
        model.log_beta.fill_(-20.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, default=None,
                        help='Resume from a complete lifecycle checkpoint')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Initialize a newborn individual from scratch using pretrained embeddings and unigram prior')
    parser.add_argument('--pretrained-embedding', type=Path, default=Path('data/gpt2_model.safetensors'))
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--decoder-bias', dest='decoder_bias', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--read-norm-init', type=float, default=0.1)
    parser.add_argument('--output', type=Path, default=Path('results/q8_fly_bptt32_continuous_100k'))
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--reference-train', type=Path, default=None,
                        help='Fixed train-only unigram reporting reference; retained across resumes')
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    parser.add_argument('--additional-tokens', type=int, default=100000)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--settle-ticks', type=int, default=None,
                        help='Quiet physical ticks after each input pulse; default retains checkpoint timing')
    parser.add_argument('--adaptive-admission', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable event-driven adaptive admission settling based on causal relaxation')
    parser.add_argument('--min-settle-ticks', type=int, default=3,
                        help='Minimum physical quiet ticks per token ensuring causal signal arrival (default: 3)')
    parser.add_argument('--flux-baseline', type=float, default=0.048,
                        help='Membrane kinetic flux dissipation baseline for attractor admission (default: 0.048)')
    parser.add_argument('--ctm-loss', dest='ctm_loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable CTM dual-anchor (min loss + max certainty) training objective')
    parser.add_argument('--checkpointing', dest='checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable gradient checkpointing per token to avoid VRAM blowup during deep settling')
    parser.add_argument('--detach-reset', action=argparse.BooleanOptionalAction, default=None,
                        help='Detach only hard-reset spike derivative; resumes retain saved convention')
    parser.add_argument('--surrogate-mode', choices=('absolute', 'threshold'), default=None,
                        help='Spike proxy width convention; resumes inherit the saved rule')
    parser.add_argument('--transmission-mode', choices=('atomic', 'incoming'), default=None,
                        help='Summation execution order; resumes inherit the saved convention')
    parser.add_argument('--writer-baseline-clock', choices=('input', 'physical'), default=None,
                        help='Packet baseline clock; default reproduces the saved implementation')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-decoder', type=float, default=None,
                        help='Decoder weight only; other read parameters retain --lr')
    parser.add_argument('--lr-synapse', type=float, default=None)
    parser.add_argument('--lr-sensory', type=float, default=None)
    parser.add_argument('--dan-plastic-lr', type=float, default=None,
                        help='Level-2: learning rate of the dopamine-gated three-factor '
                             'local rule on the DAN-innervated synapses (0 disables)')
    parser.add_argument('--read-centering', action=argparse.BooleanOptionalAction, default=None,
                        help='Center the read pathway on the running state baseline: '
                             'the readout sees only the conditional component')
    parser.add_argument('--use-read-gamma-trace', action=argparse.BooleanOptionalAction, default=None,
                        help='Continuous 2nd-order Gamma-trace temporal readout accumulator')
    parser.add_argument('--init-read-gamma', default='anatomical',
                        help='Initial decay rate for continuous Gamma-trace accumulator (default: anatomical)')
    parser.add_argument('--reinit-read-gamma', action='store_true',
                        help='Re-initialize logit_read_gamma with the new wide anatomical multi-scale timescale')
    parser.add_argument('--use-latent-predictor', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable LeJEPA latent predictive coding and SIGReg anti-collapse')
    parser.add_argument('--use-graph-observer', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable HX-1 Graph-Constrained Distributed State Observer with Predict-Arrive-Correct loop')
    parser.add_argument('--lambda-jepa', type=float, default=0.1,
                        help='Weight of LeJEPA latent dynamics loss in joint objective')
    parser.add_argument('--lambda-sigreg', type=float, default=0.2,
                        help='Weight of SIGReg anti-collapse regularizer in LeJEPA loss')
    parser.add_argument('--lambda-mcr2', type=float, default=None,
                        help='Weight of MCR^2 maximal coding rate reduction loss in joint objective (default: 0.0)')
    parser.add_argument('--eps-mcr2', type=float, default=None,
                        help='MCR^2 coding rate precision scale epsilon (default: 0.5)')
    parser.add_argument('--max-horizon', type=int, default=14,
                        help='Maximum forward conduction horizons for DAgger LeJEPA distillation (default: 14)')
    parser.add_argument('--dagger-beta', type=float, default=0.5,
                        help='DAgger interpolation parameter between teacher state and student prediction (default: 0.5)')
    parser.add_argument('--obs-init-gamma', type=float, default=0.0,
                        help='Initial value of coupling factor gamma for graph observer lookahead residual (default: 0.0)')
    parser.add_argument('--learn-stp', action='store_true', default=None,
                        help='Jointly learn existing STP release/recovery parameters; resumes inherit activation')
    parser.add_argument('--plasticity-optimizer', choices=('adamw', 'sgd'), default='adamw')
    parser.add_argument('--disable-alif', action='store_true',
                        help='Ablation: ALIF strength beta held at its lower clamp (1e-4) with zero gradient; '
                             'all other state, weights and the learner are unchanged')
    parser.add_argument('--validate-every-tokens', type=int, default=5000)
    parser.add_argument('--log-every-tokens', type=int, default=128)
    parser.add_argument('--eval-tokens', type=int, default=256)
    parser.add_argument('--replay-tokens', type=int, default=128)
    parser.add_argument('--vram-limit-mib', type=float, default=3900)
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--calibrate', action='store_true', help='One real training window, save full continuation and exit')
    parser.add_argument('--checkpoint-dir', type=Path, default=None,
                        help='Write weight checkpoints here instead of the output dir '
                             '(for output dirs on drives where large writes are unstable)')
    parser.add_argument('--extend-budget', action='store_true',
                        help='Replace an unfinished target with current count + additional fresh tokens')
    global _CHECKPOINT_DIR
    args = parser.parse_args()
    _CHECKPOINT_DIR = str(args.checkpoint_dir) if args.checkpoint_dir else None
    if args.settle_ticks is not None and args.settle_ticks < 0:
        parser.error('settle-ticks must be nonnegative')
    args.lr_synapse = args.lr if args.lr_synapse is None else args.lr_synapse
    args.lr_sensory = args.lr if args.lr_sensory is None else args.lr_sensory
    if min(args.window, args.additional_tokens, args.eval_tokens, args.replay_tokens) < 1:
        parser.error('Budgets must be positive')
    if any(value % args.window for value in (args.additional_tokens, args.eval_tokens, args.replay_tokens)):
        parser.error('Training/evaluation budgets must be multiples of the captured window')
    if not torch.cuda.is_available():
        parser.error('CUDA required')
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_storage = Path(args.checkpoint_dir) if args.checkpoint_dir else args.output
    if (ckpt_storage / 'last.pt').exists():
        if args.resume and args.resume.resolve() != (ckpt_storage / 'last.pt').resolve():
            parser.error('Output already contains a continuing individual')
        args.resume = ckpt_storage / 'last.pt'
        is_scratch_start = False
    elif (args.output / 'last.pt').exists():
        if args.resume and args.resume.resolve() != (args.output / 'last.pt').resolve():
            parser.error('Output already contains a continuing individual')
        args.resume = args.output / 'last.pt'
        is_scratch_start = False
    elif args.from_scratch:
        is_scratch_start = True
    elif args.resume is not None:
        is_scratch_start = False
    else:
        parser.error('Either --resume or --from-scratch must be specified')
    torch.manual_seed(11)
    train = np.load(args.data / 'train.npy', mmap_mode='r')
    val = np.load(args.data / 'validation.npy', mmap_mode='r')

    if is_scratch_start:
        print('Initializing newborn individual from scratch with pretrained embeddings...', flush=True)
        if args.d_model is None:
            args.d_model = 768
        if args.decoder_bias is None:
            args.decoder_bias = True
        if args.detach_reset is None:
            args.detach_reset = False
        if args.surrogate_mode is None:
            args.surrogate_mode = 'absolute'
        if args.transmission_mode is None:
            args.transmission_mode = 'atomic'
        if args.settle_ticks is None:
            args.settle_ticks = 0
        if args.writer_baseline_clock is None:
            args.writer_baseline_clock = 'physical'
        if args.read_centering is None:
            args.read_centering = False
        if args.use_read_gamma_trace is None:
            args.use_read_gamma_trace = False
        if args.use_latent_predictor is None:
            args.use_latent_predictor = False
        if args.use_graph_observer is None:
            args.use_graph_observer = False
        if args.dan_plastic_lr is None:
            args.dan_plastic_lr = 0.0
        if args.reference_train is None:
            args.reference_train = args.data / 'train.npy'

        previous_config = {
            'd_model': args.d_model, 'decoder_bias': args.decoder_bias,
            'data': str(args.data), 'reference_train': str(args.reference_train),
        }
        fmt = 'fly-bptt-v1'

        model = FlyReservoirLM(args.graph, vocab_size=50257, d_model=args.d_model,
            injection='topographic', read_surface='output', synapse_model='coba',
            use_alif=True, use_stp=True, decoder_bias=args.decoder_bias,
            read_centering=args.read_centering,
            use_read_gamma_trace=args.use_read_gamma_trace,
            init_read_gamma=args.init_read_gamma,
            use_latent_predictor=args.use_latent_predictor,
            use_graph_observer=args.use_graph_observer,
            lambda_obs=args.lambda_jepa,
            lambda_sigreg=args.lambda_sigreg,
            max_horizon=args.max_horizon,
            dagger_beta=args.dagger_beta,
            obs_init_gamma=args.obs_init_gamma,
            detach_reset=args.detach_reset, surrogate_mode=args.surrogate_mode,
            transmission_mode=args.transmission_mode).cuda()
        model.dan_plastic_lr = args.dan_plastic_lr

        from safetensors.torch import load_file
        pretrained = load_file(str(args.pretrained_embedding))
        with torch.no_grad():
            model.embedding.weight.copy_(pretrained['wte.weight'])
            model.decoder.weight.copy_(pretrained['wte.weight'])
            del pretrained
            if model.decoder.bias is not None:
                ref_train = np.load(args.reference_train, mmap_mode='r')
                ref_counts = np.ones(50257, dtype=np.float64)
                for left in range(0, len(ref_train), 1_000_000):
                    ref_counts += np.bincount(
                        ref_train[left:left+1_000_000].astype(np.int64), minlength=50257)
                prior = np.log(ref_counts / ref_counts.sum()).astype(np.float32)
                model.decoder.bias.copy_(torch.from_numpy(prior))
                del ref_train, ref_counts
            model.read_norm.weight.fill_(args.read_norm_init)

        n = model.n_neurons
        h = torch.zeros(1, n, device='cuda')
        ring = tuple(torch.zeros(1, n, device='cuda') for _ in range(4))
        ge = torch.zeros(1, n, device='cuda')
        gi = torch.zeros(1, n, device='cuda')
        b = torch.zeros(1, n, device='cuda')
        x = torch.ones(1, n, device='cuda')
        u = model.get_stp_params()[0].clone().expand_as(h).contiguous()
        baseline = torch.zeros(1, model.topographic_writer.n_total if model.topographic_writer else n, device='cuda')
        h_mean = torch.zeros(1, n, device='cuda')
        dan_gate = torch.zeros(1, n, device='cuda') if args.dan_plastic_lr > 0 else torch.empty(0, device='cuda')
        gamma_z1 = torch.zeros(1, model.n_read, device='cuda') if args.use_read_gamma_trace else torch.empty(0, device='cuda')
        gamma_z2 = torch.zeros(1, model.n_read, device='cuda') if args.use_read_gamma_trace else torch.empty(0, device='cuda')
        physical = FlyPhysicalState(h, ring, ge, gi, b, x, u, baseline, h_mean, dan_gate, gamma_z1, gamma_z2)

        names = [
            'output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias',
            'log_threshold', 'log_tau_m', 'log_beta', 'log_tau_a',
            'log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i',
            'topographic_writer.gate_linear.weight', 'topographic_writer.gate_linear.bias'
        ]
        if args.learn_stp:
            names.extend(STP_PARAMETER_NAMES)
        if args.use_read_gamma_trace:
            names.append('logit_read_gamma')
        if args.use_latent_predictor and model.latent_predictor is not None:
            for p_name, _ in model.latent_predictor.named_parameters():
                names.append(f'latent_predictor.{p_name}')
        if args.use_graph_observer and getattr(model, 'graph_observer', None) is not None:
            for p_name, _ in model.graph_observer.named_parameters():
                names.append(f'graph_observer.{p_name}')

        if args.disable_alif:
            disable_alif(model)
        learner = FlyBPTTLearner(model, physical, adam_names=names,
            lr=args.lr, lr_synapse=args.lr_synapse, lr_sensory=args.lr_sensory,
            plasticity_optimizer=args.plasticity_optimizer, lr_decoder=args.lr_decoder,
            settle_ticks=args.settle_ticks, writer_baseline_clock=args.writer_baseline_clock,
            learn_stp=bool(args.learn_stp), lambda_jepa=args.lambda_jepa,
            use_ctm_loss=bool(args.ctm_loss),
            use_checkpointing=bool(args.checkpointing),
            lambda_mcr2=float(args.lambda_mcr2 or 0.0),
            eps_mcr2=float(args.eps_mcr2 or 0.5),
            adaptive_admission=bool(args.adaptive_admission),
            min_settle_ticks=args.min_settle_ticks,
            flux_baseline=args.flux_baseline)
        learner.previous_token = int(train[0])

        initial_stp = {name: getattr(model, name).detach().clone() for name in STP_PARAMETER_NAMES}
        origin = {
            'initialization': 'born from scratch with GPT-2 embedding/unigram prior and LeJEPA Latent Predictor Form B',
            'train_tokens': 0, 'events': 0, 'updates': 0,
        }
        trained = 0
        best = math.inf
        cursor = val_cursor = 0
        recent_a = []
        recent_scores = []
        target = args.additional_tokens
        budget_start = 0
    else:
        print('Loading archived weights and continuing physical state...', flush=True)
        saved = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        fmt, previous_config = saved['format'], saved['config']
        old = saved['learner']
        previous_centering = bool(old.get('read_centering', previous_config.get('read_centering', False)))
        previous_dan_lr = float(old.get('dan_plastic_lr', previous_config.get('dan_plastic_lr', 0.0)))
        resolve_learning_modes(args, previous_config, old)
        previous_gamma = bool(old.get('use_read_gamma_trace', previous_config.get('use_read_gamma_trace', False)))
        previous_predictor = bool(old.get('use_latent_predictor', previous_config.get('use_latent_predictor', False)))
        previous_observer = bool(old.get('use_graph_observer', previous_config.get('use_graph_observer', False)))
        mode_changed = (args.read_centering != previous_centering or
                        args.dan_plastic_lr != previous_dan_lr or
                        args.use_read_gamma_trace != previous_gamma or
                        args.use_latent_predictor != previous_predictor or
                        args.use_graph_observer != previous_observer or
                        args.detach_reset != bool(previous_config.get('detach_reset', False)) or
                        args.surrogate_mode != previous_config.get('surrogate_mode', 'absolute') or
                        args.transmission_mode != previous_config.get('transmission_mode', 'atomic'))
        if args.reference_train is None:
            args.reference_train = Path(previous_config.get('reference_train', 'data/ib_owt_gpt2/train.npy'))
        if fmt not in ('fly-online-v2', 'fly-bptt-v1'):
            raise ValueError('Resume requires a complete lifecycle checkpoint')
        model = FlyReservoirLM(args.graph, vocab_size=50257, d_model=previous_config['d_model'],
            injection='topographic', read_surface='output', synapse_model='coba',
            use_alif=True, use_stp=True, decoder_bias=previous_config['decoder_bias'],
            read_centering=args.read_centering,
            use_read_gamma_trace=args.use_read_gamma_trace,
            init_read_gamma=args.init_read_gamma,
            use_latent_predictor=args.use_latent_predictor,
            use_graph_observer=args.use_graph_observer,
            lambda_obs=args.lambda_jepa,
            lambda_sigreg=args.lambda_sigreg,
            max_horizon=args.max_horizon,
            dagger_beta=args.dagger_beta,
            obs_init_gamma=getattr(args, 'obs_init_gamma', 0.0),
            detach_reset=args.detach_reset, surrogate_mode=args.surrogate_mode,
            transmission_mode=args.transmission_mode).cuda()
        model.dan_plastic_lr = args.dan_plastic_lr
        weights = {key: value for key, value in saved['model'].items()
                   if key not in ('edge_weight_e', 'edge_weight_i')}
        if args.reinit_read_gamma:
            weights.pop('logit_read_gamma', None)
            print('Re-initializing logit_read_gamma with full-causal anatomical distribution [1.0, 115.0] ticks.', flush=True)
        with torch.no_grad():
            for name in ('edge_weight_e', 'edge_weight_i'):
                getattr(model, name).copy_(saved['model'][name])
            loaded = model.load_state_dict(weights, strict=False)
            missing = [k for k in loaded.missing_keys if k != 'logit_read_gamma' and not k.startswith('latent_predictor.') and not k.startswith('graph_observer.')]
            unexpected = [k for k in loaded.unexpected_keys if not k.startswith('latent_predictor.') and not k.startswith('graph_observer.')]
            if unexpected or missing:
                raise ValueError(f'Weight migration mismatch: unexpected={unexpected}, missing={missing}')
        previous_settle_ticks = old.get('settle_ticks', 0)
        previous_writer_clock = old.get('writer_baseline_clock', 'physical')
        if args.writer_baseline_clock is None:
            args.writer_baseline_clock = previous_writer_clock
        writer_clock_changed = args.writer_baseline_clock != previous_writer_clock
        if args.settle_ticks is None:
            args.settle_ticks = previous_settle_ticks
        if fmt == 'fly-online-v2':
            if old['counters']['pending'] or old['counters']['edge_pending']:
                raise ValueError('Migration requires the archived completed update boundary')
            physical = FlyPhysicalState.from_online(old, 'cuda')
            names = list(old['grads'])
        else:
            physical = FlyPhysicalState(**{key: tuple(t.to('cuda') for t in value) if key == 'ring'
                else value.to('cuda') for key, value in old['physical'].items()})
            if physical.h_mean.numel() != physical.h.numel():
                physical.h_mean = torch.zeros_like(physical.h)
            if args.use_read_gamma_trace:
                num_read = model.n_read if model.read_surface != 'all' else model.n_neurons
                if physical.gamma_z1.numel() == 0 or args.reinit_read_gamma:
                    physical.gamma_z1 = torch.zeros(1, num_read, device='cuda')
                    physical.gamma_z2 = torch.zeros(1, num_read, device='cuda')
            if not hasattr(physical, 'observer_prior') or physical.observer_prior is None:
                physical.observer_prior = torch.empty(0, device='cuda')
            if not hasattr(physical, 'observer_history') or physical.observer_history is None:
                physical.observer_history = torch.empty(0, device='cuda')
            names = [n for n in old['adam_names']
                     if (args.use_graph_observer or not n.startswith('graph_observer.'))
                     and (args.use_latent_predictor or not n.startswith('latent_predictor.'))]
        previous_learn_stp = all(name in names for name in STP_PARAMETER_NAMES)
        args.learn_stp = bool(args.learn_stp or previous_learn_stp)
        stp_newly_trainable = [name for name in STP_PARAMETER_NAMES
                                if args.learn_stp and name not in names]
        stp_changed = bool(stp_newly_trainable)
        newly_trainable = list(stp_newly_trainable)
        if (args.reinit_read_gamma or (args.use_read_gamma_trace and 'logit_read_gamma' not in names)):
            newly_trainable.append('logit_read_gamma')
        if args.use_latent_predictor and model.latent_predictor is not None:
            for p_name, _ in model.latent_predictor.named_parameters():
                full_name = f'latent_predictor.{p_name}'
                if full_name not in names:
                    newly_trainable.append(full_name)
        if args.use_graph_observer and getattr(model, 'graph_observer', None) is not None:
            for p_name, _ in model.graph_observer.named_parameters():
                full_name = f'graph_observer.{p_name}'
                if full_name not in names:
                    newly_trainable.append(full_name)
        if args.disable_alif:
            disable_alif(model)
        learner = FlyBPTTLearner(model, physical, adam_names=names,
            lr=args.lr, lr_synapse=args.lr_synapse, lr_sensory=args.lr_sensory,
            plasticity_optimizer=args.plasticity_optimizer, lr_decoder=args.lr_decoder,
            settle_ticks=args.settle_ticks, writer_baseline_clock=args.writer_baseline_clock,
            learn_stp=args.learn_stp, lambda_jepa=args.lambda_jepa,
            use_ctm_loss=bool(args.ctm_loss),
            use_checkpointing=bool(args.checkpointing),
            lambda_mcr2=float(args.lambda_mcr2 or 0.0),
            eps_mcr2=float(args.eps_mcr2 or 0.5),
            adaptive_admission=bool(args.adaptive_admission),
            min_settle_ticks=args.min_settle_ticks,
            flux_baseline=args.flux_baseline)
        learner.load_edge_signs(old)
        learner.load_adam_state(old['optimizer'], newly_trainable=newly_trainable)
        initial_stp = {name: getattr(model, name).detach().clone() for name in STP_PARAMETER_NAMES}
        if fmt == 'fly-online-v2':
            learner.events = old['counters']['events']
            learner.previous_token = old['counters']['previous_token']
            learner.ema = old['counters']['ema']
            origin = {'checkpoint': str(args.resume), 'train_tokens': saved['train_cursor'],
                      'events': learner.events, 'old_optimizer_updates': old['counters']['updates'],
                      'migration': 'physical/weights/Adam moments retained; local eligibility and pending replay study archived; BPTT credit starts at this boundary'}
            trained = 0
            best = math.inf  # Different learner/budget: do not inherit a historical best headline.
        else:
            old_kind = old.get('plasticity_optimizer_kind', 'sgd')
            if old_kind == args.plasticity_optimizer:
                learner.sgd.load_state_dict(old['sgd'])
            else:
                print(f'Plasticity optimizer migration: {old_kind} -> {args.plasticity_optimizer}; '
                      'new plasticity moments start at zero; existing Adam moments retained.', flush=True)
            for group, rate in zip(learner.sgd.param_groups, (args.lr_sensory, args.lr_synapse)):
                group['lr'] = rate
            for key in ('events', 'updates', 'previous_token', 'ema'):
                setattr(learner, key, old[key])
            origin = saved['origin']
            trained = saved['bptt_train_tokens']
            best = saved['best_live_nll']
            if old_kind != args.plasticity_optimizer:
                # A new optimizer arm has its own live-validation best; retain the
                # historical score as provenance rather than a new-arm headline.
                origin = {**origin, 'optimizer_migration_checkpoint': str(args.resume),
                          'optimizer_migration_train_cursor': saved['train_cursor'],
                          'prior_best_live_nll': best,
                          'plasticity_optimizer_migration': f'{old_kind}->{args.plasticity_optimizer}'}
                best = math.inf
        learner.physical_ticks = old.get('physical_ticks', learner.events)
        if args.dan_plastic_lr > 0 and old.get('dan_rule') != 'delayed-pulse-ema-terminal-coactivity-v1':
            mode_changed = True
            origin = {**origin, 'dan_rule_migration': {
                'from': old.get('dan_rule', 'legacy-terminal-membrane-or-disabled'),
                'to': 'delayed-pulse-ema-terminal-coactivity-v1',
                'new_gate_history': 'starts at zero; existing physical states and moments retained'}}
        if mode_changed:
            archive = args.output/'segments'/f'learning_mode_before_{trained}'
            if not archive.resolve().is_relative_to(args.output.resolve()):
                raise ValueError('Learning-mode archive must remain within the output directory')
            archive.mkdir(parents=True, exist_ok=True)
            atomic_json(archive/'config.json', previous_config)
            # A new output arm leaves its source assets untouched. Continuing in
            # place moves its best within the SAME checkpoint volume, without a
            # second GB-sized copy or an old-best/new-config ambiguity.
            if args.output.resolve() == Path(previous_config['output']).resolve():
                storage = Path(args.checkpoint_dir or previous_config.get('checkpoint_dir') or args.output)
                old_best = storage/'best.pt'
                if old_best.exists():
                    destination = storage/'segments'/f'learning_mode_before_{trained}'/'best.pt'
                    if not destination.resolve().is_relative_to(storage.resolve()):
                        raise ValueError('Best checkpoint archive must remain within its storage')
                    if destination.exists():
                        raise FileExistsError(f'Learning-mode archive already has {destination}')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    old_best.replace(destination)
            origin = {**origin, 'learning_mode_migration': {
                'from_read_centering': previous_centering, 'to_read_centering': args.read_centering,
                'from_dan_plastic_lr': previous_dan_lr, 'to_dan_plastic_lr': args.dan_plastic_lr,
                'from_detach_reset': bool(previous_config.get('detach_reset', False)),
                'to_detach_reset': args.detach_reset,
                'from_surrogate_mode': previous_config.get('surrogate_mode', 'absolute'),
                'to_surrogate_mode': args.surrogate_mode,
                'from_transmission_mode': previous_config.get('transmission_mode', 'atomic'),
                'to_transmission_mode': args.transmission_mode,
                'train_cursor': saved['train_cursor'],
                'prior_best_live_nll': best if math.isfinite(best) else None}}
            best = math.inf
        if stp_changed:
            archive = args.output / 'segments' / f'stp_frozen_before_{trained}'
            archive.mkdir(parents=True, exist_ok=True)
            if not (archive/'config.json').exists():
                atomic_json(archive/'config.json', previous_config)
            old_best = args.output/'best.pt'
            if old_best.exists():
                destination = archive/'best.pt'
                if destination.exists():
                    raise FileExistsError(f'STP archive already has {destination}')
                old_best.replace(destination)
            origin = {**origin, 'stp_learning_migration': {
                'bptt_train_tokens': trained, 'train_cursor': saved['train_cursor'],
                'prior_best_live_nll': best if math.isfinite(best) else None,
                'newly_trainable': newly_trainable,
                'initial_parameters': {name: value.cpu().tolist() for name, value in initial_stp.items()},
                'state_and_optimizer': 'all physical values, weights, existing Adam moments and cursors retained; only new STP moments start at zero'}}
            best = math.inf
            print(f'STP joint learning activated: {newly_trainable}; existing Adam history retained.', flush=True)
        if args.settle_ticks != previous_settle_ticks:
            # Preserve the old phase's best asset without duplicating last.pt or
            # optimizer tensors. The next phase has a separate validation best.
            archive = args.output / 'segments' / f'timing_before_{trained}_settle{previous_settle_ticks}'
            root = args.output.resolve()
            if not archive.resolve().is_relative_to(root):
                raise ValueError('Timing archive must remain within the output directory')
            archive.mkdir(parents=True, exist_ok=True)
            if not (archive/'config.json').exists():
                atomic_json(archive/'config.json', previous_config)
            prior_progress = args.output/'progress.json'
            if prior_progress.exists() and not (archive/'progress.json').exists():
                atomic_json(archive/'progress.json', json.loads(prior_progress.read_text(encoding='utf-8')))
            old_best = args.output/'best.pt'
            if old_best.exists():
                destination = archive/'best.pt'
                if destination.exists():
                    raise FileExistsError(f'Timing archive already has {destination}')
                old_best.replace(destination)
            origin = {**origin, 'timing_migration': {
                'from_settle_ticks': previous_settle_ticks, 'to_settle_ticks': args.settle_ticks,
                'bptt_train_tokens': trained, 'train_cursor': saved['train_cursor'],
                'input_events': learner.events, 'physical_ticks': learner.physical_ticks,
                'prior_best_live_nll': best if math.isfinite(best) else None,
                'state_and_optimizer': 'all physical values, weights and Adam moments retained'}}
            best = math.inf
        if writer_clock_changed:
            archive = args.output / 'segments' / f'writer_clock_before_{trained}_{previous_writer_clock}'
            if not archive.resolve().is_relative_to(args.output.resolve()):
                raise ValueError('Writer clock archive must remain within the output directory')
            archive.mkdir(parents=True, exist_ok=True)
            for filename, payload in (('config.json', previous_config),
                    ('progress.json', json.loads((args.output/'progress.json').read_text(encoding='utf-8')))):
                if not (archive/filename).exists():
                    atomic_json(archive/filename, payload)
            old_best = args.output/'best.pt'
            if old_best.exists():
                destination = archive/'best.pt'
                if destination.exists():
                    raise FileExistsError(f'Writer clock archive already has {destination}')
                old_best.replace(destination)
            origin = {**origin, 'writer_clock_migration': {
                'from': previous_writer_clock, 'to': args.writer_baseline_clock,
                'bptt_train_tokens': trained, 'train_cursor': saved['train_cursor'],
                'input_events': learner.events, 'physical_ticks': learner.physical_ticks,
                'prior_best_live_nll': best if math.isfinite(best) else None,
                'state_and_optimizer': 'all physical values, weights, moments and cursors retained'}}
            best = math.inf
        learner.latent_window.copy_(old['latent_window'])
        cursor, val_cursor = saved['train_cursor'], saved['val_cursor']
        recent_a = saved.get('recent_a', [])
        recent_scores = saved.get('recent_scores', [])
    reference_train = np.load(args.reference_train, mmap_mode='r')
    reference_counts = np.ones(50257, dtype=np.float64)
    for left in range(0, len(reference_train), 1_000_000):
        reference_counts += np.bincount(
            reference_train[left:left+1_000_000].astype(np.int64), minlength=50257)
    reference_surprisal = -np.log(reference_counts/reference_counts.sum())
    reference_metadata = {
        'train': str(args.reference_train.resolve()), 'tokens': len(reference_train),
        'vocab_size': 50257, 'smoothing': 'add one', 'frozen': True,
        'validation_labels_used_for_prior': False,
        'counts_sha256': hashlib.sha256(reference_counts.tobytes()).hexdigest()}
    if is_scratch_start:
        first_pass_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
        phase_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
    else:
        if 'first_pass_meter' in saved:
            first_pass_meter = FirstPassPredictionMeter.from_state_dict(
                saved['first_pass_meter'], reference_metadata)
        else:
            # Historical sampled window logs cannot reconstruct complete totals.
            first_pass_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
        first_pass_meter.verify_cursor(trained, cursor)
        if not writer_clock_changed and not stp_changed and not mode_changed and 'phase_first_pass_meter' in saved:
            phase_meter = FirstPassPredictionMeter.from_state_dict(
                saved['phase_first_pass_meter'], reference_metadata)
        else:
            phase_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
        phase_meter.verify_cursor(trained, cursor)
        torch.set_rng_state(saved['rng_cpu'])
        torch.cuda.set_rng_state(saved['rng_cuda'])
        target = (trained + args.additional_tokens if fmt == 'fly-online-v2'
                  or args.extend_budget or trained >= saved['target_bptt_tokens'] else saved['target_bptt_tokens'])
        budget_start = trained if args.extend_budget else previous_config.get('budget_start_bptt_tokens', trained)
    del reference_train, reference_counts
    if args.data.resolve() != Path(previous_config['data']).resolve():
        old_train = np.load(Path(previous_config['data'])/'train.npy', mmap_mode='r')
        old_val = np.load(Path(previous_config['data'])/'validation.npy', mmap_mode='r')
        if len(train) < len(old_train) or len(val) < len(old_val):
            raise ValueError('Expanded stream is shorter than the continuing stream')
        for original, expanded in ((old_train, train), (old_val, val)):
            for left in range(0, len(original), 1048576):
                right = min(left+1048576, len(original))
                if not np.array_equal(original[left:right], expanded[left:right]):
                    raise ValueError('Expanded data changed the continuing stream prefix')
        del old_train, old_val
        print('Expanded train/validation prefixes verified; saved cursors retained.', flush=True)
    if cursor + (target - trained) + 1 > len(train):
        raise ValueError('Fresh OWT stream exhausted')
    evaluations_needed = target//args.validate_every_tokens - trained//args.validate_every_tokens
    if target % args.validate_every_tokens:
        evaluations_needed += 1
    if val_cursor + evaluations_needed*args.eval_tokens > len(val):
        raise ValueError('Fresh validation stream is insufficient for the full registered budget')
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(d_model=previous_config['d_model'], decoder_bias=previous_config['decoder_bias'],
        lr=args.lr, lr_synapse=args.lr_synapse, lr_sensory=args.lr_sensory,
        plasticity_optimizer=args.plasticity_optimizer,
        n_neurons=model.n_neurons, n_synapses=sum(p.numel() for p in learner.edges),
        trainable_parameters=sum(p.numel() for p in learner.trainable),
        credit=f'{args.window}-token whole-network ATan-surrogate BPTT; all within-event physical ticks included',
        physical_ticks_per_input=1+args.settle_ticks,
        event_order='one sensory input pulse, quiet propagation, output read, next target revelation',
        physical_tick_unit='unchanged graph-delay tick; no physiological milliseconds inferred from OWT',
        writer_baseline=('updates only at observed input events; held during quiet propagation'
                         if args.writer_baseline_clock == 'input'
                         else 'archived pulse update and zero-source decay at quiet ticks'),
        fixed_parameters=[name for name, parameter in model.named_parameters() if not parameter.requires_grad],
        stp_learning={'enabled': learner.learn_stp, 'parameter_names': list(STP_PARAMETER_NAMES),
                      'trainable_count': sum(getattr(model, name).numel() for name in STP_PARAMETER_NAMES)
                                         if learner.learn_stp else 0,
                      'weight_decay': 0.0, 'lr': args.lr},
        physical_state='h, delayed STP pulses, ge, gi, ALIF b, STP x/u, sensory adaptation; never reset',
        evaluation='same active learner; pre-update window scores; fresh B and A revisit with actual bridges',
        optimizer_cadence=f'one joint update per {args.window} observed input events, including active evaluation',
        origin=origin, pid=os.getpid(), budget_start_bptt_tokens=budget_start,
        fresh_training_budget=target-budget_start,
        budget_start_train_cursor=cursor if args.extend_budget else previous_config.get('budget_start_train_cursor', cursor),
        training_data_policy='monotone fresh train cursor, no wrap/shuffle/epoch replay; evaluation replay excluded from training budget')
    config['first_pass_measurement'] = {
        'reference': reference_metadata,
        'start_bptt_train_tokens': first_pass_meter.start_train_targets,
        'scope': 'every fresh training target scored before its update, including actual stream bridges; active B and A replay excluded',
        'recent_optimizer_windows': first_pass_meter.recent.maxlen,
        'ema_scope': 'historical EMA mixes training, fresh B and A replay traffic'}
    manifest = args.data / 'manifest.json'
    config['dataset_manifest'] = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
    config['train_shape'], config['validation_shape'] = list(train.shape), list(val.shape)
    config['tokenizer'] = 'GPT-2, 50257 vocabulary entries'
    if not is_scratch_start:
        del weights, old, saved
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(args.output / 'config.json', config)
    if is_scratch_start:
        atomic_save(args.output / 'birth.pt', {
            'format': 'fly-bptt-v1', 'model': dict(model.named_parameters()),
            'learner': learner.state_dict(), 'config': config, 'origin': origin,
            'train_cursor': 0, 'val_cursor': 0, 'bptt_train_tokens': 0,
            'target_bptt_tokens': target, 'best_live_nll': math.inf,
            'recent_a': [], 'recent_scores': [],
            'first_pass_meter': first_pass_meter.state_dict(),
            'phase_first_pass_meter': phase_meter.state_dict(),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()})
    (args.output / 'process.pid').write_text(str(os.getpid()))
    if not args.no_graph and not args.use_latent_predictor and not args.use_graph_observer:
        print('Capturing 32-event forward/backward; no physical reset...', flush=True)
        learner.runner = FlyBPTTGraph(learner, args.window)
    else:
        print('Running eager 32-event forward/backward; no physical reset...', flush=True)
    torch.cuda.synchronize()
    # Keep the initialization/warm-up peak in the reported budget as well.
    setup_peak_mib = torch.cuda.max_memory_allocated() / 2**20
    if max(setup_peak_mib, torch.cuda.memory_reserved()/2**20) > args.vram_limit_mib:
        raise MemoryError('BPTT setup exceeded the registered dedicated CUDA budget')
    start = time.perf_counter()
    start_trained = trained
    next_eval = ((trained // args.validate_every_tokens) + 1) * args.validate_every_tokens
    last_grad = {}

    def log(row, file='metrics.jsonl'):
        with (args.output / file).open('a', encoding='utf-8') as out:
            out.write(json.dumps(row, allow_nan=False) + '\n')

    def health():
        row = {'field_energy': float(learner.state.h.square().mean()),
            'firing_rate': float((learner.state.ring[0] > 0).float().mean()),
            'read_norm_mean': float(model.read_norm.weight.detach().mean()),
            'vram_allocated_mib': torch.cuda.memory_allocated() / 2**20,
            'vram_reserved_mib': torch.cuda.memory_reserved() / 2**20,
            'vram_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
            'setup_peak_mib': setup_peak_mib,
            'events': learner.events, 'physical_ticks': learner.physical_ticks,
            'physical_ticks_per_input': 1+learner.settle_ticks,
            'settle_ticks': learner.settle_ticks,
            'writer_baseline_clock': learner.writer_baseline_clock,
            'bptt_optimizer_updates': learner.updates,
            'stp_learning_enabled': learner.learn_stp,
            'use_checkpointing': getattr(learner, 'use_checkpointing', False),
            'sensory_resource_mean': float(learner.state.x[:, model.topographic_writer.injection_index].mean())}
        if getattr(model, 'latent_predictor', None) is not None:
            row['predictor_mix_alpha'] = float(model.latent_predictor.mix_alpha.item())
        if getattr(model, 'graph_observer', None) is not None:
            row['obs_gamma'] = float(model.graph_observer.gamma.item())
            row['obs_raw_gamma'] = float(model.graph_observer.raw_gamma.item())
            if learner.last_jepa_metrics:
                for k, v in learner.last_jepa_metrics.items():
                    row[f'obs_metric_{k}'] = float(v)
        if getattr(learner, 'adaptive_admission', False):
            row['adaptive_admission'] = True
            for k_ad, v_ad in getattr(learner, 'last_adaptive_metrics', {}).items():
                row[k_ad] = float(v_ad)
        if getattr(learner, 'use_ctm_loss', False):
            row['use_ctm_loss'] = True
            for k_ctm, v_ctm in getattr(learner, 'last_ctm_metrics', {}).items():
                row[k_ctm] = float(v_ctm)
        if getattr(learner, 'lambda_mcr2', 0.0) > 0.0:
            row['lambda_mcr2'] = learner.lambda_mcr2
            for k_mcr, v_mcr in getattr(learner, 'last_mcr2_metrics', {}).items():
                row[k_mcr] = float(v_mcr)
        for name in STP_PARAMETER_NAMES:
            parameter = getattr(model, name).detach()
            row[f'{name}_max_change_since_launch'] = float((parameter-initial_stp[name]).abs().max())
        row['stp_u0_mean'] = float(model.logit_u0.detach().sigmoid().clamp(.05, .95).mean())
        row['stp_tau_fac_mean'] = float(model.log_tau_fac.detach().exp().clamp(5, 1000).mean())
        row['stp_tau_rec_mean'] = float(model.log_tau_rec.detach().exp().clamp(5, 2000).mean())
        return row

    def prediction_accounting():
        # Keep complete historical totals and a distinct current implementation
        # phase, with no retrospective estimates and bounded memory.
        return {**first_pass_meter.summary(),
                **{f'phase_{key}': value for key, value in phase_meter.summary().items()}}

    def progress(status):
        row = {'status': status, 'bptt_train_tokens': trained, 'tokens_streamed': cursor,
            'target_bptt_tokens': target, 'ema_stream_loss': learner.ema,
            'fresh_budget_tokens_processed': trained-budget_start,
            'fresh_training_budget': target-budget_start,
            'lr': args.lr, 'lr_decoder': args.lr if args.lr_decoder is None else args.lr_decoder,
            'lr_synapse': args.lr_synapse, 'lr_sensory': args.lr_sensory,
            'best_live_nll': best if math.isfinite(best) else None,
            'speed_tokens_per_sec': (trained-start_trained) / max(time.perf_counter()-start, 1e-6),
            **health(), **last_grad, **prediction_accounting()}
        atomic_json(args.output / 'progress.json', row)
        return row

    def checkpoint(is_best=False):
        payload = {'format': 'fly-bptt-v1', 'model': dict(model.named_parameters()),
            'learner': learner.state_dict(), 'config': config, 'origin': origin,
            'train_cursor': cursor, 'val_cursor': val_cursor, 'bptt_train_tokens': trained,
            'target_bptt_tokens': target, 'best_live_nll': best,
            'recent_a': recent_a, 'recent_scores': recent_scores,
            'first_pass_meter': first_pass_meter.state_dict(),
            'phase_first_pass_meter': phase_meter.state_dict(),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()}
        atomic_save(args.output / 'last.pt', payload)
        if is_best:
            atomic_save(args.output / 'best.pt', {'format': 'fly-bptt-best-weights-v1',
                'model': dict(model.named_parameters()), 'config': config,
                'bptt_train_tokens': trained, 'best_live_nll': best})

    def observe(values):
        nonlocal last_grad
        scores = []
        for left in range(0, len(values), args.window):
            chunk, last_grad = learner.observe(np.array(values[left:left+args.window], dtype=np.int64))
            scores.extend(chunk)
            if max(torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated()) / 2**20 > args.vram_limit_mib:
                raise MemoryError('CUDA allocation/reservation exceeded the registered 3900 MiB budget')
        return scores

    try:
        while trained < target:
            values = np.array(train[cursor+1:cursor+args.window+1], dtype=np.int64)
            begin = time.perf_counter()
            scores = observe(values)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            cursor += args.window
            trained += args.window
            first_pass_meter.record(scores, reference_surprisal[values])
            first_pass_meter.verify_cursor(trained, cursor)
            phase_meter.record(scores, reference_surprisal[values])
            phase_meter.verify_cursor(trained, cursor)
            recent_a = (recent_a + values.tolist())[-args.replay_tokens:]
            recent_scores = (recent_scores + scores)[-args.replay_tokens:]
            if trained == start_trained + args.window or trained % args.log_every_tokens == 0:
                row = progress('running')
                row.update(window_prequential_nll=float(np.mean(scores)),
                    seconds_per_window=seconds, milliseconds_per_token=1000*seconds/args.window)
                log(row)
                ctm_info = ""
                if 'ctm_k_cert' in last_grad:
                    ctm_info = f" (k_cert={last_grad['ctm_k_cert']:.1f}/{learner.settle_ticks}, t0={last_grad.get('ctm_loss_tick0',0):.2f}, tS={last_grad.get('ctm_loss_tickS',0):.2f})"
                adaptive_info = ""
                if getattr(learner, 'adaptive_admission', False) and 'adaptive_ticks_mean' in last_grad:
                    adaptive_info = f" (admit_ticks={last_grad['adaptive_ticks_mean']:.1f}/{learner.settle_ticks or 14}, min={last_grad.get('adaptive_ticks_min', 0)}, max={last_grad.get('adaptive_ticks_max', 0)})"
                mcr_info = ""
                if getattr(learner, 'lambda_mcr2', 0.0) > 0.0 and getattr(learner, 'last_mcr2_metrics', None):
                    mcr_info = f" [ΔR={learner.last_mcr2_metrics.get('mcr2_delta_R', 0):.2f}]"
                print(f'BPTT {trained}/{target} | NLL {np.mean(scores):.4f}{ctm_info}{adaptive_info}{mcr_info} | {seconds:.3f}s/{args.window} | peak {row["vram_peak_mib"]:.0f} MiB', flush=True)
            if trained == start_trained + args.window:
                checkpoint()
                if args.calibrate:
                    progress('calibrated')
                    return
            if (args.output / 'STOP').exists():
                checkpoint()
                progress('paused')
                return
            if trained >= next_eval or trained == target:
                if val_cursor + args.eval_tokens > len(val):
                    raise ValueError('Fresh validation stream exhausted')
                event_start, updates_start = learner.events, learner.updates
                physical_start = learner.physical_ticks
                fresh = np.array(val[val_cursor:val_cursor+args.eval_tokens], dtype=np.int64)
                B = observe(fresh)
                val_cursor += args.eval_tokens
                A2 = observe(recent_a)
                live = float(np.mean(B))
                is_best = live < best
                best = min(best, live)
                z = learner.latent_window.detach().cpu()
                z = z-z.mean(0)
                eigenvalues = torch.linalg.eigvalsh(z@z.T).clamp_min(0)
                probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
                effective_rank = float((-(probabilities*probabilities.clamp_min(1e-30).log()).sum()).exp())
                report = {'bptt_train_tokens': trained, 'train_cursor': cursor,
                    'event_interval': [event_start, learner.events],
                    'physical_tick_interval': [physical_start, learner.physical_ticks],
                    'update_interval': [updates_start, learner.updates],
                    'fresh_validation_cursor': val_cursor, 'live_prequential_nll': live,
                    'B_curve': B, 'A1_curve': recent_scores, 'A2_replay_curve': A2,
                    'replay_tokens': recent_a, 'actual_intervening_events': args.eval_tokens,
                    'actual_intervening_physical_ticks': args.eval_tokens*(1+learner.settle_ticks),
                    'A2_prequential_nll': float(np.mean(A2)),
                    'matched_revisit_nll_change': float(np.mean(recent_scores[1:])-np.mean(A2[1:])),
                    'bridge_scoring': 'all actual bridges scored; first A token excluded from matched gap',
                    'recovery': recovery_summary(B, block_tokens=16, hold_blocks=2,
                        reference_nll=reference_surprisal[fresh]),
                    'savings': savings_summary(recent_scores, A2, intervening_events=args.eval_tokens,
                        block_tokens=16, hold_blocks=2),
                    'centered_effective_rank': effective_rank,
                    'health_scope': 'field energy, activity and centered representation structure; FTLE not measured in this branch',
                    **health(), **prediction_accounting()}
                log(report, 'lifelong_evaluation.jsonl')
                ag_val = report['recovery'].get('ag')
                ag_str = f" AG={ag_val:.4f}" if ag_val is not None else ""
                print(f'[ACTIVE EVAL {trained}] fresh B NLL={live:.4f}{ag_str}; A revisit={np.mean(A2):.4f}', flush=True)
                checkpoint(is_best)
                progress('running')
                next_eval += args.validate_every_tokens
        progress('completed')
    except BaseException as exc:
        row = progress('failed')
        row['error'] = repr(exc)
        atomic_json(args.output / 'progress.json', row)
        raise


if __name__ == '__main__':
    main()
