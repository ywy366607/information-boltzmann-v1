"""One continuing, single-tick current-token OWT learner.

Prediction precedes sensory assimilation inside FlyPipelineLearner. Active B
and A revisit use that same learner. Calibration consumes the first real
window; resumption continues its cursor rather than repeating calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.runtime.prequential_meter import FirstPassPredictionMeter
from information_boltzmann.runtime.lifelong_evaluation import recovery_summary, savings_summary

FORMAT = 'fly-pipeline-v1'
WINDOW = 32
TRAIN_UPDATES = 3000
EVAL_EVERY = 500
B_TOKENS = 256
A_TOKENS = 128
VOCAB = 50257
SOURCE_FILES = (
    'scripts/ib/train_fly_pipeline_stream.py',
    'information_boltzmann/core/fly_pipeline.py',
    'information_boltzmann/core/fly_reservoir.py',
    'information_boltzmann/core/fly_bptt_learning.py',
    'information_boltzmann/core/triton_synapse.py',
    'information_boltzmann/runtime/prequential_meter.py',
    'information_boltzmann/runtime/lifelong_evaluation.py',
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(piece)
    return result.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def atomic_checkpoint(path, payload):
    """Serialize GPU storages directly; avoid a second full CPU model/moment copy."""
    import torch
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    seen = set()

    def storage_bytes(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(storage_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(storage_bytes(item) for item in value)
        return 0

    estimate = storage_bytes(payload)
    required = estimate + max(16*2**20, estimate//100) + 256*2**20
    if shutil.disk_usage(path.parent).free < required:
        raise OSError(f'Insufficient free storage for atomic {path.name}: need {required} bytes')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        # Only the one temporary file belonging to this save is removed.
        temporary.unlink(missing_ok=True)
        raise


def load_tokens(path):
    """Use the old OWT .npy mmap convention; extended-stream .bin is uint16."""
    path = Path(path)
    if path.suffix == '.npy':
        values = np.load(path, mmap_mode='r')
    elif path.suffix == '.bin':
        if path.stat().st_size % 2:
            raise ValueError('OWT uint16 binary has an incomplete token')
        values = np.memmap(path, dtype=np.uint16, mode='r')
    else:
        raise ValueError('Expected the existing OWT .npy or uint16 .bin format')
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Token stream must be a one-dimensional integer array')
    return values


def split_path(directory, split):
    for suffix in ('.npy', '.bin'):
        candidate = Path(directory) / (split + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Missing real OWT {split} stream in {directory}')


def fixed_reference(path):
    data = load_tokens(path)
    counts = np.ones(VOCAB, dtype=np.float64)
    for offset in range(0, len(data), 1_000_000):
        chunk = np.asarray(data[offset:offset+1_000_000], dtype=np.int64)
        if len(chunk) and (chunk.min() < 0 or chunk.max() >= VOCAB):
            raise ValueError('Reference token outside GPT-2 vocabulary')
        counts += np.bincount(chunk, minlength=VOCAB)
    metadata = {
        'train': str(Path(path).resolve()), 'file_sha256': digest(path),
        'tokens': len(data), 'vocab_size': VOCAB, 'smoothing': 'add one',
        'frozen': True, 'validation_labels_used_for_prior': False,
        'counts_sha256': hashlib.sha256(counts.tobytes()).hexdigest(),
        'scope': 'Offline full training-corpus frequency reference; its data budget is separate',
    }
    return -np.log(counts/counts.sum()), metadata


def initialize_pretrained_head(model, path, surprisal, read_norm_gain):
    """Match the competing route's input table, decoder prior and initial gain.

    Only the shared GPT-2 token table is imported. It initializes two independent
    weights, never a sensory-to-decoder runtime shortcut. The static prior uses
    the declared training-only reference, with its separate offline data budget.
    """
    import torch
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as packed:
        table = packed.get_tensor('wte.weight')
    if table.shape != model.embedding.weight.shape or table.shape != model.decoder.weight.shape:
        raise ValueError('Pretrained token table must match both input and decoder dimensions')
    if not torch.isfinite(table).all():
        raise ValueError('Pretrained token table contains nonfinite values')
    prior = np.asarray(surprisal, dtype=np.float32)
    if model.decoder.bias is None or prior.shape != tuple(model.decoder.bias.shape):
        raise ValueError('Fixed prior must match the independent decoder bias')
    if not np.isfinite(prior).all() or not math.isfinite(read_norm_gain) or read_norm_gain <= 0:
        raise ValueError('Finite fixed prior and positive initial read gain required')
    with torch.no_grad():
        model.embedding.weight.copy_(table)
        model.decoder.weight.copy_(table)
        model.decoder.bias.copy_(torch.from_numpy(-prior))
        model.read_norm.weight.fill_(read_norm_gain)


@dataclass
class StreamLedger:
    """A committed optimizer-window boundary, including a partially completed eval."""
    train_cursor: int = 0
    val_cursor: int = 0
    train_updates: int = 0
    eval_updates: int = 0
    evaluations_completed: int = 0
    phase: str = 'train'
    phase_offset: int = 0
    a_tokens: list = field(default_factory=list)
    a_scores: list = field(default_factory=list)
    a_first_event_interval: list = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)
    best_live_nll: float | None = None

    @property
    def events(self):
        return WINDOW * (self.train_updates + self.eval_updates)

    @property
    def complete(self):
        return self.train_updates == TRAIN_UPDATES and self.phase == 'train' and self.evaluations_completed == 6

    def validate(self, train_start=0, val_start=0):
        if not 0 <= self.train_updates <= TRAIN_UPDATES or self.phase not in ('train', 'B', 'replay'):
            raise ValueError('Invalid registered training counter/phase')
        if self.train_cursor != train_start + WINDOW*self.train_updates:
            raise ValueError('Fresh train cursor diverges from actual training windows')
        if not 0 <= self.evaluations_completed <= 6 or not 0 <= self.phase_offset:
            raise ValueError('Invalid evaluation counter')
        if len(self.a_tokens) != len(self.a_scores) or len(self.a_tokens) != min(A_TOKENS, WINDOW*self.train_updates):
            raise ValueError('First A must retain exactly the initial fresh targets and scores')
        if self.phase == 'train':
            if self.phase_offset or self.evaluation or self.train_updates // EVAL_EVERY != self.evaluations_completed:
                raise ValueError('An evaluation due at this boundary must not be skipped')
            expected_b = self.evaluations_completed * B_TOKENS
            expected_eval_updates = self.evaluations_completed * ((B_TOKENS+A_TOKENS)//WINDOW)
        else:
            if (self.train_updates % EVAL_EVERY or
                    self.train_updates // EVAL_EVERY != self.evaluations_completed+1):
                raise ValueError('Active phase does not match its training boundary')
            limit = B_TOKENS if self.phase == 'B' else A_TOKENS
            if self.phase_offset >= limit or self.phase_offset % WINDOW:
                raise ValueError('Invalid partial evaluation cursor')
            expected_b = self.evaluations_completed*B_TOKENS + (self.phase_offset if self.phase == 'B' else B_TOKENS)
            expected_eval_updates = self.evaluations_completed*12 + self.phase_offset//WINDOW + (0 if self.phase == 'B' else 8)
            if len(self.evaluation.get('B_scores', [])) != (self.phase_offset if self.phase == 'B' else B_TOKENS):
                raise ValueError('Saved B score coverage mismatch')
            if len(self.evaluation.get('A2_scores', [])) != (0 if self.phase == 'B' else self.phase_offset):
                raise ValueError('Saved replay score coverage mismatch')
        if self.val_cursor != val_start + expected_b or self.eval_updates != expected_eval_updates:
            raise ValueError('Fresh B/replay exposure accounting mismatch')
        if self.best_live_nll is not None and not math.isfinite(self.best_live_nll):
            raise ValueError('Invalid saved monitoring best')

    def targets(self, train, validation):
        if self.phase == 'train':
            values = train[self.train_cursor:self.train_cursor+WINDOW]
        elif self.phase == 'B':
            values = validation[self.val_cursor:self.val_cursor+WINDOW]
        else:
            values = self.a_tokens[self.phase_offset:self.phase_offset+WINDOW]
        values = np.array(values, dtype=np.int64)
        if len(values) != WINDOW or values.min() < 0 or values.max() >= VOCAB:
            raise ValueError('Real stream exhausted or token outside vocabulary')
        return values

    def commit(self, values, scores):
        """Call only once after a complete real learner.observe window."""
        values, scores = list(map(int, values)), list(map(float, scores))
        if len(values) != WINDOW or len(scores) != WINDOW or any(not math.isfinite(x) or x < 0 for x in scores):
            raise ValueError('A finite scored optimizer window is required')
        event_start = self.events
        finished_eval = None
        if self.phase == 'train':
            if len(self.a_tokens) < A_TOKENS:
                if not self.a_first_event_interval:
                    self.a_first_event_interval = [event_start, event_start]
                self.a_tokens.extend(values)
                self.a_scores.extend(scores)
                self.a_first_event_interval[1] = event_start+WINDOW
            self.train_cursor += WINDOW
            self.train_updates += 1
            if self.train_updates % EVAL_EVERY == 0:
                self.phase = 'B'
                self.evaluation = {'train_updates': self.train_updates,
                    'B_cursor_start': self.val_cursor, 'B_event_start': self.events,
                    'B_scores': [], 'A2_scores': []}
        else:
            self.eval_updates += 1
            self.phase_offset += WINDOW
            if self.phase == 'B':
                self.val_cursor += WINDOW
                self.evaluation['B_scores'].extend(scores)
                if self.phase_offset == B_TOKENS:
                    self.evaluation['A2_event_start'] = self.events
                    self.phase, self.phase_offset = 'replay', 0
            else:
                expected = self.a_tokens[self.phase_offset-WINDOW:self.phase_offset]
                if values != expected:
                    raise ValueError('Replay targets differ from the registered first A')
                self.evaluation['A2_scores'].extend(scores)
                if self.phase_offset == A_TOKENS:
                    finished_eval = dict(self.evaluation)
                    finished_eval['event_end'] = self.events
                    live = float(np.mean(finished_eval['B_scores']))
                    finished_eval['is_best'] = self.best_live_nll is None or live < self.best_live_nll
                    self.best_live_nll = live if self.best_live_nll is None else min(live, self.best_live_nll)
                    self.evaluations_completed += 1
                    self.phase, self.phase_offset, self.evaluation = 'train', 0, {}
        return finished_eval


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


def review_gate(path, actual):
    review = json.loads(Path(path).read_text(encoding='utf-8'))
    if review.get('status') != 'approved' or review.get('script_sha256') != digest(__file__):
        raise ValueError('Independent implementation approval/source lock required')
    if review.get('source_hashes') != actual:
        raise ValueError('Reviewed pipeline implementation differs from current source')
    return review


def optimizer_layout(learner):
    names = {id(p): name for name, p in learner.model.named_parameters()}
    return {key: [[names[id(p)] for p in group['params']] for group in getattr(learner, key).param_groups]
            for key in ('optimizer', 'sgd')}


def parameter_digest(model):
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value.update(name.encode())
        flat = parameter.detach().reshape(-1)
        for offset in range(0, flat.numel(), 65536):
            value.update(memoryview(flat[offset:offset+65536].cpu().numpy()).cast('B'))
    return value.hexdigest()


def build_rest(model, device):
    import torch
    from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
    zero = torch.zeros(1, model.n_neurons, device=device)
    return FlyPhysicalState(zero.clone(), tuple(zero.clone() for _ in range(4)),
        zero.clone(), zero.clone(), zero.clone(), torch.ones_like(zero),
        model.get_stp_params()[0].expand_as(zero).clone(),
        torch.zeros(1, model.n_injection, device=device), zero.clone(),
        torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, device=device))


def physical_from_saved(physical, model, device):
    import torch
    from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
    required = {item.name for item in fields(FlyPhysicalState)}
    if set(physical) != required or len(physical['ring']) != 4:
        raise ValueError('Complete pipeline physical state required; no fallback initializers')
    shape = (1, model.n_neurons)
    for name in ('h', 'ge', 'gi', 'b', 'x', 'u', 'h_mean'):
        if tuple(physical[name].shape) != shape:
            raise ValueError(f'Physical shape mismatch: {name}')
    if tuple(physical['baseline'].shape) != (1, model.n_injection):
        raise ValueError('Writer baseline shape mismatch')
    if any(tuple(item.shape) != shape for item in physical['ring']):
        raise ValueError('Complete delayed ring shape mismatch')
    if any(physical[name].numel() for name in ('dan_gate', 'gamma_z1', 'gamma_z2')):
        raise ValueError('This registered pipeline has neither DAN nor Gamma state')
    for name, value in physical.items():
        for tensor in value if name == 'ring' else (value,):
            if not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite saved physical state')
    return FlyPhysicalState(**{name: tuple(t.to(device) for t in value) if name == 'ring'
                              else value.to(device) for name, value in physical.items()})


def restore_learner(saved, learner):
    import torch
    if saved.get('format') != FORMAT:
        raise ValueError('Only a full fly-pipeline-v1 continuation is accepted')
    life = saved['learner']
    current = learner.state_dict()
    for key in ('prediction_protocol', 'pipeline_version', 'settle_ticks', 'writer_baseline_clock',
                'learn_stp', 'read_centering', 'dan_plastic_lr', 'plasticity_optimizer_kind', 'adam_names'):
        if key not in life or key not in current or life[key] != current[key]:
            raise ValueError(f'Pipeline continuation changed {key}')
    named = dict(learner.model.named_parameters())
    if set(saved['model']) != set(named) or saved['optimizer_layout'] != optimizer_layout(learner):
        raise ValueError('Parameter/optimizer coverage or ordering changed')
    with torch.no_grad():
        for name, parameter in named.items():
            value = saved['model'][name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f'Parameter shape/dtype changed: {name}')
            destination, origin = parameter.reshape(-1), value.reshape(-1)
            for offset in range(0, parameter.numel(), 65536):
                destination[offset:offset+65536].copy_(origin[offset:offset+65536])
    learner.state = physical_from_saved(life['physical'], learner.model, learner.state.h.device)
    learner.load_edge_signs(life)
    learner.load_adam_state(life['optimizer'])
    learner.sgd.load_state_dict(life['sgd'])
    for optimizer in (learner.optimizer, learner.sgd):
        for parameter, state in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state and state[key].shape != parameter.shape:
                    raise ValueError('Saved optimizer moment shape mismatch')
    for name in ('events', 'updates', 'physical_ticks', 'previous_token', 'ema'):
        setattr(learner, name, life[name])
    if life['latent_window'].shape != learner.latent_window.shape:
        raise ValueError('Saved read-history shape mismatch')
    learner.latent_window.copy_(life['latent_window'])
    learner.model.topographic_writer.a_adapt.copy_(learner.state.baseline)
    torch.set_rng_state(saved['rng_cpu'])
    if learner.state.h.is_cuda:
        if isinstance(saved['rng_cuda'], torch.Tensor):
            torch.cuda.set_rng_state(saved['rng_cuda'])
        else:
            torch.cuda.set_rng_state_all(saved['rng_cuda'])


def memory_gate(limit):
    import torch
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()/2**20
    reserved = torch.cuda.max_memory_reserved()/2**20
    free, total = torch.cuda.mem_get_info()
    if max(allocated, reserved, (total-free)/2**20) > limit:
        raise MemoryError(f'Dedicated CUDA budget exceeded {limit:g} MiB')
    return {'cuda_peak_allocated_mib': allocated, 'cuda_peak_reserved_mib': reserved,
            'cuda_device_used_mib': (total-free)/2**20}


def verify_life(ledger, learner):
    if (learner.events != ledger.events or learner.updates != ledger.events//WINDOW
            or learner.physical_ticks != ledger.events):
        raise ValueError('Actual learning/exposure/single-tick counters diverged')


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')


def verify_score_journal(path, saved_events):
    """Refuse silent rollback/repeated targets when a checkpoint lags its trace."""
    cursor = 0
    if not Path(path).exists():
        if saved_events:
            raise ValueError('Continuation requires its complete existing score journal')
        return
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['event_interval'] != [cursor, cursor+WINDOW]:
                raise ValueError('Score journal is incomplete or duplicates a window')
            cursor += WINDOW
    if cursor != saved_events:
        raise ValueError('Score journal and continuation diverge; no implicit rollback or target replay')


def paired_block_interval(segments, block_tokens, *, samples=5000):
    """Conditional one-trajectory risk interval, retaining weighted partial blocks.

    Segments end at active-evaluation insertions. Resampling never forms a block
    across those gaps; a segment's short last block retains its actual weight.
    This empirical interval assumes chosen blocks adequately capture dependence.
    """
    blocks = [np.asarray(segment[start:start+block_tokens], dtype=np.float64)
              for segment in segments for start in range(0, len(segment), block_tokens)]
    if len(blocks) < 2:
        raise ValueError('Paired block interval requires at least two actual blocks')
    sums = np.asarray([block.sum() for block in blocks])
    counts = np.asarray([len(block) for block in blocks])
    draws = np.random.default_rng(0).integers(len(blocks), size=(samples, len(blocks)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return {'block_tokens': block_tokens, 'actual_blocks': len(blocks),
            'partial_block_tokens': counts[counts != block_tokens].tolist(),
            'model_minus_reference': float(sums.sum()/counts.sum()),
            'paired_percentile_95_interval': np.quantile(means, [.025, .975]).tolist(),
            'bootstrap_samples': samples, 'bootstrap_seed': 0,
            'scope': 'Conditional descriptive interval on this finite evolving trajectory; block dependence assumption'}


def terminal_risk_summary(journal):
    """One registered primary endpoint; final B is a separate direction check."""
    early, late, reference_late, segments, final_b, final_b_reference = [], [], [], [], [], []
    prior_end = None
    with Path(journal).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            scores, fixed = row['scores'], row['fixed_unigram_scores']
            if row['phase'] == 'train':
                if row['train_updates'] <= 1000:
                    early.extend(scores)
                if row['train_updates'] > TRAIN_UPDATES-1000:
                    if row['event_interval'][0] != prior_end:
                        segments.append([])
                    segments[-1].extend(np.subtract(scores, fixed).tolist())
                    late.extend(scores)
                    reference_late.extend(fixed)
                    prior_end = row['event_interval'][1]
            elif row['phase'] == 'B' and row['train_updates'] == TRAIN_UPDATES:
                final_b.extend(scores)
                final_b_reference.extend(fixed)
    if len(early) != 1000*WINDOW or len(late) != 1000*WINDOW or len(final_b) != B_TOKENS:
        raise ValueError('Registered terminal endpoint coverage incomplete')
    intervals = [paired_block_interval(segments, size) for size in (256, 128, 512)]
    supported = all(item['paired_percentile_95_interval'][1] < 0 for item in intervals)
    return {'primary_endpoint': 'Last1000 fresh-training optimizer updates only',
            'primary_tokens': len(late), 'primary_model_nll': float(np.mean(late)),
            'primary_fixed_reference_nll': float(np.mean(reference_late)),
            'primary_model_minus_reference': float(np.mean(np.subtract(late, reference_late))),
            'primary_256_token_interval': intervals[0],
            'sensitivity_128_512': intervals[1:],
            'budget_risk_advantage_supported': supported,
            'early1000_model_nll': float(np.mean(early)),
            'last1000_model_nll': float(np.mean(late)),
            'curve_scope': 'Different fresh stream segments; full journal retained; no strict monotonicity or convergence inferred',
            'final_B_direction_check': {'tokens': len(final_b), 'model_nll': float(np.mean(final_b)),
                'fixed_reference_nll': float(np.mean(final_b_reference)),
                'model_minus_reference': float(np.mean(np.subtract(final_b, final_b_reference))),
                'scores': final_b, 'fixed_reference_scores': final_b_reference,
                'scope': 'One256-token block, direction check only, no bootstrap significance claim'},
            'attribution': 'Complete candidate risk only; bias adaptation can contribute; motor use and timing benefit require separate evidence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2_31m'))
    parser.add_argument('--reference-train', type=Path, default=Path('data/ib_owt_gpt2/train.npy'))
    parser.add_argument('--pretrained-embedding', type=Path, default=Path('data/gpt2_model.safetensors'))
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    parser.add_argument('--train-offset', type=int, default=0)
    parser.add_argument('--val-offset', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4, choices=(2e-4,))
    parser.add_argument('--vram-limit-mib', type=float, default=3900., choices=(3900.,))
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--no-graph', action='store_true')
    args = parser.parse_args()
    if min(args.train_offset, args.val_offset) < 0:
        parser.error('Fresh offsets must be nonnegative')
    if Path.cwd().resolve() != ROOT:
        parser.error('Run this registered entry from the repository root so relative anatomy paths agree')
    if args.graph.resolve() != (ROOT/'data/malecns_v1/fly_reservoir_coba.npz').resolve():
        parser.error('This registered arm fixes the MaleCNS graph and its default sensory partitions')
    import torch
    hashes = source_hashes()
    review = review_gate(args.approval, hashes)
    if not torch.cuda.is_available():
        parser.error('CUDA required for this registered production training')
    train_path, val_path = split_path(args.data, 'train'), split_path(args.data, 'validation')
    reference_path = args.reference_train
    if reference_path.resolve() != (ROOT/'data/ib_owt_gpt2/train.npy').resolve():
        parser.error('This matched arm fixes the declared original training-only frequency reference')
    train, validation = load_tokens(train_path), load_tokens(val_path)
    if args.train_offset+TRAIN_UPDATES*WINDOW > len(train) or args.val_offset+6*B_TOKENS > len(validation):
        raise ValueError('Real OWT streams insufficient for the entire fixed budget')
    surprisal, reference = fixed_reference(reference_path)
    paths = {'train': train_path, 'validation': val_path, 'reference_train': reference_path,
             'pretrained_embedding': args.pretrained_embedding, 'graph': args.graph,
             'sensory_partitions': args.graph.parent/'sensory_partitions.npz'}
    for name in ('manifest.json', 'tokenizer.json'):
        if (args.data/name).exists():
            paths[name] = args.data/name
    provenance = {name: {'path': str(path.resolve()), 'sha256': digest(path)} for name, path in paths.items()}
    immutable = {'seed': 0, 'd_model': 768, 'read_centering': False, 'decoder_bias': True,
        'learn_stp': False, 'embedding': 'fixed GPT-2 token table; matched competing-route initialization',
        'decoder_initialization': 'independent GPT-2 token table plus declared static log-frequency prior',
        'read_norm_initial_gain': .1, 'lr': args.lr,
        'plasticity_optimizer': 'adamw', 'max_grad_norm': 1.0, 'window': WINDOW,
        'train_updates_target': TRAIN_UPDATES, 'eval_every_train_updates': EVAL_EVERY,
        'freshB_tokens': B_TOKENS, 'A_replay_tokens': A_TOKENS, 'train_offset': args.train_offset,
        'val_offset': args.val_offset, 'Gamma': False, 'DAN': False, 'settle_ticks': 0,
        'use_latent_predictor': False,
        'writer_baseline_clock': 'input', 'use_cuda_graph': not args.no_graph,
        'primary_endpoint': 'last1000freshupdates', 'risk_blocks': [256, 128, 512],
        'risk_partial_blocks': 'Retain all32000tokens; actual-token-weighted partial-block resampling',
        'risk_bootstrap_samples': 5000, 'risk_bootstrap_seed': 0,
        'source_hashes': hashes, 'data_provenance': provenance, 'fixed_reference': reference}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume is not None and args.resume.resolve() != (args.output/'last.pt').resolve():
        raise ValueError('Continue this output directory through last.pt; best weights cannot resume')
    if args.resume is None and any((args.output/name).exists() for name in ('last.pt', 'config.json', 'scores.jsonl')):
        raise FileExistsError('Fresh training requires an unused output directory')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    from information_boltzmann.core.fly_reservoir import FlyReservoirLM
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTGraph
    from information_boltzmann.core.fly_pipeline import FlyPipelineLearner
    model = FlyReservoirLM(args.graph, vocab_size=VOCAB, d_model=768, injection='topographic',
        read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
        decoder_bias=True, read_centering=False, use_read_gamma_trace=False,
        use_latent_predictor=False).cuda()
    model.dan_plastic_lr = 0.
    if args.resume is None:
        initialize_pretrained_head(model, args.pretrained_embedding, surprisal, .1)
    learner = FlyPipelineLearner(model, build_rest(model, 'cuda'), lr=args.lr,
        lr_synapse=args.lr, lr_sensory=args.lr, plasticity_optimizer='adamw',
        max_grad_norm=1., settle_ticks=0, writer_baseline_clock='input', learn_stp=False)
    if torch.isin(model.read_indices, model.injection_index).any():
        raise ValueError('Sensory/motor anatomical surfaces overlap')
    config = dict(immutable)
    config.update(trainable_parameters=sum(p.numel() for p in learner.trainable),
        trainable_names=[name for name,p in model.named_parameters() if p.requires_grad],
        fixed_names=[name for name,p in model.named_parameters() if not p.requires_grad],
        optimizer_layout=optimizer_layout(learner),
        initial_parameter_sha256=parameter_digest(model),
        first_target_policy='No warm-start token consumed: first target predicted before its own assimilation',
        propagation='All delayed recurrent pathways persist; numerical acceptance covers total delays 1..14 ticks',
        credit='ATan-surrogate BPTT32; detach only at optimizer boundaries; cross-boundary physical history survives',
        comparison='Matched GPT-2 table, unigram prior and read gain; distinct prediction deadline and no token-conditioned predictor',
        scoring='All targets, including actual bridges, are scored pre-update; B and replay stay active',
        main_endpoint='Fixed3000updates; last1000freshupdates only primary risk; finalB direction check; best monitor only',
        budget={'new_train_tokens': 96000, 'freshB': 1536, 'A_replay': 768,
                'total_events': 98304, 'actual_updates': 3072, 'physical_ticks': 98304},
        approval_sha256=digest(args.approval))
    ledger = StreamLedger(args.train_offset, args.val_offset)
    meter = FirstPassPredictionMeter(0, args.train_offset, reference, recent_windows=128)
    execution = {'capture_sessions': 0, 'setup_window_forwards': 0, 'setup_window_backwards': 0,
                 'live_window_forwards': 0, 'live_window_backwards': 0}
    resume_origin = None
    if args.resume is not None:
        saved = torch.load(args.resume, mmap=True, map_location='cpu', weights_only=False)
        if saved.get('format') != FORMAT or saved.get('immutable') != immutable:
            raise ValueError('Only an unchanged registered pipeline life can resume')
        config = saved['config']
        if config['trainable_names'] != [n for n,p in model.named_parameters() if p.requires_grad]:
            raise ValueError('Actual trainability changed')
        restore_learner(saved, learner)
        ledger = StreamLedger(**saved['ledger'])
        meter = FirstPassPredictionMeter.from_state_dict(saved['first_pass_meter'], reference)
        execution = saved['execution_ledger']
        resume_origin = {'path': str(args.resume.resolve()), 'sha256': digest(args.resume),
                         'events': learner.events, 'train_updates': ledger.train_updates}
        del saved
    ledger.validate(args.train_offset, args.val_offset)
    verify_life(ledger, learner)
    meter.verify_cursor(ledger.train_updates*WINDOW, ledger.train_cursor)
    verify_score_journal(args.output/'scores.jsonl', ledger.events)
    if args.calibrate and (ledger.train_updates or ledger.eval_updates):
        raise ValueError('Calibration is only the first real window; resume without --calibrate')
    atomic_json(args.output/'config.json', config)
    if resume_origin is not None:
        append_json(args.output/'resume_journal.jsonl', resume_origin)
    resource = {}
    start = time.perf_counter()
    last_metrics = {}
    in_observe = False

    def progress(status, error=None):
        row = {'status': status, 'format': FORMAT, 'ledger': asdict(ledger),
            'bptt_train_tokens': ledger.train_updates*WINDOW, 'events': learner.events,
            'physical_ticks': learner.physical_ticks, 'actual_optimizer_updates': learner.updates,
            'target_train_updates': TRAIN_UPDATES, 'elapsed_seconds': time.perf_counter()-start,
            'ema_scope': 'mixed train/freshB/replay', 'ema_stream_nll': learner.ema,
            'best_live_nll': ledger.best_live_nll, **resource, **last_metrics, **meter.summary()}
        row['execution_ledger'] = dict(execution)
        row['physical_forward_ticks_including_setup'] = WINDOW*(execution['live_window_forwards']+execution['setup_window_forwards'])
        if error is not None:
            row['error'] = repr(error)
            row['unsafe_incomplete_observe'] = in_observe
        atomic_json(args.output/'progress.json', row)

    def checkpoint(is_best=False):
        if in_observe:
            raise RuntimeError('An incomplete optimizer operation cannot be checkpointed as a continuation')
        ledger.validate(args.train_offset, args.val_offset)
        verify_life(ledger, learner)
        if source_hashes() != hashes:
            raise ValueError('Pipeline source changed during this individual lifetime')
        payload = {'format': FORMAT, 'model': dict(model.named_parameters()),
            'learner': learner.state_dict(), 'optimizer_layout': optimizer_layout(learner),
            'config': config, 'immutable': immutable, 'ledger': asdict(ledger),
            'execution_ledger': dict(execution),
            'first_pass_meter': meter.state_dict(), 'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state(), 'saved_events': learner.events}
        atomic_checkpoint(args.output/'last.pt', payload)
        if is_best:
            atomic_checkpoint(args.output/'best.pt', {'format': 'fly-pipeline-best-weights-v1',
                'model': dict(model.named_parameters()), 'config': config,
                'train_updates': ledger.train_updates, 'monitoring_B_nll': ledger.best_live_nll,
                'continuation': False, 'selection_scope': 'Monitoring artifact; final estimand uses fixed endpoint'})

    try:
        if not args.no_graph:
            learner.runner = FlyBPTTGraph(learner, WINDOW)
            execution['capture_sessions'] += 1
            execution['setup_window_forwards'] += 3
            execution['setup_window_backwards'] += 3
        resource = memory_gate(args.vram_limit_mib)
        while not ledger.complete:
            values = ledger.targets(train, validation)
            phase, before = ledger.phase, learner.events
            cursor = (ledger.train_cursor if phase == 'train' else
                      ledger.val_cursor if phase == 'B' else ledger.phase_offset)
            begin = time.perf_counter()
            in_observe = True
            scores, last_metrics = learner.observe(values)
            completed = ledger.commit(values, scores)
            if phase == 'train':
                meter.record(scores, surprisal[values])
                meter.verify_cursor(ledger.train_updates*WINDOW, ledger.train_cursor)
            verify_life(ledger, learner)
            ledger.validate(args.train_offset, args.val_offset)
            execution['live_window_forwards'] += 1
            execution['live_window_backwards'] += 1
            in_observe = False
            append_json(args.output/'scores.jsonl', {'phase': phase, 'cursor_start': cursor,
                'event_interval': [before, learner.events], 'train_updates': ledger.train_updates,
                'actual_updates': learner.updates, 'targets': values.tolist(),
                'scores': list(map(float, scores)), 'fixed_unigram_scores': surprisal[values].tolist(),
                'bridge_policy': 'All actual phase-boundary targets included'})
            resource = memory_gate(args.vram_limit_mib)
            if learner.updates == 1 or learner.updates % 25 == 0:
                progress('running')
                append_json(args.output/'metrics.jsonl', {'phase': phase,
                    'train_updates': ledger.train_updates, 'actual_updates': learner.updates,
                    'window_preupdate_nll': float(np.mean(scores)),
                    'window_fixed_unigram_nll': float(np.mean(surprisal[values])),
                    'seconds_per_window': time.perf_counter()-begin, **resource, **last_metrics})
                print(f'{phase} train {ledger.train_updates}/3000; actual {learner.updates}; NLL {np.mean(scores):.5f}', flush=True)
            if completed is not None:
                b, a2 = completed['B_scores'], completed['A2_scores']
                gap = completed['A2_event_start'] - ledger.a_first_event_interval[1]
                report = {**completed, 'A1_curve': ledger.a_scores, 'A2_replay_curve': a2,
                    'replay_tokens': ledger.a_tokens, 'A1_event_interval': ledger.a_first_event_interval,
                    'actual_intervening_events': gap, 'actual_intervening_physical_ticks': gap,
                    'B_fixed_unigram_scores': surprisal[np.asarray(validation[completed['B_cursor_start']:completed['B_cursor_start']+B_TOKENS], dtype=np.int64)].tolist(),
                    'A_fixed_unigram_scores': surprisal[np.asarray(ledger.a_tokens, dtype=np.int64)].tolist(),
                    'live_prequential_nll': float(np.mean(b)), 'A2_prequential_nll': float(np.mean(a2)),
                    'matched_revisit_nll_change': float(np.mean(ledger.a_scores[1:])-np.mean(a2[1:])),
                    'matched_gap_scope': 'Same A targets except first bridge; full A1/A2 curves retain it',
                    'recovery': recovery_summary(b, block_tokens=16, hold_blocks=2),
                    'savings': savings_summary(ledger.a_scores, a2, intervening_events=gap, block_tokens=16, hold_blocks=2)}
                append_json(args.output/'lifelong_evaluation.jsonl', report)
                checkpoint(completed['is_best'])
            if learner.updates == 1:
                checkpoint()
                if args.calibrate:
                    progress('calibrated')
                    return
            if (args.output/'STOP').exists():
                checkpoint()
                progress('paused')
                return
        checkpoint()
        atomic_json(args.output/'final_risk.json', terminal_risk_summary(args.output/'scores.jsonl'))
        progress('completed')
    except BaseException as error:
        # Complete accounted windows can be saved, but a partially failed
        # optimizer operation cannot be represented as a valid continuation.
        if not in_observe and ledger.events:
            checkpoint()
        progress('failed', error)
        raise


if __name__ == '__main__':
    main()
