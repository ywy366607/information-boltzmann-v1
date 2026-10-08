"""One full D768 OWT interval: numerical/resource calibration, zero updates."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F

from information_boltzmann.core.intrinsic_time import (
    EvolutionSchedule, IntrinsicTimePolicy, factor_characteristic_time)
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.gpu_memory import windows_gpu_memory
from information_boltzmann.runtime.training import belief_tensors
from scripts.ib.audit_medium_objective_gradients import file_hash, tensor_hash
from scripts.ib.train_plastic_conductance import unpack_belief


class DedicatedMemorySampler:
    """Read-only process counters; the CUDA allocator enforces its own hard cap."""

    def __init__(self, cap_bytes):
        self.cap_bytes = cap_bytes
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.exceeded = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                value = windows_gpu_memory()
                if value is not None:
                    self.samples.append({'seconds': time.perf_counter(), **value})
                    self.exceeded |= value['dedicated_bytes'] > self.cap_bytes
            except Exception as error:
                self.errors.append(str(error))
            self.stop.wait(.25)

    def check(self):
        if self.exceeded:
            raise MemoryError('Sampled dedicated GPU use exceeded the declared cap')

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=46)
        return {'sample_count': len(self.samples),
                'sampled_peak_dedicated_mib': None if not self.samples else
                    max(x['dedicated_bytes'] for x in self.samples) / 2**20,
                'sampled_peak_shared_mib': None if not self.samples else
                    max(x['shared_bytes'] for x in self.samples) / 2**20,
                'counter_errors': self.errors, 'cap_exceeded': self.exceeded,
                'last_sample': None if not self.samples else self.samples[-1],
                'peak_is_sampled': True}


def norm_record(value):
    if value is None:
        return {'connected': False, 'finite': None, 'norm': None}
    value = value.detach()
    squared = 0.
    for start in range(0, value.numel(), 2**20):
        piece = value.reshape(-1)[start:start + 2**20].double()
        squared += float(piece.square().sum().cpu())
    return {'connected': True, 'finite': bool(torch.isfinite(value).all()),
            'norm': math.sqrt(squared)}


def group_gradients(model):
    selectors = {
        'medium': lambda n: n.startswith('medium.'),
        'material': lambda n: n.startswith('medium.material.'),
        'write': lambda n: n.startswith('write_agent.'),
        'read': lambda n: n.startswith('readout.'),
        'temporal': lambda n: n.startswith('temporal_readout.'),
        'decoder': lambda n: n.startswith(('decoder.', 'read_norm.')),
        'intrinsic_time_head': lambda n: n.startswith('intrinsic_time.'),
        'all_trainable': lambda n: True}
    report = {}
    for group, selector in selectors.items():
        entries = [(name, parameter) for name, parameter in model.named_parameters()
                   if parameter.requires_grad and selector(name)]
        records = [(name, parameter.numel(), norm_record(parameter.grad))
                   for name, parameter in entries]
        connected = [record for _, _, record in records if record['connected']]
        report[group] = {
            'parameter_count': sum(size for _, size, _ in records),
            'connected_parameter_count': sum(size for _, size, record in records
                                             if record['connected']),
            'finite': all(record['finite'] for record in connected),
            'norm': math.sqrt(sum(record['norm']**2 for record in connected)),
            'unused_names': [name for name, _, record in records if not record['connected']]}
    return report


def calibrate(saved, train, args, report):
    # First load all old tensors strictly, in their original time coordinates.
    # The new zero-initialized clock is an explicit candidate branch only.
    model = PlasticMediumPorts3D(**saved['config']['constructor'])
    model.load_state_dict(saved['model'], strict=True)
    if saved['config'].get('pretrained_vocabulary', {}).get('embedding_frozen'):
        model.source.embedding.weight.requires_grad_(False)
    initial_cpu = unpack_belief(saved['belief'], torch.device('cpu')).detach()
    with torch.no_grad():
        factor = model.medium.current_transport_factor(initial_cpu.medium)
        spacing = [1. / n for n in model.medium.shape]
        scales = factor_characteristic_time(factor, cell_spacing=spacing)
        cell = float(scales.cell_crossing_time)
        reference = float(scales.torus_crossing_time)
        speed = float(torch.linalg.matrix_norm(factor, ord=2).amax())
        resolved_rate = float((factor / factor.new_tensor(spacing)[:, None]).square()
                              .sum((-2, -1)).sqrt().amax())
    if not math.isfinite(reference) or not math.isfinite(cell):
        raise ValueError('The saved factor has no finite transport time reference')
    old_duration = float(saved['config']['event_duration'])
    solver_step, observer_step = cell / args.solver_divisor, cell
    schedule = EvolutionSchedule.for_duration(reference, solver_max_step=solver_step,
        observer_max_step=observer_step, max_steps=args.max_steps)
    report.update({
        'time_reference': {
            'chosen': 'torus_crossing_time', 'reference_duration': reference,
            'cell_crossing_time': cell, 'torus_crossing_time': reference,
            'max_singular_speed': speed, 'max_resolved_rotation_rate': resolved_rate,
            'cell_spacing': spacing, 'domain_lengths': [1., 1., 1.],
            'factor_shape': list(factor.shape), 'factor_sha256': tensor_hash([('B', factor)]),
            'old_event_duration': old_duration, 'duration_to_old_ratio': reference / old_duration,
            'interpretation': 'Global fastest-speed crossing estimates are reference scales only; '
                'they do not establish arrival through slow channels or useful propagation.'},
        'resolution': {'solver_max_step': solver_step, 'observer_max_step': observer_step,
            'solver_divisor_of_characteristic_cell': args.solver_divisor,
            'observer_count': schedule.observer_count, 'solver_substeps': schedule.solver_substeps,
            'total_solver_steps': schedule.observer_count * schedule.solver_substeps,
            'actual_observer_interval': reference / schedule.observer_count,
            'actual_solver_step': reference / schedule.observer_count / schedule.solver_substeps,
            'max_evolution_steps': args.max_steps,
            'choice': 'One observer per characteristic cell scale; solver at most one quarter '
                'of that scale. Declared numerical resolution, not an accuracy/convergence claim.'},
        'material_coefficients_shape': list(model.medium.material.coefficients.shape),
        'material_reference_shape': list(model.medium.material.reference_shape),
        'structure_posterior_enabled': model.medium.structural_posterior is not None,
        'frequency_time_reference': float(model.temporal_readout.bank.frequency_time_reference)})
    if model.medium.structural_posterior is not None:
        raise ValueError('The resource calibration must not add structural-posterior attribution')
    del factor, initial_cpu
    model.intrinsic_time = IntrinsicTimePolicy(reference)
    model.solver_max_step, model.observer_max_step = solver_step, observer_step
    model.max_evolution_steps = args.max_steps
    candidate_parameters = {'intrinsic_time_reference': reference, 'intrinsic_max_duration': None,
        'solver_max_step': solver_step, 'observer_max_step': observer_step,
        'max_evolution_steps': args.max_steps}
    report['candidate_constructor'] = {**saved['config']['constructor'], **candidate_parameters}
    report['candidate_added_parameters'] = candidate_parameters
    # Eager arithmetic avoids compiling an adaptive event or allocating compiler workspaces.
    model.medium.execution_backend, model.port_execution = 'native', 'native'
    if model.medium.short_term_plasticity is not None:
        model.medium.short_term_plasticity.fuse_execution = False
    model = model.to('cuda').train()
    initial = unpack_belief(saved['belief'], torch.device('cuda')).detach()
    model_before = tensor_hash(model.named_parameters())
    belief_before = tensor_hash((str(i), value) for i, value in enumerate(belief_tensors(initial)))
    initial.medium.field.requires_grad_(True)
    cursor, carry = int(saved['cursor']), int(saved['learner']['carry_token'])
    target = int(train[cursor])
    ids = torch.tensor([carry], device='cuda')
    targets = torch.tensor([target], device='cuda')
    report.update({'target_source_interval': [cursor, cursor + 1], 'input_token_ids': [carry],
        'target_token_ids': [target], 'target_min': target, 'target_max': target,
        'carry_matches_train_predecessor': carry == int(train[cursor - 1]),
        'input_target_sha256': hashlib.sha256(np.array([carry, target], dtype=np.int64).tobytes()).hexdigest(),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initial_physical_time': initial.medium.elapsed.detach().cpu().tolist(),
        'initial_temporal_time': initial.temporal.elapsed.detach().cpu().tolist()})
    torch.set_rng_state(saved['cpu_rng'])
    if saved.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(saved['cuda_rng'])
    monitor = DedicatedMemorySampler(args.vram_cap_mib * 2**20)
    monitor.thread.start()
    try:
        torch.cuda.synchronize()
        begin = time.perf_counter()
        table = F.normalize(model.source.embedding.weight, dim=-1)
        prepared = model.medium.prepare_evolution()
        # These are training_event's same causal operations, split only so the
        # entry-state and differentiable duration gradients can be retained.
        duration = model.event_time(initial, old_duration)
        duration.retain_grad()
        written, write = model.assimilate(initial, ids, token_features=table, diagnostics=False)
        written.medium.field.retain_grad()
        outgoing, _, motion = model.advance(written, duration, prepared=prepared,
            diagnostics=False, return_motion=True, activation_checkpointing=True)
        outgoing.medium.field.retain_grad()
        feature, _ = model.read(outgoing, decode=False, prepared=prepared,
                                diagnostics=False, motion=motion)
        logits = model.decode(feature)
        task = F.cross_entropy(logits, targets)
        port = write['_write_free_energy']
        joint = task + port
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(joint).all()):
            raise FloatingPointError('Nonfinite endpoint prediction/objective')
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        # Publish the actually observed endpoint before backward so a capped
        # backward failure does not erase successful forward/clock evidence.
        report.update({'forward_completed': True, 'forward_seconds': forward_end - begin,
            'requested_duration': float(duration.detach()),
            'task_nll': float(task.detach()), 'port_objective': float(port.detach()),
            'joint_objective': float(joint.detach()),
            'objective_scale': 'One task CE plus one W4 port objective, weights 1:1',
            'logits': {'shape': list(logits.shape), 'finite': True,
                'minimum': float(logits.detach().min()), 'maximum': float(logits.detach().max())},
            'final_physical_time': outgoing.medium.elapsed.detach().cpu().tolist(),
            'final_temporal_time': outgoing.temporal.elapsed.detach().cpu().tolist(),
            'physical_time_increment': (outgoing.medium.elapsed - initial.medium.elapsed).detach().cpu().tolist(),
            'temporal_time_increment': (outgoing.temporal.elapsed - initial.temporal.elapsed).detach().cpu().tolist(),
            'clock_difference_max': float((outgoing.medium.elapsed - outgoing.temporal.elapsed).detach().abs().max())})
        monitor.check()
        print(f'FULL INTERVAL FORWARD: {schedule.observer_count} observers, '
              f'{schedule.observer_count * schedule.solver_substeps} solver steps; BACKWARD', flush=True)
        joint.backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        monitor.check()
        report.update({'status': 'completed_full_interval_one_forward_one_backward',
            'task_nll': float(task.detach()), 'port_objective': float(port.detach()),
            'joint_objective': float(joint.detach()),
            'objective_scale': 'One task CE plus one W4 port objective, weights 1:1',
            'port_nll': None if 'port_nll' not in write else float(write['port_nll'].detach()),
            'write_action_kl': None if 'write_action_kl' not in write else float(write['write_action_kl'].detach()),
            'logits': {'shape': list(logits.shape), 'finite': True,
                'minimum': float(logits.detach().min()), 'maximum': float(logits.detach().max())},
            'requested_duration': float(duration.detach()),
            'duration_gradient': norm_record(duration.grad),
            'duration_gradient_value': None if duration.grad is None else float(duration.grad),
            'full_interval_entry_field_gradient': norm_record(written.medium.field.grad),
            'endpoint_field_gradient': norm_record(outgoing.medium.field.grad),
            'initial_field_gradient': norm_record(initial.medium.field.grad),
            'gradients': group_gradients(model),
            'intrinsic_head_gradient_values': {name: None if p.grad is None else p.grad.detach().cpu().tolist()
                for name, p in model.intrinsic_time.named_parameters()},
            'final_physical_time': outgoing.medium.elapsed.detach().cpu().tolist(),
            'final_temporal_time': outgoing.temporal.elapsed.detach().cpu().tolist(),
            'physical_time_increment': (outgoing.medium.elapsed - initial.medium.elapsed).detach().cpu().tolist(),
            'temporal_time_increment': (outgoing.temporal.elapsed - initial.temporal.elapsed).detach().cpu().tolist(),
            'clock_difference_max': float((outgoing.medium.elapsed - outgoing.temporal.elapsed).detach().abs().max()),
            'forward_seconds': forward_end - begin, 'backward_seconds': backward_end - forward_end,
            'forward_backward_seconds': backward_end - begin,
            'resource_rate_one_token_per_second': 1. / (backward_end - begin),
            'rate_scope': 'This single checkpointed long interval only; no extrapolation to BPTT32 throughput.',
            'parameter_hash_before_after': model_before,
            'parameters_unchanged': tensor_hash(model.named_parameters()) == model_before,
            'initial_belief_hash_before_after': belief_before,
            'initial_belief_unchanged': tensor_hash((str(i), value) for i, value in
                enumerate(belief_tensors(initial))) == belief_before})
    finally:
        report['memory'] = {'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            **monitor.finish()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vram-cap-mib', type=int, default=3072)
    parser.add_argument('--driver-headroom-mib', type=int, default=1024)
    parser.add_argument('--solver-divisor', type=float, default=4.)
    parser.add_argument('--max-steps', type=int, default=4096)
    parser.add_argument('--append-attempt', action='store_true',
                        help='Preserve the existing first attempt and record one explicit second attempt')
    args = parser.parse_args()
    if not math.isfinite(args.solver_divisor) or args.solver_divisor <= 0:
        parser.error('Positive finite solver divisor required')
    budget = args.vram_cap_mib - args.driver_headroom_mib
    if budget <= 0 or args.vram_cap_mib > 3072:
        parser.error('Keep the dedicated cap at most 3072 MiB with positive allocator room')
    previous = None
    if args.append_attempt:
        previous = json.loads(args.output.read_text(encoding='utf-8'))
        if 'first_attempt' in previous or 'second_attempt' in previous:
            parser.error('Appending is limited to one explicit second attempt')
    torch.set_num_threads(4)
    begin = time.perf_counter()
    print('HASH / MMAP CPU LOAD / ACTUAL B TIME SCALES', flush=True)
    checkpoint_hash = file_hash(args.checkpoint)
    if previous is not None and previous['checkpoint_sha256_before_after'] != checkpoint_hash:
        raise ValueError('Second attempt requires the same immutable checkpoint')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    train_path = args.data_dir / 'train.npy'
    data_hash = file_hash(train_path)
    if data_hash != saved['config']['data_sha256']['train.npy']:
        raise ValueError('Real OWT data hash differs from the saved individual')
    manifest_path = args.data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(budget * 2**20 / properties.total_memory)
    torch.cuda.reset_peak_memory_stats()
    sources = ('scripts/ib/audit_medium_long_interval.py',
        'scripts/ib/audit_medium_objective_gradients.py',
        'information_boltzmann/core/intrinsic_time.py',
        'information_boltzmann/core/plastic_ports.py',
        'information_boltzmann/core/plastic_medium.py',
        'information_boltzmann/core/temporal_probes.py',
        'information_boltzmann/runtime/training.py')
    report = {'purpose': 'D768 full-long-interval numerical/resource calibration; no NLL or capability comparison',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256_before_after': checkpoint_hash,
        'checkpoint_step': saved['step'], 'saved_optimizer_updates': saved['learner']['optimizer_updates'],
        'saved_pending_events': saved['learner']['pending'], 'optimizer_updates_executed': 0,
        'optimizer_instantiated': False, 'checkpoint_written': False, 'window_tokens': 1,
        'execution': 'Native eager FP32 arithmetic; nonreentrant activation checkpoint per observer interval; '
            'no compilation of the adaptive event; one complete first-order joint backward.',
        'saved_constructor': saved['config']['constructor'], 'dataset_source': manifest['source'],
        'dataset_revision': manifest['revision'], 'dataset_split': 'train; next unconsumed target only',
        'train_sha256': data_hash, 'manifest_sha256': file_hash(manifest_path),
        'vram_cap_mib': args.vram_cap_mib, 'driver_headroom_mib': args.driver_headroom_mib,
        'tensor_allocator_cap_mib': budget, 'gpu_name': properties.name,
        'source_hashes': {path: file_hash(path) for path in sources},
        'attempts': 1, 'fallback_permitted': False}
    train = np.load(train_path, mmap_mode='r')
    try:
        calibrate(saved, train, args, report)
    except (torch.cuda.OutOfMemoryError, MemoryError, FloatingPointError) as error:
        report.update(status='safe_failure_without_retry', error_type=type(error).__name__,
                      error=str(error), completed_full_backward=False)
    else:
        report['completed_full_backward'] = True
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report.update(checkpoint_immutable=file_hash(args.checkpoint) == checkpoint_hash,
                  elapsed_seconds=time.perf_counter() - begin,
                  cuda_allocated_after_release_mib=torch.cuda.memory_allocated() / 2**20,
                  cuda_reserved_after_release_mib=torch.cuda.memory_reserved() / 2**20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    published = report if previous is None else {
        'purpose': 'Same D768/one-token/full-interval resource calibration; preserve both explicit attempts',
        'adjustment': 'First attempt reached the 2048 MiB allocator protection line while sampled '
            'dedicated usage was 2124.64 MiB. The authorized second attempt increases only the '
            'allocator budget to 2700 MiB, retaining a 3072 MiB dedicated cap and 372 MiB driver headroom.',
        'first_attempt': previous, 'second_attempt': report,
        'latest_status': report['status'], 'optimizer_updates_executed_total': 0,
        'checkpoint_immutable': report['checkpoint_immutable'] and previous['checkpoint_immutable']}
    args.output.write_text(json.dumps(published, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'status': report['status'],
        'memory': report.get('memory'), 'gradients': report.get('gradients'),
        'seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
