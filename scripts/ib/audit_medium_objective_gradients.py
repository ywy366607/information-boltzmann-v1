"""One real next-stream window: task/port gradients, no optimizer or checkpoint write."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.gpu_memory import total_gpu_memory, windows_gpu_memory
from information_boltzmann.runtime.training import belief_tensors, training_event
from scripts.ib.train_plastic_conductance import unpack_belief


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def gradient_statistics(names, parameters, task, port, predicate):
    selected = [i for i, name in enumerate(names) if predicate(name)]
    task_sq = port_sq = dot = 0.
    for i in selected:
        left, right = task[i], port[i]
        # FP64 chunk reductions avoid retaining double-sized vocabulary tensors.
        size = parameters[i].numel()
        for start in range(0, size, 2**20):
            a = None if left is None else left.reshape(-1)[start:start + 2**20].double()
            b = None if right is None else right.reshape(-1)[start:start + 2**20].double()
            if a is not None:
                task_sq += float(torch.dot(a, a))
            if b is not None:
                port_sq += float(torch.dot(b, b))
            if a is not None and b is not None:
                dot += float(torch.dot(a, b))
    task_norm, port_norm = math.sqrt(task_sq), math.sqrt(port_sq)
    return {'parameter_count': sum(parameters[i].numel() for i in selected),
            'task_connected_parameters': sum(parameters[i].numel() for i in selected if task[i] is not None),
            'port_connected_parameters': sum(parameters[i].numel() for i in selected if port[i] is not None),
            'task_gradient_norm': task_norm, 'port_gradient_norm': port_norm,
            'task_port_dot': dot,
            'task_port_cosine': None if task_norm * port_norm == 0 else dot / (task_norm * port_norm),
            'port_to_task_norm_ratio': None if task_norm == 0 else port_norm / task_norm,
            'port_projection_on_task': None if task_sq == 0 else dot / task_sq,
            'unclipped_joint_gradient_norm': math.sqrt(max(0., task_sq + port_sq + 2 * dot)),
            'task_unused_names': [names[i] for i in selected if task[i] is None],
            'port_unused_names': [names[i] for i in selected if port[i] is None]}


def evaluate(model, initial, ids, targets, duration, use_checkpoint):
    table = F.normalize(model.source.embedding.weight, dim=-1)
    prepared = model.medium.prepare_evolution()
    belief, features, port_terms, port_nll, action_kl = initial, [], [], [], []

    def event(current, token):
        return training_event(model, current, token, table, prepared, duration, 1, False)

    for index in range(ids.shape[1]):
        if use_checkpoint:
            _, belief, write, _, feature = checkpoint(event, belief, ids[:, index],
                use_reentrant=False, preserve_rng_state=False)
        else:
            _, belief, write, _, feature = event(belief, ids[:, index])
        features.append(feature)
        port_terms.append(write['_write_free_energy'])
        if 'port_nll' in write:
            port_nll.append(write['port_nll'])
        if 'write_action_kl' in write:
            action_kl.append(write['write_action_kl'])
        if (index + 1) % 8 == 0:
            print(f'FORWARD {index + 1}/{ids.shape[1]}', flush=True)
    logits = model.decode(torch.stack(features, 1))
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Nonfinite logits')
    token_nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none')
    diagnostics = {'logits_shape': list(logits.shape), 'logits_finite': True,
                   'logit_min': float(logits.detach().min()),
                   'logit_max': float(logits.detach().max()),
                   'logit_std': float(logits.detach().double().std(unbiased=False)),
                   'token_nll': token_nll.detach().cpu().tolist(),
                   'port_nll': None if not port_nll else float(torch.stack(port_nll).mean()),
                   'write_action_kl': None if not action_kl else float(torch.stack(action_kl).mean()),
                   'final_physical_time': belief.medium.elapsed.detach().cpu().tolist()}
    return token_nll.mean(), torch.stack(port_terms).mean(), diagnostics


def audit_window(saved, tokens, train, device, use_checkpoint):
    constructor = dict(saved['config']['constructor'])
    model = PlasticMediumPorts3D(**constructor)
    model.load_state_dict(saved['model'], strict=True)
    if saved['config'].get('pretrained_vocabulary', {}).get('embedding_frozen'):
        model.source.embedding.weight.requires_grad_(False)
    # Arithmetic-only eager execution avoids an unbudgeted new compiler graph.
    model.medium.execution_backend = 'native'
    model.port_execution = 'native'
    if model.medium.short_term_plasticity is not None:
        model.medium.short_term_plasticity.fuse_execution = False
    model = model.to(device).train()
    if (model.solver_max_step is not None or model.observer_max_step is not None
            or model.intrinsic_time is not None
            or float(model.temporal_readout.bank.frequency_time_reference) != 1.):
        raise ValueError('Saved-baseline audit must retain its legacy solver and time coordinates')
    initial = unpack_belief(saved['belief'], device).detach()
    cursor, carry = int(saved['cursor']), int(saved['learner']['carry_token'])
    target_array = np.array(train[cursor:cursor + tokens], dtype=np.int64)
    if len(target_array) != tokens:
        raise ValueError('Next train window exceeds the saved stream')
    observed = np.concatenate((np.array([carry], dtype=np.int64), target_array[:-1]))
    targets = torch.from_numpy(target_array).to(device)[None]
    ids = torch.from_numpy(observed).to(device)[None]
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    names, values = zip(*parameters)
    model_before = tensor_hash(model.named_parameters())
    belief_before = tensor_hash((str(i), value) for i, value in enumerate(belief_tensors(initial)))
    torch.set_rng_state(saved['cpu_rng'])
    if device.type == 'cuda' and saved.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(saved['cuda_rng'])
    duration = float(saved['config']['event_duration'])
    task_loss, port_loss, diagnostics = evaluate(model, initial, ids, targets, duration, use_checkpoint)
    task_value, port_value = float(task_loss), float(port_loss)
    print('TASK GRADIENT', flush=True)
    task_gpu = torch.autograd.grad(task_loss, values, allow_unused=True)
    task = tuple(None if value is None else value.detach().cpu() for value in task_gpu)
    del task_gpu, task_loss, port_loss
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Two independent forwards from exactly the same immutable initial state
    # avoid checkpoint's retained recomputation graph across objective backwards.
    torch.set_rng_state(saved['cpu_rng'])
    if device.type == 'cuda' and saved.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(saved['cuda_rng'])
    check_task, port_loss, verification = evaluate(model, initial, ids, targets, duration, use_checkpoint)
    forward_differences = {'task': abs(float(check_task) - task_value),
                           'port': abs(float(port_loss) - port_value),
                           'token_nll_max': max(abs(a - b) for a, b in
                                                zip(verification['token_nll'], diagnostics['token_nll']))}
    if forward_differences['task'] > 1e-6 or forward_differences['port'] > 1e-6:
        raise RuntimeError('Same-state objective forwards disagree')
    del check_task
    print('PORT GRADIENT', flush=True)
    port_gpu = torch.autograd.grad(port_loss, values, allow_unused=True)
    port = tuple(None if value is None else value.detach().cpu() for value in port_gpu)
    del port_gpu
    print('GRADIENT REDUCTIONS', flush=True)
    groups = {'medium': lambda n: n.startswith('medium.'),
              'write': lambda n: n.startswith('write_agent.'),
              'read': lambda n: n.startswith('readout.'),
              'temporal': lambda n: n.startswith('temporal_readout.'),
              'decoder': lambda n: n.startswith(('decoder.', 'read_norm.')),
              'source_other': lambda n: n.startswith('source.'),
              'all_trainable': lambda n: True}
    for label, prefixes in {
            'medium_material': ('medium.material.',),
            'medium_transport': ('medium.log_speed.', 'medium.transport_shear.'),
            'medium_conduction': ('medium.conduction_plasticity.',),
            'medium_stp': ('medium.short_term_plasticity.',),
            'medium_response': ('medium.conductance_response.',),
            'medium_collision': ('medium.collision_rate.',)}.items():
        groups[label] = lambda name, prefixes=prefixes: name.startswith(prefixes)
    report = {label: gradient_statistics(names, values, task, port, selector)
              for label, selector in groups.items()}
    assert all(parameter.grad is None for parameter in values)
    assert tensor_hash(model.named_parameters()) == model_before
    assert tensor_hash((str(i), value) for i, value in enumerate(belief_tensors(initial))) == belief_before
    memory = None
    if device.type == 'cuda':
        torch.cuda.synchronize()
        process_memory = windows_gpu_memory()
        memory = {'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                  'total_gpu': total_gpu_memory(), 'process_gpu': process_memory}
    return {'window_tokens': tokens, 'next_target_cursor': cursor,
            'target_source_interval': [cursor, cursor + tokens],
            'carry_token': carry, 'carry_matches_train_predecessor': carry == int(train[cursor - 1]),
            'input_token_ids': observed.tolist(), 'target_token_ids': target_array.tolist(),
            'input_target_sha256': hashlib.sha256(observed.tobytes() + target_array.tobytes()).hexdigest(),
            'target_min': int(target_array.min()), 'target_max': int(target_array.max()),
            'initial_physical_time': initial.medium.elapsed.cpu().tolist(),
            'event_duration': duration, 'physical_credit_horizon': tokens * duration,
            'task_nll': task_value, 'port_objective': port_value,
            'joint_objective': task_value + port_value, 'groups': report,
            'forward': diagnostics, 'memory': memory,
            'separate_same_state_forwards': 2,
            'objective_forward_agreement_abs': forward_differences,
            'parameter_hash_before_after': model_before, 'initial_belief_hash_before_after': belief_before,
            'parameters_unchanged': True, 'initial_belief_unchanged': True,
            'parameter_grad_buffers_untouched': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--fallback-tokens', type=int, default=8)
    parser.add_argument('--prior-capped-window', type=int,
                        help='Record a larger window that failed its capped allocator in a previous process')
    parser.add_argument('--vram-cap-mib', type=int, default=3072)
    parser.add_argument('--driver-headroom-mib', type=int, default=1024)
    args = parser.parse_args()
    if min(args.tokens, args.fallback_tokens) < 1 or args.fallback_tokens > args.tokens:
        parser.error('Positive tokens and fallback no larger than the requested window required')
    torch.set_num_threads(4)
    start = time.perf_counter()
    print('CHECKPOINT HASH/LOAD', flush=True)
    checkpoint_hash = file_hash(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    data_path = args.data_dir / 'train.npy'
    data_hash = file_hash(data_path)
    if data_hash != saved['config']['data_sha256']['train.npy']:
        raise ValueError('OWT train data differs from the saved individual')
    train = np.load(data_path, mmap_mode='r')
    device = torch.device(args.device)
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        # Hard allocator ceiling leaves separately declared driver/context room.
        tensor_budget = args.vram_cap_mib - args.driver_headroom_mib
        if tensor_budget <= 0:
            parser.error('GPU cap must leave CUDA-context headroom')
        torch.cuda.set_per_process_memory_fraction(tensor_budget * 2**20 / properties.total_memory)
        torch.cuda.reset_peak_memory_stats()
    fallback = (None if args.prior_capped_window is None else
        {'requested_tokens': args.prior_capped_window,
         'reason': 'Prior 32-token task backward completed; port backward exceeded the 2560 MiB tensor allocator. Fresh process releases that graph before this smaller prefix.',
         'status': 'one smaller local diagnostic, not the complete BPTT32 gradient'})
    try:
        result = audit_window(saved, args.tokens, train, device, True)
    except torch.cuda.OutOfMemoryError as error:
        if device.type != 'cuda' or args.fallback_tokens == args.tokens:
            raise
        fallback = {'requested_tokens': args.tokens, 'reason': str(error),
                    'status': 'full window exceeded the capped tensor allocator; one smaller local diagnostic'}
        result = None
    # Exit the except block first: its traceback otherwise retains the old graph.
    if result is None:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f'ONE FALLBACK: {args.fallback_tokens} tokens', flush=True)
        result = audit_window(saved, args.fallback_tokens, train, device, True)
    assert file_hash(args.checkpoint) == checkpoint_hash
    if result['memory'] and result['memory']['process_gpu']:
        if result['memory']['process_gpu']['dedicated_bytes'] > args.vram_cap_mib * 2**20:
            raise MemoryError('Observed dedicated GPU usage exceeded the declared cap')
    manifest_path = args.data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result.update({'purpose': 'one real OWT local objective-gradient diagnostic; no capability verdict',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256_before_after': checkpoint_hash,
        'checkpoint_immutable': True, 'checkpoint_step': saved['step'],
        'saved_optimizer_updates': saved['learner']['optimizer_updates'],
        'saved_pending_events': saved['learner']['pending'],
        'optimizer_updates_executed': 0, 'optimizer_instantiated': False,
        'checkpoint_written': False, 'device': args.device, 'vram_cap_mib': args.vram_cap_mib,
        'driver_headroom_mib': args.driver_headroom_mib,
        'tensor_allocator_cap_mib': args.vram_cap_mib - args.driver_headroom_mib,
        'fallback': fallback, 'dataset_source': manifest['source'], 'dataset_revision': manifest['revision'],
        'dataset_split': 'train; exactly the next unconsumed target interval',
        'train_sha256': data_hash, 'manifest_sha256': file_hash(manifest_path),
        'saved_constructor': saved['config']['constructor'],
        'solver': 'legacy one substep, fixed saved event duration; intrinsic/adaptive time disabled',
        'execution': 'native eager FP32 with non-reentrant activation checkpointing; same physical laws',
        'frequency_time_reference': 1., 'objective_scale': 'mean task CE and mean W4 port objective, weights 1:1',
        'gradient_reductions': 'unclipped; FP64 norm/dot accumulation; none denotes a disconnected group',
        'source_hashes': {path: file_hash(path) for path in (
            'scripts/ib/audit_medium_objective_gradients.py',
            'information_boltzmann/core/plastic_ports.py',
            'information_boltzmann/core/plastic_medium.py',
            'information_boltzmann/core/temporal_probes.py',
            'information_boltzmann/runtime/training.py')},
        'elapsed_seconds': time.perf_counter() - start})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'tokens': result['window_tokens'],
                      'task_nll': result['task_nll'], 'port_objective': result['port_objective'],
                      'medium': result['groups']['medium'],
                      'material': result['groups']['medium_material'], 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
