"""Full real-data CUDA BPTT32 identity and warm-update resource acceptance.

Uses a saved individual only as an immutable reference. No checkpoint or training
cursor is committed. Both compared executions start at the identical snapshot.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime import training
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.medium_health import conditional_state_response
from information_boltzmann.runtime.optimization import make_medium_optimizer
from scripts.ib.train_plastic_conductance import unpack_belief
from scripts.ib.audit_medium_long_interval import DedicatedMemorySampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--vram-limit-mib', type=int, default=3072)
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--reference-dense', action='store_true',
                        help='Measure the previous dense keys plus auxiliary final read')
    parser.add_argument('--dense-keys-only', action='store_true',
                        help='Keep dense keys but omit the unused auxiliary final read')
    args = parser.parse_args()
    torch.set_num_threads(2)
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction((args.vram_limit_mib - 192) / total_mib)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    constructor = dict(saved['config']['constructor'])
    constructor['read_key_execution'] = 'dense'
    # Native port GEMMs isolate the algebraic comparison from compiler cache
    # specialization; the physical kernel and all 32-event credit stay live.
    constructor['port_execution'] = 'native'
    model = PlasticMediumPorts3D(**constructor).cuda().train()
    model.source.embedding.weight.requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    initial = unpack_belief(saved['belief'], 'cuda')
    posterior = model.medium.structural_posterior
    model._structural_window_events = 32
    noise = torch.zeros_like(posterior.mean)
    train = np.load('data/ib_owt_gpt2/train.npy', mmap_mode='r')
    target = torch.from_numpy(np.array(train[saved['cursor']:saved['cursor'] + 32], dtype=np.int64)).cuda()
    carry = saved['learner']['carry_token']
    observed = torch.cat((target.new_tensor([carry]), target[:-1]))[None]
    original_aux = training._writer_auxiliary_gradients
    monitor = DedicatedMemorySampler(args.vram_limit_mib * 2**20)
    monitor.thread.start()
    report = {'scope': 'actual D7688^3 real OWT, full32 credit, execution only',
              'checkpoint': str(args.checkpoint), 'checkpoint_step': saved['step'],
              'checkpoint_cursor': saved['cursor'],
              'production_checkpoint_written': False, 'runs': []}
    reference = None
    try:
        for key_execution in (() if args.timing_only else ('dense', 'support')):
            model.readout.compact_key_execution = key_execution
            model.read_key_execution = key_execution
            model.zero_grad(set_to_none=True)
            posterior.window_active.fill_(False)
            posterior.begin_window(noise)
            belief = initial.detach()
            for value in training.belief_tensors(belief):
                if value.is_floating_point() or value.is_complex():
                    value.requires_grad_(True)
            def auxiliary(*values, **options):
                options['read_feature'] = key_execution == 'dense'
                return original_aux(*values, **options)
            training._writer_auxiliary_gradients = auxiliary
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            print(f'BPTT32 identity: {key_execution}', flush=True)
            result = training.quiet_training_chunk(model, observed, target[None], belief,
                event_duration=saved['config']['event_duration'],
                activation_checkpointing=True, checkpoint_granularity='event',
                return_token_nll=True)
            loss, final, nll, token_nll = result
            loss.backward()
            torch.cuda.synchronize()
            values = {'loss': loss.detach().cpu(), 'nll': nll.detach().cpu(),
                      'token_nll': token_nll.detach().cpu(),
                      'states': [x.detach().cpu().clone() for x in training.belief_tensors(final)],
                      'input_gradients': [None if x.grad is None else x.grad.detach().cpu().clone()
                                          for x in training.belief_tensors(belief)],
                      'parameter_gradients': {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                              for name, p in model.named_parameters()}}
            row = {'read_key_execution': key_execution,
                   'auxiliary_final_read': key_execution == 'dense',
                   'forward_backward_seconds': time.perf_counter() - started,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
            report['runs'].append(row)
            print(json.dumps(row), flush=True)
            if reference is None:
                reference = values
            else:
                maximum = {'state': 0., 'input_gradient': 0., 'parameter_gradient': 0.}
                for key in ('loss', 'nll', 'token_nll'):
                    torch.testing.assert_close(values[key], reference[key], atol=5e-6, rtol=5e-4)
                def compare(actual, expected, family, name):
                    assert (actual is None) == (expected is None), name
                    if expected is not None:
                        assert bool(torch.isfinite(actual).all()), name
                        maximum[family] = max(maximum[family], float((actual - expected).abs().max()))
                        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-4, msg=str(name))
                for index, (actual, expected) in enumerate(zip(values['states'], reference['states'])):
                    compare(actual, expected, 'state', index)
                for index, (actual, expected) in enumerate(zip(values['input_gradients'], reference['input_gradients'])):
                    compare(actual, expected, 'input_gradient', index)
                for name, expected in reference['parameter_gradients'].items():
                    compare(values['parameter_gradients'][name], expected, 'parameter_gradient', name)
                report['identity'] = {'passed': True, 'maximum_absolute_differences': maximum,
                                      'parameter_tensors': len(values['parameter_gradients']),
                                      'state_tensors': len(values['states'])}
            del result, loss, final, nll, token_nll, values, belief
            model.zero_grad(set_to_none=True)
            posterior.window_active.fill_(False)
            gc.collect()
            monitor.check()
        # Native complete-state JVP must work after compiled physical training;
        # diagnostic execution is observational and never mutates this individual.
        model.port_execution = 'fused'
        if not args.timing_only:
            print('Complete-state forward AD diagnostic with fused ports enabled', flush=True)
            report['conditional_response'] = conditional_state_response(
                model, initial.detach(), observed[:, 0],
                event_duration=saved['config']['event_duration'], directions=1)
        model.readout.compact_key_execution = (
            'dense' if args.reference_dense or args.dense_keys_only else 'support')
        model.read_key_execution = model.readout.compact_key_execution
        training._writer_auxiliary_gradients = original_aux
        if args.reference_dense:
            def dense_auxiliary(*values, **options):
                options['read_feature'] = True
                return original_aux(*values, **options)
            training._writer_auxiliary_gradients = dense_auxiliary
        report['timing_policy'] = {'read_key_execution': model.read_key_execution,
                                   'auxiliary_final_read': args.reference_dense,
                                   'first_update_includes_compilation': True,
                                   'health_update_snapshot': False}
        optimizer = make_medium_optimizer(model, lr=2e-4, saved_state=saved['optimizer'])
        learner = ActiveMediumTrainer(model, optimizer, initial.detach(),
            carry_token=carry, event_duration=saved['config']['event_duration'],
            chunk_tokens=32, tokens_per_update=32, activation_checkpointing=True,
            optimizer_state_offload=True, checkpoint_granularity='event')
        learner.load_state_dict(saved['learner'])
        # Independent timing processes replay the same structural samples as
        # the saved individual, rather than consuming constructor/JVP randomness.
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        report['timing_rng'] = 'restored checkpoint CPU/CUDA RNG before both windows'
        report['complete_updates'] = []
        prior = np.zeros(model.decoder.out_features, dtype=np.float64)
        for index in range(2):
            targets = torch.from_numpy(np.array(
                train[saved['cursor'] + 32 * index:saved['cursor'] + 32 * (index + 1)],
                dtype=np.int64)).cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            print(f'Production complete update {index + 1}/2', flush=True)
            rows = learner.consume(targets, phase='train_first_pass', prior_nll=prior)
            torch.cuda.synchronize()
            entry = {'window': index + 1, 'tokens': len(rows),
                     'seconds': time.perf_counter() - started,
                     'optimizer_updates': learner.optimizer_updates,
                     'prequential_nll': sum(row[2] for row in rows) / len(rows)}
            report['complete_updates'].append(entry)
            print(json.dumps(entry), flush=True)
            monitor.check()
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        training._writer_auxiliary_gradients = original_aux
        report['memory'] = monitor.finish()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
