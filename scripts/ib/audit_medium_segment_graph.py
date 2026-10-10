"""Full real-data CUDA microstep graph identity and warm-update acceptance.

Uses a saved individual only as an immutable reference. No checkpoint or training
cursor is committed. Both compared executions start at the identical snapshot.
"""
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

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime import training
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.medium_health import conditional_state_response, MediumHealthAuditor
from information_boltzmann.runtime.optimization import make_medium_optimizer, initialize_hopf_branch
from scripts.ib.train_plastic_conductance import unpack_belief
from scripts.ib.audit_medium_long_interval import DedicatedMemorySampler


def restore_calibration_runtime(model, config, data_dir):
    """Validate the measured stream and restore non-constructor learning controls."""
    data_dir = Path(data_dir)
    observed = {}
    for name, expected in config['data_sha256'].items():
        digest = hashlib.sha256()
        with (data_dir / name).open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        observed[name] = digest.hexdigest()
        if observed[name] != expected:
            raise ValueError(f'Calibration data differs from saved individual: {name}')
    manifest_hash = hashlib.sha256((data_dir / 'manifest.json').read_bytes()).hexdigest()
    if manifest_hash != config['manifest_sha256']:
        raise ValueError('Calibration manifest differs from saved individual')
    dual_rate = config.get('structural_learning', {}).get('dual_learning_rate')
    if dual_rate is not None and (not math.isfinite(dual_rate) or dual_rate < 0):
        raise ValueError('Saved structural dual learning rate is invalid')
    model.structure_dual_learning_rate = dual_rate
    return {'verified_data_sha256': observed, 'verified_manifest_sha256': manifest_hash,
            'structural_dual_learning_rate': dual_rate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--vram-limit-mib', type=int, default=3072)
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--segment-graph', action='store_true')
    parser.add_argument('--shared-credit', action='store_true')
    parser.add_argument('--credit-identity', action='store_true',
                        help='Compare separate and shared credit on identical complete BPTT32 inputs')
    parser.add_argument('--anomaly', action='store_true')
    parser.add_argument('--deferred-writer-credit', action='store_true')
    parser.add_argument('--credit-history-offload', action='store_true')
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--fused-port-forward', action='store_true')
    parser.add_argument('--health', action='store_true', help='Include production fourth-pillar collection/update snapshots')
    parser.add_argument('--updates', type=int, default=4)
    parser.add_argument('--adopt-junction', action='store_true',
                        help='Disposable exact-zero Gen3 migration from the saved Gen2 individual')
    parser.add_argument('--keep-moments-on-gpu', action='store_true',
                        help='Measure persistent GPU Adam moments inside the same memory cap')
    args = parser.parse_args()
    if args.deferred_writer_credit and not args.shared_credit:
        parser.error('Deferred writer credit requires --shared-credit')
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
    if args.credit_identity or args.shared_credit:
        import torch._functorch.config as aot_config
        aot_config.donated_buffer = False
    torch.set_num_threads(2)
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction((args.vram_limit_mib - 192) / total_mib)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    constructor = dict(saved['config']['constructor'])
    if args.adopt_junction:
        constructor['hopf_recomposition'] = True
    constructor['read_key_execution'] = 'support'
    # Native port GEMMs isolate the algebraic comparison from compiler cache
    # specialization; the physical kernel and all 32-event credit stay live.
    constructor['port_execution'] = 'native'
    model = PlasticMediumPorts3D(**constructor).cuda().train()
    provenance = restore_calibration_runtime(model, saved['config'], args.data_dir)
    if args.deferred_writer_credit:
        from information_boltzmann.core.deferred_linear_credit import DeferredWriterCredit
        model.deferred_writer_credit = True
        model.deferred_credit = DeferredWriterCredit(model.write_agent, model.readout)
    model.shared_credit_execution = args.shared_credit
    model.credit_history_offload = args.credit_history_offload
    model.source.embedding.weight.requires_grad_(False)
    optimizer_state = saved['optimizer']
    if args.adopt_junction:
        optimizer_state = initialize_hopf_branch(model, saved['model'], optimizer_state, saved['learner'])
    else:
        model.load_state_dict(saved['model'], strict=True)
    initial = unpack_belief(saved['belief'], 'cuda')
    posterior = model.medium.structural_posterior
    model._structural_window_events = 32
    noise = torch.zeros_like(posterior.mean)
    train = np.load(args.data_dir / 'train.npy', mmap_mode='r')
    target = torch.from_numpy(np.array(train[saved['cursor']:saved['cursor'] + 32], dtype=np.int64)).cuda()
    carry = saved['learner']['carry_token']
    observed = torch.cat((target.new_tensor([carry]), target[:-1]))[None]
    monitor = DedicatedMemorySampler(args.vram_limit_mib * 2**20)
    monitor.thread.start()
    report = {'scope': 'actual D7688^3 real OWT, full32 credit, execution only',
              'checkpoint': str(args.checkpoint), 'checkpoint_step': saved['step'],
              'checkpoint_cursor': saved['cursor'],
              'production_checkpoint_written': False, 'runs': [],
              'junction_enabled': args.adopt_junction,
              'junction': None if model.hopf_pathway is None else model.hopf_pathway.descriptions(),
              'source_data_sha256': saved['config']['data_sha256'],
              'runtime_restoration': provenance}
    reference = None
    try:
        modes = (() if args.timing_only else
                 (('separate', 'shared') if args.credit_identity else ('fused', 'segment')))
        for key_execution in modes:
            model.medium.segment_graph_enabled = args.segment_graph if args.credit_identity else key_execution == 'segment'
            model.shared_credit_execution = key_execution == 'shared' if args.credit_identity else args.shared_credit
            if args.deferred_writer_credit:
                model.deferred_writer_credit = model.shared_credit_execution
                model.deferred_credit.enabled = model.shared_credit_execution
            model.port_execution = ('fused' if args.fused_port_forward
                                    and key_execution == 'shared' else 'native')
            model.readout.compact_key_execution = 'support'
            model.read_key_execution = 'support'
            model.zero_grad(set_to_none=True)
            posterior.window_active.fill_(False)
            posterior.begin_window(noise)
            belief = initial.detach()
            for value in training.belief_tensors(belief):
                if value.is_floating_point() or value.is_complex():
                    value.requires_grad_(True)
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
            model.flush_deferred_credit()
            torch.cuda.synchronize()
            values = {'loss': loss.detach().cpu(), 'nll': nll.detach().cpu(),
                      'token_nll': token_nll.detach().cpu(),
                      'states': [x.detach().cpu().clone() for x in training.belief_tensors(final)],
                      'input_gradients': [None if x.grad is None else x.grad.detach().cpu().clone()
                                          for x in training.belief_tensors(belief)],
                      'parameter_gradients': {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                              for name, p in model.named_parameters()}}
            row = {'physical_execution': key_execution,
                   'port_execution': model.port_execution,
                   'auxiliary_final_read': False,
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
                    assert (actual is None) == (expected is None), (
                        family, name, 'actual_none', actual is None, 'expected_none', expected is None)
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
        model.shared_credit_execution = args.shared_credit
        if args.deferred_writer_credit:
            model.deferred_writer_credit = True
            model.deferred_credit.enabled = True
        # Native complete-state JVP must work after compiled physical training;
        # diagnostic execution is observational and never mutates this individual.
        model.port_execution = 'fused'
        if args.deferred_writer_credit and not args.fused_port_forward:
            model.port_execution = 'native'
        if not args.timing_only:
            print('Complete-state forward AD diagnostic with fused ports enabled', flush=True)
            report['conditional_response'] = conditional_state_response(
                model, initial.detach(), observed[:, 0],
                event_duration=saved['config']['event_duration'], directions=1)
        model.readout.compact_key_execution = 'support'
        model.read_key_execution = model.readout.compact_key_execution
        model.medium.segment_graph_enabled = args.segment_graph
        report['timing_policy'] = {'read_key_execution': model.read_key_execution,
                                   'auxiliary_final_read': False,
                                   'first_update_includes_compilation': True,
                                   'health_update_snapshot': args.health, 'segment_graph': args.segment_graph,
                                   'optimizer_state_offload': not args.keep_moments_on_gpu}
        report['timing_policy']['shared_credit'] = args.shared_credit
        report['timing_policy']['deferred_writer_credit'] = args.deferred_writer_credit
        report['timing_policy']['port_execution'] = model.port_execution
        report['timing_policy']['credit_history_offload'] = args.credit_history_offload
        optimizer = make_medium_optimizer(model, lr=2e-4, saved_state=optimizer_state,
                                          fused=args.fused_optimizer)
        report['timing_policy']['fused_optimizer'] = args.fused_optimizer
        learner = ActiveMediumTrainer(model, optimizer, initial.detach(),
            carry_token=carry, event_duration=saved['config']['event_duration'],
            chunk_tokens=32, tokens_per_update=32, activation_checkpointing=True,
            optimizer_state_offload=not args.keep_moments_on_gpu, checkpoint_granularity='event')
        if args.health:
            learner.health = MediumHealthAuditor(model, window_tokens=256, block_tokens=32)
            from information_boltzmann.runtime.medium_health import ChunkHealthCapture
            learner.health_capture = ChunkHealthCapture(model, 32)
        learner.load_state_dict(saved['learner'])
        # Independent timing processes replay the same structural samples as
        # the saved individual, rather than consuming constructor/JVP randomness.
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        report['timing_rng'] = 'restored checkpoint CPU/CUDA RNG before both windows'
        report['complete_updates'] = []
        from scripts.ib.train_medium_active_stream import train_only_reference
        prior, prior_hash = train_only_reference(args.data_dir / 'train.npy', model.decoder.out_features)
        if prior_hash != saved['config']['prior_counts_sha256']:
            raise ValueError('Calibration unigram reference differs from saved individual')
        report['runtime_restoration']['verified_prior_counts_sha256'] = prior_hash
        for index in range(args.updates):
            targets = torch.from_numpy(np.array(
                train[saved['cursor'] + 32 * index:saved['cursor'] + 32 * (index + 1)],
                dtype=np.int64)).cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            print(f'Production complete update {index + 1}/{args.updates}', flush=True)
            rows = learner.consume(targets, phase='train_first_pass', prior_nll=prior)
            torch.cuda.synchronize()
            entry = {'window': index + 1, 'tokens': len(rows),
                     'seconds': time.perf_counter() - started,
                     'optimizer_updates': learner.optimizer_updates,
                     'prequential_nll': sum(row[2] for row in rows) / len(rows)}
            report['complete_updates'].append(entry)
            print(json.dumps(entry), flush=True)
            monitor.check()
        report['segment_replays'] = getattr(getattr(model.medium, '_segment_advance', None), 'replays', 0)
        report['rhs_segment_replays'] = getattr(getattr(model.medium, '_segment_rhs', None), 'replays', 0)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['memory'] = monitor.finish()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
