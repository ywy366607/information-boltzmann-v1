"""Compile existing native contact families and verify a full individual.

No new experience or molecular identity is invented. The optional archive is a
lossless, zero-exposure execution migration, not a trained capability result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import (
    restore_native, pending_queue_tape, clone_state, sha256,
)
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from scripts.ib.fly_interface.native_system_matrix import deterministic_replay
from information_boltzmann.core.fly_native_generative import _preserve_rollout
from information_boltzmann.core.fly_native_runtime import (
    attach_program_document, validate_program_bindings, runtime_inventory,
)
from information_boltzmann.core.fly_native_templates import compile_existing_kc_routes
from information_boltzmann.core import fly_bptt_learning as fl


def run(checkpoint: Path, output: Path, *, migrate=False):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'last.pt'
    if migrate and destination.exists():
        raise FileExistsError('Refuse to overwrite an existing complete individual')
    source_sha = sha256(checkpoint)
    learner, saved, args, changed = restore_native(checkpoint, output, frozen=True, device='cpu')
    if changed:
        raise ValueError('An execution compiler must not migrate the learning algorithm')
    model = learner.model
    document = compile_existing_kc_routes(model)
    programme = attach_program_document(model, document)
    validate_program_bindings(model)
    if programme.layout.size or document['materials']:
        raise ValueError('This lossless compiler must add no physical state or parameters')
    tape = pending_queue_tape(saved, 1, 1)[0]
    job = saved['learner'].get('native_context_state', {}).get('pending_teaching') or {}
    source = job.get('source', tape['forced']['sources'][0])
    duration = tape['forced']['intervals'][0]
    initial_digest = state_digest(learner.state.state_dict())
    weights_digest = state_digest(model.state_dict())
    initial_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state().clone()
    eyes = None
    results = {}
    with deterministic_replay():
        for label, enabled in (('legacy', False), ('sham', False), ('compiled', True), ('restored', False)):
            model.native_program_enabled = enabled
            started = time.perf_counter()
            with _preserve_rollout(model), torch.no_grad():
                current = clone_state(learner.state)
                current.native_state = programme.layout.initialize(device=current.h.device)
                current = fl.prepare_sensory_expectation(model, current, learner.previous_token, source)
                out = fl.advance_fly_token_stream(model, current,
                    torch.tensor([int(learner.previous_token)], device=current.h.device),
                    [duration], ticks=duration, hold_input=learner.hold_input,
                    writer_baseline_clock=learner.writer_baseline_clock,
                    fresh=True, sensory_on=True, voice=tape['last_voice'],
                    executed_action=source == 'self_hearing',
                    reward_ticks=fl.reward_ticks_left(model, tape['last_reward'], 0),
                    ai_g=learner.ai_g.detach().clone(), ai_tape=[False] * duration,
                    read_every=True, eye_forced=eyes, update_expectation=False)
                if eyes is None:
                    eyes = out[-4].detach().cpu().reshape(-1).tolist()
                current = fl.expect_next(model, out[0], out[1])
                results[label] = {
                    'physical': {name: state_digest(value) for name, value in current.state_dict().items()},
                    'muscles': state_digest(out[-1]),
                    'logits': state_digest(model.decoder(model.read_norm(out[1]))),
                    'elapsed_seconds': time.perf_counter() - started,
                }
                del current, out
            print(json.dumps({'arm': label, 'ticks': duration,
                              'seconds': results[label]['elapsed_seconds']}), flush=True)
            gc.collect()
    model.native_program_enabled = True
    comparisons = {}
    for label in ('sham', 'compiled', 'restored'):
        comparisons[label] = {
            'different_physical_fields': [name for name, value in results['legacy']['physical'].items()
                                         if results[label]['physical'][name] != value],
            'muscles_exact': results[label]['muscles'] == results['legacy']['muscles'],
            'logits_exact': results[label]['logits'] == results['legacy']['logits'],
        }
    source_unchanged = (initial_digest == state_digest(learner.state.state_dict())
                        and weights_digest == state_digest(model.state_dict())
                        and torch.equal(initial_rng, torch.get_rng_state())
                        and torch.equal(initial_cuda_rng, torch.cuda.get_rng_state())
                        and source_sha == sha256(checkpoint))
    passed = source_unchanged and all(not row['different_physical_fields']
        and row['muscles_exact'] and row['logits_exact'] for row in comparisons.values())
    report = {'at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(checkpoint), 'source_sha256': source_sha,
        'status': 'exact_delivery_passed' if passed else 'not_accepted',
        'trained': int(saved['bptt_train_tokens']), 'new_training_inputs': 0,
        'clock_source': tape['scope'], 'intervals': [duration], 'ticks': duration,
        'word': int(learner.previous_token), 'target': tape['targets'][0], 'source': source,
        'comparisons': comparisons, 'source_state_parameters_rng_preserved': source_unchanged,
        'arm_seconds': {name: row['elapsed_seconds'] for name, row in results.items()},
        'execution_device': 'cpu', 'cpu_threads': torch.get_num_threads(),
        'new_parameters': 0, 'new_physical_state_elements': 0,
        'compiled_instructions': len(programme.instructions),
        'compiled_routes': document['owned_routes'],
        'biological_completeness_verified': False,
        'sensory_return_edges_activated': 0,
        'scope': 'Exact lowering of existing native KC delivery; no new biology or learning/exploration benefit claim',
        'runtime_inventory': runtime_inventory(model, learner.state)}
    (output / 'program.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not passed:
        raise RuntimeError('Compiled delivery diverges; do not publish a candidate')
    if migrate:
        if shutil.disk_usage(output).free < 2 * checkpoint.stat().st_size + 1024 ** 3:
            raise OSError('Insufficient complete-archive and atomic-replacement reserve')
        # Reuse the ORIGINAL mmap archive, never the diagnostic learner whose
        # optimizers were cleared for frozen replay. Every old value persists.
        payload = {**saved, 'model': {**saved['model'],
            'native_program_digest': model.native_program_digest.detach().cpu()},
            'learner': {**saved['learner'], 'native_runtime_document': document,
                'physical': {**saved['learner']['physical'], 'native_state': torch.empty(0)}},
            'config': {**saved['config'], 'native_runtime_program': str(output / 'program.json')}}
        from scripts.ib.train_fly_bptt_stream import atomic_save
        atomic_save(destination, payload)
        migrated = torch.load(destination, map_location='cpu', mmap=True, weights_only=False)
        unchanged_top = [key for key in saved if key not in ('model', 'learner', 'config')]
        unequal = [key for key in unchanged_top if state_digest(saved[key]) != state_digest(migrated[key])]
        for section in ('model', 'learner', 'config'):
            for key, value in saved[section].items():
                actual = migrated[section][key]
                if section == 'learner' and key == 'physical':
                    actual = {name: actual[name] for name in value}
                if state_digest(value) != state_digest(actual):
                    unequal.append(section + '.' + key)
        report['candidate'] = {'path': str(destination), 'sha256': sha256(destination),
            'bytes': destination.stat().st_size, 'old_fields_changed': unequal,
            'full_trainer_restore': 'pending_separate_check', 'new_training_inputs': 0}
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if unequal:
            raise RuntimeError('Source archive field changed during execution migration')
    print(json.dumps({key: report[key] for key in ('status', 'ticks', 'compiled_routes', 'new_parameters')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--migrate', action='store_true')
    options = parser.parse_args()
    run(options.checkpoint, options.output, migrate=options.migrate)
