"""Check a full native checkpoint through the current trainer without exposure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import ensure_gpu_idle, restore_native, sha256
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
from information_boltzmann.core.fly_mother_world import MotherWorld
from information_boltzmann.core.fly_mb_terminal import MB_TERMINAL_STATE_FIELDS
from information_boltzmann.core.fly_apl_cable import APL_CABLE_STATE_FIELDS
from information_boltzmann.core.fly_kc_contact_memory import KC_CONTACT_STATE_FIELDS
from information_boltzmann.core.fly_kc_dopamine import KC_DOPAMINE_STATE_FIELDS
from information_boltzmann.core.fly_dan_release import DAN_RELEASE_STATE_FIELDS
from information_boltzmann.core.fly_dan_credit import DAN_CREDIT_FIELDS
from information_boltzmann.core.fly_kc_terminals import KC_TERMINAL_STATE_FIELDS
from information_boltzmann.core.fly_apl_local_release import APL_LOCAL_RELEASE_STATE_FIELDS
from information_boltzmann.core.fly_kc_lateral import KC_LATERAL_STATE_FIELDS


def verify(checkpoint: Path, output: Path, native_credit_cpu=None, device='cuda'):
    ensure_gpu_idle([])
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(checkpoint)
    overrides = {'native_credit_cpu': native_credit_cpu} if native_credit_cpu is not None else None
    learner, saved, args, changed = restore_native(checkpoint, output, frozen=False, overrides=overrides, device=device)
    if changed:
        raise ValueError('Full restore check must not migrate the algorithm')
    weights = checkpoint_model_weights(learner.model)
    source_weights = saved['model']
    missing = sorted(set(source_weights) - set(weights))
    extra = sorted(set(weights) - set(source_weights))
    changed_weights = [name for name in source_weights.keys() & weights.keys()
                       if state_digest(source_weights[name]) != state_digest(weights[name])]
    current = learner.state_dict()
    expected = dict(saved['learner'])
    optimizer_backend_changes = []
    if device == 'cpu':
        # Parameter names, moments, steps and all hyperparameters stay under
        # exact comparison. Only the CUDA fused-kernel selector is executable
        # placement metadata, explicitly reported rather than silently ignored.
        groups = []
        for index, group in enumerate(expected['optimizer']['param_groups']):
            actual = current['optimizer']['param_groups'][index]
            updated = dict(group)
            if group.get('fused') is True and actual.get('fused') is False:
                updated['fused'] = False
                optimizer_backend_changes.append({'group': index, 'key': 'fused',
                                                  'archived': True, 'cpu_execution': False})
            groups.append(updated)
        expected['optimizer'] = {**expected['optimizer'], 'param_groups': groups}
    physical = dict(expected['physical'])
    schema_extension = []
    from information_boltzmann.core.fly_native_runtime import state_owners
    for name in state_owners():
        if name not in physical and current['physical'][name].numel() == 0:
            physical[name] = current['physical'][name]
            schema_extension.append(name)
    expected['physical'] = physical
    mismatches = [key for key in saved['learner'] if not (key == 'native_credit_cpu' and overrides)
                  if key not in current or state_digest(current[key]) != state_digest(expected[key])]
    rng_same = (torch.equal(saved['rng_cpu'], torch.get_rng_state())
                and torch.equal(saved['rng_cuda'], torch.cuda.get_rng_state().cpu()))
    world = MotherWorld(SimpleNamespace(usage={}), tiktoken.get_encoding('gpt2'), opening=None)
    world.load_state_dict(saved['mother_world'])
    world_same = state_digest(world.state_dict()) == state_digest(saved['mother_world'])
    mother_rng = np.random.default_rng()
    mother_rng.bit_generator.state = saved['mother_rng']
    mother_rng_same = state_digest(mother_rng.bit_generator.state) == state_digest(saved['mother_rng'])
    unchanged = sha256(checkpoint) == fingerprint
    report = {'checkpoint': str(checkpoint), 'sha256': fingerprint,
        'trained': int(saved['bptt_train_tokens']), 'scope': 'zero-exposure complete trainer restore',
        'storage_only_override': overrides, 'execution_device': device,
        'optimizer_backend_changes': optimizer_backend_changes,
        'empty_physical_schema_fields_added': schema_extension,
        'missing_model_entries': missing, 'extra_model_entries': extra,
        'changed_model_entries': changed_weights, 'changed_learner_entries': mismatches,
        'cpu_cuda_rng_exact': rng_same, 'source_unchanged': unchanged,
        'mother_world_roundtrip_exact': world_same, 'mother_rng_exact': mother_rng_same,
        'mother_world_scope': 'real MotherWorld state restored without new exposure or network call',
        'status': 'exact_restore_passed' if not (missing or extra or changed_weights or mismatches)
                  and rng_same and unchanged and world_same and mother_rng_same else 'not_accepted'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))
    if report['status'] != 'exact_restore_passed':
        raise RuntimeError('Full native restore mismatch; inspect report before continuation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-credit-cpu', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    verify(args.checkpoint, args.output, args.native_credit_cpu, args.device)
