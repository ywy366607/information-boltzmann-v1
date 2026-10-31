"""Zero-exposure, full-state activation of the native OA/5-HT chemical route.

Only an archive without amine parameters may use this migration. It preserves
all previous parameters, named optimizer moments, state, world and RNG. This
is a mechanism migration, never evidence of a learning or behavioural benefit.
"""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import restore_native, ensure_gpu_idle, sha256
from scripts.ib.fly_interface.action_loop_v2_audit import cpu_copy, state_digest
from information_boltzmann.core.fly_amine_modulation import AMINE_PARAMETER_NAMES, AMINE_KINETICS_VERSION
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
from scripts.ib.train_fly_bptt_stream import atomic_save


def named_optimizer(state):
    return {name: state_digest(state['state'].get(index))
            for group in state['param_groups']
            for name, index in zip(group['parameter_names'], group['params'])}


def verify_saved_activation(checkpoint: Path, output: Path):
    """Independent CPU/mmap verification, including recovery after a report OOM."""
    fingerprint = sha256(checkpoint)
    destination_hash = sha256(output/'last.pt')
    old = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    new = torch.load(output/'last.pt', map_location='cpu', mmap=True, weights_only=False)
    modified_top = [k for k in old if k not in ('model', 'learner', 'config')
                    and state_digest(old[k]) != state_digest(new[k])]
    modified_weights = [k for k in old['model'] if state_digest(old['model'][k]) != state_digest(new['model'].get(k))]
    modified_state = [k for k in old['learner'] if k not in ('optimizer', 'adam_names')
                      and state_digest(old['learner'][k]) != state_digest(new['learner'].get(k))]
    a, b = named_optimizer(old['learner']['optimizer']), named_optimizer(new['learner']['optimizer'])
    modified_moments = [k for k in a if a[k] != b.get(k)]
    config_changes = {k for k in old['config'] if old['config'][k] != new['config'].get(k)}
    additions = set(new['model']) - set(old['model'])
    if (modified_top or modified_weights or modified_state or modified_moments
            or config_changes - {'amine_modulation', 'amine_kinetics_version'}
            or additions != set(AMINE_PARAMETER_NAMES)
            or new['config'].get('amine_kinetics_version') != AMINE_KINETICS_VERSION
            or set(new['learner']['adam_names']) - set(old['learner']['adam_names']) != set(AMINE_PARAMETER_NAMES)):
        raise RuntimeError('Saved activation changed previous full state')
    report = {'at': datetime.now(timezone.utc).isoformat(), 'source': str(checkpoint),
              'source_sha256': fingerprint, 'checkpoint': str(output/'last.pt'),
              'sha256': destination_hash, 'trained': int(new['bptt_train_tokens']),
              'amine_kinetics_version': AMINE_KINETICS_VERSION,
              'new_parameters': sorted(additions), 'modified_previous_parameters': modified_weights,
              'modified_previous_learner_fields': modified_state,
              'modified_previous_adam_moments': modified_moments,
              'modified_previous_top_level_fields': modified_top,
              'scope': 'independent zero-exposure complete-payload equivalence',
              'status': 'exact_activation_passed'}
    (output/'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


def migrate(checkpoint: Path, output: Path):
    ensure_gpu_idle([])
    if output.exists():
        raise ValueError('Migration requires a new isolated output directory')
    if shutil.disk_usage(output.parent).free < 2 * checkpoint.stat().st_size + 512 * 2**20:
        raise OSError('Insufficient room for full state and the next atomic replacement')
    fingerprint = sha256(checkpoint)
    learner, saved, args, changed = restore_native(checkpoint, output, frozen=False,
                                                   overrides={'amine_modulation': True})
    if saved['config'].get('amine_modulation') or any(k in saved['model'] for k in AMINE_PARAMETER_NAMES):
        raise ValueError('Only previously absent amine routes can be activated here')
    weights = checkpoint_model_weights(learner.model)
    modified_weights = [name for name in saved['model']
                        if state_digest(weights[name]) != state_digest(saved['model'][name])]
    extra_weights = sorted(set(weights) - set(saved['model']))
    current = learner.state_dict()
    modified_state = [name for name in saved['learner'] if name not in ('optimizer', 'adam_names')
                      and state_digest(current[name]) != state_digest(saved['learner'][name])]
    old_optimizer = named_optimizer(saved['learner']['optimizer'])
    new_optimizer = named_optimizer(current['optimizer'])
    modified_moments = [name for name, value in old_optimizer.items() if new_optimizer.get(name) != value]
    extra_names = set(current['adam_names']) - set(saved['learner']['adam_names'])
    rng_exact = (torch.equal(torch.get_rng_state(), saved['rng_cpu'])
                 and torch.equal(torch.cuda.get_rng_state().cpu(), saved['rng_cuda']))
    if (not changed or modified_weights or modified_state or modified_moments or not rng_exact
            or set(extra_weights) != set(AMINE_PARAMETER_NAMES) or extra_names != set(AMINE_PARAMETER_NAMES)):
        raise RuntimeError('Migration changed existing state; refuse to save')
    model = learner.model
    contacts = {suffix: int((model.amine_channel == channel).sum())
                for channel, suffix in enumerate(('oa', '5ht'))}
    report = {'at': datetime.now(timezone.utc).isoformat(), 'source': str(checkpoint),
              'source_sha256': fingerprint, 'trained': int(saved['bptt_train_tokens']),
              'amine_kinetics_version': AMINE_KINETICS_VERSION,
              'new_parameters': extra_weights, 'modified_previous_parameters': modified_weights,
              'modified_previous_learner_fields': modified_state,
              'modified_previous_adam_moments': modified_moments, 'cpu_cuda_rng_exact': rng_exact,
              'contacts': contacts, 'source_cells': int(model.amine_source_mask.sum()),
              'response_initialization': 'signed zero, to learn through existing neural dynamics',
              'scope': 'zero-exposure activation, no behavioural or learning benefit claim'}
    # Do not reinterpret unseen queued dialogue as exposure or settle pending
    # credit during migration. Every old top-level payload is carried forward.
    payload = {**saved, 'model': cpu_copy(weights), 'learner': cpu_copy(current),
               'config': {**saved['config'], 'amine_modulation': True,
                          'amine_kinetics_version': AMINE_KINETICS_VERSION}}
    output.mkdir(parents=True)
    (output/'migration_pending.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    atomic_save(output/'last.pt', payload)
    # Full CPU exports coexist with GPU model/optimizer state during save.
    # Release them before allocating hash buffers on this 16GB Windows host.
    del payload, learner, saved, weights, current, model
    gc.collect()
    torch.cuda.empty_cache()
    verify_saved_activation(checkpoint, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-saved', action='store_true')
    args = parser.parse_args()
    if args.verify_saved:
        verify_saved_activation(args.checkpoint, args.output)
    else:
        migrate(args.checkpoint, args.output)
