"""Read-only CPU audit of an existing RULER life, not a training launcher.

Checks actual saved local participation and its production read program.
Functional delivery and strict old-source attribution are separate obligations;
passing serialization tests never supplies either of those obligations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    torch.set_num_threads(1)
    run = json.loads(args.training_report.read_text(encoding='utf-8'))
    original = Path(run['source'])
    before = {str(p): sha256(p) for p in (args.checkpoint, original)}
    owner = load_readonly_checkpoint(args.checkpoint)
    state = owner['learner']
    source = load_readonly_checkpoint(original)['learner']
    driver = owner['branch_driver']
    native_same = all(torch.equal(value, source['native'][name])
                      for name, value in state['native'].items())
    group = source['optimizer']['param_groups'][0]['params']
    adam_same = True
    for identity in group:
        old = source['optimizer']['state'].get(identity, {})
        new = state['optimizer']['state'].get(identity, {})
        adam_same &= old.keys() == new.keys()
        for key in old:
            adam_same &= (torch.equal(old[key], new[key]) if torch.is_tensor(old[key])
                          else old[key] == new[key])
    layers = {}
    invariants = []
    for layer, stored in state['memories'].items():
        cfg, extra = stored['memory._extra_state'], stored['_extra_state']
        capsules = extra['capsules']
        occupied = stored['memory.occupied']
        versions = stored['memory.read_credit_version']
        birth = stored['memory.birth']
        clock = int(stored['memory.clock'])
        records = []
        for slot in occupied.nonzero().flatten().tolist():
            capsule = capsules[slot]
            if capsule is None:
                invariants.append(f'layer{layer+1} slot{slot}: missing occupied capsule')
                continue
            prefix = capsule.inputs.shape[1]
            tail = getattr(capsule, 'continuation', None)
            tail_length = 0 if tail is None else tail.shape[1]
            finite = bool(torch.isfinite(capsule.inputs).all())
            if tail is not None:
                finite &= bool(torch.isfinite(tail).all())
            same_version = capsule.version == int(versions[slot])
            if not finite or not same_version or prefix != extra['replay_tokens']:
                invariants.append(f'layer{layer+1} slot{slot}: nonfinite/version/length failure')
            records.append(dict(slot=slot, birth=int(birth[slot]), age=clock-int(birth[slot]),
                stored_inputs=prefix+tail_length, default_replayed_inputs=prefix,
                stored_but_not_replayed=tail_length, version_matches=same_version,
                finite=finite, has_joint_archive=getattr(capsule, 'joint', None) is not None))
        changes = {}
        for name in ('encoder.weight', 'encoder.bias', 'inlet.projection.weight', 'null_key', 'null_bias'):
            current, previous = stored[name], source['memories'][layer][name]
            difference = current.float() - previous.float()
            changes[name] = dict(changed_elements=int((difference != 0).sum()),
                                 delta_norm=float(difference.norm()))
        layers[str(layer+1)] = dict(capacity=cfg['capacity'], latent_width=stored['encoder.weight'].shape[0],
            occupied=len(records), capture_full_window=extra.get('capture_full_window', False),
            read_interval=extra.get('read_interval'), placement=extra.get('recall_placement', 'input'),
            null_enabled=extra.get('null_read_enabled', False), retention_rule=cfg['consolidation_rule'],
            records=records, parameter_changes=changes,
            remaining_default_replay_tail_tokens=sum(r['stored_but_not_replayed'] for r in records))
    source_preserved = all(sha256(p) == value for p, value in before.items())
    # Labels reflect checks actually executed, not hoped-for learning outcomes.
    result = dict(status='blocked_before_training', checkpoint=str(args.checkpoint),
        checkpoint_sha256=before[str(args.checkpoint)], original_source=str(original),
        original_source_sha256=before[str(original)], source_preserved=source_preserved,
        completed_at_checkpoint=driver['completed'], logged_examples=len(run['steps']),
        native_parameters_held=native_same, native_adam_history_held=bool(adam_same),
        cumulative_updates=state['updates'], adaptation_updates=state['updates']-source['updates'],
        seen=state['cache']['seen_tokens'], pending_inputs=len(owner['stream']['inputs']),
        native_state_layers=len(state['cache']['states']), layers=layers,
        serialization_identity_errors=invariants, cuda_initialized=torch.cuda.is_initialized(),
        body_updates=0, training_ready=False,
        unresolved=[
            'Production replay consumes capsule.inputs only; stored continuation is not executed.',
            'Whole-window fact presence is not a label for the executed65-input value.',
            'Complete128 end-response and single-layer query were previously tested without task delivery; length alone is not a repair.',
            'Natural addressing plus local delivery has not passed frozen unseen-question functional checks.',
            'This adaptation explicitly holds native parameters and disables native-S qualification; it is not a delayed-credit comparison.',
            'Native-S raw/cut currently does not resolve source ages; strictly>128-only intervention remains unimplemented.',
        ])
    if not source_preserved or invariants or not native_same or not adam_same or torch.cuda.is_initialized():
        result['status'] = 'invariant_failure'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
