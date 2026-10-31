"""Actual-state APL subsystem delivery; NOT a whole-brain clock replay.

The archived afferent boundary is held fixed for a few local operator ticks.
This isolates calcium/resource causality without claiming future afferents,
KC dynamics, action changes, training admission or behavioral benefit.
"""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import restore_native, sha256
from scripts.ib.fly_interface.native_system_matrix import intervene
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from information_boltzmann.core.fly_apl_cable import advance_apl_cable
from information_boltzmann.core.fly_apl_local_release import APL_LOCAL_RELEASE_STATE_FIELDS
from information_boltzmann.core.fly_native_generative import _preserve_rollout


def sparse_kc_delivery(model, packet):
    """The existing receptor routing/reduction at its specified future lag.

    Return sparse target vectors, without allocating a source-by-4M-node field.
    Neither this readout nor its weighted sums advance the rest of the brain.
    """
    result = {}
    for lag in range(1, 5):
        prefix = f'kct_receptor_{lag}'
        node = getattr(model, prefix + '_pre_node')
        valid = node >= 0
        values = packet.new_zeros(node.numel())
        values[valid] = packet[node[valid]]
        values = values * getattr(model, prefix + '_fraction') * getattr(model, prefix + '_gaba')
        result[lag] = torch.segment_reduce(values[getattr(model, prefix + '_order')], 'sum',
                                          lengths=getattr(model, prefix + '_count'))
    return result


def audit(checkpoint, output, ticks=3):
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(checkpoint)
    learner, saved, _, changed = restore_native(checkpoint, output, frozen=True, device='cpu')
    if changed or not getattr(learner.model, 'apl_local_release_enabled', False):
        raise ValueError('Require the already migrated enabled candidate')
    model, start = learner.model, learner.state
    del learner, saved
    gc.collect()
    before = state_digest(start.state_dict())
    parameter_before = state_digest(model.state_dict())
    results, baseline = {}, None
    for arm in ('intact', 'sham', 'apl_calcium_entry', 'apl_release_resources', 'restored'):
        # replace() below creates only new local coordinates, never mutating the
        # archived incoming spike/KC packet boundary or unrelated brain states.
        current = start
        with _preserve_rollout(model), intervene(model, arm), torch.no_grad():
            for _ in range(ticks):
                step = advance_apl_cable(model, current, None)
                current = replace(current, apl_voltage=step[1], apl_ge=step[2], apl_gi=step[3],
                                  apl_packets=step[4],
                                  **dict(zip(APL_LOCAL_RELEASE_STATE_FIELDS, step[5:])))
            packet = current.apl_packets[0].clone()
            delivery = sparse_kc_delivery(model, packet)
        snapshot = (packet, delivery, current.apl_release_calcium, current.apl_release_resources)
        if baseline is None:
            baseline = snapshot
        results[arm] = {
            'calcium_max': float(snapshot[2].max()), 'resources_min': float(snapshot[3].min()),
            'release_packet_absmax_delta': float((packet - baseline[0]).abs().max()),
            'kc_contact_input_absmax_delta_by_lag': {
                str(lag): float((value - baseline[1][lag]).abs().max()) if value.numel() else 0.
                for lag, value in delivery.items()},
            'kc_contact_input_nonzero_targets_by_lag': {
                str(lag): int(torch.count_nonzero(value)) for lag, value in delivery.items()},
        }
    controls = all(results[a]['release_packet_absmax_delta'] == 0 and
                   all(v == 0 for v in results[a]['kc_contact_input_absmax_delta_by_lag'].values())
                   for a in ('sham', 'restored'))
    untouched = before == state_digest(start.state_dict()) and parameter_before == state_digest(model.state_dict())
    report = {'checkpoint': str(checkpoint), 'sha256': fingerprint,
              'scope': 'actual-state isolated APL subsystem; fixed archived afferents, not whole-brain real-clock replay',
              'operator_ticks': ticks, 'execution_device': 'cpu', 'arms': results,
              'controls_passed': controls, 'original_state_and_parameters_unchanged': untouched,
              'source_unchanged': fingerprint == sha256(checkpoint),
              'whole_brain_causal_replay': 'unaccepted: separate full replay exhausted host memory',
              'learning_or_action_benefit': 'unmeasured',
              'status': 'subsystem_controls_passed' if controls and untouched else 'not_accepted'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not controls or not untouched:
        raise RuntimeError('APL subsystem controls failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=3)
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error('Use positive operator ticks')
    audit(args.checkpoint, args.output, args.ticks)
