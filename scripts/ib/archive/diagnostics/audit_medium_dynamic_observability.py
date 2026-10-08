"""Re-run the original hidden-flux diagnostic on explicit dynamic read arms."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.runtime.optimization import initialize_dynamic_read_branch
from information_boltzmann.runtime.training import belief_tensors
from scripts.ib.audit_medium_dynamic_read import digest
from scripts.ib.audit_medium_time_read_credit import read_observability, relative
from scripts.ib.audit_medium_read_closure import first_order_response, spatial_mask
from scripts.ib.train_plastic_conductance import unpack_belief


def motion_measurements(model, belief):
    recorded = []
    handle = model.readout.motion_merge.register_forward_pre_hook(
        lambda _module, values: recorded.append(values[0]))
    try:
        feature, _ = model.read(belief, decode=False)
    finally:
        handle.remove()
    reader = model.readout
    width = reader.queries * reader.d
    ordered = [part.reshape(feature.shape[0], reader.queries, reader.heads,
                            reader.head_dim).transpose(1, 2)
               for part in (recorded[0][:, :width], recorded[0][:, width:])]
    return feature, torch.cat(ordered, dim=-1)


def raw_and_remote_checks(model, belief):
    base_feature, raw = motion_measurements(model, belief)
    rows = []
    for scale in (.99, 1.01):
        changed = PlasticBelief(replace(belief.medium,
            flux=tuple(x * scale for x in belief.medium.flux)), belief.precision)
        feature, actual = motion_measurements(model, changed)
        per_probe = (actual - raw).norm(dim=-1) / raw.norm(dim=-1).clamp_min(1e-12)
        rows.append(dict(scale=scale, per_probe_relative_change=per_probe[0].tolist(),
                         responding_probes=int(((actual - raw).abs().amax(-1) > 0).sum()),
                         feature_relative_change=relative(feature, base_feature)))
    supports = model.readout.footprint().reshape(
        model.readout.heads * model.readout.queries, *model.medium.shape) > 0
    probe = min(range(len(supports)), key=lambda i: int(supports[i].sum()))
    sites = supports[probe]
    altered_flux = tuple(value * torch.where(
        spatial_mask(('flux_x', 'flux_y', 'flux_z')[axis], sites, value),
        value.new_tensor(1.), value.new_tensor(1.01))
        for axis, value in enumerate(belief.medium.flux))
    remote_perturbation_norm = float(sum((left - right).square().sum()
        for left, right in zip(altered_flux, belief.medium.flux)).sqrt())
    assert remote_perturbation_norm > 0.
    _, altered_raw = motion_measurements(model, PlasticBelief(
        replace(belief.medium, flux=altered_flux), belief.precision))
    remote_change = float((altered_raw.flatten(1, 2)[:, probe]
                          - raw.flatten(1, 2)[:, probe]).abs().max())
    assert remote_change == 0.
    return dict(perturbations=rows, remote_flux_control=dict(
        probe=probe, held_aperture_sites=int(sites.sum()),
        held_local_and_incident_edges=True, external_flux_perturbation_norm=remote_perturbation_norm,
        raw_probe_max_change=remote_change))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    constructor = dict(saved['config']['constructor'])
    constructor.update(medium_execution='native', port_execution='native')
    models = {'old_instantaneous': PlasticMediumPorts3D(**constructor).train()}
    models['old_instantaneous'].load_state_dict(saved['model'])
    for name in ('zero_migrated', 'active_fanin_711'):
        model = PlasticMediumPorts3D(**constructor, read_mode='dynamic').train()
        initialize_dynamic_read_branch(model, saved['model'])
        if name == 'active_fanin_711':
            torch.manual_seed(711)
            for layer in ('motion_policy', 'motion_keys', 'motion_merge'):
                getattr(model.readout, layer).reset_parameters()
        models[name] = model
        for key, weight in saved['model'].items():
            assert torch.equal(model.state_dict()[key], weight), key
    initial_hashes = {name: digest(model) for name, model in models.items()}
    duration = saved['config']['event_duration']
    belief = unpack_belief(saved['belief'], 'cpu')
    corpus = np.load('data/ib_owt_gpt2/train.npy', mmap_mode='r')
    targets = torch.tensor(np.array(corpus[saved['cursor']:saved['cursor'] + 32], dtype=np.int64))
    ids = torch.cat((targets.new_tensor([saved['learner']['carry_token']]), targets[:-1]))
    old = models['old_instantaneous']
    samples = []
    with torch.no_grad():
        table = F.normalize(old.source.embedding.weight, dim=-1)
    for event, token in enumerate(ids):
        with torch.no_grad():
            written, _ = old.assimilate(belief, token.reshape(1), token_features=table,
                                       diagnostics=False, training_terms=False)
        if event in (0, 8, 16, 24):
            print(f'Same diagnostic retest: real event {event}', flush=True)
            results = {name: read_observability(model, written, int(targets[event]), duration)
                       for name, model in models.items()}
            with torch.no_grad():
                reference, _ = old.read(written, decode=False)
                migrated, _ = models['zero_migrated'].read(written, decode=False)
                assert torch.equal(reference, migrated)
                raw = {name: raw_and_remote_checks(model, written)
                       for name, model in models.items() if model.read_mode == 'dynamic'}
                old_future, _ = old.advance(written, duration, diagnostics=False)
                for model in models.values():
                    future, _ = model.advance(written, duration, diagnostics=False)
                    for left, right in zip(belief_tensors(old_future), belief_tensors(future)):
                        assert torch.equal(left, right)
            samples.append(dict(event=event, observed=int(token), target=int(targets[event]),
                original_diagnostic=results, raw_motion_measurement=raw,
                zero_migration_exact=True,
                unchanged_physical_jet=first_order_response(old, written, duration)))
        with torch.no_grad():
            belief, _ = old.advance(written, duration, diagnostics=False)
    summary = {}
    for name in models:
        rows = [row for sample in samples
                for row in sample['original_diagnostic'][name]['flux_perturbations']]
        partials = [sample['original_diagnostic'][name]['state_credit'][key]['current']['norm']
                    for sample in samples for key in ('flux_x', 'flux_y', 'flux_z')]
        summary[name] = dict(
            immediate_feature_max_abs_change_mean=float(np.mean([r['current_feature_max_difference'] for r in rows])),
            immediate_feature_max_abs_change_max=float(max(r['current_feature_max_difference'] for r in rows)),
            immediate_flux_task_gradient_norm_mean=float(np.mean(partials)),
            immediate_flux_task_gradient_norm_max=float(max(partials)))
    assert summary['old_instantaneous']['immediate_feature_max_abs_change_max'] == 0
    assert summary['zero_migrated']['immediate_feature_max_abs_change_max'] == 0
    assert summary['active_fanin_711']['immediate_feature_max_abs_change_mean'] > 0
    report = dict(purpose='Retest a diagnosed structural blind direction, not task improvement',
        original_tool='scripts/ib/audit_medium_time_read_credit.py::read_observability',
        checkpoint=str(args.checkpoint.resolve()), checkpoint_step=saved['step'],
        corpus_offset=saved['cursor'], actual_forward_events=32, selected_events=[0, 8, 16, 24],
        optimizer_updates=0, active_initialization='seed711 standard fan-in for ONLY 3 new additive paths',
        original_weights_equal=True, selected_physical_transitions_equal=True,
        parameters_unchanged={name: initial_hashes[name] == digest(model) for name, model in models.items()},
        summary=summary, samples=samples, elapsed_seconds=time.perf_counter() - started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preregistration='docs/information_boltzmann/MEDIUM_DYNAMIC_OBSERVABILITY_RETEST_20261008.md')
    assert all(report['parameters_unchanged'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(summary=summary, elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
