"""Distinguish existing dynamic-state observability from extra history, without fitting."""
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
from scripts.ib.audit_medium_time_read_credit import components, differentiable, relative
from scripts.ib.train_plastic_conductance import unpack_belief


def measure(model, belief):
    captured = []
    handle = model.readout.merge.register_forward_pre_hook(
        lambda _module, values: captured.append(values[0]))
    try:
        feature, _ = model.read(belief, decode=False)
    finally:
        handle.remove()
    batch = feature.shape[0]
    reader = model.readout
    width = reader.queries * reader.heads * reader.head_dim
    mean, variance = captured[0][:, :width], captured[0][:, width:]
    def probes(value):
        return value.reshape(batch, reader.queries, reader.heads,
            reader.head_dim).transpose(1, 2).flatten(1, 2)
    return feature, torch.cat((probes(mean), probes(variance)), dim=-1)


def halo(sites):
    return sites | torch.stack([torch.roll(sites, direction, axis)
        for axis in range(3) for direction in (-1, 1)]).any(0)


def incident_edges(sites):
    return torch.stack([sites | torch.roll(sites, -1, axis) for axis in range(3)], -1)


def spatial_mask(name, sites, value):
    edge = incident_edges(sites)
    if name.startswith('flux_'):
        axis = ('flux_x', 'flux_y', 'flux_z').index(name)
        result = edge[..., axis][None, ..., None]
    elif name == 'conduction':
        result = edge[None]
    elif name == 'transmission':
        result = edge[None, ..., None]
    elif name == 'receptors':
        result = sites[None, ..., None, None]
    elif name == 'field':
        result = sites[None, ..., None]
    else:
        return None
    return result.expand_as(value)


def support_record(name, value, gradient, sites):
    mask = spatial_mask(name, sites, value)
    if mask is None:
        return dict(spatial=False, connected=gradient is not None)
    if gradient is None:
        return dict(spatial=True, connected=False, squared_norm=0., outside_aperture=0.,
                    outside_one_halo=0.)
    squared = gradient.double().square()
    total = float(squared.sum())
    outside = float(squared.masked_select(~mask).sum())
    outside_halo = float(squared.masked_select(~spatial_mask(name, halo(sites), value)).sum())
    return dict(spatial=True, connected=True, squared_norm=total,
                outside_aperture=outside / total if total else 0.,
                outside_one_halo=outside_halo / total if total else 0.)


def boundary_sensitivity(model, belief, duration, supports, directions):
    entry = differentiable(belief)
    items = [(k, v) for k, v in components(entry).items() if v is not None]
    evolved, _ = model.advance(entry, duration, diagnostics=False)
    _, measurement = measure(model, evolved)
    rows = []
    for probe, sites in enumerate(supports):
        for direction, vector in enumerate(directions):
            projection = (measurement[0, probe] * vector).sum()
            gradients = torch.autograd.grad(projection, [v for _, v in items],
                                            retain_graph=True, allow_unused=True)
            rows.append(dict(probe=probe, direction=direction,
                site_count=int(sites.sum()), one_halo_site_count=int(halo(sites).sum()),
                components={name: support_record(name, value, gradient, sites)
                            for (name, value), gradient in zip(items, gradients)}))
    return rows


def closure_counterfactual(model, belief, duration, supports):
    # Geometry selects the test probe before any response/label is examined.
    probe = min(range(len(supports)), key=lambda i: int(supports[i].sum()))
    sites = supports[probe]
    _, immediate = measure(model, belief)
    future, _ = model.advance(belief, duration, diagnostics=False)
    _, baseline_future = measure(model, future)
    rows = []
    for region_name, region in (('aperture', sites), ('aperture_plus_one_halo', halo(sites))):
        for scale in (.99, 1.01):
            altered_flux = []
            max_inside_change = 0.
            for axis, value in enumerate(belief.medium.flux):
                name = ('flux_x', 'flux_y', 'flux_z')[axis]
                mask = spatial_mask(name, region, value)
                changed = value * torch.where(mask, value.new_tensor(1.), value.new_tensor(scale))
                max_inside_change = max(max_inside_change,
                    float((changed - value).masked_select(mask).abs().max()))
                altered_flux.append(changed)
            altered = PlasticBelief(replace(belief.medium, flux=tuple(altered_flux)), belief.precision)
            _, immediate_changed = measure(model, altered)
            future_changed, _ = model.advance(altered, duration, diagnostics=False)
            _, response = measure(model, future_changed)
            rows.append(dict(held_region=region_name, external_flux_scale=scale,
                held_flux_max_difference=max_inside_change,
                current_probe_max_difference=float((immediate_changed[:, probe] - immediate[:, probe]).abs().max()),
                future_probe_relative_difference=relative(response[:, probe], baseline_future[:, probe])))
    return dict(probe=probe, aperture_sites=int(sites.sum()), halo_sites=int(halo(sites).sum()), tests=rows)


def first_order_response(model, belief, duration):
    with torch.no_grad():
        prepared = model.medium.prepare_evolution()
        baseline_feature, baseline_measurement = measure(model, belief)
    def field_flow(tau):
        evolved, _ = model.advance(belief, tau, prepared=prepared, diagnostics=False)
        return evolved.medium.field
    # Derivative of the autonomous dynamics; no adjacent-token write is included.
    _, derivative = torch.autograd.functional.jvp(field_flow,
        (belief.medium.field.new_zeros(1),), (belief.medium.field.new_ones(1),))
    if not torch.isfinite(derivative).all():
        raise FloatingPointError('Nonfinite autonomous field derivative')
    rows = []
    with torch.no_grad():
        for divisor in (4, 8, 16):
            epsilon = duration / divisor
            evolved, _ = model.advance(belief, epsilon, prepared=prepared, diagnostics=False)
            true_feature, true_measurement = measure(model, evolved)
            jet = PlasticBelief(belief.medium.with_field(belief.medium.field + epsilon * derivative),
                                belief.precision)
            jet_feature, jet_measurement = measure(model, jet)
            rows.append(dict(epsilon=epsilon,
                field_hold_error=relative(belief.medium.field, evolved.medium.field),
                field_jet_error=relative(jet.medium.field, evolved.medium.field),
                feature_hold_error=relative(baseline_feature, true_feature),
                feature_jet_error=relative(jet_feature, true_feature),
                probe_hold_error=relative(baseline_measurement, true_measurement),
                probe_jet_error=relative(jet_measurement, true_measurement)))
    for index in range(len(rows) - 1):
        for quantity in ('field_hold_error', 'field_jet_error', 'feature_hold_error', 'feature_jet_error'):
            rows[index][quantity + '_halving_ratio'] = rows[index][quantity] / max(rows[index + 1][quantity], 1e-20)
    return dict(autonomous_derivative_norm=float(derivative.norm()), samples=rows,
                meaning='Existing-state first-order jet forecasts physical read response; it is not a fitted language readout.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    constructor = dict(saved['config']['constructor'])
    constructor.update(medium_execution='native', port_execution='native')
    model = PlasticMediumPorts3D(**constructor).train()
    model.load_state_dict(saved['model'])
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    belief = unpack_belief(saved['belief'], 'cpu')
    duration = saved['config']['event_duration']
    corpus = np.load(args.data_dir / 'train.npy', mmap_mode='r')
    targets = torch.from_numpy(np.array(corpus[saved['cursor']:saved['cursor'] + 32], dtype=np.int64))
    ids = torch.cat((targets.new_tensor([saved['learner']['carry_token']]), targets[:-1]))
    with torch.no_grad():
        table = F.normalize(model.source.embedding.weight, dim=-1)
        footprint = model.readout.footprint().reshape(model.readout.heads * model.readout.queries, *model.medium.shape)
        supports = footprint > 0
    generator = torch.Generator().manual_seed(449)
    directions = F.normalize(torch.randn(2, model.readout.head_dim * 2, generator=generator), dim=-1)
    records = []
    for event, token in enumerate(ids):
        with torch.no_grad():
            written, _ = model.assimilate(belief, token.reshape(1), token_features=table,
                                         diagnostics=False, training_terms=False)
        if event in (0, 8, 16, 24):
            print(f'Closure audit: actual event{event},16 compact probes', flush=True)
            sensitivity = boundary_sensitivity(model, written, duration, supports, directions)
            with torch.no_grad():
                counterfactual = closure_counterfactual(model, written, duration, supports)
            jet = first_order_response(model, written, duration)
            records.append(dict(event=event, observed=int(token),
                boundary_vjp=sensitivity, boundary_counterfactual=counterfactual,
                first_order_response=jet))
        with torch.no_grad():
            belief, _ = model.advance(written, duration, diagnostics=False)
    unchanged = all(torch.equal(value.detach(), original[name]) for name, value in model.named_parameters())
    if not unchanged:
        raise AssertionError('Read-only model parameters changed')
    report = dict(purpose='Local physical observation closure, not linguistic information/capability comparison',
        checkpoint=str(args.checkpoint.resolve()), checkpoint_step=saved['step'],
        checkpoint_actual_events=saved['learner']['events'], corpus_offset=saved['cursor'],
        duration=duration, actual_forward_events=32, optimizer_updates=0,
        parameters_unchanged=unchanged, device='cpu', aperture_union_sites=int(supports.any(0).sum()),
        samples=records, elapsed_seconds=time.perf_counter() - started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preregistration='docs/information_boltzmann/MEDIUM_READ_CLOSURE_PREREG_20261008.md')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('optimizer_updates', 'parameters_unchanged', 'elapsed_seconds', 'aperture_union_sites')}, indent=2))
    print(json.dumps(records[0]['first_order_response'], indent=2))


if __name__ == '__main__':
    main()
