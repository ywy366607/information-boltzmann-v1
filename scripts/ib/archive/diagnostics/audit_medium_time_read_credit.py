"""Read-only real-corpus timing, observability and credit diagnostics."""
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
from information_boltzmann.runtime.training import quiet_training_chunk
from scripts.ib.audit_medium_learning_chain import credit_summary
from scripts.ib.train_plastic_conductance import unpack_belief


def components(belief):
    state = belief.medium
    return dict(field=state.field, flux_x=state.flux[0], flux_y=state.flux[1],
                flux_z=state.flux[2], conduction=state.conduction,
                receptors=state.receptors, transmission=state.transmission,
                precision=belief.precision)


def differentiable(belief):
    clone = lambda x: None if x is None else x.detach().clone().requires_grad_(True)
    state = belief.medium
    return PlasticBelief(replace(state, field=clone(state.field),
        flux=tuple(clone(x) for x in state.flux), conduction=clone(state.conduction),
        receptors=clone(state.receptors), transmission=clone(state.transmission)),
        clone(belief.precision))


def relative(left, right):
    return float((left - right).norm() / right.norm().clamp_min(1e-12))


def grad_record(value, gradient):
    if gradient is None:
        return dict(connected=False, norm=0., rms=0., relative_direction_sensitivity=0.)
    return dict(connected=True, norm=float(gradient.norm()),
                rms=float(gradient.square().mean().sqrt()),
                relative_direction_sensitivity=float((gradient * value.detach()).norm()))


def score(model, belief, target):
    feature, _ = model.read(belief, decode=False)
    logits = model.decode(feature)
    log_probability = logits.log_softmax(-1)
    ce = -log_probability[0, target]
    entropy = -(log_probability.exp() * log_probability).sum()
    return feature, ce, entropy


def time_response(model, belief, ids, targets, duration, table):
    rows = []
    fractions = [0., .5, 1., 2., 4.]
    prepared = model.medium.prepare_evolution()
    for index, (observed, target) in enumerate(zip(ids, targets)):
        written, _ = model.assimilate(belief, observed.reshape(1),
            token_features=table, diagnostics=False, training_terms=False)
        baseline, _ = model.advance(written, duration, prepared=prepared, diagnostics=False)
        base_feature, _, _ = score(model, baseline, int(target))
        points = []
        current = written
        for fraction in fractions:
            if fraction == .5:
                current, _ = model.advance(written, .5 * duration, prepared=prepared, diagnostics=False)
            elif fraction == 1.:
                current = baseline
            elif fraction > 1.:
                for _ in range(int(fraction - points[-1]['duration_multiple'])):
                    current, _ = model.advance(current, duration, prepared=prepared, diagnostics=False)
            feature, ce, entropy = score(model, current, int(target))
            points.append(dict(duration_multiple=fraction, nll=float(ce), entropy=float(entropy),
                feature_relative_to_default=relative(feature, base_feature)))
        resolutions = []
        for steps in (1, 2, 4):
            evolved, _ = model.advance(written, duration, substeps=steps,
                                      prepared=prepared, diagnostics=False)
            feature, ce, _ = score(model, evolved, int(target))
            resolutions.append(dict(substeps=steps, nll=float(ce),
                field_relative_to_one=relative(evolved.medium.field, baseline.medium.field),
                feature_relative_to_one=relative(feature, base_feature)))
        rows.append(dict(event=index, observed=int(observed), target=int(target),
            current_bias_only_nll=float(F.cross_entropy(model.decoder.bias[None], target.reshape(1))),
            times=points, fixed_duration_resolution=resolutions))
        belief = baseline
        if (index + 1) % 8 == 0:
            print(f'Timing audit: {index + 1}/{len(ids)} real events', flush=True)
    baseline_losses = np.array([row['times'][2]['nll'] for row in rows])
    summary = []
    for j, fraction in enumerate(fractions):
        losses = np.array([row['times'][j]['nll'] for row in rows])
        prior = np.array([row['current_bias_only_nll'] for row in rows])
        summary.append(dict(duration_multiple=fraction, nll=float(losses.mean()),
            gain_over_current_bias=float((prior - losses).mean()),
            difference_from_default=float((losses - baseline_losses).mean()),
            improved_events=int((losses < baseline_losses).sum())))
    curves = np.array([[p['nll'] for p in row['times']] for row in rows])
    confidence_indices = [int(np.argmin([p['entropy'] for p in row['times']])) for row in rows]
    confidence_ce = np.array([row['times'][j]['nll'] for row, j in zip(rows, confidence_indices)])
    return dict(events=rows, predetermined_time_summary=summary,
        fixed_duration_resolution_summary=[dict(substeps=steps,
            mean_nll=float(np.mean([r['fixed_duration_resolution'][j]['nll'] for r in rows])),
            mean_feature_relative=float(np.mean([r['fixed_duration_resolution'][j]['feature_relative_to_one'] for r in rows])))
            for j, steps in enumerate((1, 2, 4))],
        descriptive_only=dict(label_selected_min_nll=float(curves.min(1).mean()),
            entropy_selected_nll=float(confidence_ce.mean()),
            entropy_selected_counts={str(f): confidence_indices.count(j) for j, f in enumerate(fractions)},
            warning='Post-hoc label minimum is an oracle; additional durations change external-arrival timing and compute.'))


def read_observability(model, belief, target, duration):
    initial = differentiable(belief)
    names, tensors = zip(*[(k, v) for k, v in components(initial).items() if v is not None])
    feature, ce, _ = score(model, initial, target)
    direct = torch.autograd.grad(ce, tensors, retain_graph=True, allow_unused=True)
    evolved, _ = model.advance(initial, duration, diagnostics=False)
    later_feature, later_ce, _ = score(model, evolved, target)
    later = torch.autograd.grad(later_ce, tensors, allow_unused=True)
    table = {name: dict(current=grad_record(value, g0), evolved=grad_record(value, g1))
             for name, value, g0, g1 in zip(names, tensors, direct, later)}
    perturbations = []
    with torch.no_grad():
        base_feature, base_ce, _ = score(model, belief, target)
        base_later, _ = model.advance(belief, duration, diagnostics=False)
        base_later_feature, base_later_ce, _ = score(model, base_later, target)
        for scale in (.99, 1.01):
            altered = PlasticBelief(replace(belief.medium,
                flux=tuple(x * scale for x in belief.medium.flux)), belief.precision)
            changed_feature, changed_ce, _ = score(model, altered, target)
            changed_later, _ = model.advance(altered, duration, diagnostics=False)
            changed_later_feature, changed_later_ce, _ = score(model, changed_later, target)
            perturbations.append(dict(flux_scale=scale,
                current_feature_max_difference=float((changed_feature - base_feature).abs().max()),
                current_nll_difference=float(changed_ce - base_ce),
                evolved_feature_relative_difference=relative(changed_later_feature, base_later_feature),
                evolved_nll_difference=float(changed_later_ce - base_later_ce)))
    return dict(state_credit=table, flux_perturbations=perturbations)


def supervision(model, belief, ids, targets, duration, table):
    initial = differentiable(belief)
    belief = initial
    states = [belief]
    features, auxiliary = [], []
    durations = torch.full((len(ids),), duration, requires_grad=True)
    prepared = model.medium.prepare_evolution()
    for index, observed in enumerate(ids):
        belief, info = model.assimilate(belief, observed.reshape(1),
                                       token_features=table, diagnostics=False)
        auxiliary.append(info['_write_free_energy'])
        belief, _ = model.advance(belief, durations[index:index + 1],
                                 prepared=prepared, diagnostics=False)
        feature, _ = model.read(belief, decode=False)
        states.append(belief)
        features.append(feature)
    expression = torch.cat(features, 0)
    token_losses = F.cross_entropy(model.decode(expression), targets, reduction='none')
    task, port = token_losses.mean(), torch.stack(auxiliary).mean()
    entries = [(step, name, value) for step, state in enumerate(states)
               for name, value in components(state).items() if value is not None]
    final_credit = torch.autograd.grad(token_losses[-1], [x[2] for x in entries],
                                      retain_graph=True, allow_unused=True)
    temporal = []
    for step in range(len(states)):
        temporal.append(dict(state_after_events=step, lag=len(ids) - step,
            components={name: grad_record(value, grad)
                        for (at, name, value), grad in zip(entries, final_credit) if at == step}))
    time_task = torch.autograd.grad(task, durations, retain_graph=True)[0]
    time_port = torch.autograd.grad(port, durations, retain_graph=True)[0]
    pairs = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names, parameters = zip(*pairs)
    task_grads = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(port, parameters, allow_unused=True)
    groups, unused = credit_summary(names, parameters, task_grads, aux_grads)
    detached = belief.detach()
    # A future CE has no autograd edge back to any previous state after detach.
    later, _ = model.advance(detached, duration, diagnostics=False)
    _, later_ce, _ = score(model, later, int(targets[-1]))
    boundary = torch.autograd.grad(later_ce, tuple(components(initial).values()), allow_unused=True)
    # Stored outputs can be ancestors of other components within the same step.
    # Independently clone each selected full slice to obtain a proper partial
    # derivative of its future suffix, instead of confusing internal graph-node
    # sensitivities with the Markov state's component-wise adjoint.
    independent = []
    with torch.no_grad():
        frozen_table = F.normalize(model.source.embedding.weight, dim=-1)
        frozen_prepared = model.medium.prepare_evolution()
    for lag in (0, 1, 2, 4, 8, 16, 24, 31, 32):
        step = len(ids) - lag
        entry = differentiable(states[step])
        suffix = entry
        for index in range(step, len(ids)):
            suffix, _ = model.assimilate(suffix, ids[index:index + 1],
                token_features=frozen_table, diagnostics=False, training_terms=False)
            suffix, _ = model.advance(suffix, duration, prepared=frozen_prepared, diagnostics=False)
        _, suffix_ce, _ = score(model, suffix, int(targets[-1]))
        items = [(name, value) for name, value in components(entry).items() if value is not None]
        gradients = torch.autograd.grad(suffix_ce, [value for _, value in items], allow_unused=True)
        independent.append(dict(lag=lag, state_after_events=step,
            terminal_nll=float(suffix_ce.detach()),
            components={name: grad_record(value, gradient)
                        for (name, value), gradient in zip(items, gradients)}))
    with torch.no_grad():
        sides = []
        for multiplier in (.99, 1.01):
            total, _, nll = quiet_training_chunk(model, ids[None], targets[None],
                initial.detach(), event_duration=duration * multiplier)
            sides.append(dict(duration_multiplier=multiplier, task_nll=float(nll),
                              auxiliary_loss=float(total - nll)))
    analytic_task = float(time_task.detach().sum() * duration)
    analytic_aux = float(time_port.detach().sum() * duration)
    return dict(task_nll=float(task.detach()), auxiliary_loss=float(port.detach()),
        terminal_loss_graph_node_credit=temporal,
        terminal_loss_independent_state_credit=independent,
        graph_node_warning='Graph-node sensitivities include dependencies between components created inside the same step; independent suffix replay is the Markov component partial.',
        parameter_groups=groups,
        disconnected_parameters=unused,
        task_elasticity_per_event=(time_task.detach() * duration).tolist(),
        auxiliary_elasticity_per_event=(time_port.detach() * duration).tolist(),
        uniform_duration_elasticity=dict(analytic_task=analytic_task,
            central_difference_task=(sides[1]['task_nll'] - sides[0]['task_nll']) / .02,
            analytic_auxiliary=analytic_aux,
            central_difference_auxiliary=(sides[1]['auxiliary_loss'] - sides[0]['auxiliary_loss']) / .02,
            no_update_finite_difference=sides),
        exact_detach_contract=dict(all_previous_state_gradients_absent=all(g is None for g in boundary),
            forward_field_difference=float((belief.medium.field.detach() - detached.medium.field).abs().max()),
            note='This confirms truncation only; the extra read uses a diagnostic label, not a valid stream forecast.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--events', type=int, default=32)
    args = parser.parse_args()
    if args.events != 32:
        parser.error('This preregistration specifies32 actual events')
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    constructor = dict(saved['config']['constructor'])
    constructor.update(medium_execution='native', port_execution='native')
    model = PlasticMediumPorts3D(**constructor).train()
    model.load_state_dict(saved['model'])
    belief = unpack_belief(saved['belief'], 'cpu')
    duration = saved['config']['event_duration']
    corpus = np.load(args.data_dir / 'train.npy', mmap_mode='r')
    offset = saved['cursor']
    targets = torch.from_numpy(np.array(corpus[offset:offset + args.events], dtype=np.int64))
    ids = torch.cat((targets.new_tensor([saved['learner']['carry_token']]), targets[:-1]))
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        table = F.normalize(model.source.embedding.weight, dim=-1)
        timing = time_response(model, belief, ids, targets, duration, table)
        observed_belief, _ = model.assimilate(belief, ids[:1], token_features=table,
                                            diagnostics=False, training_terms=False)
    print('Checking immediate and evolved read observability', flush=True)
    observability = read_observability(model, observed_belief, int(targets[0]), duration)
    observability['alignment'] = dict(observed=int(ids[0]), target=int(targets[0]),
                                     state='after assimilating observed token, before advance')
    print('Computing exact32-event state credit without parameter updates', flush=True)
    table = F.normalize(model.source.embedding.weight, dim=-1)
    credit = supervision(model, belief, ids, targets, duration, table)
    unchanged = all(torch.equal(p.detach(), original[n]) for n, p in model.named_parameters())
    if not unchanged:
        raise AssertionError('Read-only audit modified weights')
    report = dict(purpose='Read-only mechanism audit, not active evaluation or training benefit',
        checkpoint=str(args.checkpoint.resolve()), checkpoint_step=saved['step'],
        checkpoint_actual_events=saved['learner']['events'], corpus_offset=offset,
        corpus_sha256=saved['config']['data_sha256']['train.npy'],
        elapsed_at_entry=float(belief.medium.elapsed[0]), duration=duration,
        optimizer_updates=0, parameters_unchanged=unchanged, device='cpu',
        timing=timing, read_observability=observability, supervision=credit,
        elapsed_seconds=time.perf_counter() - started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preregistration='docs/information_boltzmann/MEDIUM_TIME_READ_CREDIT_PREREG_20261007.md')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('optimizer_updates', 'parameters_unchanged', 'elapsed_seconds')}, indent=2))
    print(json.dumps(timing['predetermined_time_summary'], indent=2))


if __name__ == '__main__':
    main()
