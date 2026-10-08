"""Fourth evaluation pillar: actual energy, structure and response, not entropy flux.

Observers never supply gradients or change dynamics. Storage is bounded by the
measurement window, independent of the age of the continuous individual.
"""
from __future__ import annotations

from collections import deque
import copy
import math

import numpy as np
import torch
from torch.autograd import forward_ad


ENERGY_KEYS = ('energy_before', 'energy_after_write', 'energy_after',
               'incident_energy', 'reflected_energy', 'response_source_work',
               'dissipated_energy', 'write_balance_residual', 'evolution_balance_residual',
               'field_dc_energy', 'field_ac_energy', 'flux_dc_energy', 'flux_ac_energy',
               'transport_energy_change', 'transport_spatial_energy_change',
               'collision_energy_change', 'collision_spatial_energy_change',
               'bath_energy_change', 'bath_spatial_energy_change')
DECODE_KEYS = ('feature_rms_before_norm', 'feature_rms_after_norm',
               'logit_standard_deviation', 'prediction_entropy_nats')


@torch.no_grad()
def spatial_medium_snapshot(model, belief):
    """Bounded read-only spatial telemetry; one record per actual training site."""
    state = belief.medium
    material = model.medium.material_field()
    speed = model.medium.edge_log_speeds(state, material=material).exp()
    field_energy = .5 * state.field.square().sum(-1)
    flux_energy = .5 * sum(x.square().sum(-1) for x in state.flux)
    shear = (None if model.medium.transport_shear is None else
             model.medium.transport_shear(material))
    ports = model.port_snapshot(belief)
    factor = model.medium.current_transport_factor(state)
    energy_current = model.medium.transport_energy_current(state, factor)
    from ..core.local_ports import balanced_grid
    def initial_ports(count):
        grid = balanced_grid(count)
        axes = [(torch.arange(n, device=state.field.device, dtype=state.field.dtype) + .5) / n
                for n in grid]
        centers = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        return (centers + .5 / state.field.new_tensor(model.medium.shape)).remainder(1.)
    write_coords, read_coords = ports['write_port_coords'], ports['read_port_coords']
    write_initial, read_initial = initial_ports(len(write_coords)), initial_ports(len(read_coords))
    def drift(current, initial):
        return ((current - initial + .5).remainder(1.) - .5).norm(dim=-1)
    def values(tensor):
        return tensor.detach().cpu().tolist()
    return {'protocol': 'medium_spatial_snapshot_v1',
            'shape': list(model.medium.shape),
            'physical_time': float(state.elapsed[0]),
            'coordinates': values(model.medium.coordinates.reshape(-1, 3)),
            'field_energy': values(field_energy[0].flatten()),
            'flux_energy': values(flux_energy[0].flatten()),
            'speed': values(speed[0].reshape(-1, 3)),
            'material': values(material.reshape(-1, material.shape[-1])),
            'shear': None if shear is None else values(shear.reshape(-1, 3)),
            'effective_transport_factor': values(factor[0].reshape(-1, 3, 3)),
            'transport_energy_current': values(energy_current[0].reshape(-1, 3)),
            'energy_current_scope': 'instantaneous discrete conservative transport generator; display traces frozen snapshots',
            'transport_capacity_budget': model.medium.transport_capacity_limit,
            'write_initial_coords': values(write_initial),
            'read_initial_coords': values(read_initial),
            'write_port_displacement': values(drift(write_coords, write_initial)),
            'read_port_displacement': values(drift(read_coords, read_initial)),
            'write_gate': values(ports['write_bank_gate_from_current_belief'][0]),
            'read_attention_by_head': values(ports['read_attention_per_head']),
            'read_footprint_by_probe': values(model.readout.weights()),
            'read_heads': model.readout.heads,
            'read_queries': model.readout.queries,
            'read_map_scope': 'instantaneous attention within current finite apertures; persistent temporal history is an additional read branch',
            'write_coords': values(ports['write_port_coords']),
            'read_coords': values(ports['read_port_coords']),
            'rendering': f'{math.prod(model.medium.shape)} actual sites; interpolated clouds and energy-current tracers are display only'}


def spatial_energy(value):
    """Quadrature DC and non-DC energies, preserving channel identity."""
    mean = value.mean((1, 2, 3), keepdim=True)
    dc = 0.5 * mean.square().sum(-1).mean()
    ac = 0.5 * (value - mean).square().sum(-1).mean()
    return dc, ac


def primal(value):
    return forward_ad.unpack_dual(value).primal.detach()


class EventHealthCapture:
    """Fixed-address CUDA-Graph-compatible outputs from the actual event."""

    def __init__(self, model):
        parameter = next(model.parameters())
        self.values = parameter.new_zeros(len(ENERGY_KEYS))
        self.feature = parameter.new_zeros(model.medium.channels)
        self.decode_values = parameter.new_zeros(len(DECODE_KEYS))

    @torch.no_grad()
    def record_decode(self, model, features, logits):
        """Observe the already-computed pre-target logits; no second vocab GEMM."""
        features, logits = primal(features), primal(logits)
        normalized = model.read_norm(features)
        log_probability = logits.log_softmax(-1)
        entropy = -(log_probability.exp() * log_probability).sum(-1).mean()
        self.decode_values.copy_(torch.stack((features.square().mean().sqrt(),
            normalized.square().mean().sqrt(), logits.std(-1, unbiased=False).mean(), entropy)))

    @torch.no_grad()
    def record(self, model, incoming, written, outgoing, write_info, evolution_info, feature):
        energy = model.medium.energy
        before, after_write, after = (primal(energy(b.medium)).mean()
                                     for b in (incoming, written, outgoing))
        incident, reflected = (primal(write_info[k]).mean()
                               for k in ('incident_energy', 'reflected_energy'))
        source = primal(evolution_info.get('response_source_work', before.new_zeros(()))).mean()
        dissipated = primal(evolution_info['bath_out_energy']).mean()
        field_dc, field_ac = spatial_energy(primal(outgoing.medium.field))
        flux_parts = [spatial_energy(primal(x)) for x in outgoing.medium.flux]
        values = (before, after_write, after, incident, reflected, source, dissipated,
                  after_write - before - incident + reflected,
                  after - after_write - source + dissipated,
                  field_dc, field_ac, sum(p[0] for p in flux_parts), sum(p[1] for p in flux_parts))
        values += tuple(primal(evolution_info[k]).mean() for k in ENERGY_KEYS[13:])
        self.values.copy_(torch.stack(values).to(self.values))
        self.feature.copy_(primal(feature).reshape(-1).to(self.feature))


class ChunkHealthCapture(EventHealthCapture):
    """One fixed-address observation per token, transferred after chunk replay."""

    def __init__(self, model, tokens):
        super().__init__(model)
        self.event_values = self.values.new_zeros(tokens, len(ENERGY_KEYS))
        self.event_features = self.feature.new_zeros(tokens, self.feature.numel())
        self.event_decode = self.decode_values.new_zeros(tokens, len(DECODE_KEYS))
        self.select(0)

    def select(self, index):
        self.values = self.event_values[index]
        self.feature = self.event_features[index]

    @torch.no_grad()
    def record_decode(self, model, features, logits):
        features, logits = primal(features), primal(logits)
        normalized = model.read_norm(features)
        log_probability = logits.log_softmax(-1)
        entropy = -(log_probability.exp() * log_probability).sum(-1)
        values = torch.stack((features.square().mean(-1).sqrt(),
            normalized.square().mean(-1).sqrt(), logits.std(-1, unbiased=False), entropy), -1)
        self.event_decode[:features.shape[1]].copy_(values.mean(0))

    def observations(self, count):
        return zip(self.event_values[:count].cpu().double().tolist(),
                   self.event_features[:count].cpu().double().tolist(),
                   self.event_decode[:count].cpu().double().tolist())


def representation_summary(features):
    if len(features) < 2:
        return {'status': 'insufficient_samples'}
    z = np.asarray(features, dtype=np.float64)
    centered = z - z.mean(0)
    variance = float(np.mean(np.sum(centered ** 2, axis=1)))
    roughness = float(0.5 * np.mean(np.sum(np.diff(z, axis=0) ** 2, axis=1)))
    eigenvalues = np.linalg.svd(centered, compute_uv=False) ** 2
    total = float(eigenvalues.sum())
    bound = min(z.shape[0] - 1, z.shape[1])
    if total == 0:
        return {'status': 'constant_representation', 'variance': variance,
                'temporal_roughness': roughness, 'roughness_to_variance': None,
                'spectral_entropy': None, 'normalized_spectral_entropy': None,
                'effective_rank': 0.0, 'participation_rank': 0.0, 'rank_bound': bound}
    p = eigenvalues[eigenvalues > 0] / total
    entropy = float(-np.sum(p * np.log(p)))
    return {'status': 'measured', 'variance': variance, 'temporal_roughness': roughness,
            'roughness_to_variance': roughness / variance,
            'spectral_entropy': entropy,
            'normalized_spectral_entropy': entropy / math.log(bound) if bound > 1 else None,
            'effective_rank': math.exp(entropy), 'participation_rank': float(1 / np.sum(p ** 2)),
            'rank_bound': bound}


def risk_trend(rows, block_tokens):
    """Complete consecutive blocks within the SAME experience role only.

    HAC uncertainty for a descriptive linear trend, not a convergence verdict.
    The cube-root bandwidth is a declared statistical measurement convention.
    """
    groups, blocks = [], []
    for row in rows:
        role = (row['phase'], row['novel'])
        if not groups or groups[-1][0] != role:
            groups.append((role, []))
        groups[-1][1].append(row)
    for segment_index, (role, segment) in enumerate(groups):
        for start in range(0, len(segment) - block_tokens + 1, block_tokens):
            part = segment[start:start + block_tokens]
            blocks.append({'phase': role[0], 'novel': role[1], 'segment': segment_index,
                           'event_midpoint': sum(r['event'] for r in part) / block_tokens,
                           'mean_nll': sum(r['nll'] for r in part) / block_tokens})
    result = {'blocks': blocks, 'by_phase': {}}
    for segment_index, (role, _) in enumerate(groups):
        label = role[0] + ('/first_pass' if role[1] else '/replay')
        if label in result['by_phase']:
            label += f'/segment{segment_index}'
        selected = [b for b in blocks if b['segment'] == segment_index]
        item = {'segment': segment_index, 'complete_blocks': len(selected), 'slope_nats_per_event': None,
                'slope_hac_standard_error': None, 'hac_lags': None}
        if len(selected) >= 3:
            x = np.array([b['event_midpoint'] for b in selected], dtype=np.float64)
            x -= x.mean()
            y = np.array([b['mean_nll'] for b in selected], dtype=np.float64)
            design = np.column_stack((np.ones_like(x), x))
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ beta
            scores = design * residual[:, None]
            meat = scores.T @ scores
            lags = min(len(x) - 1, int(len(x) ** (1 / 3)))
            for lag in range(1, lags + 1):
                cross = scores[lag:].T @ scores[:-lag]
                meat += (1 - lag / (lags + 1)) * (cross + cross.T)
            bread = np.linalg.inv(design.T @ design)
            covariance = bread @ meat @ bread * len(x) / (len(x) - 2)
            item.update(slope_nats_per_event=float(beta[1]), hac_lags=lags,
                        slope_hac_standard_error=math.sqrt(max(0., covariance[1, 1])))
        result['by_phase'][label] = item
    return result


class MediumHealthAuditor:
    """Bounded live measurements; phase boundaries do not reset the auditor."""

    def __init__(self, model, *, window_tokens=256, block_tokens=32):
        if window_tokens < 2 or block_tokens < 1:
            raise ValueError('Positive block and at least two window observations required')
        self.capture = EventHealthCapture(model)
        self.window_tokens, self.block_tokens = int(window_tokens), int(block_tokens)
        self.rows = deque(maxlen=window_tokens)
        self.features = deque(maxlen=window_tokens)
        self.updates = deque(maxlen=window_tokens)
        self.events = 0
        self.energy_totals = {key: 0.0 for key in ('write_work', 'response_source_work',
                                                 'dissipated_energy', 'balance_residual')}
        self.birth_energy = None
        self.last_energy = None

    def record_event(self, nll, *, novel, phase, observation=None):
        if observation is None:
            values = self.capture.values.cpu().double().tolist()
            feature = self.capture.feature.cpu().double().tolist()
            decode = self.capture.decode_values.cpu().double().tolist()
        else:
            values, feature, decode = observation
        if not all(math.isfinite(v) for v in (*values, *feature, *decode, nll)):
            raise FloatingPointError('Nonfinite fourth-pillar observation')
        row = dict(zip(ENERGY_KEYS, values))
        row.update(zip(DECODE_KEYS, decode))
        self.events += 1
        row.update(event=self.events, nll=float(nll), novel=bool(novel), phase=str(phase))
        if self.birth_energy is None:
            self.birth_energy = row['energy_before']
        # Continuity is independent of summing per-event conservation residuals.
        row['energy_continuity_residual'] = (0.0 if self.last_energy is None
                                            else row['energy_before'] - self.last_energy)
        self.last_energy = row['energy_after']
        self.energy_totals['write_work'] += row['energy_after_write'] - row['energy_before']
        self.energy_totals['response_source_work'] += row['response_source_work']
        self.energy_totals['dissipated_energy'] += row['dissipated_energy']
        self.energy_totals['balance_residual'] += row['evolution_balance_residual']
        self.rows.append(row)
        self.features.append(feature)
        while self.updates and self.updates[0]['event'] < self.rows[0]['event']:
            self.updates.popleft()

    def record_update(self, norm_before_clip, update_norm, parameter_norm):
        self.updates.append({'event': self.events, 'gradient_norm_before_clip': float(norm_before_clip),
                             'actual_update_norm': float(update_norm), 'parameter_norm': float(parameter_norm),
                             'relative_update_norm': float(update_norm / parameter_norm) if parameter_norm else None})

    def summary(self):
        rows = list(self.rows)
        if not rows:
            return {'protocol': 'persistent_medium_health_v1', 'status': 'no_observations'}
        last = rows[-1]
        structure = {key: last[key] for key in ('field_dc_energy', 'field_ac_energy',
                                                'flux_dc_energy', 'flux_ac_energy')}
        field_energy = structure['field_dc_energy'] + structure['field_ac_energy']
        structure['field_spatial_fraction'] = structure['field_ac_energy'] / field_energy if field_energy else None
        structure['latest_operator_changes'] = {key: last[key] for key in ENERGY_KEYS[13:]}
        structure['window_operator_changes'] = {key: sum(row[key] for row in rows) for key in ENERGY_KEYS[13:]}
        # Never compute temporal differences across experience-role boundaries.
        segments = []
        for row, feature in zip(rows, self.features):
            role = (row['phase'], row['novel'])
            if not segments or segments[-1]['role'] != role:
                segments.append({'role': role, 'features': []})
            segments[-1]['features'].append(feature)
        representations = [{'phase': s['role'][0], 'novel': s['role'][1],
                            'samples': len(s['features']), **representation_summary(s['features'])}
                           for s in segments]
        ledger = dict(self.energy_totals)
        ledger.update(initial_energy=self.birth_energy, current_energy=self.last_energy,
                      cumulative_balance_residual=self.last_energy - self.birth_energy
                      - ledger['write_work'] - ledger['response_source_work'] + ledger['dissipated_energy'],
                      max_abs_write_residual=max(abs(r['write_balance_residual']) for r in rows),
                      max_abs_evolution_residual=max(abs(r['evolution_balance_residual']) for r in rows),
                      max_abs_continuity_residual=max(abs(r['energy_continuity_residual']) for r in rows))
        decoding = {key: {'latest': last.get(key),
                         'window_mean': float(np.mean([r[key] for r in rows if key in r]))}
                    for key in DECODE_KEYS if any(key in r for r in rows)}
        return {'protocol': 'persistent_medium_health_v1', 'events': self.events,
                'window_events': len(rows), 'energy': ledger, 'structure': structure,
                'representation_segments': representations, 'predictive_risk': risk_trend(rows, self.block_tokens),
                'decoding': decoding,
                'recent_optimizer_updates': list(self.updates),
                'interpretation': 'Descriptive health measurements; no noise-expulsion, criticality or convergence verdict.'}

    def state_dict(self):
        return {'version': 1, 'window_tokens': self.window_tokens, 'block_tokens': self.block_tokens,
                'events': self.events, 'rows': list(self.rows), 'features': list(self.features),
                'updates': list(self.updates), 'energy_totals': dict(self.energy_totals),
                'birth_energy': self.birth_energy, 'last_energy': self.last_energy}

    def load_state_dict(self, saved):
        if (saved['version'] != 1 or saved['window_tokens'] != self.window_tokens
                or saved['block_tokens'] != self.block_tokens):
            raise ValueError('Fourth-pillar continuation measurement mismatch')
        for name in ('events', 'energy_totals', 'birth_energy', 'last_energy'):
            setattr(self, name, copy.deepcopy(saved[name]))
        for name in ('rows', 'features', 'updates'):
            setattr(self, name, deque(copy.deepcopy(saved[name]), maxlen=self.window_tokens))


def conditional_state_response(model, belief, observed, *, event_duration, substeps=1, directions=2):
    """Full-state directional JVP of THIS event at the live parameter snapshot.

    Includes field, all fluxes, structural conduction, receptor gates, STP and
    precision, including the pre-input clock's dependence on this state.
    Excludes elapsed-time coordinates and learning/optimizer variables.
    Equal per-component RMS metric in model coordinates; not a maximal FTLE,
    branching estimator, SOC test, or claim of long-time stability.
    """
    from .online_credit import credit_tensors, replace_credit
    from dataclasses import replace
    if directions < 1 or not math.isfinite(event_duration) or event_duration <= 0 or substeps < 1:
        raise ValueError('Positive directional and physical measurement budgets required')
    def response_tensors(state):
        temporal = (() if state.temporal is None else
                    (state.temporal.value.real, state.temporal.value.imag))
        return (*credit_tensors(state), *temporal)

    def response_belief(values):
        count = len(credit_tensors(belief))
        physical = replace_credit(belief, values[:count])
        history = (None if belief.temporal is None else
                   replace(belief.temporal,
                           value=torch.complex(values[count], values[count + 1])))
        return replace(physical, temporal=history)

    incoming = tuple(t.detach() for t in response_tensors(belief))
    generator = torch.Generator(device=incoming[0].device).manual_seed(0)
    gains, durations = [], []
    with torch.no_grad():
        for _ in range(directions):
            tangent = tuple(torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=generator)
                            for t in incoming)
            norm = sum(t.square().mean() for t in tangent).sqrt()
            tangent = tuple(t / norm for t in tangent)
            with forward_ad.dual_level():
                dual = tuple(forward_ad.make_dual(x, u) for x, u in zip(incoming, tangent))
                current = response_belief(dual)
                # Match training: decide time from the incoming local history,
                # before the new observation is assimilated. Keep the dual
                # duration intact so its state dependence enters the JVP.
                duration = model.event_time(current, event_duration)
                actual_duration = (float(primal(duration)) if isinstance(duration, torch.Tensor)
                                   else float(duration))
                if not math.isfinite(actual_duration) or actual_duration <= 0:
                    raise ValueError('Finite positive actual response duration required')
                written, _ = model.assimilate(current, observed,
                                              diagnostics=True, training_terms=False)
                evolved, _ = model.advance(written, duration, substeps=substeps, diagnostics=True)
                response = [forward_ad.unpack_dual(t).tangent for t in response_tensors(evolved)]
                gain = sum(t.square().mean() for t in response if t is not None).sqrt()
                gains.append(float(gain))
                durations.append(actual_duration)
    if not all(math.isfinite(g) for g in gains):
        raise FloatingPointError('Nonfinite full-state directional response')
    return {'scope': 'conditional_complete_physical_state_one_event',
            'directions': directions, 'duration': durations[0],
            'direction_durations': durations, 'nominal_event_duration': float(event_duration),
            'duration_policy': 'model.event_time from pre-observation physical state; clock JVP included',
            'state_shapes': [list(t.shape) for t in incoming],
            'metric': 'sum of component mean-square distances in model coordinates',
            'gains': gains, 'log_gain_per_physical_time':
            [math.log(g) / duration if g > 0 else None for g, duration in zip(gains, durations)],
            'claim': 'Directional local response, not maximal Lyapunov exponent or criticality.'}
