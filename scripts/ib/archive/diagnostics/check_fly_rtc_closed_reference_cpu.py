"""Autonomous full-state numerical ceiling, not a fast learned surrogate."""
from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ib.check_fly_rtc_cpu import responses, FlyReservoirLM, TickResponseStudent
from information_boltzmann.core.fly_rtc_motor_flux import MotorFluxForecaster


def main():
    output = ROOT/'results/published/fly_rtc_closed_reference_cpu_20261007.json'
    registration = dict(
        purpose='Independent local-equation implementation with all upstream physical states retained',
        horizon=14, seed=11, updates=0,
        prediction='Complete causal state plus known zero future drive reproduces original physical integration within2e-6',
        approval='rtc_contract_review approved deterministic autonomous full-state ceiling',
        scope='Numerical reproducibility, not learned predictive ability or acceleration',
        stop='Any physical field error exceeding2e-6 fails closure')
    output.with_name(output.stem+'_preregistered.json').write_text(
        json.dumps(registration, indent=2), encoding='utf-8')
    torch.set_num_threads(2)
    torch.manual_seed(11)
    graph = ROOT/'results/fly_rtc_cpu_numerical/real_induced_graph.npz'
    graph_sha256 = hashlib.sha256(graph.read_bytes()).hexdigest()
    earlier = json.loads((ROOT/'results/published/fly_rtc_flux_cpu_20261007.json').read_text(encoding='utf-8'))
    if graph_sha256 != earlier['graph']['graph_sha256']:
        raise ValueError('Reference graph differs from registered hybrid graph')
    model = FlyReservoirLM(graph, vocab_size=16, d_model=16, injection='sensory',
                           read_surface='output', synapse_model='coba', use_alif=True, use_stp=True)
    model.rtc_student = TickResponseStudent.from_model(
        model, latent_dim=32, sample_per_region=32, horizon=14, seed=11)
    data = responses(model, 14, 11, return_surface=True, return_physical=True)
    # Extend the exact physical subdomain to every neuron. Consequently all
    # anatomical edges are internal, and there is no unknown boundary flux.
    complete = copy.deepcopy(model)
    complete.read_indices = torch.arange(model.n_neurons)
    reference = MotorFluxForecaster(complete)
    coefficients = reference.coefficients(complete)
    maxima = {f.name: 0. for f in fields(reference.local_state(data[8][0]))}
    tick_errors = torch.zeros(14)
    with torch.no_grad():
        for origin, targets in zip(data[8], data[9]):
            current = reference.local_state(origin)
            zero_boundary = torch.zeros(*current.h.shape, 2)
            # Only current and the declared zero boundary reach this loop.
            # targets are used after each predicted update for comparison.
            for tick, target in enumerate(targets):
                current = reference.integrate(current, zero_boundary, coefficients)
                actual = reference.local_state(target)
                for field in fields(current):
                    if field.name == 'ring':
                        error = max(float((a-b).abs().max()) for a,b in zip(current.ring, actual.ring))
                    else:
                        error = float((getattr(current, field.name)-getattr(actual, field.name)).abs().max())
                    maxima[field.name] = max(maxima[field.name], error)
                tick_errors[tick] = max(float(tick_errors[tick]),
                    float((current.h[:, model.read_indices]-target.h[:, model.read_indices]).abs().max()))
    report = dict(preregistration=registration, device='cpu', neurons=model.n_neurons,
                  graph_sha256=graph_sha256,
                  maximum_absolute_error=maxima, motor_max_error_by_tick=tick_errors.tolist(),
                  future_teacher_inputs=0, retained_state='All neuron h/ge/gi/b/x/u/mean and four pulse slots',
                  learned_updates=0, passed=max(maxima.values())<=2e-6,
                  compute_limit='Retains all physical nodes and edges; duplicates full simulation arithmetic')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('Complete-state autonomous closure failed')


if __name__ == '__main__':
    main()
