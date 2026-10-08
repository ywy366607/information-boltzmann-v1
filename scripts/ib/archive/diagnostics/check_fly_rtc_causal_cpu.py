"""Registered finite-horizon causal response repair; numerical CPU assay only."""
from __future__ import annotations

from dataclasses import fields
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ib.check_fly_rtc_cpu import responses, FlyReservoirLM, TickResponseStudent
from information_boltzmann.core.fly_rtc_learning import physical_options, quiet_teacher_step
from information_boltzmann.core.fly_rtc_causal import CausalMotorForecaster
from information_boltzmann.core.fly_rtc_motor_flux import MotorLocalState
from information_boltzmann.core.fly_rtc_surface import MotorSurfaceForecaster


def median_ms(function):
    function()
    times = []
    for _ in range(7):
        start = time.perf_counter()
        function()
        times.append(1000*(time.perf_counter()-start))
    return statistics.median(times)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fused', action='store_true', help='Compile all signed delay tiers into one sparse operator')
    args = parser.parse_args()
    name = 'fly_rtc_causal_fused_cpu_20261007.json' if args.fused else 'fly_rtc_causal_cpu_20261007.json'
    output = ROOT/'results/published'/name
    graph = ROOT/'results/fly_rtc_cpu_numerical/real_induced_graph.npz'
    fingerprint = hashlib.sha256(graph.read_bytes()).hexdigest()
    old = json.loads((ROOT/'results/published/fly_rtc_flux_cpu_20261007.json').read_text(encoding='utf-8'))
    if fingerprint != old['graph']['graph_sha256']:
        raise ValueError('Graph differs from registered preceding assay')
    prereg = dict(graph_sha256=fingerprint, seed=11, horizons=[1, 2, 4, 7, 14],
        fused_transmission=args.fused,
        updates=0, scope='Same256nodeinducedgraph; deterministic response under declared zero future drive',
        reviewer='rtc_contract_review approved corrected sum-delay cone plus all-domain origin boundary queues',
        prediction='Allmotorphysicalfields and transmittedpulse decisions matchfulloracle within2e-6; exactness needs no fitting',
        cost_prediction='Retained state/edges and matched CPUtime measured; smallworldclosure mayremove speed advantage',
        stop='Any motorfield error>2e-6 blocks promotion; no further student fit or production training')
    output.with_name(output.stem+'_preregistered.json').write_text(json.dumps(prereg, indent=2), encoding='utf-8')
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = FlyReservoirLM(graph, vocab_size=16, d_model=16, injection='sensory',
        read_surface='output', synapse_model='coba', use_alif=True, use_stp=True)
    model.rtc_student = TickResponseStudent.from_model(model, latent_dim=32, sample_per_region=32, horizon=14, seed=11)
    data = responses(model, 14, 11, return_surface=True, return_physical=True)
    results = []
    origin = data[8][0]
    options = physical_options(model)
    for horizon in prereg['horizons']:
        compile_start = time.perf_counter()
        predictor = CausalMotorForecaster(model, horizon=horizon, fused_transmission=args.fused)
        coefficients = predictor.coefficients(model)
        compile_ms = 1000*(time.perf_counter()-compile_start)
        maxima = {f.name: 0. for f in fields(predictor.local_state(origin))}
        tick_max = [0.]*horizon
        errors, hold_errors, pulse_mismatches = [], [], 0
        for root, future in zip(data[8], data[9]):
            predicted = predictor.rollout(root, coefficients)
            for tick in range(1, horizon+1):
                target = future[tick-1]
                motor = model.read_indices
                for field in fields(predicted[tick]):
                    if field.name == 'ring':
                        error = max(float((a-b[:, motor]).abs().max()) for a,b in zip(predicted[tick].ring, target.ring))
                    else:
                        name = 'h_mean' if field.name == 'mean' else field.name
                        error = float((getattr(predicted[tick], field.name)-getattr(target, name)[:, motor]).abs().max())
                    maxima[field.name] = max(maxima[field.name], error)
                pulse_mismatches += int(((predicted[tick].ring[0] > 0) != (target.ring[0][:, motor] > 0)).sum())
                tick_max[tick-1] = max(tick_max[tick-1], float((predicted[tick].h-target.h[:, motor]).abs().max()))
                response = MotorSurfaceForecaster.decode(model, predicted[tick].h[4:], predicted[tick].mean[4:])
                actual_y = MotorSurfaceForecaster.decode(model, target.h[4:, motor], target.h_mean[4:, motor])
                root_y = MotorSurfaceForecaster.decode(model, root.h[4:, motor], root.h_mean[4:, motor])
                errors.append((response-actual_y).square().mean())
                hold_errors.append((root_y-actual_y).square().mean())
        def complete_rollout():
            def surface(state):
                motor = model.read_indices
                return MotorLocalState(*(getattr(state, n)[:, motor] for n in ('h','ge','gi','b','x','u')),
                    state.h_mean[:, motor], tuple(p[:, motor] for p in state.ring))
            state = origin
            outputs = [surface(state)]
            for _ in range(horizon):
                state = quiet_teacher_step(model, state, options)
                outputs.append(surface(state))
            return outputs
        candidate_ms = median_ms(lambda: predictor.rollout(origin, coefficients))
        complete_ms = median_ms(complete_rollout)
        scatter = CausalMotorForecaster(model, horizon=horizon, fused_transmission=False)
        scatter_coefficients = scatter.coefficients(model)
        scatter_ms = median_ms(lambda: scatter.rollout(origin, scatter_coefficients))
        operator_bytes = 0
        if args.fused:
            operator = predictor.arrival_operator
            operator_bytes = sum(t.numel()*t.element_size() for t in
                (operator.values(), operator.crow_indices(), operator.col_indices()))
        result = dict(**predictor.budget(), maximum_absolute_error=maxima,
            motor_max_error_by_tick=tick_max, pulse_decision_mismatches=pulse_mismatches,
            heldout_response_error_ratio=float(torch.stack(errors).mean()/torch.stack(hold_errors).mean()),
            compiled_ms=compile_ms, candidate_median_ms=candidate_ms, full_median_ms=complete_ms,
            scatter_median_ms=scatter_ms, fused_vs_scatter_ratio=scatter_ms/candidate_ms,
            sparse_operator_bytes=operator_bytes,
            timing_ratio=complete_ms/candidate_ms,
            timing_scope='CPU float32 batch8, warmup1 +median7, samehorizon/everytickmotorfields; allreadimmutableorigin, candidatequeue/projectionincluded',
            passed=max(maxima.values())<=2e-6 and pulse_mismatches==0)
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(preregistration=prereg, graph=old['graph'], results=results,
        future_teacher_inputs=0, weights_updated=0, checkpoints_written=0,
        passed=all(r['passed'] for r in results),
        scope='Physical no-loss quiet response; no learnedfuture-stimulus prediction, language gain or GPUtiming')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError('Finite-horizon physical closure failed')


if __name__ == '__main__':
    main()
