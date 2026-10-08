"""Matched CPU exploration: complete motor state versus zero motor head input.

Same real induced graph, physical oracle, eight continuing stimulus lanes and
120-update budget as the prior assay. Origin motor is observed; future motor
is predicted autonomously. No language labels, GPU, or production changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ib.check_fly_rtc_cpu import (
    responses, subset, FlyReservoirLM, TickResponseStudent,
)
from information_boltzmann.core.fly_rtc_surface import MotorSurfaceForecaster


def surface_subset(data, lanes):
    base = subset(data[:5], lanes)
    motor = data[5][:, lanes].flatten(0, 1)
    target = data[6][:, :, lanes].transpose(0, 1).flatten(1, 2)
    mean = data[7][:, lanes].flatten(0, 1)
    return base + (motor, target, mean)


def evaluate(student, model, data):
    history, target_z, target_y, root_y, motor, target_m, mean = data
    with torch.no_grad():
        z, m, means = student(history, motor, mean)
        y = student.decode(model, m, means)
        raw_err = (m[1:] - target_m).square().mean(dim=(1, 2))
        raw_hold = (target_m - motor[None]).square().mean(dim=(1, 2))
        read_err = (y[1:] - target_y).square().mean(dim=(1, 2))
        read_hold = (target_y - root_y[None]).square().mean(dim=(1, 2))
        latent_error = (z[1:] - target_z).square().mean()
        latent_hold = (target_z - history[0][None]).square().mean()
        return dict(motor_error_ratio=float(read_err.mean() / read_hold.mean()),
                    raw_motor_error_ratio=float(raw_err.mean() / raw_hold.mean()),
                    latent_error_ratio=float(latent_error / latent_hold),
                    motor_delta_mse=float(read_err.mean()),
                    hold_motor_delta_mse=float(read_hold.mean()),
                    raw_motor_mse=float(raw_err.mean()), hold_raw_motor_mse=float(raw_hold.mean()),
                    zero_tick_max_error=float((y[0] - root_y).abs().max()),
                    per_tick=[dict(tick=k + 1, motor=float(read_err[k]),
                                   hold_motor=float(read_hold[k]), raw_motor=float(raw_err[k]),
                                   hold_raw_motor=float(raw_hold[k])) for k in range(len(raw_err))])


def fit(student, model, data, updates):
    history, target_z, target_y, root_y, motor, target_m, mean = data
    z_scale = (target_z - history[0][None]).square().mean().clamp_min(1e-12)
    m_scale = (target_m - motor[None]).square().mean().clamp_min(1e-12)
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-3, weight_decay=0)
    curve = []
    start = time.perf_counter()
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        z, m, _ = student(history, motor, mean)
        loss = F.mse_loss(z[1:], target_z) / z_scale + F.mse_loss(m[1:], target_m) / m_scale
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite surface fit; stop numerical arm')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        if not torch.isfinite(gradient):
            raise RuntimeError('Nonfinite gradient; stop numerical arm')
        optimizer.step()
        if update == 0 or (update + 1) % 20 == 0:
            point = dict(update=update + 1, train_normalized_loss=float(loss.detach()))
            curve.append(point)
            print(json.dumps(dict(use_motor_state=student.use_motor_state, **point)), flush=True)
    return dict(curve=curve, seconds=time.perf_counter() - start,
                train_after=evaluate(student, model, data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=ROOT / 'results/fly_rtc_cpu_numerical/real_induced_graph.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/published/fly_rtc_surface_cpu_20261007.json')
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    prior = json.loads((ROOT / 'results/published/fly_rtc_cpu_response_20261007.json').read_text())
    if hashlib.sha256(args.graph.read_bytes()).hexdigest() != prior['graph']['graph_sha256']:
        raise ValueError('Induced graph does not match registered previous numerical oracle')
    prereg = dict(scope='Bounded numerical assay, not joint language learning',
                 independent_review='rtc_contract_review approved paired CPU assay',
                 graph_sha256=prior['graph']['graph_sha256'], seed=args.seed,
                 updates_per_arm=args.updates, horizon=14, lr=2e-3,
                 train_lanes=[0, 1, 2, 3], heldout_lanes=[4, 5, 6, 7],
                 arms=['complete_motor_input', 'same_parameters_zero_motor_input'],
                 prediction='Complete surface guarantees exact zero-tick read; future response gain requires heldout error below persistence.',
                 failure_action='If exact read passes but future prediction fails, retain observable repair and investigate dynamics/fit before long task training.',
                 loss='train-normalized latent and full raw motor MSE; identical across paired arms',
                 old_comparison_limit='Prior student used latent and decoded-feature MSE; paired arms isolate input preservation, old comparison is descriptive.',
                 future_teacher_use='Labels only; every future motor input is an autonomous draft',
                 stopping='Fixed budget; no tuning or model selection using heldout results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(args.output.stem + '_preregistered.json').write_text(
        json.dumps(prereg, indent=2), encoding='utf-8')
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model = FlyReservoirLM(args.graph, vocab_size=16, d_model=16, injection='sensory',
                           read_surface='output', synapse_model='coba', use_alif=True, use_stp=True)
    model.rtc_student = TickResponseStudent.from_model(
        model, latent_dim=32, sample_per_region=32, horizon=14, seed=args.seed)
    model.eval()
    data = responses(model, 14, args.seed, return_surface=True)
    train, heldout = surface_subset(data, slice(0, 4)), surface_subset(data, slice(4, 8))
    template = MotorSurfaceForecaster(model.rtc_student, len(model.read_indices))
    complete, control = copy.deepcopy(template), copy.deepcopy(template)
    control.use_motor_state = False
    for left, right in zip(complete.parameters(), control.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    results = {}
    for label, student in (('complete_motor_input', complete), ('zero_motor_input', control)):
        before = evaluate(student, model, heldout)
        if before['zero_tick_max_error'] != 0:
            raise RuntimeError('Zero-tick native read interface failed')
        fitted = fit(student, model, train, args.updates)
        results[label] = dict(before=before, **fitted, after=evaluate(student, model, heldout),
                             heldout_by_lane=[evaluate(student, model, surface_subset(data, slice(i, i + 1)))
                                              for i in range(4, 8)])
    report = dict(preregistration=prereg, graph=prior['graph'], device='cpu',
                  updates_per_arm=args.updates, total_updates=2 * args.updates,
                  checkpoints_written=0, teacher_motor_counterfactual_energy=data[4],
                  parameter_count=sum(p.numel() for p in complete.parameters()),
                  old_student_parameter_count=sum(p.numel() for p in model.rtc_student.parameters()),
                  teacher='Fresh fixed COBA-ALIF-STP, same seed as prior; current motor-only read fixed',
                  results=results, previous_motor_error_ratio=prior['after']['motor_error_ratio'],
                  elapsed_seconds=time.perf_counter() - start)
    report['verdict'] = ('useful_bounded_surface_prediction' if
        results['complete_motor_input']['after']['motor_error_ratio'] < 1 else
        'exact_observable_preserved_but_future_gain_unestablished')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(verdict=report['verdict'], elapsed_seconds=report['elapsed_seconds'],
          heldout_ratios={label: result['after']['motor_error_ratio'] for label, result in results.items()},
          report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
