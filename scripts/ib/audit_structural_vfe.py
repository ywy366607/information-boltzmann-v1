"""CPU numerical identities for the proposed structural VFE; no training/task.

Checks variational stationarity, simplex resources, paired skew coupling,
cost identifiability and the zero-coupling birth barrier. None measures memory
or predictive capability. Existing production modules are not changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def project_simplex(value: np.ndarray, budget: float) -> np.ndarray:
    ordered = np.sort(value)[::-1]
    threshold = (np.cumsum(ordered) - budget) / np.arange(1, len(value) + 1)
    active = np.flatnonzero(ordered > threshold)[-1]
    return np.maximum(value - threshold[active], 0.0)


def audit() -> dict:
    rng = np.random.default_rng(449)
    # Conditional structural variational objective, with fixed A_0 and A_1.
    a0, a1, pi = 1.7, 0.8, 0.23
    logit = a0 - a1 + math.log(pi / (1 - pi))
    r = 1 / (1 + math.exp(-logit))
    derivative = a1 - a0 + math.log(r / (1-r)) - math.log(pi / (1-pi))
    assert abs(derivative) < 1e-12

    # The last component is idle capacity. Projection admits boundary values.
    allocation = project_simplex(np.array([1.2, -0.4, 0.7, 0.2]), 1.0)
    assert allocation.min() >= 0 and abs(allocation.sum() - 1.0) < 1e-12
    # A fixed algebraic objective with known L=1; this is not a capability task.
    start = np.full(4, 0.25)
    target = np.array([0.7, -0.4, 0.1, 0.6])
    updated = project_simplex(start - 0.5 * (start - target), 1.0)
    objective = lambda x: 0.5 * np.sum((x-target)**2)
    decrease = float(objective(start) - objective(updated))
    assert decrease > 0

    # Arbitrary paired couplings preserve Euclidean activity energy.
    d, n = rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    x = rng.normal(size=8)
    def generator(a):
        operator = d + a*n
        return np.block([[np.zeros((5, 5)), -operator.T],
                         [operator, np.zeros((3, 3))]])
    k = generator(0.8)
    energy_derivative = float(x @ k @ x)
    assert abs(energy_derivative) < 1e-12
    probe = rng.normal(size=8)
    dk = np.block([[np.zeros((5, 5)), -n.T], [n, np.zeros((3, 3))]])
    analytic = float(probe @ dk @ x)
    h = 1e-5
    numeric = float(probe @ (generator(0.8+h)-generator(0.8-h)) @ x / (2*h))
    assert abs(analytic-numeric) < 1e-9

    # A separate, unbounded baseline defeats a penalty charged only to a gate.
    raw = rng.normal(size=(3, 3))
    g = np.array([0.4, 0.8, 0.6])
    cost_cheat_error = float(np.max(np.abs(g[:, None]*raw - (g/2)[:, None]*(2*raw))))
    assert cost_cheat_error == 0
    row_norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    direction = raw / row_norm
    assert np.max(np.abs(np.linalg.norm(direction, axis=-1)-1)) < 1e-12

    # Every fast modulation is a utilization fraction of installed capacity.
    utilization = (1 / (1 + np.exp(-rng.normal(size=3)))) * rng.random(3) * rng.random(3)
    installed = g[:, None] * direction
    effective = utilization[:, None] * installed
    capacity_violation = float(np.maximum(np.linalg.norm(effective, axis=-1)-g, 0).max())
    assert capacity_violation == 0

    # Softmax produces an exact local relative-value competition derivative.
    logits, resource, costs = rng.normal(size=4), 1.7, rng.normal(size=4)
    def soft_resources(value):
        weight = np.exp(value - value.max())
        return resource * weight / weight.sum()
    resources = soft_resources(logits)
    competition = resources * (costs - resources @ costs / resource)
    competition_numeric = np.empty(4)
    for index in range(4):
        perturbation = np.eye(4)[index] * h
        competition_numeric[index] = (soft_resources(logits+perturbation) @ costs
                                      - soft_resources(logits-perturbation) @ costs) / (2*h)
    competition_error = float(np.max(np.abs(competition-competition_numeric)))
    assert competition_error < 1e-9

    # Gaussian coefficient KL and an OU prior have closed finite-dimensional forms.
    mean, prior_mean = rng.normal(size=5), rng.normal(size=5)
    variance, prior_variance = np.exp(rng.normal(size=5)), np.exp(rng.normal(size=5))
    def gaussian_kl(value):
        return 0.5 * np.sum((variance + (value-prior_mean)**2) / prior_variance
                            - 1 + np.log(prior_variance/variance))
    kl_gradient = (mean-prior_mean) / prior_variance
    kl_numeric = np.empty(5)
    for index in range(5):
        perturbation = np.eye(5)[index] * h
        kl_numeric[index] = (gaussian_kl(mean+perturbation)
                             - gaussian_kl(mean-perturbation)) / (2*h)
    kl_error = float(np.max(np.abs(kl_gradient-kl_numeric)))
    assert kl_error < 1e-8
    rho = math.exp(-0.16/3.0)
    # 3.0 here is an arbitrary numerical identity input, not a production prior.
    stationary_variance_error = float(np.max(np.abs(
        rho**2 * prior_variance + (1-rho**2)*prior_variance - prior_variance)))
    assert stationary_variance_error < 1e-12

    # One exact conservative rotation: starting with zero store, field change
    # begins at O(a^2). A linear maintenance fee creates a genuine birth barrier.
    price = 0.1
    birth_objective = lambda a: 0.5*(math.cos(a)+1)**2 + price*a
    assert birth_objective(1e-3) > birth_objective(0)
    assert birth_objective(math.pi) < birth_objective(0)

    # A time-unit reparameterization preserves the filter, changing only the
    # optimizer coordinate; no physical acceleration is claimed.
    omega, dt, reference = 78.53981633974483, 0.005, 0.005
    same_filter_error = abs(np.exp(1j*omega*dt) - np.exp(1j*(omega*reference)*(dt/reference)))
    assert same_filter_error < 1e-12

    return {
        'scope': 'CPU numerical identities only; zero training or capability tasks',
        'bernoulli_stationarity_residual': derivative,
        'simplex_allocation': allocation.tolist(),
        'simplex_sum_residual': float(allocation.sum()-1),
        'fixed_objective_projected_step_decrease': decrease,
        'paired_skew_energy_derivative': energy_derivative,
        'coupling_derivative_abs_error': abs(analytic-numeric),
        'unbounded_baseline_gate_cost_cheat_error': cost_cheat_error,
        'fast_utilization_capacity_violation': capacity_violation,
        'softmax_competition_derivative_abs_error': competition_error,
        'gaussian_kl_mean_gradient_abs_error': kl_error,
        'ou_prior_stationary_variance_abs_error': stationary_variance_error,
        'birth_barrier': {'at_zero': birth_objective(0),
                         'small_positive': birth_objective(1e-3),
                         'finite_proposal': birth_objective(math.pi)},
        'dimensionless_filter_equivalence_error': float(same_filter_error),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('results/published/medium_structural_vfe_math_20261008.json'))
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
