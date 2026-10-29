"""Deterministic CLO6 interface checks; no training or capability comparison.

Read-only with respect to optimizer sources. Source hashes delimit the inspected
revision while Claude is working. Results report counterexamples, not a new arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
FILES = [ROOT / "scripts" / name for name in (
    "run_optimizer_continual.py", "run_optimizer_continual_v4.py",
    "run_optimizer_continual_v5.py", "run_optimizer_continual_v6.py")]

# Admission is denied: useful retention and temporal recovery have not been
# connected to this executor. Instantaneous nonzero shares are NOT required.
# See the Lean review and clo6_single_decision_admission_20261027.json.
SINGLE_DECISION_ADMITTED = False


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}


def budgeted_reference(g, h, uncertainty, write_cost, radius, budget):
    """One-compartment convex reference, not an optimizer or a learned model.

    Minimize .5*sum(h*u^2) + sum((uncertainty+write_cost)*abs(u))
    - sum(g*u), with abs(u)<=radius and .5*sum(u^2)<=budget.
    h and uncertainty are supplied certificates, not estimated here. The
    quadratic budget is a declared surrogate, not measured physical energy.
    """
    if not SINGLE_DECISION_ADMITTED:
        raise RuntimeError("Single-decision reference has not passed the full Lean admission gate")
    assert torch.all(h > 0) and torch.all(uncertainty >= 0)
    assert torch.all(write_cost >= 0) and radius >= 0 and budget >= 0
    power = (g.abs() - uncertainty - write_cost).clamp(min=0)

    def step(price):
        return g.sign() * (power / (h + price)).clamp(max=radius)

    def usage(u):
        return float(0.5 * u.square().sum())

    if budget == 0:
        u, price = torch.zeros_like(g), None
    elif usage(step(0.0)) <= budget:
        u, price = step(0.0), 0.0
    else:
        low, high = 0.0, 1.0
        while usage(step(high)) > budget:
            high *= 2
        for _ in range(80):
            mid = (low + high) / 2
            if usage(step(mid)) > budget:
                low = mid
            else:
                high = mid
        u, price = step(high), high
    gain = float((g * u - 0.5 * h * u.square()
                  - (uncertainty + write_cost) * u.abs()).sum())
    assert usage(u) <= budget + 1e-12 and gain >= -1e-12
    return dict(step=u.tolist(), resource_price=price,
                quadratic_budget_used=usage(u), certified_net_gain=gain)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    before = hashes()
    from run_optimizer_continual_v6 import CLO6

    opt = CLO6((2, 2))
    opt.tau2 = torch.tensor([[1.0, 9.0]])
    old_q, old_tau = opt.Q.clone(), opt.tau2.clone()
    angle = math.pi / 4
    opt._rebase(torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]]))
    transform = old_q.T @ opt.Q
    expected = old_tau @ transform.square()
    actual = opt.tau2.clone()

    # Inspect whether the column statistic excludes the same element it shrinks.
    a, b = CLO6((2, 1)), CLO6((2, 1))
    a.m = torch.tensor([[1.0], [2.0]])
    b.m = torch.tensor([[5.0], [2.0]])
    p = torch.ones(2, 1)
    fa, fb = a._frac(p), b._frac(p)

    # Inspect allocated tensors, not a simulated training run or a GPU peak.
    storage = []
    for rows, cols in ((8, 32), (32, 32)):
        fresh = CLO6((rows, cols))
        fields = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                          bytes=v.numel() * v.element_size())
                  for k, v in vars(fresh).items() if isinstance(v, torch.Tensor)}
        allocated = sum(v["bytes"] for v in fields.values())
        expected_bytes = 69 * rows * cols + 8 * cols * cols + 32 * rows
        assert allocated == expected_bytes
        storage.append(dict(shape=[rows, cols], allocated_tensor_bytes=allocated,
                            formula_bytes=expected_bytes, fields=fields))

    # Coupled curvature does not make a non-diagonal optimizer necessary for
    # convergence. This is a matrix identity, not a capability benchmark.
    H = torch.tensor([[2., 1.], [1., 2.]], dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64) - 0.5 * H
    assert torch.equal(A.T @ A, 0.25 * torch.eye(2, dtype=torch.float64))

    # One-column, two-row allocation with equal available reservoirs. The
    # current row/column budgets equalize squared steps in this special case.
    # Both scalar proposals are safe; the joint reallocation can raise loss.
    witness = CLO6((2, 1))
    witness.rho = 10.0
    g = torch.ones(2, 1, dtype=torch.float64)
    h = torch.tensor([[1.], [1000.]], dtype=torch.float64)
    proposed = witness._bound(g / h)
    allocated_step = torch.ones_like(proposed) * (proposed.square().sum() / 2).sqrt()
    before_decrease = float((g * proposed - 0.5 * h * proposed.square()).sum())
    after_decrease = float((g * allocated_step - 0.5 * h * allocated_step.square()).sum())
    assert before_decrease > 0 and after_decrease < 0
    reference_budget = float(0.5 * proposed.square().sum())
    reference_check = {
        "admitted": False,
        "executed": False,
        "reason": "Useful latch, temporal recovery and functional diversity remain unconnected to this executor. Small or zero instantaneous shares are allowed; no per-step equal-share requirement is imposed.",
    }
    after = hashes()
    if before != after:
        raise RuntimeError("Optimizer sources changed during audit; rerun on a stable snapshot")
    result = {
        "scope": "deterministic algebra/state checks; no training or capability evidence",
        "source_sha256": before,
        "initial_optimizer_storage": {
            "measurements": storage,
            "formula_bytes_for_K_by_D": "69*K*D + 8*D*D + 32*K",
            "scope": "Initial direct tensor fields only; excludes weights, gradients, later state, temporaries, Python objects and allocator overhead. Includes inherited unused fields; no 1B allocation performed.",
        },
        "coupled_quadratic_diagonal_step": {
            "H": H.tolist(), "scalar_step": 0.5,
            "error_transition": A.tolist(), "squared_norm_factor": 0.25,
            "scope": "Exact one-step identity for this SPD quadratic; refutes necessity of whitening for convergence, not comparative efficiency on language.",
        },
        "allocation_breaks_scalar_descent": {
            "gradient": g.tolist(), "diagonal_curvature": h.tolist(),
            "rho": witness.rho, "bounded_proposal": proposed.tolist(),
            "equal_row_budget_step": allocated_step.tolist(),
            "quadratic_decrease_before_allocation": before_decrease,
            "quadratic_decrease_after_allocation": after_decrease,
            "scope": "Algebraic special case of one-column equal-row allocation, not a claim that a measured CLO6 trajectory reaches this state; complete-step descent needs its own certificate.",
        },
        "single_decision_reference": reference_check,
        "float32_age_resolution": {
            "age": 2**24,
            "increment_changes_value": bool(torch.tensor(float(2**24)) + 1 != torch.tensor(float(2**24))),
            "scope": "Exact arithmetic check on current float32 counter representation; does not claim unlimited fixed-bit counters are needed.",
        },
        "column_prior_rebase": {
            "actual": actual.tolist(), "diagonal_transport_reference": expected.tolist(),
            "max_abs_difference": float((actual - expected).abs().max()),
            "matches_existing_diagonal_transport_convention": bool(torch.allclose(actual, expected)),
            "qualification": "Diagonal transport is itself an approximation for general rotations.",
        },
        "column_prior_uses_own_element": {
            "original_prior": a.tau2.tolist(), "changed_prior": b.tau2.tolist(),
            "original_gain": fa.tolist(), "changed_gain": fb.tolist(),
            "same_element_changes_its_prior": not torch.equal(a.tau2, b.tau2),
        },
        "gaussian_posterior_identity": {
            "prior_variance": 1, "noise_variance": 1,
            "expected_squared_posterior_mean": 0.5, "posterior_variance": 0.5,
            "subtraction": 0, "sum": 1,
            "qualification": "Exact conjugate-model counterexample to a general unbiasedness claim; not a claim that CLO6 is an exact Gaussian posterior.",
        },
        "scalar_newton_boundary": {
            str(r): abs(1 - 1 / r) for r in (0.4, 0.5, 1.0, 2.0)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
