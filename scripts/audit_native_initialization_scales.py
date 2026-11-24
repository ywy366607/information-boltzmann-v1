"""Audit initialization transport using existing logs and CPU equation checks.

No checkpoint load, CUDA work, world calls, or optimizer updates. Numerical
witnesses exercise production operators; they are not capability experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def summarize_events(path):
    rows = []
    with path.open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("autonomous_event", {}).get("kind") == "external":
                rows.append(row)
    first, last = rows[0], rows[-1]
    result = dict(
        path=str(path), external_records=len(rows),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cycle_range=[first["cycle"], last["cycle"]],
        scope="external-boundary snapshots; not all optimizer update receipts",
    )
    result["dynamics_roles"] = {}
    for role in ("freq", "context", "coupling", "pair"):
        gates = [r["adaptive_dynamics"]["budget_write_fraction"][role] for r in rows]
        result["dynamics_roles"][role] = dict(
            zero_gate_snapshots=sum(x == 0 for x in gates),
            mean_gate=mean(gates), first_gate=gates[0], final_gate=gates[-1],
        )
    for key in ("coupling", "context_gain", "content_coefficients", "frequency"):
        def flatten(value):
            return [z for x in value for z in (flatten(x) if isinstance(x, list) else [x])]
        a = flatten(first["adaptive_dynamics"][key])
        b = flatten(last["adaptive_dynamics"][key])
        result[key] = dict(first_min=min(a), first_max=max(a), first_mean=mean(a),
                           final_min=min(b), final_max=max(b), final_mean=mean(b),
                           max_absolute_change=max(abs(x-y) for x, y in zip(a, b)))
    result["coupling"]["mean_relative_change"] = (
        result["coupling"]["final_mean"] / result["coupling"]["first_mean"] - 1)
    result["momentum_read"] = dict(
        first=first["momentum_wood"], final=last["momentum_wood"],
        weight_update_nonzero_snapshots=sum(r["momentum_wood"]["weight_step"] > 0 for r in rows),
    )
    result["question_read"] = dict(
        final_decision_byte_tv=last["question_context"]["decision_byte_probability_tv"],
        weight_update_nonzero_snapshots=sum(r["question_context"]["weight_step"] > 0 for r in rows),
    )
    result["scales"] = {}
    for key in ("relation_ridge", "memory_temperature"):
        values = [r["learnable_precision_scales"][key] for r in rows]
        result["scales"][key] = dict(first=values[0], final=values[-1])
    result["spectrum_revision"] = last["adaptive_dynamics"]["spectrum_revision"]
    return result


def equation_checks():
    import torch
    from scripts.ib_local.adaptive_hamiltonian import potential, force_analytic
    from information_boltzmann.budgeted_adamw import coherence_gate

    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(231123)
    # A small matrix-operator witness, not a learning task or a simulated life.
    heads = 3
    state = torch.randn(1, 1, heads, 4, 4, generator=generator, dtype=torch.float64) * .1
    theta = torch.zeros(1, heads+4, dtype=torch.float64)
    theta[:, heads+1] = math.log(math.expm1(.005))
    u = potential(state, theta, heads)
    force = force_analytic(state, theta, heads)
    checks = []
    for scale in (.5, 2.):
        scaled_u = potential(scale*state, theta, heads)
        scaled_force = force_analytic(scale*state, theta, heads)
        u_error = float((scaled_u-scale**4*u).abs().max()/u.abs().max())
        f_error = float((scaled_force-scale**3*force).abs().max()/force.abs().max())
        assert u_error < 1e-11 and f_error < 1e-11
        checks.append(dict(state_scale=scale, energy_expected_scale=scale**4,
                           force_expected_scale=scale**3,
                           force_to_linear_expected_scale=scale**2,
                           energy_relative_error=u_error, force_relative_error=f_error))
    probe_state = state.clone().requires_grad_()
    grad, = torch.autograd.grad(potential(probe_state, theta, heads).sum(), probe_state)
    analytic_error = float((grad-force).abs().max()/grad.abs().max())
    assert analytic_error < 1e-11
    raw = theta.clone().requires_grad_()
    local_grad, = torch.autograd.grad(potential(state, raw, heads).sum(), raw)
    assert float(local_grad[:, heads+1].abs().max()) > 0
    assert float(local_grad[:, heads+2:].abs().max()) > 0
    coupling = .005
    elasticity = -math.expm1(-coupling)/coupling
    # Exact homogeneity of the coherence controller under a constant rescale.
    signal, energy, noise = .7, 1., .1
    base_gate = coherence_gate(torch.tensor(signal, dtype=torch.float64),
                               torch.tensor(energy, dtype=torch.float64), noise, wiener=True)
    gate_checks = []
    for scale in (1e-3, 1., 1e3):
        gate = coherence_gate(torch.tensor(signal*scale**2, dtype=torch.float64),
                             torch.tensor(energy*scale**2, dtype=torch.float64), noise, wiener=True)
        error = abs(float(gate-base_gate))
        assert error < 1e-12
        gate_checks.append(dict(gradient_scale=scale, gate=float(gate), error=error))
    # Adam m/sqrt(v) cancels the same constant gradient scale when epsilon=0.
    adam = []
    for scale in (1e-3, 1., 1e3):
        direction = scale*.7 / math.sqrt(scale**2)
        assert abs(direction-.7) < 1e-12
        adam.append(dict(gradient_scale=scale, epsilon_zero_direction=direction,
                         epsilon_1e8_direction=scale*.7/(abs(scale)+1e-8)))
    # Zero final linear weights admit learning of W, but block the first x cotangent.
    x = torch.randn(4, generator=generator, dtype=torch.float64, requires_grad=True)
    weight = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    cotangent = torch.tensor([1., -.3], dtype=torch.float64)
    dw, dx = torch.autograd.grad((weight@x*cotangent).sum(), (weight, x))
    assert float(dw.norm()) > 0 and float(dx.norm()) == 0
    return dict(
        operator="production adaptive_hamiltonian.potential/force_analytic, CPU float64",
        fixed_pair_weight_homogeneity=checks, analytic_force_relative_error=analytic_error,
        small_coupling=dict(value=coupling, raw_softplus=math.log(math.expm1(coupling)),
                            d_log_coupling_d_raw=elasticity,
                            d_potential_d_raw_coupling=float(local_grad[0, heads+1]),
                            d_potential_d_zero_pair_parameters=local_grad[0, heads+2:].tolist()),
        wiener_constant_gradient_scale_invariance=gate_checks,
        adam_constant_gradient_scale=adam,
        zero_linear_port=dict(weight_gradient_norm=float(dw.norm()), input_gradient_norm=float(dx.norm())),
        premise="quartic/cubic identities fix pair weights (theta_pair=0); learned state-dependent weights add terms",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    prior = json.loads((project/"results/published/cgb_batch2_proposals_review_20261123.json").read_text(encoding="utf8"))
    result = dict(schema=1, optimizer_updates=0, world_calls=0, checkpoint_loads=0,
                  execution="CPU-only algebra witnesses plus existing event logs; live source untouched",
                  log_receipts=summarize_events(args.run/"events.jsonl"),
                  actual_boundary_force=prior["coupling_at_saved_boundary"],
                  equation_checks=equation_checks(),
                  code_bindings=dict(
                      body_birth="native_navigation_body.initialize_native: all Linear/Embedding weights reinitialized nonzero, residual std follows actual depth",
                      dynamic_birth="adaptive_hamiltonian.AdaptiveHamiltonian: positive softplus coupling, zero context/pair raw coefficients; gradient bridges carry later credit",
                      clocks="adaptive_spectrum.admit: inherited state retained, dyadic grid and solver admission",
                      optimizer="BudgetedAdamW.coherence_gate: signal/noise ratio homogeneous; gate modulates actual step; independent-noise interpretation is an extra premise",
                      workset="PastPresentField.exchange: sigmoid precision gain and shared language operators; constructor matrices nonzero",
                      memory="LatentNavigationV3: read_gain=.05; hot-key temperature and relation ridge are trainable positive log scales",
                      linear_ports="momentum_to_boundary and question_to_hidden constructor zeros are overwritten by complete native birth; additive migration may retain zero start",
                  ),
                  theorem_bindings=[
                      dict(file="lean/CGBAxioms/CGBAxioms/GenerationBudgetedPlasticity.lean",
                           declarations=["closed_gate_preserves_weight", "executed_displacement", "coherent_evidence_reopens"],
                           premise="actual gate and Adam direction supplied; no task-optimality conclusion",
                           binding="BudgetedAdamW executed update and per-role boundary gate receipts"),
                  ])
    paths = ["scripts/ib_local/adaptive_hamiltonian.py",
             "scripts/ib_local/native_navigation_body.py",
             "scripts/ib_local/adaptive_spectrum.py",
             "information_boltzmann/budgeted_adamw.py"]
    result["code_sha256"] = {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in paths}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(dict(report=str(args.report), logs=result["log_receipts"],
                          equation_checks=result["equation_checks"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
