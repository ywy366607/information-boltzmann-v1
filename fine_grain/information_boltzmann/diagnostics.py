"""Full phase-space response diagnostics; finite-time evidence, not criticality proofs."""
import math

import torch

from .state import PhaseState


def phase_difference(a: PhaseState, b: PhaseState, kappa: float) -> torch.Tensor:
    """Fixed energy-coordinate norm of ALL particles and BOTH position/velocity."""
    return torch.cat((math.sqrt(kappa) * (a.x - b.x), a.v - b.v), -1)


def renormalize(reference: PhaseState, shadow: PhaseState, epsilon: float,
                kappa: float) -> tuple[PhaseState, float]:
    delta = phase_difference(shadow, reference, kappa)
    distance = float(torch.linalg.vector_norm(delta))
    if not math.isfinite(distance) or distance <= torch.finfo(delta.dtype).tiny:
        raise FloatingPointError("Unresolved perturbation: increase precision/epsilon; do not report lambda=0")
    ratio = epsilon / distance
    return PhaseState(reference.x + ratio * (shadow.x - reference.x),
                      reference.v + ratio * (shadow.v - reference.v), reference.time), distance


def conditional_response(model, state, tokens, generator, epsilon=1e-5, burn_in=16):
    """Common-noise full-vector Benettin finite response with frozen physical parameters.

    State-dependent jump acceptances can differ. Thus epsilon-convergence is required
    before interpreting this finite response as a conditional Lyapunov exponent.
    """
    if model.adaptive_gamma:
        raise ValueError("Freeze adaptive_gamma during response measurement")
    shadow = PhaseState(state.x.clone(), state.v.clone(), state.time)
    shadow.x[0, 0] += epsilon / math.sqrt(model.force.kappa)
    rates = []
    with torch.no_grad():
        for token in tokens:
            before = generator.get_state()
            base, _, _ = model._advance_steps(state, int(token), generator)
            after = generator.get_state()
            generator.set_state(before)
            try:
                perturbed, _, _ = model._advance_steps(shadow, int(token), generator)
            finally:
                generator.set_state(after)
            shadow, distance = renormalize(base, perturbed, epsilon, model.force.kappa)
            rates.append(math.log(distance / epsilon) / model.event_interval)
            state = base
    retained = rates[burn_in:]
    if not retained:
        raise ValueError("Need events beyond burn-in")
    return {"finite_response_rate": sum(retained) / len(retained), "rates": rates,
            "epsilon": epsilon, "burn_in": burn_in, "events": len(rates),
            "claim": "finite_common_noise_full_phase_response_not_proof_of_criticality"}
