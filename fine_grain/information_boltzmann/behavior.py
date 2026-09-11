"""Finite continuation probes, not a proof of probabilistic bisimulation."""
import torch


@torch.no_grad()
def continuation_distance(model, left, right, continuations, seed=0):
    """Compare predictive distributions before/after identical input suffixes.

    Parameters are frozen; each pair uses common random numbers. The measured
    total variation is conditional on sampled noise, not an exact marginal over
    the stochastic transition kernel. Inputs and caller RNG are not mutated.
    """
    rows = []
    for index, suffix in enumerate(continuations):
        a, b = left, right
        ga = torch.Generator(device=left.x.device).manual_seed(seed + index)
        gb = torch.Generator(device=left.x.device).manual_seed(seed + index)
        distances = []
        for token in [None, *suffix]:
            if token is not None:
                a, _, _ = model._advance_steps(a, int(token), ga)
                b, _, _ = model._advance_steps(b, int(token), gb)
            pa = model.decode(a).softmax(-1)
            pb = model.decode(b).softmax(-1)
            distances.append(float((pa - pb).abs().sum() / 2))
        rows.append(distances)
    return {"total_variation_by_suffix": rows,
            "claim": "finite_common_noise_behavior_probe_not_bisimulation_proof"}
