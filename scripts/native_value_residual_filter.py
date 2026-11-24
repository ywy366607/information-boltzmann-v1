"""Iterated ridge on the ACTIVE persistent scalar-value bank.

For G=D K D, A=G+lambda I, r=D(target-prior), production uses
alpha=A^-1 r. A second residual fit uses alpha+lambda A^-1 alpha.
The same factor, labels, features, source weights and prior are reused.
These are research functions; production selection is not patched here.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def coefficients(factor, residual, ridge, steps=2):
    """Return iterated-ridge coefficients; steps=1 is the original solver."""
    if steps < 1 or int(steps) != steps or ridge <= 0:
        raise ValueError('positive integer steps and positive ridge required')
    column = residual.reshape(len(factor), -1)
    term = torch.cholesky_solve(column, factor)
    answer = term.clone()
    for _ in range(1, steps):
        term = ridge*torch.cholesky_solve(term, factor)
        answer = answer+term
    return answer


def spectral_report(gram, residual, ridge):
    """Separate representability from the original ridge's spectral attenuation."""
    values, vectors = torch.linalg.eigh((gram+gram.T)/2)
    coordinates = vectors.T@residual
    energy = coordinates.square()
    contraction = ridge/(values+ridge)
    total = energy.sum().clamp_min(1e-30)
    return dict(eigenvalues=values.tolist(), minimum_eigenvalue=float(values.min()),
        condition=float(values.max()/values.min().clamp_min(1e-30)),
        residual_energy=float(total), weak_modes=int((values<ridge).sum()),
        weak_target_energy_fraction=float(energy[values<ridge].sum()/total),
        one_step_remaining_fraction=float((energy*contraction.square()).sum()/total),
        two_step_remaining_fraction=float((energy*contraction.pow(4)).sum()/total),
        modal_residual=coordinates.tolist(), one_step_contraction=contraction.tolist())


@torch.no_grad()
def prediction_delta(engine, packets):
    """Second-fit increment in today's active scalar-bank coordinates.

    Call the original predictor first so its versioned factor exists. This
    operation changes neither its stored coefficients nor its novelty law.
    """
    from scripts.ib_local import native_amortized_decision as amort
    cache = engine._native_fit_cache
    if cache is None:
        raise ValueError('original active predictor must populate its cache first')
    if cache['x'] is None:
        return engine.model.encoder.weight.new_zeros(len(packets))
    query = amort.features(engine, packets)
    cross = amort._kernel(query, cache['x'])*cache['scale'][None]
    delta_alpha = engine.relation_ridge*torch.cholesky_solve(
        cache['coefficient'].double(), cache['factor'].double())
    return (cross.double()@delta_alpha)[:, 0].to(query)


@contextmanager
def refined_root_values(engine):
    """Experimental final-root refinement on the ORIGINAL formed menu.

    Internal planning and candidate construction keep their original readout.
    The actual root successor reads get the second residual fit. Original
    teacher selection, arrived-world learning and full nominal actor learning
    then consume the values produced by this route. Exit restores both ports.
    """
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local import native_amortized_decision as amort
    from scripts.ib_local import persistent_rhythm as rhythm
    from scripts.ib_local.competing_action_teachers import action_key
    if not rhythm.enabled(engine):
        raise ValueError('root residual refinement requires the active persistent route')
    original_value, original_read = planner.value, rhythm.read_plans
    active = []
    stats = dict(root_calls=0, nodes=0, extra_teacher_calls=0, steps=2)

    @torch.no_grad()
    def read(e, states, sources, *, carried=None):
        values, cases, count = original_read(e, states, sources, carried=carried)
        if e is not engine or not active:
            return values, cases, count
        packets = [amort.recipe(e, carried or state, source, selected=state)
                   for state, source in zip(states, sources)]
        # The original read may have been completely served by persisted
        # values. Populate today's factor independently of that small cache.
        if e._native_fit_cache is None:
            amort.predict(e, packets)
        stats['nodes'] += len(packets)
        return values+prediction_delta(e, packets), cases, count

    @torch.no_grad()
    def value(e, state, source, action, mother, distribution, momentum):
        roots = getattr(e, '_competing_action_roots', None)
        root = e is engine and roots and action_key(source, action) in roots
        if root:
            active.append(True)
            stats['root_calls'] += 1
        try:
            return original_value(e, state, source, action, mother, distribution, momentum)
        finally:
            if root:
                active.pop()

    rhythm.read_plans, planner.value = read, value
    try:
        yield stats
    finally:
        rhythm.read_plans, planner.value = original_read, original_value
