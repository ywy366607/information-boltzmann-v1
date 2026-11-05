"""Numerical admission reference for the existing reinvestment + G14 contract.

This is not a language body, training runner, or voxel refinement. In particular,
an additive energy readout cannot use mixture responsibilities by renaming fields.
All evidence below uses one PRE-ARRIVAL snapshot; autograd is only the audit oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def evidence(log_share: torch.Tensor, logits: torch.Tensor, target: int) -> dict:
    """One compartment: [contacts], [contacts, labels]; positive normalized laws.

    GenerationReinvestment.credit_eq_new_share + Generation.log_softmax_deriv:
    d log P(y)/d z[i,b] = r[i] * (1[b=y] - p[i,b]).
    For routing LOGITS the derivative is r-a; for raw masses it is different.
    """
    log_a = log_share.log_softmax(-1)
    log_p = logits.log_softmax(-1)
    log_prediction = torch.logsumexp(log_a[:, None] + log_p, dim=0)
    log_r = (log_a + log_p[:, target]).log_softmax(-1)
    a, p, r = log_a.exp(), log_p.exp(), log_r.exp()
    residual = -p
    residual[:, target] += 1
    return dict(log_prediction=log_prediction, log_r=log_r,
                share_direction=r - a, content_direction=r[:, None] * residual)


def reference_step(log_share: torch.Tensor, logits: torch.Tensor,
                   target: int, content_step: float) -> dict:
    """Exact reinvestment plus explicit-logit content learning, in parallel.

    Only applicable when logits really ARE the editable content coordinates.
    A body generating logits must instead pull content_direction back through
    that body's actual event transition. No past graph or new history tag is used.
    The content step is an explicit numerical setting, not a derived optimum.
    """
    if content_step < 0:
        raise ValueError("content_step must be nonnegative")
    event = evidence(log_share, logits, target)
    return dict(event, next_log_share=event["log_r"],
                next_logits=logits + content_step * event["content_direction"])


def audit_joint_law(device: str) -> dict:
    """Check the common-law bridge, not an implementable latent-state enumerator.

    A finite joint energy yields a marginal prediction, posterior responsibilities
    and a two-phase score derivative. Probability normalization is NOT a proof of
    conserved structural capacity. The enumerated table is an algebraic oracle.
    """
    rng = torch.Generator().manual_seed(732)

    def sample(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.float64).to(device)

    def close(a, b):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-12)

    features = sample(4, 5, 256)
    theta = sample(4).requires_grad_()
    energy = sample(5, 256) - torch.einsum("e,eib->ib", theta, features)
    log_q = -energy - torch.logsumexp(-energy.flatten(), 0)
    log_a = torch.logsumexp(log_q, 1)
    log_p = log_q - log_a[:, None]
    log_prediction = torch.logsumexp(log_q, 0)
    target = 81
    log_r = log_q[:, target] - log_prediction[target]
    event = evidence(log_a, log_p, target)
    close(event["log_prediction"], log_prediction)
    close(event["log_r"], log_r)

    # Posterior is the minimizer of one information objective; gap = KL(q || r).
    trial_log = sample(5).log_softmax(0)
    trial = trial_log.exp()
    objective = (trial * (trial_log - log_a - log_p[:, target])).sum()
    gap = (trial * (trial_log - log_r)).sum()
    close(objective + log_prediction[target], gap)
    optimum = (log_r.exp() * (log_r - log_a - log_p[:, target])).sum()
    close(optimum, -log_prediction[target])

    # The SAME local interaction changes both marginal participation and content.
    oracle, = torch.autograd.grad(log_prediction[target], theta)
    positive = torch.einsum("i,ei->e", log_r.exp(), features[:, :, target])
    negative = torch.einsum("ib,eib->e", log_q.exp(), features)
    close(oracle, positive - negative)

    # At fixed energy, symmetric conductance changes kinetics, not the Gibbs law.
    e = sample(6)
    raw = sample(6, 6).exp()
    conductance = (raw + raw.T) / 2
    conductance.fill_diagonal_(0)
    rate = conductance * torch.exp((e[:, None] - e[None, :]) / 2)
    rate = rate - torch.diag(rate.sum(1))
    stationary = (-e).softmax(0)
    close(stationary @ rate, torch.zeros_like(stationary))
    close(stationary @ (2 * rate), torch.zeros_like(stationary))
    nonstationary = torch.zeros_like(stationary)
    nonstationary[0] = 1
    assert not torch.allclose(nonstationary @ rate, nonstationary @ (2 * rate))
    return dict(checks=4, latent_states=5, labels=256,
                joint_mixture_error=(event["log_prediction"]-log_prediction).abs().max().item(),
                variational_identity_error=abs((objective+log_prediction[target]-gap).item()),
                common_interaction_gradient_error=(oracle-positive+negative).abs().max().item(),
                fixed_energy_stationarity_error=(stationary @ rate).abs().max().item(),
                scope="finite algebra oracle; no hidden-state enumeration proposed for a body",
                lean_bridge_proved=False, structural_conservation_proved=False)


def audit_generation_limits(device: str) -> dict:
    """Algebraic counterexamples to overreading the common-law identities."""
    dtype = torch.float64
    a = torch.tensor([0.3, 0.7], dtype=dtype, device=device)
    p = torch.full((2, 256), 1 / 256, dtype=dtype, device=device)
    event = evidence(a.log(), p.log(), 81)
    torch.testing.assert_close(event["log_r"].exp(), a)
    other = evidence(a.log(), p.log(), 82)
    torch.testing.assert_close(event["log_prediction"], other["log_prediction"])
    # This same Q works for every context: neither context nor history is encoded.

    occupied = torch.tensor([1., 0.], dtype=dtype, device=device)
    target = torch.tensor([0.4, 0.6], dtype=dtype, device=device)
    updated = occupied + 0.5 * target * (1 - occupied.sum())
    torch.testing.assert_close(updated, occupied)  # Release zero: permanent winner.

    # Symmetric two-state steps have the same equilibrium but different readouts.
    rate = torch.tensor([0.1, 0.4], dtype=dtype, device=device)
    stationary = rate / (2 * rate)
    first_read = rate  # Initial up fraction = 0; p' = p+k(1-p)-kp.
    torch.testing.assert_close(stationary, torch.full_like(rate, 0.5))
    assert first_read[0] != first_read[1]

    # Conservation alone: omitted rate bounds permit an impossible free fraction.
    overfilled = 2.0 * target
    free = 1 - overfilled.sum()
    torch.testing.assert_close(overfilled.sum() + free, free.new_tensor(1.))
    assert free < 0
    return dict(checks=4, kind="counterexample_identity_checks_not_capability",
                context_free_joint_law_admitted_by_static_identities=True,
                full_pool_with_zero_release_keeps_winner=True,
                identical_equilibrium=stationary.tolist(),
                distinct_finite_time_readouts=first_read.tolist(),
                conservation_without_bounds_free_fraction=free.item())


def audit_recurrent_exchange(device: str) -> dict:
    """Algebraic reference for section 7.8, NOT a replacement language body.

    s is an explicitly retained, predictive allocation. Pair flows are only an
    audit decomposition: production must not allocate the dense flow matrix.
    New results here are paper derivations, not newly compiled Lean theorems.
    """
    dtype = torch.float64
    rng = torch.Generator().manual_seed(733)

    def sample(*shape):
        return torch.randn(shape, generator=rng, dtype=dtype).to(device)

    def close(a, b):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-12)

    def kl(a, b):
        return (a * (a.log() - b.log())).sum()

    checks = []
    s = sample(7).softmax(0)
    factor = sample(7, 3)
    kernel = factor @ factor.T  # PSD is sufficient for the discrete descent claim.
    logits = sample(7, 256)
    tau, gamma, target = 0.8, 0.3, 81
    event = evidence(kernel @ s / tau, logits, target)
    r = event["log_r"].exp()
    updated = (1 - gamma) * s + gamma * r
    flow = gamma * r[:, None] * s[None, :]  # destination, source
    close(updated, s + flow.sum(1) - flow.sum(0))
    close(updated.sum(), s.sum())
    assert (updated > 0).all()
    close((updated - s).clamp_min(0).sum(), (s - updated).clamp_min(0).sum())
    checks.append("bounded_retained_allocation_equals_balanced_pair_exchange")

    # Its same state participates in the next prediction; no external resource account.
    future = evidence(kernel @ updated / tau, logits, target)["log_prediction"]
    predictive_effect = (future - event["log_prediction"]).abs().max().item()
    assert predictive_effect > 1e-4
    checks.append("exchanged_state_changes_subsequent_prediction")

    # Exact current-event pullback; not a claim of full historical gradients.
    k = kernel.detach().requires_grad_()
    z = logits.detach().requires_grad_()
    score = evidence(k @ s / tau, z, target)
    dk, dz = torch.autograd.grad(score["log_prediction"][target], (k, z))
    close(dk, score["share_direction"][:, None] * s[None, :] / tau)
    close(dz, score["content_direction"])
    checks.append("same_boundary_score_updates_interaction_and_content")

    # Frozen K and frozen evidence e: this is not a Lyapunov function for learning.
    e = tau * logits.log_softmax(-1)[:, target]

    def potential(x):
        return tau * (x * x.log()).sum() - 0.5 * x @ kernel @ x - e @ x

    x = s.detach().requires_grad_()
    grad, = torch.autograd.grad(potential(x), (x,))
    dissipation = (grad * (r - s)).sum()
    close(dissipation, -tau * (kl(s, r) + kl(r, s)))
    assert dissipation < 0
    checks.append("frozen_field_continuous_dissipation_is_symmetric_kl")
    delta = r - s
    close(potential(r) - potential(s), -tau * kl(s, r) - 0.5 * delta @ kernel @ delta)
    assert potential(updated) <= potential(s)  # convex entropy extends to relaxation.
    checks.append("psd_frozen_field_discrete_potential_decreases")

    # One parameter set jointly supports distinct stable memories and evidence release.
    gain = 2 * torch.log(s.new_tensor(3.))
    fixed = s.new_tensor([0.25, 0.5, 0.75])
    mapped = torch.sigmoid(gain * (2 * fixed - 1))
    close(mapped, fixed)
    slope = 1 - gamma + gamma * 2 * gain * fixed * (1 - fixed)
    assert slope[0] < 1 and slope[1] > 1 and slope[2] < 1
    # The branches predict different labels with the SAME content and present input.
    negative_evidence = -(gain + s.new_tensor(3.).log())
    rare = 0.5 * negative_evidence.exp()
    content = torch.stack([torch.stack([s.new_tensor(0.25), rare, 0.75 - rare]),
                           s.new_tensor([0.25, 0.5, 0.25])])
    close(content.sum(1), torch.ones(2, dtype=dtype, device=device))
    close(content[0, 0].log() - content[1, 0].log(), s.new_tensor(0.))
    left = s.new_tensor([0.25, 0.75]) @ content
    right = s.new_tensor([0.75, 0.25]) @ content
    assert (right - left).abs().max() > 0.2
    checks.append("same_predictive_state_has_two_locally_stable_latches")

    # A literal positive normalized language likelihood realizes this log ratio.
    observed_probability = content[:, 1]  # Same content matrix: another arriving symbol.
    close(observed_probability[0].log() - observed_probability[1].log(), negative_evidence)
    occupied = s.new_tensor(1.)
    steps = 0
    while occupied >= 0.5:
        release_target = torch.sigmoid(gain * (2 * occupied - 1) + negative_evidence)
        assert release_target <= 0.25 + 1e-12
        occupied = (1 - gamma) * occupied + gamma * release_target
        steps += 1
        assert occupied <= 0.25 + 0.75 * (1 - gamma) ** steps + 1e-12
        assert steps < 100
    checks.append("language_likelihood_counterevidence_releases_same_latch")

    # Restricted-family deletions: remove K -> no permanent bistability; remove
    # contrastive evidence -> one-dimensional map cannot move .75 below .5.
    distance = s.new_tensor(0.5)
    for _ in range(12):
        distance *= 1 - gamma
    close(distance, s.new_tensor(0.5 * (1 - gamma) ** 12))
    close(torch.sigmoid(gain * (2 * fixed[2] - 1)), fixed[2])
    checks.append("interaction_and_boundary_evidence_deletion_limits")

    # Restoring a pattern is not exact transport of every causal label. With
    # gamma=1 and a frozen target, all material-tag contrasts are erased.
    tag = sample(7)
    tag -= tag.mean()
    transported_tag = r * tag.sum()
    close(transported_tag, torch.zeros_like(tag))
    checks.append("full_refresh_erases_material_tags_despite_pattern_latching")
    return dict(checks=len(checks), passed=checks, status="paper_construction_numerically_checked",
                persistent_extra_controller_fields=0,
                state_dimension=7, boundary_labels=256,
                predictive_log_probability_change=predictive_effect,
                binary_gain=gain.item(), relaxed_fixed_point_slopes=slope.tolist(),
                counterevidence_release_steps=steps, release_gamma=gamma,
                full_body_admitted=False, learned_self_organization_demonstrated=False,
                exact_long_credit_demonstrated=False)


def run(device: str) -> dict:
    generator = torch.Generator().manual_seed(731)

    def sample(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64).to(device)

    def close(a, b, atol=2e-12):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=atol)

    # Actual 256-label boundary dimensions; random numbers test identities only.
    u = sample(7).requires_grad_()
    z = sample(7, 256).requires_grad_()
    y = 81
    event = evidence(u, z, y)
    du, dz = torch.autograd.grad(event["log_prediction"][y], (u, z))
    close(du, event["share_direction"])
    close(dz, event["content_direction"])
    gradient_error = max((du - event["share_direction"]).abs().max().item(),
                         (dz - event["content_direction"]).abs().max().item())

    # A shared physical coordinate may change BOTH participation and content.
    # The full instantaneous pullback must include both terms, with no detach.
    v, w = sample(7), sample(7, 256)
    theta = torch.zeros((), dtype=z.dtype, device=device, requires_grad=True)
    score = evidence(u.detach() + theta * v, z.detach() + theta * w, y)
    oracle, = torch.autograd.grad(score["log_prediction"][y], (theta,))
    predicted = (du * v).sum() + (dz * w).sum()
    close(oracle, predicted)
    h = 1e-5
    plus = evidence(u.detach() + h*v, z.detach() + h*w, y)["log_prediction"][y]
    minus = evidence(u.detach() - h*v, z.detach() - h*w, y)["log_prediction"][y]
    close((plus - minus) / (2*h), predicted, atol=2e-9)

    # Budget is a count of existing capacity, not a food/reward account.
    step = reference_step(u.detach(), z.detach(), y, 0.1)
    before, after = 123 * u.detach().softmax(-1), 123 * step["next_log_share"].exp()
    close(before.sum(), after.sum())
    close((after-before).clamp_min(0).sum(), (before-after).clamp_min(0).sum())
    close(step["next_logits"].softmax(-1).sum(-1), torch.ones_like(after))
    close(step["log_prediction"], event["log_prediction"])
    close(step["next_logits"] - z.detach(), 0.1 * dz)

    # Jointly wrong equal predictors: allocation freezes but content can move.
    same_u = torch.zeros(2, dtype=z.dtype, device=device)
    same_z = torch.zeros(2, 256, dtype=z.dtype, device=device)
    same_z[:, y] = -5
    repaired = reference_step(same_u, same_z, y, 0.1)
    close(repaired["share_direction"], torch.zeros_like(same_u))
    old = repaired["log_prediction"][y]
    new = evidence(repaired["next_log_share"], repaired["next_logits"], y)["log_prediction"][y]
    assert new > old  # A single numerical case, not a future-language guarantee.

    # The score distribution must precede the target and be target independent.
    other = reference_step(u.detach(), z.detach(), (y + 1) % 256, 0.1)
    close(step["log_prediction"], other["log_prediction"])
    assert not torch.allclose(step["next_logits"], other["next_logits"])

    # Mixture and additive-energy laws give distinct credit on the SAME logits.
    energy_z = z.detach().clone().requires_grad_()
    energy_score = (u.detach().softmax(-1)[:, None] * energy_z).sum(0).log_softmax(-1)[y]
    energy_grad, = torch.autograd.grad(energy_score, (energy_z,))
    mismatch = (energy_grad - dz).abs().max().item()
    assert mismatch > 1e-4

    # Log-domain representation avoids epsilon floors in the reference.
    extreme_z = torch.zeros_like(same_z)
    extreme_z[:, y] = -1000
    extreme = reference_step(same_u, extreme_z, y, 0.1)
    assert all(torch.isfinite(value).all() for value in extreme.values())
    assert extreme["log_prediction"][y] < -999

    joint = audit_joint_law(device)
    limits = audit_generation_limits(device)
    exchange = audit_recurrent_exchange(device)
    return dict(status="passed", kind="numerical_identity_audit_not_capability",
                device=device, dtype="float64", seed=731, labels=256, checks=15 + exchange["checks"],
                joint_law_bridge=joint,
                generation_limits=limits,
                recurrent_exchange=exchange,
                max_gradient_error=gradient_error,
                joint_pullback_error=abs((oracle-predicted).item()),
                conserved_budget_error=abs((before.sum()-after.sum()).item()),
                identical_wrong_predictors_current_log_score_gain=(new-old).item(),
                additive_readout_credit_mismatch=mismatch,
                body_admitted=False, training_runs=0, new_production_theorems=0)


def audit_association_transport(device: str) -> dict:
    """Algebraic checks of the existing delta write, not a learning experiment.

    History arrays below are proof-side expansion oracles. A runtime needs only
    the current matrix; the expansion is not an exact historical-gradient rule.
    """
    rng = torch.Generator().manual_seed(735)

    def sample(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.float64).to(device)

    def close(a, b):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-12)

    state = sample(4, 6)
    initial = state.clone()
    eye = torch.eye(6, dtype=state.dtype, device=device)
    query = sample(6)
    head, bias = sample(256, 4), sample(256)
    head = head / head.abs().sum(-1, keepdim=True)
    fixed_read_change = torch.zeros(4, dtype=state.dtype, device=device)
    bound = torch.zeros((), dtype=state.dtype, device=device)
    operators, writes = [], []
    moving_errors = []
    for t in range(17):
        # A realized key can depend on the evolving state. The expansion is
        # still an identity, but differentiating frozen keys omits that path.
        key = sample(6) + 0.2 * state.sum(0)
        key = key / key.norm()
        value = sample(4)
        beta = 0.2 + 0.6 * (t % 5) / 4
        innovation = value - state @ key
        op = eye - beta * torch.outer(key, key)
        write = beta * torch.outer(value, key)
        updated = state + beta * torch.outer(innovation, key)
        close(updated, state @ op + write)
        close(updated @ key - value, (1 - beta) * (state @ key - value))
        close(updated @ query - state @ query, beta * innovation * (key @ query))
        orthogonal = query - (query @ key) * key
        close(updated @ orthogonal, state @ orthogonal)
        assert torch.linalg.matrix_norm(op, ord=2) <= 1 + 1e-12

        q2, h2, b2 = sample(6), sample(256, 4), sample(256)
        old_logits = head @ (state @ query) + bias
        new_logits = h2 @ (updated @ q2) + b2
        delta = ((b2 - bias) + (h2 - head) @ (state @ query)
                 + h2 @ (state @ (q2 - query) + beta * innovation * (key @ q2)))
        close(new_logits - old_logits, delta)
        logloss_change = (-new_logits.log_softmax(0) + old_logits.log_softmax(0)).abs()
        assert logloss_change.max() <= 2 * delta.abs().max() + 1e-11
        moving_errors.append((new_logits - old_logits - delta).abs().max().item())
        fixed_read_change += beta * innovation * (key @ query)
        bound += beta * innovation.abs().max() * (key @ query).abs()
        operators.append(op)
        writes.append(write)
        state = updated

    # Independently expand all actual rank-one writes through later operators.
    product = eye.clone()
    expanded_writes = torch.zeros_like(state)
    for op, write in reversed(list(zip(operators, writes))):
        expanded_writes += write @ product
        product = op @ product
    expanded = initial @ product + expanded_writes
    close(state, expanded)
    close((state - initial) @ query, fixed_read_change)
    assert fixed_read_change.abs().max() <= bound + 1e-11
    old_logits = head @ (initial @ query) + bias
    new_logits = head @ (state @ query) + bias
    nll_change = (-new_logits.log_softmax(0) + old_logits.log_softmax(0)).abs().max()
    assert nll_change <= 2 * bound + 1e-11

    # Row/column budget is NOT preserved merely by a correct associative write.
    mass = torch.full((2, 2), 0.5, dtype=state.dtype, device=device)
    key = torch.tensor([1., 0.], dtype=state.dtype, device=device)
    target = torch.tensor([1., 0.], dtype=state.dtype, device=device)
    changed = mass + 0.5 * torch.outer(target - mass @ key, key)
    close(changed.sum(0), mass.sum(0))
    assert not torch.allclose(changed.sum(1), mass.sum(1))
    return dict(status="passed", kind="numerical_identity_audit_not_capability",
                device=device, dtype="float64", seed=735, labels=256,
                identity_families=["affine_delta_factorization", "same_key_contraction",
                    "cross_query_interference", "orthogonal_read_preserved",
                    "realized_path_expansion", "finite_prefix_language_bound",
                    "moving_query_and_head_accounted", "double_budget_counterexample"],
                realized_path_length=17,
                expansion_error=(state - expanded).abs().max().item(),
                moving_readout_identity_error=max(moving_errors),
                language_change=float(nll_change), language_change_bound=float(2 * bound),
                body_admitted=False, training_runs=0, new_production_theorems=0,
                historical_gradient_claim=False, conserved_allocation_bridge=False)


def audit_content_budget(device: str) -> dict:
    """One admitted read/write operator; no language training or latch claim."""
    from scripts.ib_local.gated_deltanet import admit_association_content

    torch.manual_seed(736)
    dev = torch.device(device)
    u = torch.randn(2, 3, 16, 24, device=dev, dtype=torch.float64)
    bounded = admit_association_content(u, 1.0)
    row, col = bounded.abs().sum(-1), bounded.abs().sum(-2)
    assert row.max() <= 1 + 1e-12 and col.max() <= 1 + 1e-12
    torch.testing.assert_close(admit_association_content(bounded, 1.0), bounded)
    query = torch.randn(2, 3, 16, device=dev, dtype=u.dtype)
    old_read = torch.einsum('bhij,bhi->bhj', u, query)
    new_read = torch.einsum('bhij,bhi->bhj', bounded, query)
    correction = torch.einsum('bhij,bhi->bhj', bounded-u, query)
    torch.testing.assert_close(new_read-old_read, correction)
    head = torch.randn(256, 24, device=dev, dtype=u.dtype)
    old_logits, new_logits = old_read @ head.T, new_read @ head.T
    nll_change = (new_logits.log_softmax(-1)-old_logits.log_softmax(-1)).abs().amax(-1)
    nll_bound = 2 * (correction @ head.T).abs().amax(-1)
    assert (nll_change <= nll_bound + 1e-10).all()
    # In float32, measure violation instead of claiming exact arithmetic.
    single = admit_association_content(u.float(), 1.0)
    violation = max(0., float(single.abs().sum(-1).max()-1),
                    float(single.abs().sum(-2).max()-1))
    assert violation < 1e-6
    timing = None
    if device == 'cuda':
        # Same CUDA-graph launch boundary, measured core operator only.
        shape = (1, 8, 64, 64)
        state = torch.randn(shape, device=dev)
        delta = torch.randn_like(state) * 0.01
        durations = {}
        for name, constrained in [('delta_proposal', False), ('budgeted_delta', True)]:
            for _ in range(3):
                proposal = 0.99 * state + delta
                result = admit_association_content(proposal, 1.0) if constrained else proposal
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                proposal = 0.99 * state + delta
                result = admit_association_content(proposal, 1.0) if constrained else proposal
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200):
                graph.replay()
            end.record()
            end.synchronize()
            durations[name] = start.elapsed_time(end) * 1000 / 200
        timing = dict(shape=shape, graph_replays=200, microseconds_per_call=durations,
                      scope='core operator only, not full language training throughput')
    return dict(status='passed', kind='implementation_admission_not_capability',
                device=device, seed=736, reference_dtype='float64',
                reference='lean/CGBAxioms/reviews/budgeted_association_write_20261103.lean',
                constraints='row and column L1 upper bounds with derived slack; not equal marginals',
                max_row_mass=float(row.max()), max_column_mass=float(col.max()),
                float32_max_budget_violation=violation,
                read_correction_error=float((new_read-old_read-correction).abs().max()),
                nll_change_max=float(nll_change.max()), nll_bound_max=float(nll_bound.max()),
                additional_persistent_elements=0, additional_parameters=0,
                timing=timing, training_runs=0, body_admitted=False,
                nonlinear_evidence_exchange_connected=False, healthy_latch_proved=False)


def audit_joint_content_exchange(device: str) -> dict:
    """Actual runtime primitive: conditional dynamical audit, not task performance."""
    from scripts.ib_local.gated_deltanet import exchange_association_content as step

    dev, dtype = torch.device(device), torch.float64
    torch.manual_seed(737)
    value = torch.tensor([0.2, -0.1], device=dev, dtype=dtype)
    gamma = value.new_tensor([0.3])
    gain = 2 * value.new_tensor([3.]).log()
    zero = torch.zeros(2, device=dev, dtype=dtype)
    query = value.new_tensor([1., -1.])
    radii, reads, recovery_errors, release_steps = [], [], [], []

    # Coordinates p,M00,M01,M10,M11 keep a=(p,1-p) on the simplex.
    def update(z, address=zero):
        a = torch.stack((z[0], 1-z[0]))
        m = z[1:].reshape(2, 2)
        m2, a2 = step(m, a, address, value, gain, gamma, 1.0)
        return torch.cat((a2[:1], m2.flatten()))

    for p in (0.25, 0.75):
        share = value.new_tensor([p, 1-p])
        m = share[:, None] * value
        fixed = torch.cat((share[:1], m.flatten()))
        torch.testing.assert_close(update(fixed), fixed, rtol=1e-12, atol=1e-13)
        jac = torch.autograd.functional.jacobian(update, fixed)
        radius = torch.linalg.eigvals(jac).abs().max().item()
        assert radius < 1
        radii.append(radius)
        reads.append((query @ m).tolist())
        disturbed = fixed + value.new_tensor([0.001, 0.0002, -0.0001, -0.0001, 0.0002])
        initial_error = (disturbed-fixed).norm()
        for _ in range(200):
            disturbed = update(disturbed)
        final_error = (disturbed-fixed).norm()
        assert final_error < initial_error * 1e-3
        recovery_errors.append(float(final_error))
        # Same gain/rate/value. Only boundary address evidence changes.
        address = zero.clone()
        address[0] = (1 if p < 0.5 else -1) * (gain.item()+math.log(3.))
        state = fixed.clone()
        for n in range(1, 81):
            state = update(state, address)
            if (p < 0.5 and state[0] > 0.5) or (p > 0.5 and state[0] < 0.5):
                break
        else:
            raise AssertionError('counterevidence failed to release')
        release_steps.append(n)
        opposite = 1-share
        expected = torch.cat((opposite[:1], (opposite[:, None]*value).flatten()))
        for _ in range(250):
            state = update(state)  # remove forcing: the new branch must persist
        torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)
    assert reads[0] != reads[1]

    a = torch.randn(3, 7, device=dev, dtype=dtype).softmax(-1)
    c = torch.randn(3, 7, 5, device=dev, dtype=dtype)
    c = c / c.abs().sum(-1, keepdim=True).clamp_min(1.)
    m = a.unsqueeze(-1)*c
    v = torch.randn(3, 5, device=dev, dtype=dtype)
    address = torch.randn_like(a)
    g, rate = a.new_full((3, 1), 4.), a.new_full((3, 1), 0.4)
    m2, a2 = step(m, a, address, v, g, rate, 1.)
    torch.testing.assert_close(a2.sum(-1), a.sum(-1))
    assert (m2.abs().sum(-1) <= a2+1e-12).all()
    assert (m2.abs().sum(-2) <= 1+1e-12).all()
    assert (m2-m).abs().max() > 1e-3
    # Hold m fixed, vary valid allocation: the next content differs. a cannot
    # in general be recovered from signed m, so it is charged as real state.
    common_m = value.new_tensor([[0.02, -0.01], [0.02, -0.01]])
    m_left, _ = step(common_m, value.new_tensor([0.25, 0.75]), zero, value, gain, gamma, 1.)
    m_right, _ = step(common_m, value.new_tensor([0.75, 0.25]), zero, value, gain, gamma, 1.)
    assert not torch.allclose(m_left, m_right)
    # Unsupported rows can be recruited; zero allocation is not absorbing.
    restored_m, restored_a = step(torch.zeros_like(common_m), value.new_tensor([1., 0.]),
                                 zero, value, gain, gamma, 1.)
    assert restored_a[1] > 0 and restored_m[1].abs().sum() > 0
    timing = None
    if device == 'cuda':
        shape = (1, 8, 64, 64)
        occupancy = torch.full(shape[:-1], 1/64, device=dev)
        contents = torch.randn(shape, device=dev)
        contents = contents / contents.abs().sum(-1, keepdim=True).clamp_min(1.)
        contents *= occupancy.unsqueeze(-1)
        payload, location = torch.randn(1, 8, 64, device=dev), torch.randn_like(occupancy)
        interaction = torch.full((1, 8, 1), 8., device=dev)
        rate_arg = torch.full_like(interaction, 0.3)
        for _ in range(3):
            output = step(contents, occupancy, location, payload, interaction, rate_arg, 1.)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = step(contents, occupancy, location, payload, interaction, rate_arg, 1.)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            graph.replay()
        end.record()
        end.synchronize()
        timing = dict(shape=shape, graph_replays=200,
                      microseconds_per_call=start.elapsed_time(end)*1000/200,
                      scope='joint core operator only, not complete training throughput')
    return dict(status='passed', kind='conditional_joint_dynamics_not_language_capability',
                device=device, seed=737, gain=float(gain), rate=float(gamma),
                joint_fixed_points=[0.25, 0.75], full_joint_jacobian_radii=radii,
                actual_reads=reads, perturbation_recovery_errors=recovery_errors,
                counterevidence_release_steps=release_steps,
                switched_branch_persists_after_forcing_removed=True, timing=timing,
                extra_scalars_per_key_value_matrix='key_dimension',
                state_overhead_at_64_by_64=1/64,
                row_content_bound_violation=max(0., float((m2.abs().sum(-1)-a2).max())),
                same_content_different_share_changes_update=True, zero_share_recruitable=True,
                formal_scope='normalization, weighted-content capacity, readout, conditional fixed lift; stability/release numerical',
                world_evidence='Gaussian latent value residual plus learned address, not exact byte posterior',
                training_runs=0, body_admitted=False, general_anti_monopoly_proved=False)


def audit_shared_memory_learning(device: str) -> dict:
    """Same-state write/retain/read/current-teaching bridge; no training.

    The old memory is detached. Autograd checks current-event formulas only;
    neither the audit nor the proposed interpretation reconstructs its history.
    The production layer is checked as well as the isolated analytic identities.
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer

    rng = torch.Generator().manual_seed(739)
    errors = {}

    def sample(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.float64).to(device)

    def close(name, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
        errors[name] = float((actual - expected).detach().abs().max())

    # Exact partials at fixed pre-event state. Downstream CE supplies delta.
    memory = sample(6, 4).detach()
    raw_k, raw_q = sample(6), sample(6)
    k = (raw_k / raw_k.norm()).requires_grad_()
    q = (raw_q / raw_q.norm()).requires_grad_()
    v = sample(4).requires_grad_()
    alpha = torch.tensor(.73, dtype=memory.dtype, device=device, requires_grad=True)
    beta = torch.tensor(.42, dtype=memory.dtype, device=device, requires_grad=True)
    head, bias = sample(256, 4), sample(256)
    innovation = v - alpha * (memory.T @ k)
    following = alpha * memory + beta * torch.outer(k, innovation)
    output = following.T @ q
    logits = head @ output + bias
    loss = -logits.log_softmax(0)[81]
    residual = logits.detach().softmax(0)
    residual[81] -= 1
    signal = head.T @ residual
    overlap = k @ q
    expected = (
        beta * (q * (signal @ innovation) - alpha * overlap * (memory @ signal)),
        following @ signal,
        beta * overlap * signal,
        signal @ (memory.T @ q - beta * overlap * (memory.T @ k)),
        overlap * (signal @ innovation),
    )
    actual = torch.autograd.grad(loss, (k, q, v, alpha, beta))
    for name, got, want in zip(('key', 'query', 'value', 'retention', 'write_gain'), actual, expected):
        close(name + '_teaching', got, want)

    # Normalization is part of the implementation, not silently omitted.
    for name, raw, normalized, covector in (
            ('key', raw_k, k, expected[0]), ('query', raw_q, q, expected[1])):
        raw = raw.detach().requires_grad_()
        weighted = (torch.nn.functional.normalize(raw, dim=0) * covector.detach()).sum()
        got, = torch.autograd.grad(weighted, raw)
        want = (covector - normalized * (normalized @ covector)) / raw.norm()
        close(name + '_normalization_pullback', got, want)

    # Full existing layer: compare closed read algebra and every parameter's
    # current CE gradient, including sigmoid gates, RMSNorm, SiLU and SwiGLU.
    # No convolution in this check; its history/pullback is not certified here.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(739)
        layer = GatedDeltaNetLayer(d=8, heads=2, recurrence='decay_then_delta',
                                   mlp_hidden=16).to(device=device, dtype=torch.float64)
    x = sample(1, 1, 8)
    old = sample(1, 2, 4, 4).detach()
    actual_output, actual_state, _ = layer(x, old)
    norm = layer.norm(x)
    query = torch.nn.functional.normalize(layer.q_proj(norm).reshape(1, 2, 4), dim=-1)
    key = torch.nn.functional.normalize(layer.k_proj(norm).reshape(1, 2, 4), dim=-1)
    value = layer.v_proj(norm).reshape(1, 2, 4)
    a = layer.alpha_proj(norm).sigmoid().reshape(1, 2, 1)
    b = layer.beta_proj(norm).sigmoid().reshape(1, 2, 1)
    old_read = lambda vector: (old.transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)
    error = value - a * old_read(key)
    read = a * old_read(query) + b * (key * query).sum(-1, keepdim=True) * error
    proposed = x + layer.o_proj(read.reshape(1, 1, 8) * torch.nn.functional.silu(layer.g_proj(norm)))
    inner = layer.mlp_norm(proposed)
    proposed = proposed + layer.mlp['down'](
        torch.nn.functional.silu(layer.mlp['gate'](inner)) * layer.mlp['up'](inner))
    close('runtime_output', actual_output, proposed)
    close('runtime_state', actual_state,
          a.unsqueeze(-1) * old + b.unsqueeze(-1) * key.unsqueeze(-1) * error.unsqueeze(-2))
    language_head, language_bias = sample(256, 8), sample(256)
    score = lambda out: -(language_head @ out.flatten() + language_bias).log_softmax(0)[81]
    named = list(layer.named_parameters())
    got = torch.autograd.grad(score(actual_output), [p for _, p in named])
    want = torch.autograd.grad(score(proposed), [p for _, p in named])
    minimum_gradient_norm = min(float(g.detach().norm()) for g in got)
    assert minimum_gradient_norm > 0, 'Every checked parameter tensor must receive an active signal'
    for (name, _), left, right in zip(named, got, want):
        close('runtime_parameter:' + name, left, right)

    # Equal current predictions can still require different current query
    # updates, supplied by the FULL retained state after history is detached.
    directions, predictions = [], []
    for sign in (1., -1.):
        stored = torch.tensor([[0.], [sign]], dtype=torch.float64, device=device).detach()
        raw_query = torch.tensor([1., 0.], dtype=torch.float64, device=device, requires_grad=True)
        query = torch.nn.functional.normalize(raw_query, dim=0)
        read = (stored.T @ query).squeeze()
        scores = torch.stack((read, -read))
        current_loss = -scores.log_softmax(0)[0]
        gradient, = torch.autograd.grad(current_loss, raw_query)
        directions.append(gradient)
        predictions.append(scores.detach().softmax(0))
    close('equal_prediction', predictions[0], predictions[1])
    close('opposite_current_teaching', directions[0], -directions[1])
    assert float(directions[0].norm()) > .99
    assert not old.requires_grad and old.grad_fn is None
    return dict(status='passed', kind='numerical_same_state_current_learning_audit_not_capability',
                device=device, dtype='float64', seed=739, errors=errors,
                runtime='GatedDeltaNetLayer(decay_then_delta, d=8, heads=2, mlp_hidden=16, conv_kernel=0)',
                all_layer_parameter_tensors_checked=len(named), pre_event_memory_detached=True,
                minimum_parameter_gradient_norm=minimum_gradient_norm,
                same_prediction_opposite_query_learning=True, extra_runtime_state_elements=0,
                scope='Current-event chain rule and normalization; conditional on actual stored history and arrived signal.',
                not_claimed=['arbitrary historical parameter credit', 'learned history sufficiency',
                             'healthy latch', 'conserved Medium realization', 'active life closure',
                             'convolution cache gradient verification'],
                historical_gradient_claim=False, training_runs=0, body_admitted=False)


def audit_retention_retrieval(device: str) -> dict:
    """Conditional storage algebra and an optimizer-interface replay, not training.

    Random tensors are numerical equation tests, not a synthetic memory task.
    The single AdamW step exercises parameter drift; no performance or healthy
    latch claim is made from its direction. Audit snapshots are not body state.
    """
    import copy
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer

    rng = torch.Generator().manual_seed(740)
    errors = {}

    def sample(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.float64).to(device)

    def close(name, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
        errors[name] = float((actual - expected).detach().abs().max())

    # One same-state update: release selected association, retain other reads,
    # account for query motion. No orthogonality assumption in the bound.
    state = sample(6, 4)
    q = torch.nn.functional.normalize(sample(6), dim=0)
    q_next = torch.nn.functional.normalize(sample(6), dim=0)
    key = torch.nn.functional.normalize(sample(6), dim=0)
    value = sample(4)
    alpha, beta = .83, .61
    innovation = value - alpha * (state.T @ key)
    following = alpha * state + beta * torch.outer(key, innovation)
    decay = (alpha - 1) * (state.T @ q)
    interference = beta * (key @ q) * innovation
    drift = following.T @ (q_next - q)
    change = following.T @ q_next - state.T @ q
    close('joint_read_decomposition', change, decay + interference + drift)
    close('selected_key_rewrite', following.T @ key - value,
          (1 - beta) * (alpha * (state.T @ key) - value))
    bound = decay.abs() + interference.abs() + drift.abs()
    assert bool((change.abs() <= bound + 1e-12).all())

    # Exact conditional compatibility, not a promise that sigmoid gates attain
    # endpoints or language learning finds orthogonal keys.
    protected = q - (q @ key) * key
    rewritten = state + torch.outer(key, value - state.T @ key)
    close('same_rule_full_rewrite', rewritten.T @ key, value)
    close('same_rule_orthogonal_preservation', rewritten.T @ protected, state.T @ protected)
    release_key = protected / protected.norm()
    release_target = sample(4)
    released = rewritten + torch.outer(release_key, release_target - rewritten.T @ release_key)
    close('same_rule_later_release', released.T @ release_key, release_target)
    assert float(((released - rewritten).T @ protected).norm()) > 0

    # Old-history differences under COMMON realized future controls. With
    # adaptive differing controls an extra term is required; do not omit it.
    other = state + sample(6, 4)
    difference, transported = state - other, state - other
    initial_norm, factor = float(difference.norm()), 1.0
    for t in range(12):
        key = torch.nn.functional.normalize(sample(6), dim=0)
        value, a, b = sample(4), .8 + .01 * t, .1 + .05 * t
        step = lambda s: a * s + b * torch.outer(key, value - a * (s.T @ key))
        state, other = step(state), step(other)
        transported = a * (transported - b * torch.outer(key, transported.T @ key))
        factor *= a
        close(f'common_control_history_{t}', state - other, transported)
        assert float((state - other).norm()) <= factor * initial_norm + 1e-11

    # An actual layer, including its fixed convolution cache and nonlinear
    # readout. One numerical optimizer step follows the runtime's AdamW and
    # dynamics-group clip; the training scheduler/CUDA graph is not certified.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(740)
        layer = GatedDeltaNetLayer(d=8, heads=2, recurrence='decay_then_delta',
                                   mlp_hidden=16, conv_kernel=4).to(device=device, dtype=torch.float64)
    before = copy.deepcopy(layer)
    x, probe = sample(1, 1, 8), sample(1, 1, 8)
    old = sample(1, layer.matrix_size + layer.history_size).detach()
    output, next_state, _ = layer(x, old)
    head, bias = sample(256, 8), sample(256)
    loss = -(head @ output.flatten() + bias).log_softmax(0)[81]
    optimizer = torch.optim.AdamW(layer.parameters(), lr=3e-4, foreach=True)
    loss.backward()
    gradients = [p.grad for p in layer.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    gradient_norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0, foreach=True)
    optimizer.step()
    carried = next_state.detach().clone()
    assert carried.grad_fn is None
    matrix_only = old.clone()
    matrix_only[:, :layer.matrix_size] = carried[:, :layer.matrix_size]
    with torch.no_grad():
        def logits(model, memory):
            out, _, _ = model(probe, memory)
            return head @ out.flatten() + bias

        z0 = logits(before, old)
        zm = logits(before, matrix_only)
        zc = logits(before, carried)
        z1 = logits(layer, carried)
        terms = {'matrix_transition': zm-z0, 'convolution_cache_transition': zc-zm,
                 'actual_parameter_step': z1-zc}
        total = sum(terms.values())
        close('full_runtime_read_drift', z1-z0, total)
        nll_change = (z1.log_softmax(0)-z0.log_softmax(0)).abs().max()
        nll_bound = 2 * total.abs().max()
        assert nll_change <= nll_bound + 1e-11
        component_norms = {name: float(term.abs().max()) for name, term in terms.items()}
        assert all(value > 0 for value in component_norms.values())
        parameter_drift = max(float((p - p0).abs().max())
                              for p, p0 in zip(layer.parameters(), before.parameters()))
        assert parameter_drift > 0
    return dict(status='passed', kind='conditional_retention_and_runtime_interface_not_capability',
                device=device, dtype='float64', seed=740, errors=errors,
                read_change_bound=float(bound.max()),
                conditional_history_steps=12, common_control_retention_upper_factor=factor,
                history_difference_norm=float(transported.norm()),
                runtime='GatedDeltaNetLayer(decay_then_delta, d=8, heads=2, mlp_hidden=16, conv_kernel=4)',
                optimizer_interface_steps=1, optimizer='AdamW(lr=3e-4, weight_decay=0.01); clip=1',
                parameter_drift_max=parameter_drift, preclip_gradient_norm=float(gradient_norm),
                ordered_readout_components_max=component_norms,
                decomposition_order=['matrix', 'convolution_cache', 'parameters'],
                decomposition_scope='Telescoping identity; order-dependent, not independent causal benefits.',
                all_label_nll_change_max=float(nll_change), all_label_nll_change_bound=float(nll_bound),
                pre_event_state_detached=True, additional_persistent_state_elements=0,
                training_runs=0, body_admitted=False, healthy_latch_proved=False,
                not_claimed=['learned long-term retention', 'regenerative latch',
                             'historical parameter gradient', 'full trainer certification',
                             'AdamW step always preserves useful memory'])


def audit_feedback_renewal(device: str) -> dict:
    """Equation audit of a conditional read-back family; no installed body.

    The small mode is a formal feasibility witness, not a memory capability
    task. Matrix Jacobians are temporary audit oracles, never learner state.
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM

    rng = torch.Generator().manual_seed(741)
    errors = {}

    def sample(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.float64).to(device)

    def close(name, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
        errors[name] = float((actual - expected).detach().abs().max())

    # Structural check of the ACTUAL existing stack: a higher layer's state
    # cannot alter an earlier layer's transition at fixed parameters/input.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(741)
        model = GatedDeltaNetLM(vocab_size=256, d=8, layers=3, heads=2,
                                recurrence='decay_then_delta', mlp_hidden=16,
                                conv_kernel=4).to(device=device, dtype=torch.float64)
    state = sample(*model.state_shape(1))
    tokens = torch.tensor([[81]], device=device)
    with torch.no_grad():
        _, actual, _ = model.predict_logits(tokens, state)
        for layer_index in (1, 2):
            changed = state.clone()
            changed[layer_index] += sample(*changed[layer_index].shape)
            _, following, _ = model.predict_logits(tokens, changed)
            close(f'no_fast_upstream_feedback_{layer_index}',
                  actual[:layer_index], following[:layer_index])

    # Same matrix, with nonlinear read-back INSIDE its target value. Existing
    # memory is both the content read and the source of the regenerating term.
    key = torch.nn.functional.normalize(sample(6), dim=0)
    modes = key.new_tensor([5/12, -5/12, 5/12, -5/12])
    fixed = torch.outer(key, modes)

    def renew(memory, cue, strength=1., value=None):
        read = memory.T @ key
        drive = 4 * read + cue
        generated = drive / (1 + drive.abs())
        incoming = torch.zeros_like(read) if value is None else value
        target = (1-strength) * incoming + strength * generated
        return .5 * memory + .5 * torch.outer(key, target - .5 * read)

    def scalar(read, cue):
        drive = 4 * read + cue
        return read/4 + drive/(1 + drive.abs())/2

    matrix = sample(6, 4)
    cue = sample(4)
    updated = renew(matrix, cue)
    incoming = sample(4)
    close('zero_feedback_is_original_delta', renew(matrix, cue, strength=0., value=incoming),
          .5 * matrix + .5 * torch.outer(key, incoming - .5 * (matrix.T @ key)))
    close('readback_coordinate_identity', updated.T @ key, scalar(matrix.T @ key, cue))
    projector = torch.eye(6, dtype=key.dtype, device=device) - torch.outer(key, key)
    close('unselected_component', projector @ updated, .5 * (projector @ matrix))
    close('joint_fixed_matrix', renew(fixed, torch.zeros_like(modes)), fixed)
    jac = torch.autograd.functional.jacobian(
        lambda flat: renew(flat.reshape(6, 4), torch.zeros_like(modes)).flatten(), fixed.flatten())
    radius = float(torch.linalg.eigvals(jac).abs().max())
    close('full_matrix_local_gain', key.new_tensor(radius), key.new_tensor(17/32))
    disturbed = fixed + .01 * sample(6, 4)
    initial_distance = float((disturbed-fixed).norm())
    for _ in range(40):
        disturbed = renew(disturbed, torch.zeros_like(modes))
    restored_distance = float((disturbed-fixed).norm())
    assert restored_distance < initial_distance * 1e-8
    switched = renew(fixed, -8 * modes.sign())
    assert bool(((switched.T @ key + modes).abs() <= .25 + 1e-12).all())
    for _ in range(40):
        switched = renew(switched, torch.zeros_like(modes))
    close('release_then_opposite_restore', switched, -fixed)
    # Continuous evidence response remains nonzero in the held basin.
    cue = torch.zeros_like(modes, requires_grad=True)
    response = torch.autograd.functional.jacobian(lambda u: renew(fixed, u).T @ key, cue)
    close('held_evidence_response', response,
          torch.eye(4, dtype=key.dtype, device=device) * (9/128))
    without_feedback = .5 * fixed + .5 * torch.outer(key, -.5 * (fixed.T @ key))
    close('feedback_deletion_loses_rest', without_feedback, .25 * fixed)
    # Algebraic family checks, not a hyperparameter/capability sweep. Neutral
    # input yields nonzero roots exactly above eta=(1-a)/(beta*g)=3/8.
    branch_checks = []
    for strength in (.5, .75, 1.):
        magnitude = 2 * strength / 3 - .25
        branch = torch.outer(key, modes.sign() * magnitude)
        close(f'generated_branch_{strength}',
              renew(branch, torch.zeros_like(modes), strength=strength), branch)
        branch_checks.append(dict(strength=strength, rest_magnitude=magnitude,
                                  local_coordinate_gain=.25 + .75**2/(2*strength)))
    # TTT interpretation: an explicit potential reproduces the SAME update.
    # Its gradient is an audit oracle only. The runtime needs only the existing
    # read and softsign, not logs, a second matrix, or an inner autograd tape.
    def potential(memory, key, cue, value, alpha, beta, strength, gain):
        read = memory.T @ key
        drive = gain * read + cue
        primitive = drive.abs() - torch.log1p(drive.abs())
        return ((1-alpha)/(2*beta) * memory.square().sum()
                + alpha/2 * read.square().sum()
                - (1-strength) * (value * read).sum()
                - strength/gain * primitive.sum())

    potential_checks = []
    for alpha, beta, strength, gain in ((.5, .5, 0., 4.), (.5, .5, .75, 4.),
                                       (.5, .5, 1., 4.), (.8, .2, .6, 2.)):
        coefficient = (1-alpha)/beta
        lipschitz = max(abs(coefficient), abs(coefficient+alpha),
                        abs(coefficient+alpha-strength*gain))
        assert beta*lipschitz < 2
        largest_descent_residual = -float('inf')
        for trial in range(12):
            memory = sample(6, 4).requires_grad_()
            # Include the primitive's origin in first-order verification.
            drive_cue = -(gain * (memory.T @ key)).detach() if trial == 0 else sample(4)
            value = sample(4)
            energy = potential(memory, key, drive_cue, value, alpha, beta, strength, gain)
            gradient, = torch.autograd.grad(energy, memory)
            read = memory.T @ key
            drive = gain*read+drive_cue
            target = (1-strength)*value + strength*drive/(1+drive.abs())
            following = alpha*memory + beta*torch.outer(key, target-alpha*read)
            close(f'potential_gradient_{alpha}_{strength}_{trial}',
                  following, memory-beta*gradient)
            next_energy = potential(following, key, drive_cue, value,
                                    alpha, beta, strength, gain)
            # Descent lemma for this fixed event, not across changed language/cues.
            residual = (next_energy-energy + beta*(1-beta*lipschitz/2)
                        * gradient.square().sum()).detach().item()
            assert residual <= 1e-10, residual
            largest_descent_residual = max(largest_descent_residual, residual)
        potential_checks.append(dict(alpha=alpha, beta=beta, strength=strength,
                                     gain=gain, gradient_lipschitz_bound=lipschitz,
                                     largest_descent_inequality_residual=largest_descent_residual))
    z = key.new_tensor(.1).requires_grad_()
    target = 4*z/(1+(4*z).abs())
    residual_gradient, = torch.autograd.grad((z-target).square()/2, z)
    frozen_target_gradient = (z-target).detach()
    assert float(residual_gradient*frozen_target_gradient) < 0
    potential_bridge = dict(
        status='passed', matrix_gradient_checks=48,
        primitive='Phi(x)=abs(x)-log(1+abs(x)); Phi_prime=softsign',
        potential='(1-alpha)/(2*beta)*||S||_F^2 + alpha/2*||S.T*k||^2 - (1-eta)*v.T*S.T*k - eta/g*sum(Phi(g*S.T*k+u))',
        assumptions='unit fixed key; fixed event coefficients/cue/value; beta>0, g>0, 0<=alpha,eta<=1',
        checked_parameters=potential_checks,
        endogenous_squared_error_gradient=float(residual_gradient),
        frozen_target_squared_error_gradient=float(frozen_target_gradient),
        opposite_direction_counterexample=True,
        language_loss_descent_proved=False,
        primitive_derivative_scope='Lean constructs integral primitive and derives derivative via FTC; log closed form analytic/numerical only.',
        lean_scope='Selected-coordinate potential derivative and exact step; matrix formula and descent bound analytic/numerical only.',
        runtime_log_evaluations_needed=0, additional_persistent_arrays=0)
    return dict(status='passed', kind='formal_construction_equation_audit_not_language_capability',
                device=device, dtype='float64', seed=741, errors=errors,
                ttt_potential_bridge=potential_bridge,
                unchanged_runtime='3-layer GatedDeltaNetLM, conv4, MLP16, d8, 2 heads',
                fast_upstream_feedback_observed=False,
                construction='S+ = alpha*S + beta*k*((1-eta)*v+eta*softsign(g*S.T*k+u)-alpha*S.T*k).T',
                witness_coefficients=dict(alpha=.5, beta=.5, gain=4),
                witness_feedback_strength=1., original_delta_at_feedback_strength_zero=True,
                neutral_feedback_threshold=3/8, branch_checks=branch_checks,
                coefficient_scope='One feasible certificate, not architecture hyperparameters or trained values.',
                full_matrix_jacobian_radius=radius, perturbation_initial_distance=initial_distance,
                perturbation_restored_distance=restored_distance,
                held_cue_response_diagonal=float(response.diagonal().min()),
                cue_condition='Magnitude 8 in the selected value coordinates; language reachability unproved.',
                restoration_steps=40, cue_steps=1, additional_memory_arrays=0,
                gain_parameterization='Not installed; gain/strength and cue parameterization still to be chosen and charged.',
                old_language_runtime_changed=False, training_runs=0, body_admitted=False,
                historical_gradient_claim=False,
                unresolved=['language-derived cue usefulness/reachability', 'moving learned key compatibility',
                            'simultaneous multiple-mode interference', 'full capacity/P2a refinement',
                            'gradient/optimizer and scan cost of installed feedback', 'action-world closure'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--out", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--association-only", action="store_true",
                        help="Audit the read/write/retention bridge independently of exchange candidates")
    modes.add_argument("--content-budget-only", action="store_true",
                       help="Audit actual-content capacity admission and readout correction")
    modes.add_argument("--joint-content-only", action="store_true",
                       help="Audit joint evidence allocation and actual weighted content")
    modes.add_argument("--shared-memory-only", action="store_true",
                       help="Audit retained memory's influence on current write/read/gate learning")
    modes.add_argument("--retention-only", action="store_true",
                        help="Audit joint rewrite/retention/retrieval and actual optimizer readout drift")
    modes.add_argument("--feedback-only", action="store_true",
                        help="Audit a conditional same-matrix regenerative read-back construction")
    args = parser.parse_args()
    result = (audit_feedback_renewal(args.device) if args.feedback_only else
              audit_retention_retrieval(args.device) if args.retention_only else
              audit_shared_memory_learning(args.device) if args.shared_memory_only else
              audit_joint_content_exchange(args.device) if args.joint_content_only else
              audit_content_budget(args.device) if args.content_budget_only else
              audit_association_transport(args.device) if args.association_only else run(args.device))
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).relative_to(root).as_posix(),
               "lean/CGBAxioms/CGBAxioms/Generation.lean",
               "lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean",
               "lean/CGBAxioms/CGBAxioms/GenerationLattice.lean",
               "lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean",
               "lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean",
               "information_boltzmann/cgb/voxel_body.py"]
    if args.association_only:
        sources = [Path(__file__).relative_to(root).as_posix(),
                   "lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean",
                   "lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean",
                   "lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean",
                   "lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean"]
    if (args.content_budget_only or args.joint_content_only or args.shared_memory_only
            or args.retention_only or args.feedback_only):
        sources = [Path(__file__).relative_to(root).as_posix(),
                   "scripts/ib_local/gated_deltanet.py", "tests/test_gated_deltanet.py",
                   "lean/CGBAxioms/reviews/budgeted_association_write_20261103.lean"]
    if args.retention_only:
        sources += ["lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean",
                    "lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean",
                    "lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean",
                    "lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean",
                    "scripts/ib_local/train_cbim_malecns_internal_time.py"]
    if args.feedback_only:
        sources += ["lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean",
                    "lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean",
                    "lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean",
                    "lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean"]
    result["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
    }
    result["torch_version"] = torch.__version__
    result["hardware"] = torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
    rendered = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
