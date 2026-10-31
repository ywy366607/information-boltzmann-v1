import numpy as np
import torch

from fine_grain.active_gdn2 import ActiveInferenceGDN2
from fine_grain.cascade_gdn_reference import CascadeActiveInferenceGDN2
from fine_grain.capability_tasks import CAPABILITY_CASES, collate_capability, capability_sample
from tests.test_capability_closure import _model


def test_association_readout_and_probe_error_use_declared_keys():
    memory = ActiveInferenceGDN2(d_model=3, n_slots=4, res=4)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    state.precision.fill_(1.0)
    state.natural[0] = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ])
    keys = torch.zeros(1, 2, 4)
    keys[0, 0, 1] = 1.0
    keys[0, 1, 3] = 1.0
    targets = torch.tensor([[[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]])
    readout = memory.association_readout(state, keys)
    assert torch.equal(readout, targets)
    assert torch.equal(memory.association_probe_error(state, keys, targets), torch.zeros(1))


def test_initial_address_depends_on_position_not_content_but_can_learn_content():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=4, res=4)
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    same_center = memory.atlas[0].view(1, 1, 2).expand(1, 2, 2)
    initial = memory.address(tokens, same_center)
    assert torch.equal(initial[:, 0], initial[:, 1])

    with torch.no_grad():
        memory.content_address.weight[0, 0] = 1.0
    changed = memory.address(tokens, same_center)
    assert not torch.allclose(changed[:, 0], changed[:, 1])
    changed[0, 0, 0].backward()
    assert memory.content_address.weight.grad is not None
    assert memory.content_address.weight.grad.abs().sum() > 0


def test_natural_write_is_associative_matrix_transpose():
    torch.manual_seed(7)
    memory = ActiveInferenceGDN2(d_model=5, n_slots=4, res=4)
    state = memory.initial_state(2, torch.device("cpu"), torch.float32)
    tokens = torch.randn(2, 3, 5)
    centers = torch.randn(2, 3, 2)
    obs_pi = torch.tensor([1.0, 0.5])
    address = memory.address(tokens, centers)
    local_pi = torch.nn.functional.softplus(memory.log_likelihood_precision(tokens))
    event_pi = obs_pi.reshape(2, 1, 1) * local_pi / 3
    expected_add = memory.association_write(address, tokens, event_pi)
    updated = memory.assimilate(state, tokens, centers, obs_pi, update_motion=False)
    assert torch.allclose(
        memory.association_state(updated) - memory.association_state(state),
        expected_add,
        atol=1e-6,
    )


def test_cascade_gdn_reference_has_geometric_scales_and_mean_preserving_decay():
    cascade = CascadeActiveInferenceGDN2(4, 4, 4, n_levels=3, base_tau=1.0, gamma=4.0)
    state = cascade.initial_state(1, torch.device("cpu"), torch.float32)
    assert torch.allclose(cascade.time_constants, torch.tensor([1.0, 4.0, 16.0]))
    level = state.levels[0]
    level.natural.fill_(2.0)
    level.precision.fill_(3.0)
    decayed = cascade.decay_state(level, torch.tensor(1.0))
    assert torch.allclose(decayed.natural / decayed.precision, level.natural / level.precision)
    assert torch.all(decayed.precision < level.precision)


def test_zero_observation_precision_is_exact_no_write():
    torch.manual_seed(0)
    memory = ActiveInferenceGDN2(d_model=8, n_slots=4, res=8)
    state = memory.initial_state(2, torch.device("cpu"), torch.float32)
    tokens = torch.randn(2, 4, 8)
    centers = memory.atlas.unsqueeze(0).expand(2, -1, -1)
    updated = memory.assimilate(state, tokens, centers, torch.zeros(2))
    assert torch.equal(updated.natural, state.natural)
    assert torch.equal(updated.precision, state.precision)


def test_assimilation_uses_state_dependent_slot_gain_and_trainable_precision():
    torch.manual_seed(41)
    memory = ActiveInferenceGDN2(d_model=3, n_slots=4, res=4)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    state.precision = torch.tensor([[[0.1], [0.3], [0.8], [1.5]]])
    old_mean = torch.tensor([[[0.2, -0.1, 0.4], [-0.3, 0.5, 0.2],
                              [0.7, -0.4, 0.1], [0.0, 0.6, -0.2]]])
    state.natural = state.precision * old_mean
    tokens = torch.tensor([[[1.0, -0.5, 0.7], [-0.4, 0.8, 0.3]]])
    centers = memory.atlas[:2].unsqueeze(0)
    updated = memory.assimilate(state, tokens, centers, torch.ones(1), update_motion=False)
    weight = memory.address(tokens, centers).unsqueeze(-1) * (
        torch.nn.functional.softplus(memory.log_likelihood_precision(tokens)) / 2
    ).unsqueeze(-2)
    added_precision = weight.sum(dim=1)
    evidence_mean = (weight * tokens.unsqueeze(-2)).sum(dim=1) / added_precision
    gain = added_precision / (state.precision + added_precision)
    expected = old_mean + gain * (evidence_mean - old_mean)

    assert torch.allclose(updated.natural / updated.precision, expected, atol=1e-6)
    assert torch.all((gain >= 0) & (gain < 1))
    assert not torch.allclose(gain[:, 0], gain[:, -1])
    queries = memory.address(tokens, centers)
    weighted_values = (weight * tokens.unsqueeze(-2)).sum(dim=1)
    expected_change = torch.einsum(
        "bmk,bkd->bmd", queries,
        (weighted_values - added_precision * old_mean) / (state.precision + added_precision),
    )
    observed_change = memory.association_readout(updated, queries) - memory.association_readout(state, queries)
    assert torch.allclose(observed_change, expected_change, atol=1e-6)
    old_read = memory.association_readout(state, queries)[0, 0, 0]
    new_read = memory.association_readout(updated, queries)[0, 0, 0]
    delta = observed_change[0, 0, 0]
    assert abs(float(delta.detach())) > 1e-5
    for target in (old_read, new_read):
        actual_loss_change = (new_read - target).square() - (old_read - target).square()
        predicted_loss_change = delta * (2 * (old_read - target) + delta)
        assert torch.allclose(actual_loss_change, predicted_loss_change, atol=1e-6)
    assert (new_read - old_read).square() > (old_read - old_read).square()
    assert (new_read - new_read).square() < (old_read - new_read).square()
    updated.natural.div(updated.precision).square().sum().backward()
    assert memory.log_likelihood_precision.weight.grad is not None
    assert memory.log_likelihood_precision.weight.grad.abs().sum() > 0


def test_equal_mean_different_precision_changes_same_future_observation():
    """The current mean alone is not closed under actual GDN assimilation."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        memory = ActiveInferenceGDN2(d_model=2, n_slots=1, res=1).to(device).double()
        states = [memory.initial_state(1, device, torch.float64) for _ in range(2)]
        for state, precision in zip(states, (0.5, 2.0)):
            state.precision.fill_(precision)
            state.natural.zero_()
        query = torch.ones(1, 1, 1, dtype=torch.float64, device=device)
        tokens = torch.tensor([[[1.0, -0.5]]], dtype=torch.float64, device=device)
        centers = memory.atlas.unsqueeze(0)
        with torch.no_grad():
            assert torch.equal(memory.association_readout(states[0], query),
                               memory.association_readout(states[1], query))
            updated = [memory.assimilate(state, tokens, centers, 1.0, update_motion=False)
                       for state in states]
            first = memory.association_readout(updated[0], query)
            second = memory.association_readout(updated[1], query)
            assert not torch.allclose(first, second, atol=1e-10, rtol=0)
            w = torch.nn.functional.softplus(memory.log_likelihood_precision(tokens)).item()
            assert torch.allclose(first, tokens * (w / (0.5 + w)), atol=1e-12)
            assert torch.allclose(second, tokens * (w / (2.0 + w)), atol=1e-12)
            # One declared affine readout can expose the future difference.
            head = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64, device=device)
            assert not torch.allclose(torch.softmax(first @ head.T, dim=-1),
                                      torch.softmax(second @ head.T, dim=-1), atol=1e-10, rtol=0)


def test_process_variance_prevents_gain_collapse_under_repeated_observation():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=1, res=1)
    tokens = torch.tensor([[[1.0, -1.0]]])
    centers = memory.atlas.unsqueeze(0)
    with torch.no_grad():
        states = [memory.initial_state(1, torch.device("cpu"), torch.float32)
                  for _ in range(2)]
        gains = [[], []]
        q = float(torch.nn.functional.softplus(memory.process_variance_raw))
        for _ in range(1000):
            for i, horizon in enumerate((0.0, 1.0)):
                prior = memory.dynamic_prior(states[i], horizon)
                updated = memory.assimilate(prior, tokens, centers, 1.0, update_motion=False)
                w = updated.precision - prior.precision
                gains[i].append(float((w / updated.precision).item()))
                states[i] = updated
        wmin = float(torch.nn.functional.softplus(memory.log_likelihood_precision(tokens)).item())
        lower_bound = wmin / (1.0 / q + wmin)
    assert gains[0][-1] < 0.002
    assert min(gains[1]) >= lower_bound - 1e-6
    assert gains[1][-1] > 0.05


def test_predict_assimilate_query_matches_shared_slot_contract():
    """One physical update satisfies the existing gain and query identities."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        memory = ActiveInferenceGDN2(d_model=3, n_slots=4, res=4).to(device).double()
        state = memory.initial_state(1, device, torch.float64)
        state.precision = torch.tensor([[[0.2], [0.5], [1.2], [2.0]]],
                                       device=device, dtype=torch.float64)
        old_mean = torch.tensor([[[0.3, -0.2, 0.6], [-0.4, 0.1, 0.8],
                                  [0.9, -0.7, 0.2], [0.1, 0.5, -0.6]]],
                                device=device, dtype=torch.float64)
        state.natural = state.precision * old_mean
        tokens = torch.tensor([[[0.7, -0.5, 0.4], [-0.3, 0.8, 0.2]]],
                              device=device, dtype=torch.float64)
        centers = memory.atlas[:2].unsqueeze(0)
        horizon = 0.7
        with torch.no_grad():
            prior = memory.dynamic_prior(state, horizon)
            q = horizon * torch.nn.functional.softplus(memory.process_variance_raw)
            expected_prior_precision = state.precision / (1 + q * state.precision)
            assert torch.allclose(prior.precision, expected_prior_precision, atol=1e-12)
            assert torch.allclose(prior.natural / prior.precision, old_mean, atol=1e-12)

            keys = memory.address(tokens, centers)
            event_precision = torch.nn.functional.softplus(
                memory.log_likelihood_precision(tokens)) / tokens.shape[1]
            weights = keys.unsqueeze(-1) * event_precision.unsqueeze(-2)
            weight = weights.sum(dim=1)
            weighted = (weights * tokens.unsqueeze(-2)).sum(dim=1)
            updated = memory.assimilate(prior, tokens, centers, 1.0, update_motion=False)
            updated_mean = updated.natural / updated.precision
            expected_mean = (expected_prior_precision * old_mean + weighted) / (
                expected_prior_precision + weight)
            assert torch.allclose(updated_mean, expected_mean, atol=1e-12)

            gain = weight / (expected_prior_precision + weight)
            gain_floor = weight / (1 / q + weight)
            assert torch.all(weight > 0) and torch.all(q > 0)
            assert torch.all(gain >= gain_floor - 1e-12)
            query_change = memory.association_readout(updated, keys) - (
                memory.association_readout(state, keys))
            expected_change = torch.einsum(
                "bmk,bkd->bmd", keys,
                (weighted - weight * old_mean) / (expected_prior_precision + weight),
            )
            assert torch.allclose(query_change, expected_change, atol=1e-12)
            old_logits = memory.association_readout(state, keys)
            new_logits = memory.association_readout(updated, keys)
            old_probability = torch.softmax(old_logits, dim=-1)
            new_probability = torch.softmax(new_logits, dim=-1)
            multiplier = torch.exp(expected_change)
            average_multiplier = (old_probability * multiplier).sum(dim=-1, keepdim=True)
            assert torch.allclose(new_probability,
                                  old_probability * multiplier / average_multiplier,
                                  atol=1e-12)
            margin = multiplier - average_multiplier
            assert torch.any(margin > 1e-8) and torch.any(margin < -1e-8)
            assert torch.equal(new_probability > old_probability, margin > 0)
            # A fixed affine language head may have a different output width.
            head = torch.tensor([[0.5, -0.2, 0.1], [-0.3, 0.7, 0.4],
                                 [0.2, 0.1, -0.6], [0.8, 0.3, 0.2],
                                 [-0.4, -0.1, 0.9]], dtype=torch.float64, device=device)
            bias = torch.tensor([0.1, -0.2, 0.3, 0.0, -0.1],
                                dtype=torch.float64, device=device)
            z0 = old_logits @ head.T + bias
            z1 = new_logits @ head.T + bias
            dz = expected_change @ head.T
            prob0, prob1 = torch.softmax(z0, dim=-1), torch.softmax(z1, dim=-1)
            factor = torch.exp(dz)
            mean_factor = (prob0 * factor).sum(dim=-1, keepdim=True)
            assert torch.allclose(prob1, prob0 * factor / mean_factor, atol=1e-12)
            assert torch.any(factor > mean_factor + 1e-8)
            assert torch.any(factor < mean_factor - 1e-8)
            assert torch.equal(prob1 > prob0, factor > mean_factor)
            head_shift = torch.tensor([[0.03, 0.01, -0.02], [-0.02, 0.01, 0.01],
                                       [0.01, -0.02, 0.02], [0.0, 0.02, -0.01],
                                       [-0.01, 0.01, 0.03]], dtype=torch.float64,
                                      device=device)
            bias_shift = torch.tensor([-0.03, 0.02, -0.01, 0.03, 0.0],
                                      dtype=torch.float64, device=device)
            next_head = head + head_shift
            z2 = new_logits @ next_head.T + bias + bias_shift
            dz2 = old_logits @ head_shift.T + expected_change @ next_head.T + bias_shift
            assert torch.allclose(z2 - z0, dz2, atol=1e-12)
            prob2 = torch.softmax(z2, dim=-1)
            factor2 = torch.exp(dz2)
            mean_factor2 = (prob0 * factor2).sum(dim=-1, keepdim=True)
            assert torch.allclose(prob2, prob0 * factor2 / mean_factor2, atol=1e-12)
            assert torch.any(factor2 > mean_factor2 + 1e-8)
            assert torch.any(factor2 < mean_factor2 - 1e-8)
            assert torch.equal(prob2 > prob0, factor2 > mean_factor2)
            next_keys = keys.roll(1, dims=1)
            next_read = memory.association_readout(updated, next_keys)
            query_effect = torch.einsum("bmk,bkd->bmd", next_keys - keys, old_mean)
            state_effect = torch.einsum(
                "bmk,bkd->bmd", next_keys,
                (weighted - weight * old_mean) / (expected_prior_precision + weight),
            )
            assert torch.allclose(next_read - old_logits, query_effect + state_effect, atol=1e-12)
            z3 = next_read @ next_head.T + bias + bias_shift
            dz3 = (old_logits @ head_shift.T +
                   (query_effect + state_effect) @ next_head.T + bias_shift)
            assert torch.allclose(z3 - z0, dz3, atol=1e-12)
            prob3 = torch.softmax(z3, dim=-1)
            factor3 = torch.exp(dz3)
            mean_factor3 = (prob0 * factor3).sum(dim=-1, keepdim=True)
            assert torch.allclose(prob3, prob0 * factor3 / mean_factor3, atol=1e-12)
            assert torch.any(factor3 > mean_factor3 + 1e-8)
            assert torch.any(factor3 < mean_factor3 - 1e-8)
            assert torch.equal(prob3 > prob0, factor3 > mean_factor3)


def test_history_current_frame_and_future_state_time_are_explicit(monkeypatch):
    """Audit the current physical-time contract without changing the model."""
    torch.manual_seed(42)
    model = _model(use_active_gdn2=True).eval()
    memory = model.mot_stack.active_gdn2
    original_assimilate = memory.assimilate
    original_prior = memory.dynamic_prior
    observations, horizons = [], []

    def capture_assimilate(state, tokens, centers, observation_precision, **kwargs):
        result = original_assimilate(state, tokens, centers, observation_precision, **kwargs)
        observations.append((tokens.detach().clone(),
                             (result.precision - state.precision).detach().clone()))
        return result

    def capture_prior(state, horizon, **kwargs):
        horizons.append(torch.as_tensor(horizon).detach().clone())
        return original_prior(state, horizon, **kwargs)

    monkeypatch.setattr(memory, "assimilate", capture_assimilate)
    monkeypatch.setattr(memory, "dynamic_prior", capture_prior)
    image = torch.rand(1, 3, 16, 16)
    history = torch.stack((image * 0.5, image), dim=1)
    with torch.no_grad():
        out = model(image, ["Predict next frame"],
                    image_precision=torch.ones(1), text_precision=torch.ones(1),
                    target_time=torch.ones(1), history_images=history,
                    history_precision=torch.ones(1, 2), action=torch.zeros(1, 2),
                    action_precision=torch.zeros(1))
    # The final history image is the current image in the supplied convention.
    # Present code assimilates it twice; this test documents the observed
    # precision accounting rather than declaring the duplication desirable.
    assert len(observations) == 3
    assert torch.allclose(observations[-2][0], observations[-1][0])
    assert torch.allclose(observations[-2][1], observations[-1][1], atol=1e-6)
    assert len(horizons) == 1 and torch.equal(horizons[0], torch.ones(1))
    assert out["causal_state"] is model.mot_stack._last_causal_state


def test_dynamic_prior_releases_precision_without_moving_mean():
    torch.manual_seed(1)
    memory = ActiveInferenceGDN2(d_model=8, n_slots=4, res=8)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.randn(1, 4, 8)
    centers = memory.atlas.unsqueeze(0)
    state = memory.assimilate(state, tokens, centers, torch.ones(1))
    old_mean = state.natural / state.precision
    predicted = memory.dynamic_prior(
        state, torch.ones(1), action=torch.tensor([[1.0, 0.0]]),
        action_precision=torch.ones(1),
    )
    new_mean = predicted.natural / predicted.precision
    assert torch.allclose(new_mean, old_mean, atol=1e-6)
    assert torch.all(predicted.precision < state.precision)


def test_delta_update_erases_current_key_prediction_before_write():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=3, res=3)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    state.precision.fill_(1.0)
    key = torch.tensor([[0.5, 1.0, 0.0]])
    key = key / key.norm(dim=-1, keepdim=True)
    first = torch.tensor([[2.0, -1.0]])
    second = torch.tensor([[0.25, 3.0]])
    state = memory.association_delta_update(state, key, first)
    updated = memory.association_delta_update(state, key, second)
    readout = memory.association_readout(updated, key.unsqueeze(1)).squeeze(1)
    assert torch.allclose(readout, second, atol=1e-5)


def test_one_hot_precision_assimilation_matches_delta_reference():
    """Controlled one-slot address; no claim about soft-address training."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        memory = ActiveInferenceGDN2(d_model=2, n_slots=3, res=3).to(device).double()
        state = memory.initial_state(1, device, torch.float64)
        state.precision.fill_(2.0)
        state.natural = state.precision * torch.tensor(
            [[[0.1, -0.5], [0.3, 0.2], [-0.4, 0.8]]],
            dtype=torch.float64, device=device)
        key = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64, device=device)
        value = torch.tensor([[0.7, -0.3]], dtype=torch.float64, device=device)
        centers = memory.atlas[:1].unsqueeze(0)
        memory.address = lambda tokens, centers: key.unsqueeze(1)
        with torch.no_grad():
            event_pi = torch.nn.functional.softplus(
                memory.log_likelihood_precision(value.unsqueeze(1))).item()
            gain = event_pi / (2.0 + event_pi)
            assimilated = memory.assimilate(
                state, value.unsqueeze(1), centers, 1.0, update_motion=False)
            delta = memory.association_delta_update(state, key, value, beta=gain)
            assert torch.allclose(assimilated.natural / assimilated.precision,
                                  delta.natural / delta.precision, atol=1e-12)


def test_delta_reference_updates_mean_with_heterogeneous_slot_precision():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=3, res=3)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    state.precision = torch.tensor([[[0.2], [2.0], [5.0]]])
    state.natural = state.precision * torch.tensor(
        [[[0.5, -0.5], [1.0, 0.0], [-0.5, 0.2]]])
    key = torch.tensor([[0.6, 0.8, 0.0]])
    value = torch.tensor([[2.0, -1.0]])
    prior_mean = state.natural / state.precision
    prior_prediction = memory.association_readout(state, key.unsqueeze(1)).squeeze(1)
    updated = memory.association_delta_update(state, key, value)
    expected_mean = prior_mean + torch.einsum("bd,bk->bkd", value - prior_prediction, key)
    assert torch.allclose(updated.natural / updated.precision, expected_mean, atol=1e-6)
    assert torch.equal(updated.precision, state.precision)
    assert torch.allclose(memory.association_readout(updated, key.unsqueeze(1)).squeeze(1),
                          value, atol=1e-6)


def test_delta_reference_reuses_fixed_state_for_repeated_arrivals():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=3, res=3)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    state.precision.fill_(1.0)
    key = torch.tensor([[0.6, 0.8, 0.0]])
    for i in range(20):
        value = torch.tensor([[float(i % 3), float(-(i % 4))]])
        state = memory.association_delta_update(state, key, value, beta=1.0)
        readout = memory.association_readout(state, key.unsqueeze(1)).squeeze(1)
        assert torch.allclose(readout, value, atol=1e-5)
        assert state.natural.shape == (1, 3, 2)


def test_delta_write_cross_read_follows_key_overlap_and_prediction_error():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=3, res=3)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    state.precision = torch.tensor([[[0.2], [2.0], [5.0]]])
    state.natural[0] = state.precision[0] * torch.tensor(
        [[0.2, -0.3], [0.7, 0.4], [0.1, -0.5]])
    key = torch.tensor([[1.0, 0.0, 0.0]])
    queries = torch.tensor([[[0.0, 1.0, 0.0], [0.6, 0.8, 0.0]]])
    value = torch.tensor([[1.5, -0.7]])
    beta = 0.75
    old = memory.association_readout(state, queries)
    prediction = memory.association_readout(state, key.unsqueeze(1)).squeeze(1)
    updated = memory.association_delta_update(state, key, value, beta=beta)
    observed = memory.association_readout(updated, queries) - old
    overlap = torch.einsum("bqk,bk->bq", queries, key)
    expected = beta * overlap.unsqueeze(-1) * (value - prediction).unsqueeze(1)
    assert torch.allclose(observed, expected, atol=1e-6)
    assert torch.allclose(observed[:, 0], torch.zeros_like(observed[:, 0]), atol=1e-6)
    assert observed[:, 1].abs().max() > 0


def test_soft_addresses_have_overlapping_delta_effects():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=3, res=3)
    state = memory.initial_state(1, torch.device("cpu"), torch.float64)
    memory = memory.double()
    state.precision.fill_(1.0)
    tokens = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], dtype=torch.float64)
    centers = memory.atlas[:2].unsqueeze(0)
    addresses = memory.address(tokens, centers)
    assert torch.all(addresses > 0)
    key = addresses[:, 0]
    query = addresses[:, 1:2]
    before = memory.association_readout(state, query)
    value = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    changed = memory.association_delta_update(state, key, value, beta=1.0)
    difference = memory.association_readout(changed, query) - before
    overlap = (query * key.unsqueeze(1)).sum(dim=-1)
    assert torch.all(overlap > 0)
    assert torch.allclose(difference, overlap.unsqueeze(-1) * value.unsqueeze(1), atol=1e-12)
    assert torch.all(difference != 0)


def test_normalized_assimilation_uses_effective_keys_and_prior_residual():
    """Actual natural writes compose with per-slot precision and old state."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        torch.manual_seed(31)
        memory = ActiveInferenceGDN2(d_model=4, n_slots=9, res=9).to(device).double()
        tokens = torch.tanh(torch.randn(1, 4, 4, dtype=torch.float64, device=device))
        centers = memory.atlas[:4].unsqueeze(0)
        state = memory.initial_state(1, device, torch.float64)
        state.precision = torch.linspace(0.3, 1.7, 9, dtype=torch.float64,
                                         device=device).reshape(1, 9, 1)
        state.natural = state.precision * torch.linspace(
            -0.2, 0.3, 36, dtype=torch.float64, device=device).reshape(1, 9, 4)
        with torch.no_grad():
            keys = memory.address(tokens, centers)
            event_pi = torch.nn.functional.softplus(
                memory.log_likelihood_precision(tokens)) / tokens.shape[1]
            updated = memory.assimilate(state, tokens, centers, 1.0, update_motion=False)
            effective_keys = keys * event_pi / updated.precision.transpose(1, 2)
            normalized = updated.natural / updated.precision
            prior = state.natural / updated.precision
            expected_write = memory.association_write(effective_keys, tokens).transpose(1, 2)
            assert torch.allclose(normalized, prior + expected_write, atol=1e-12)

            dual = memory.association_dual_keys(effective_keys)
            actual_read = memory.association_readout(updated, dual)
            prior_read = torch.einsum("bkd,bmk->bmd", prior, dual)
            assert torch.allclose(actual_read, tokens + prior_read, atol=1e-9)
            assert torch.linalg.vector_norm(prior_read) > 0


def test_effective_keys_track_two_writes_across_actual_process_prior():
    """Historical key bookkeeping is a reference, not part of model state."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        torch.manual_seed(55)
        memory = ActiveInferenceGDN2(d_model=4, n_slots=9, res=9).to(device).double()
        values = torch.tanh(torch.randn(1, 4, 4, dtype=torch.float64, device=device))
        state = memory.initial_state(1, device, torch.float64)
        state.precision = torch.linspace(0.3, 1.7, 9, device=device,
                                         dtype=torch.float64).reshape(1, 9, 1)
        state.natural = state.precision * torch.linspace(
            -0.2, 0.3, 36, device=device, dtype=torch.float64).reshape(1, 9, 4)
        initial_natural = state.natural.clone()
        with torch.no_grad():
            early, late = values[:, :2], values[:, 2:]
            early_centers = memory.atlas[:2].unsqueeze(0)
            late_centers = memory.atlas[2:4].unsqueeze(0)
            early_keys = memory.address(early, early_centers)
            early_weight = torch.nn.functional.softplus(memory.log_likelihood_precision(early)) / 2
            first = memory.assimilate(state, early, early_centers, 1.0, update_motion=False)
            prior = memory.dynamic_prior(first, 0.7)
            retention = prior.precision / first.precision
            late_keys = memory.address(late, late_centers)
            late_weight = torch.nn.functional.softplus(memory.log_likelihood_precision(late)) / 2
            last = memory.assimilate(prior, late, late_centers, 1.0, update_motion=False)

            denominator = last.precision.transpose(1, 2)
            effective_early = early_keys * early_weight * retention.transpose(1, 2) / denominator
            effective_late = late_keys * late_weight / denominator
            effective = torch.cat((effective_early, effective_late), dim=1)
            prior_residual = initial_natural * retention / last.precision
            reconstructed = prior_residual + memory.association_write(
                effective, values).transpose(1, 2)
            assert torch.allclose(last.natural / last.precision, reconstructed, atol=1e-12)
            dual = memory.association_dual_keys(effective)
            read = memory.association_readout(last, dual)
            residual = torch.einsum("bkd,bmk->bmd", prior_residual, dual)
            assert torch.allclose(read, values + residual, atol=1e-9)


def test_gram_dual_readout_recovers_nonorthogonal_associations():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=3, res=3)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    keys = torch.tensor([[[1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]])
    values = torch.tensor([[[2.0, -1.0], [0.25, 3.0]]])
    state.precision.fill_(1.0)
    state.natural = memory.association_write(keys, values).transpose(-1, -2)
    dual = memory.association_dual_readout(state, keys)
    assert torch.allclose(dual, values, atol=1e-6)


def test_bounded_decoder_bridge_uses_actual_gdn_addresses_and_state():
    """The abstract decoder contract is exercised on the real address path.

    `address` supplies the key family and `association_write` supplies the
    persistent V Kᵀ state.  For an independent finite probe family, the Gram
    dual is the runtime linear decoder, so the decoded values are recovered
    up to floating-point error without changing the state representation.
    """
    torch.manual_seed(31)
    memory = ActiveInferenceGDN2(d_model=4, n_slots=9, res=9)
    tokens = torch.tanh(torch.randn(1, 4, 4))
    centers = memory.atlas[:4].unsqueeze(0)
    keys = memory.address(tokens, centers)
    values = tokens.clone()
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    state.precision.fill_(1.0)
    state.natural = memory.association_write(keys, values).transpose(-1, -2)

    gram = keys @ keys.transpose(-1, -2)
    decoder_residual = gram @ torch.linalg.solve(gram, torch.eye(4).unsqueeze(0)) - torch.eye(4).unsqueeze(0)
    decoded = memory.association_dual_readout(state, keys)

    assert torch.linalg.matrix_rank(keys[0]) == 4
    assert float(decoder_residual.detach().abs().amax()) < 1e-5
    assert float((decoded - values).detach().abs().amax()) < 1e-5


def test_interference_bound_matches_actual_assimilation_readout():
    """Check the uniform algebraic bound against the actual GDN update."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        torch.manual_seed(42)
        memory = ActiveInferenceGDN2(d_model=4, n_slots=9, res=9).to(device=device, dtype=torch.float64)
        values = torch.tanh(torch.randn(1, 4, 4, device=device, dtype=torch.float64))
        centers = memory.atlas[:4].unsqueeze(0)
        state = memory.initial_state(1, torch.device(device), torch.float64)
        old_natural = state.natural.clone()
        keys = memory.address(values, centers)
        weights = torch.nn.functional.softplus(memory.log_likelihood_precision(values)) / values.shape[1]
        state = memory.assimilate(state, values, centers, 1.0, update_motion=False)
        effective = keys * weights / state.precision.transpose(1, 2)
        # These are the actual address queries, not an assumed Gram inverse.
        queries = keys
        bound = memory.association_interference_bound(effective, queries)
        prior = torch.einsum("bkd,bmk->bmd", old_natural / state.precision, queries)
        read = memory.association_readout(state, queries)
        error = (read - prior - values).abs()
        assert torch.all(error <= bound.unsqueeze(-1) + 1e-12)
        # The old-state residual is common to both references, so the
        # certified difference isolates interference from the new values.
        head = torch.randn(256, 4, device=device, dtype=torch.float64)
        bias = torch.randn(256, device=device, dtype=torch.float64)
        actual_logits = torch.nn.functional.linear(read, head, bias)
        ideal_logits = torch.nn.functional.linear(values + prior, head, bias)
        actual_nll = -torch.log_softmax(actual_logits, dim=-1)[..., 65]
        ideal_nll = -torch.log_softmax(ideal_logits, dim=-1)[..., 65]
        head_bound = head.abs().sum(dim=-1).amax()
        assert torch.all((actual_nll - ideal_nll).abs() <= 2 * bound * head_bound + 1e-11)


def test_dynamic_prior_increases_probe_uncertainty_without_changing_mean():
    torch.manual_seed(8)
    memory = ActiveInferenceGDN2(d_model=4, n_slots=4, res=4)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.randn(1, 4, 4)
    centers = memory.atlas.unsqueeze(0)
    state = memory.assimilate(state, tokens, centers, torch.ones(1), update_motion=False)
    keys = memory.address(tokens, centers)
    before_mean = state.natural / state.precision
    before_var = memory.association_probe_uncertainty(state, keys)
    erased = memory.dynamic_prior(state, torch.tensor([8.0]))
    after_mean = erased.natural / erased.precision
    after_var = memory.association_probe_uncertainty(erased, keys)
    assert torch.allclose(before_mean, after_mean, atol=1e-6)
    assert torch.all(after_var >= before_var)
    assert torch.any(after_var > before_var)


def test_action_changes_source_address_not_transient_slice_identity():
    torch.manual_seed(2)
    memory = ActiveInferenceGDN2(d_model=4, n_slots=9, res=9)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.randn(1, 9, 4)
    centers = memory.atlas.unsqueeze(0)
    state = memory.assimilate(state, tokens, centers, torch.ones(1))
    _, _, still = memory.query(
        state, tokens, centers, horizon=torch.ones(1),
        action=torch.zeros(1, 2), action_precision=torch.ones(1),
    )
    moved, _, active = memory.query(
        state, tokens, centers, horizon=torch.ones(1),
        action=torch.tensor([[1.0, 0.0]]), action_precision=torch.ones(1),
    )
    assert not torch.equal(active["source_centers"], still["source_centers"])
    assert torch.allclose(
        active["displacement_yx"][:, 1],
        torch.full((1,), 2.0 / 3.0), atol=1e-5,
    )
    moved.square().mean().backward()
    assert memory.action_scale_raw.grad is not None
    assert float(memory.action_scale_raw.grad.abs()) > 0.0


def test_prior_action_landing_is_exact_identity_at_initialization_and_trainable():
    torch.manual_seed(20)
    memory = ActiveInferenceGDN2(d_model=8, n_slots=4, res=8)
    current = torch.randn(2, 4, 8)
    prior = torch.randn_like(current)
    landed = memory.apply_prior_action(current, prior)
    assert torch.equal(landed, current)
    landed.square().mean().backward()
    grad = memory.prior_channel_gain.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_persistent_atlas_deslice_is_spatial_and_content_independent():
    memory = ActiveInferenceGDN2(d_model=8, n_slots=9, res=9)
    axis = torch.linspace(-1.0, 1.0, 9)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([yy, xx], dim=-1).reshape(1, -1, 2)
    weights = memory.point_to_atlas_weights(points)
    assert weights.shape == (1, 81, 9)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 81), atol=1e-6)
    # A persistent address map must resolve space; uniform assignments would
    # recreate the rejected transient-Deslice bottleneck.
    assert float(weights.detach().std(dim=1).mean()) > 0.05
    shifted = memory.point_to_atlas_weights(points.flip(1))
    assert torch.equal(shifted, weights.flip(1))


def test_predict_step_transports_belief_before_observation_correction():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=9, res=9)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    # Put a sharp identifiable belief at the atlas center.
    center = int(torch.argmin(memory.atlas.square().sum(dim=-1)))
    state.precision.fill_(100.0)
    state.natural.zero_()
    state.natural[0, center, 0] = 100.0
    moved = memory.transport_state(state, torch.tensor([[0.0, 2.0 / 3.0]]))
    mean = moved.natural / moved.precision
    peak = int(mean[0, :, 0].argmax())
    assert float(memory.atlas[peak, 1]) > float(memory.atlas[center, 1])


def test_unified_future_vfe_trains_causal_memory_in_same_slice_chart():
    torch.manual_seed(3)
    model = _model(
        action_tokens=True,
        action_rel_bias=True,
        transition_gate=True,
        target_time_adaln=False,
        transition_loss_coef=0.5,
        use_active_gdn2=True,
        causal_memory_loss_coef=0.25,
    )
    rng = np.random.default_rng(3)
    batch = collate_capability([
        capability_sample(rng, 16, case, digit="3", color="red", place="middle_center")
        for case in CAPABILITY_CASES
    ])
    zeros = torch.zeros(len(CAPABILITY_CASES))
    batch["t"] = zeros
    out = model.forward_with_future_posterior(
        batch["image"], batch["prompt"], batch["target_rgb"], t=zeros,
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
        target_time=batch["target_time"],
        history_images=batch["history_images"],
        history_precision=batch["history_precision"],
        action=batch["action"], action_precision=batch["action_precision"],
    )
    assert out["causal_prior_mu"].shape == (4, 8, 32)
    assert out["transition_q_causal_mu"].shape == (4, 8, 32)
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert meta["causal_transition_kl"] >= 0.0
    loss.backward()
    grad = model.mot_stack.active_gdn2.content_address.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_present_boundary_keeps_active_memory_write_path_exactly_closed():
    torch.manual_seed(4)
    model = _model(
        action_tokens=True,
        action_rel_bias=True,
        transition_gate=True,
        use_active_gdn2=True,
    ).eval()
    image = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        model(
            image, ["Reconstruct current frame"] * 2,
            image_precision=torch.ones(2), text_precision=torch.ones(2),
            target_time=torch.zeros(2), action=torch.tensor([[1.0, 0.0]]).expand(2, -1),
            action_precision=torch.ones(2),
        )
    gate = model.mot_stack.layers[0].last_causal_gate
    assert gate is not None
    assert torch.equal(gate, torch.zeros_like(gate))


def test_semantic_generation_does_not_open_physical_prior_gate():
    torch.manual_seed(41)
    model = _model(use_active_gdn2=True).eval()
    with torch.no_grad():
        model(
            torch.zeros(1, 3, 16, 16),
            ["Draw digit 3 with a thin red stroke at middle center"],
            image_precision=torch.zeros(1), text_precision=torch.ones(1),
            target_time=torch.zeros(1), action=torch.zeros(1, 2),
            action_precision=torch.zeros(1),
        )
    gate = model.mot_stack.layers[0].last_causal_gate
    assert gate is not None
    assert torch.equal(gate, torch.zeros_like(gate))


def test_enabling_active_memory_does_not_advance_global_rng_stream():
    torch.manual_seed(99)
    plain = _model(use_active_gdn2=False)
    plain_tail = plain.seg_head[-1].weight.detach().clone()
    torch.manual_seed(99)
    active = _model(use_active_gdn2=True)
    active_tail = active.seg_head[-1].weight.detach().clone()
    assert torch.equal(plain_tail, active_tail)


def test_assimilate_explicit_shared_views_match_legacy_path_exactly():
    """Separating value/address/likelihood ports preserves the shared default."""
    torch.manual_seed(23)
    memory = ActiveInferenceGDN2(d_model=3, n_slots=4, res=4).double()
    legacy = memory.initial_state(2, torch.device("cpu"), torch.float64)
    explicit = memory.initial_state(2, torch.device("cpu"), torch.float64)
    tokens = torch.randn(2, 2, 3, dtype=torch.float64)
    centers = memory.atlas[:2].double().unsqueeze(0).expand(2, -1, -1).clone()
    observation_precision = torch.tensor([0.7, 1.3], dtype=torch.float64)

    legacy = memory.assimilate(
        legacy, tokens, centers, observation_precision, update_motion=False,
    )
    explicit = memory.assimilate(
        explicit, tokens, centers, observation_precision, update_motion=False,
        address_tokens=tokens, likelihood_tokens=tokens,
    )
    for name in ("natural", "precision", "velocity_mean", "velocity_precision",
                 "last_tokens", "last_centers", "last_observation_precision"):
        torch.testing.assert_close(getattr(explicit, name), getattr(legacy, name),
                                   rtol=0, atol=0)
