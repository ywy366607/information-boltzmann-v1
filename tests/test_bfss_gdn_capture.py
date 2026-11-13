"""Numerical interface checks, not a language capability experiment."""
import copy
import torch

from scripts.ib_local.bfss_gdn_capture import matrix_view, observed_momentum, quiet_packed
from scripts.ib_local.bfss_gdn_reference import event_step
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM, admit_association_content
from scripts.ib_local.navigation_bfss_dynamics import configure_momentum
from test_navigation_internal_rollout import engine


def test_observed_layer_replay_matches_real_packed_write_with_admission():
    torch.manual_seed(21)
    base = GatedDeltaNetLM(256, 16, 1, 2, 'decay_then_delta', 32, 4, .02,
                           decay_init='log_timescale')
    layer = base.layers[0]
    x = torch.randn(1, 9, 16)
    initial = base.initial_state(1)[0]
    p = torch.randn_like(matrix_view(layer, initial))
    _, actual, _ = layer(x, initial)
    s, carried, correction = observed_momentum(layer, x, initial, p)
    torch.testing.assert_close(s, matrix_view(layer, actual), atol=2e-7, rtol=2e-5)
    assert correction > 0 and torch.isfinite(carried).all()


def test_chunking_produces_identical_momentum_and_state():
    torch.manual_seed(22)
    base = GatedDeltaNetLM(256, 16, 1, 2, 'decay_then_delta', 32, 4, .03,
                           decay_init='log_timescale')
    layer = base.layers[0]
    x = torch.randn(1, 11, 16)
    start = base.initial_state(1)[0]
    p0 = torch.randn_like(matrix_view(layer, start))
    whole_s, whole_p, _ = observed_momentum(layer, x, start, p0)
    _, middle, _ = layer(x[:, :5], start)
    _, split_p, _ = observed_momentum(layer, x[:, :5], start, p0)
    split_s, split_p, _ = observed_momentum(layer, x[:, 5:], middle, split_p)
    torch.testing.assert_close(whole_s, split_s, atol=1e-7, rtol=2e-5)
    torch.testing.assert_close(whole_p, split_p, atol=1e-7, rtol=2e-5)


def test_zero_internal_time_preserves_complete_state_and_quiet_keeps_conv_history():
    torch.manual_seed(23)
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32.,
                           decay_init='log_timescale')
    state = torch.randn_like(base.initial_state(1))
    momenta = [torch.randn_like(matrix_view(layer, state[i])) for i, layer in enumerate(base.layers)]
    same, same_p = quiet_packed(base, state, momenta, dt=0., frequency=.8, coupling=.1, steps=1)
    assert torch.equal(same, state)
    assert all(torch.equal(a, b) for a, b in zip(momenta, same_p))
    changed, _ = quiet_packed(base, state, momenta, dt=.01, frequency=.8, coupling=.1, steps=1)
    for i, layer in enumerate(base.layers):
        assert torch.equal(changed[i, :, layer.matrix_size:], state[i, :, layer.matrix_size:])
    assert not torch.equal(changed, state)


def test_actual_body_migration_preserves_existing_state_and_captures_objective():
    e = engine()
    e.recurrent = e.model.body.base.initial_state(1)
    before = copy.deepcopy(e.model.state_dict())
    original = e.recurrent.clone()
    configure_momentum(e, dt=.01)
    assert torch.equal(e.recurrent, original)
    assert all(torch.equal(before[k], v) for k, v in e.model.state_dict().items()
               if torch.is_tensor(v))
    e.model._momentum_active = True
    ids = torch.randint(0, 256, (1, 4))
    logp = torch.full((1, 4, 256), -5.54517744448)
    directions = torch.randn(8, 4)
    loss, next_state, _, _ = e.model.objective(ids, ids, e.recurrent, logp, directions)
    assert torch.isfinite(loss)
    assert len(e.model._momentum_captured) == len(e.model.body.base.layers)
    assert all(p.norm() > 0 for p, _, _ in e.model._momentum_captured)
    assert next_state.shape == e.recurrent.shape
