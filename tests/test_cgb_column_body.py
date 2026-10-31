import torch

from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody, ColumnBodyState


def _body() -> CGBColumnLanguageBody:
    torch.manual_seed(7)
    return CGBColumnLanguageBody(6, (3, 4), active_columns=3)


def test_common_state_is_exactly_h_c_s_and_readouts_add_no_history_page():
    body = _body()
    state = body.initial_state(2)
    assert state.activity.shape == (2, 3, 4, 6)
    assert state.natural.shape == state.activity.shape
    assert state.precision.shape == (2, 3, 4, 1)
    assert state.participation.shape == (2, 3, 4, 4, 2)
    assert set(state.__dataclass_fields__) == {
        "activity", "natural", "precision", "participation",
    }
    demand = body.demand(state)
    trace = body.trace(state)
    assert demand.shape == state.precision.shape
    assert trace.shape == state.participation.shape
    assert torch.equal(trace, state.participation * demand.unsqueeze(-2))


def test_source_type_is_retained_in_local_s_without_a_separate_action_log():
    body = _body()
    state = body.initial_state(1)
    token = torch.tensor([31], dtype=torch.long)
    external = body.step(state, token, elapsed=0.0, source="external")
    self_event = body.step(state, token, elapsed=0.0, source="self")

    assert torch.equal(external.route, self_event.route)
    assert torch.equal(external.state.activity, self_event.state.activity)
    assert torch.equal(external.state.natural, self_event.state.natural)
    assert torch.equal(external.state.precision, self_event.state.precision)
    assert not torch.equal(external.state.participation, self_event.state.participation)
    assert external.state.participation[..., 0].sum() > 0
    assert external.state.participation[..., 1].sum() == 0
    assert self_event.state.participation[..., 1].sum() > 0
    assert self_event.state.participation[..., 0].sum() == 0


def test_prediction_is_pre_update_and_zero_elapsed_only_writes_selected_carrier_columns():
    body = _body()
    state = body.initial_state(1)
    token = torch.tensor([9], dtype=torch.long)
    first = body.step(state, token, elapsed=0.0)
    second = body.step(first.state, token, elapsed=0.0)

    # Same old state produces an identical prepared prediction; updating it
    # changes the next prediction only after the event was written.
    replay = body.step(state, token, elapsed=0.0)
    assert torch.equal(first.logits, replay.logits)
    assert not torch.equal(first.logits, second.logits)

    active = first.route > 0
    carrier_change = (first.state.natural - state.natural).abs().amax(dim=-1) > 0
    assert torch.equal(carrier_change, active)
    assert torch.equal(first.state.precision.squeeze(-1) > state.precision.squeeze(-1), active)


def test_implicit_edge_flux_has_zero_total_and_no_edge_index_state():
    body = _body()
    state = body.initial_state(2)
    state = body.step(state, torch.tensor([4, 5]), elapsed=0.0).state
    step = body.step(state, torch.tensor([6, 7]), elapsed=0.0)
    east, south = step.edge_flux[..., 0], step.edge_flux[..., 1]
    net = east.roll(1, dims=2) - east + south.roll(1, dims=1) - south
    assert torch.allclose(net.sum(dim=(1, 2)), torch.zeros(2), atol=1e-6)
    assert not any("edge_index" in field or "history" in field
                   for field in step.state.__dataclass_fields__)


def test_state_detach_preserves_values_and_removes_autograd_history():
    body = _body()
    state = body.initial_state(1)
    step = body.step(state, torch.tensor([12]), elapsed=1.0)
    loss = step.logits.square().mean() + step.state.natural.square().mean()
    loss.backward()
    detached = body.detach_state(step.state)
    for before, after in zip(
        (step.state.activity, step.state.natural, step.state.precision, step.state.participation),
        (detached.activity, detached.natural, detached.precision, detached.participation),
    ):
        assert torch.equal(before, after)
        assert not after.requires_grad


def test_zero_elapsed_language_arrival_leaves_nonparticipants_unchanged():
    body = _body()
    initial = body.initial_state(1)
    state = ColumnBodyState(
        activity=torch.empty_like(initial.activity).uniform_(-0.7, 0.7),
        natural=torch.randn_like(initial.natural),
        precision=initial.precision,
        participation=torch.rand_like(initial.participation),
    )
    step = body.step(state, torch.tensor([23]), elapsed=0.0)
    selected = step.route > 0
    neighbourhood = body._local_neighbourhood(step.route)

    assert torch.equal(step.state.activity[~neighbourhood], state.activity[~neighbourhood])
    assert torch.equal(step.state.natural[~selected], state.natural[~selected])
    assert torch.equal(step.state.precision[~selected], state.precision[~selected])
    assert torch.equal(step.state.participation[~selected], state.participation[~selected])
