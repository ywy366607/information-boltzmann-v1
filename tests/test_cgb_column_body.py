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
    assert state.participation.shape == (2, 3, 4, 4, 2, 1)
    assert set(state.__dataclass_fields__) == {
        "activity", "natural", "precision", "participation",
    }
    demand = body.demand(state)
    trace = body.trace(state)
    assert demand.shape == state.precision.shape
    assert trace.shape == state.participation.shape
    assert torch.equal(trace, state.participation * demand.unsqueeze(-1).unsqueeze(-1))


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
    assert external.state.participation[..., 0, :].sum() > 0
    assert external.state.participation[..., 1, :].sum() == 0
    assert self_event.state.participation[..., 1, :].sum() > 0
    assert self_event.state.participation[..., 0, :].sum() == 0


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


def test_selected_local_contacts_are_reciprocal_without_a_feedback_state_page():
    body = _body()
    state = body.initial_state(1)
    step = body.step(state, torch.tensor([39]), elapsed=0.0, source="external")

    mismatch = body.reciprocal_participation_mismatch(step.state.participation)
    assert torch.allclose(mismatch, torch.zeros_like(mismatch), atol=1e-7)
    assert step.state.participation[..., 0, :].sum() > 0
    assert step.state.participation[..., 1, :].sum() == 0


def test_contact_basis_rank_expands_only_s_not_the_number_of_persistent_pages():
    torch.manual_seed(7)
    body = CGBColumnLanguageBody(6, (3, 4), active_columns=3, contact_rank=3)
    state = body.initial_state(1)
    assert state.participation.shape == (1, 3, 4, 4, 2, 3)
    assert set(state.__dataclass_fields__) == {
        "activity", "natural", "precision", "participation",
    }
    step = body.step(state, torch.tensor([39]), elapsed=0.0)
    mismatch = body.reciprocal_participation_mismatch(step.state.participation)
    assert torch.allclose(mismatch, torch.zeros_like(mismatch), atol=1e-7)
    assert body.trace(step.state).shape == state.participation.shape


def test_old_contact_state_changes_next_local_activity_but_not_its_prepared_prediction():
    body = _body()
    with torch.no_grad():
        body.contact_to_activity.weight.fill_(0.1)
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        participation=torch.full_like(baseline.participation, 0.5),
    )
    token = torch.tensor([41], dtype=torch.long)
    no_contact = body.step(baseline, token, elapsed=0.0)
    with_contact = body.step(charged, token, elapsed=0.0)

    # The byte is scored from old H/C.  S can affect the succeeding local
    # dynamics, but cannot make the arrived byte choose its own prediction.
    assert torch.equal(no_contact.route, with_contact.route)
    assert torch.equal(no_contact.features, with_contact.features)
    assert torch.equal(no_contact.logits, with_contact.logits)
    assert torch.equal(no_contact.state.natural, with_contact.state.natural)
    assert torch.equal(no_contact.state.precision, with_contact.state.precision)
    assert not torch.equal(no_contact.state.activity, with_contact.state.activity)


def test_muting_contact_readout_preserves_the_s_write_but_removes_its_h_effect():
    body = _body()
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        participation=torch.full_like(baseline.participation, 0.5),
    )
    with torch.no_grad():
        body.contact_to_activity.weight.zero_()
    token = torch.tensor([42], dtype=torch.long)
    no_contact = body.step(baseline, token, elapsed=0.0)
    muted_contact = body.step(charged, token, elapsed=0.0)

    # The carrier still evolves through its reciprocal local write, but cannot
    # influence H when its only readout edge has been muted.
    assert torch.equal(no_contact.state.activity, muted_contact.state.activity)
    assert torch.equal(no_contact.state.natural, muted_contact.state.natural)
    assert torch.equal(no_contact.state.precision, muted_contact.state.precision)
    assert not torch.equal(no_contact.state.participation, muted_contact.state.participation)


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
    # A contact is local to both endpoints: the routed column and a direct
    # neighbour share its reciprocal S update.  No remote contact changes.
    assert torch.equal(step.state.participation[~neighbourhood], state.participation[~neighbourhood])


def test_g14_readout_update_improves_the_arrived_byte_without_persistent_receipt_state():
    body = _body()
    state = body.initial_state(1)
    prepared = body.step(state, torch.tensor([44]), elapsed=0.0)
    target = torch.tensor([17], dtype=torch.long)
    before = torch.log_softmax(prepared.logits, dim=-1)[0, target.item()]
    body.update_readout_from_arrival(prepared, target, step_size=0.01)
    after = torch.log_softmax(body.readout(prepared.features.detach()), dim=-1)[0, target.item()]
    assert after > before
    assert "features" not in prepared.state.__dataclass_fields__


def test_complete_hcs_snapshot_restores_values_and_reports_only_persistent_state():
    body = _body()
    state = body.step(body.initial_state(1), torch.tensor([81]), elapsed=0.25).state
    snapshot = body.clone_state(state)
    footprint = body.persistent_state_bytes(snapshot)
    assert footprint["total"] == sum(footprint[name] for name in (
        "H_activity", "C_natural", "C_precision", "S_participation",
    ))
    assert footprint["total"] == sum(value.numel() * value.element_size() for value in (
        snapshot.activity, snapshot.natural, snapshot.precision, snapshot.participation,
    ))
    state.activity.add_(1)
    assert not torch.equal(state.activity, snapshot.activity)
    assert all(not value.requires_grad for value in (
        snapshot.activity, snapshot.natural, snapshot.precision, snapshot.participation,
    ))


def test_arriving_byte_cannot_select_its_own_pre_update_prediction_route():
    body = _body()
    state = body.initial_state(1)
    first = body.step(state, torch.tensor([3]), elapsed=0.0)
    second = body.step(state, torch.tensor([201]), elapsed=0.0)
    assert torch.equal(first.route, second.route)
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.logits, second.logits)
    assert not torch.equal(first.state.natural, second.state.natural)


def test_empty_elapsed_time_releases_only_c_precision_and_keeps_h_s_exactly():
    body = _body()
    state = body.step(body.initial_state(1), torch.tensor([70]), elapsed=0.0).state
    advanced = body.advance_time(state, 2.0)
    assert torch.equal(advanced.activity, state.activity)
    assert torch.equal(advanced.participation, state.participation)
    assert torch.all(advanced.precision < state.precision)
    assert torch.allclose(advanced.natural / advanced.precision,
                          state.natural / state.precision, atol=1e-6)


def test_compilable_tensor_step_is_exactly_the_validated_public_transition():
    """The CUDA execution kernel may remove Python overhead, never semantics."""
    body = _body()
    state = body.step(body.initial_state(1), torch.tensor([70]), elapsed=0.0).state
    token = torch.tensor([19], dtype=torch.long)
    public = body.step(state, token, elapsed=0.25, source="external")
    elapsed = body._as_elapsed(0.25, 1, state.precision).reshape(1)
    source = body._source_index("external", 1, state.activity)
    actual = body._step_tensors(
        state.activity, state.natural, state.precision, state.participation,
        token, elapsed, source,
    )
    expected = (
        public.state.activity, public.state.natural, public.state.precision,
        public.state.participation, public.features, public.route, public.edge_flux,
    )
    for from_kernel, from_public in zip(actual, expected):
        torch.testing.assert_close(from_kernel, from_public)
