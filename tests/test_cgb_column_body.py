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
    # C comprises its numerical natural/precision fields and one 8-bit-scale
    # causal query identity.  It remains one carrier, not a fourth state page.
    assert set(state.__dataclass_fields__) == {
        "activity", "natural", "precision", "causal_token", "participation",
    }
    assert torch.equal(state.causal_token, torch.full((2,), -1, dtype=torch.long))
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
    assert external.state.participation[..., 1, :].sum() == 0


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
    # C comprises its numerical natural/precision fields and one 8-bit-scale
    # causal query identity.  It remains one carrier, not a fourth state page.
    assert set(state.__dataclass_fields__) == {
        "activity", "natural", "precision", "causal_token", "participation",
    }
    assert torch.equal(state.causal_token, torch.full((1,), -1, dtype=torch.long))
    step = body.step(state, torch.tensor([39]), elapsed=0.0)
    mismatch = body.reciprocal_participation_mismatch(step.state.participation)
    assert torch.allclose(mismatch, torch.zeros_like(mismatch), atol=1e-7)
    assert body.trace(step.state).shape == state.participation.shape


def test_existing_s_rank_is_a_generated_geometric_release_bank():
    """Broad-delay capacity changes retention, never the state-page count.

    `GenerationCascadeMemory` admits a bank only when its rank coordinates
    really have separated time scales.  One local release and one trainable
    ratio generate those scales here; no rank-specific clock, eligibility
    tensor, or historical event list is introduced.
    """
    torch.manual_seed(7)
    body = CGBColumnLanguageBody(6, (3, 4), active_columns=3, contact_rank=3)
    base_retain = torch.full((1, 3, 4, 4), 0.2)
    retain = body._rank_retention(base_retain)
    release = 1.0 - retain
    scale = body.participation_scale()
    assert retain.shape == (1, 3, 4, 4, 3)
    torch.testing.assert_close(release[..., 1], release[..., 0] / scale)
    torch.testing.assert_close(release[..., 2], release[..., 1] / scale)
    scale.backward()
    assert body.participation_scale_raw.grad is not None
    assert body.participation_scale_raw.grad.item() > 0

    # Under equally local repeated use, occupied fast coordinates release
    # first while the slow coordinates remain readable.  The complement of
    # every two-state pool remains implicit and exact.
    occupied = torch.ones(1, 3, 4, 4, 2, 3)
    drive = torch.zeros_like(occupied)
    active = torch.ones(1, 3, 4, 4)
    after = body._adaptive_phase_exchange(occupied, drive, retain, active)
    assert torch.all(after[..., 0] < after[..., 1])
    assert torch.all(after[..., 1] < after[..., 2])
    assert torch.all((0.0 <= after) & (after <= 1.0))
    state = body.initial_state(1)
    assert set(state.__dataclass_fields__) == {
        "activity", "natural", "precision", "causal_token", "participation",
    }


def test_old_contact_state_changes_next_local_activity_but_not_its_prepared_prediction():
    body = _body()
    with torch.no_grad():
        body.contact_to_activity.weight.fill_(0.1)
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        causal_token=baseline.causal_token,
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


def test_s_participation_reaches_a_later_language_prediction_through_h():
    """S has a causal route to a later score, without a hidden BPTT buffer."""
    torch.manual_seed(7)
    body = CGBColumnLanguageBody(6, (3, 4), active_columns=3, contact_rank=3)
    with torch.no_grad():
        body.contact_to_activity.weight.fill_(0.1)
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        causal_token=baseline.causal_token,
        participation=torch.full_like(baseline.participation, 0.5),
    )

    # The first score cannot inspect the current S readout; S influences the
    # post-arrival H transition.  That changed H is then visible at the next
    # score, establishing the actual S -> H -> language boundary path needed
    # before an S teaching rule could be admitted.
    first_plain = body.step(baseline, torch.tensor([41]), elapsed=0.0)
    first_charged = body.step(charged, torch.tensor([41]), elapsed=0.0)
    assert torch.equal(first_plain.logits, first_charged.logits)
    second_plain = body.step(first_plain.state, torch.tensor([73]), elapsed=0.0)
    second_charged = body.step(first_charged.state, torch.tensor([73]), elapsed=0.0)
    assert not torch.equal(second_plain.logits, second_charged.logits)


def test_muting_contact_readout_preserves_the_s_write_but_removes_its_h_effect():
    body = _body()
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        causal_token=baseline.causal_token,
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
        (step.state.activity, step.state.natural, step.state.precision, step.state.causal_token,
         step.state.participation),
        (detached.activity, detached.natural, detached.precision, detached.causal_token,
         detached.participation),
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
        causal_token=initial.causal_token,
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


def test_s_arrival_is_a_bounded_adaptive_two_state_eligibility_exchange():
    """Existing S occupancy creates the trace; no additional trace page exists."""
    occupied = torch.tensor([[[[[[0.0], [0.4]]]]]])
    drive = torch.tensor([[[[[0.8], [0.3]]]]])
    retain = torch.tensor([[[[0.75]]]])
    active = torch.tensor([[[[0.5]]]])
    actual = CGBColumnLanguageBody._adaptive_phase_exchange(
        occupied, drive, retain, active,
    )
    k_plus = active.unsqueeze(-1).unsqueeze(-1) * drive
    k_minus = active.unsqueeze(-1).unsqueeze(-1) * (1.0 - retain.unsqueeze(-1).unsqueeze(-1))
    expected = occupied + k_plus * (1.0 - occupied) - k_minus * occupied

    # This is exactly the existing Lean two-state step, with adaptive local
    # rates.  Its implicit complementary population remains conserved.
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close((1.0 - actual) + actual, torch.ones_like(actual))
    assert torch.all((0.0 <= actual) & (actual <= 1.0))

    # A contact that is not routed neither decays nor changes.  At saturation,
    # an up event has no remaining capacity: the same finite pool creates the
    # refractory/anti-repetition effect without a counter variable.
    inactive = CGBColumnLanguageBody._adaptive_phase_exchange(
        occupied, drive, retain, torch.zeros_like(active),
    )
    torch.testing.assert_close(inactive, occupied)
    saturated = CGBColumnLanguageBody._adaptive_phase_exchange(
        torch.ones_like(occupied), torch.ones_like(drive), torch.ones_like(retain), active,
    )
    torch.testing.assert_close(saturated, torch.ones_like(occupied))


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
        "H_activity", "C_natural", "C_precision", "C_causal_token", "S_participation",
    ))
    assert footprint["total"] == sum(value.numel() * value.element_size() for value in (
        snapshot.activity, snapshot.natural, snapshot.precision, snapshot.causal_token, snapshot.participation,
    ))
    state.activity.add_(1)
    assert not torch.equal(state.activity, snapshot.activity)
    assert all(not value.requires_grad for value in (
        snapshot.activity, snapshot.natural, snapshot.precision, snapshot.causal_token, snapshot.participation,
    ))


def test_arriving_byte_cannot_select_its_own_pre_update_prediction_route():
    body = _body()
    state = body.initial_state(1)
    first = body.step(state, torch.tensor([3]), elapsed=0.0)
    second = body.step(state, torch.tensor([201]), elapsed=0.0)
    # `route` reports the post-score write address and may depend on the
    # arrived byte.  The only forbidden dependency is from that byte into the
    # already issued prediction.
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.logits, second.logits)
    assert not torch.equal(first.route, second.route)
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
        state.activity, state.natural, state.precision, state.causal_token,
        state.participation, token, elapsed, source,
    )
    expected = (
        public.state.activity, public.state.natural, public.state.precision,
        public.state.causal_token, public.state.participation,
        public.features, public.route, public.edge_flux,
        public.credit_token, public.value_eligibility,
    )
    for from_kernel, from_public in zip(actual, expected):
        torch.testing.assert_close(from_kernel, from_public)


def test_live_arrival_updates_only_causal_query_for_the_next_event():
    body = _body()
    born = body.initial_state(2)
    token = torch.tensor([17, 203], dtype=torch.long)
    step = body.step(born, token, elapsed=0.0)

    # This is CGB.GDNLife.positive_observation_sets_last_byte at the column
    # carrier boundary: a current byte cannot affect its already issued score,
    # but it is retained as the next event's causal query identity.
    assert torch.equal(step.state.causal_token, token)
    replay = body.step(born, token, elapsed=0.0)
    assert torch.equal(step.logits, replay.logits)
    advanced = body.advance_time(step.state, 3.0)
    assert torch.equal(advanced.causal_token, token)

    # The stored identity is read on the following event, through a value-
    # independent address map.  Force distinct address codes so this remains
    # a causal contract rather than a lucky initialization property.
    with torch.no_grad():
        body.causal_address.weight.zero_()
        body.causal_address.weight[17, 0] = 1
        body.causal_address.weight[203, 1] = 1
        body.column_keys.zero_()
        body.column_keys[0, 0, 0] = 1
        body.column_keys[-1, -1, 1] = 1
    next_a = body.step(step.state, torch.tensor([7, 7]), elapsed=0.0)
    alternate = ColumnBodyState(step.state.activity, step.state.natural,
                                step.state.precision,
                                torch.tensor([203, 17], dtype=torch.long),
                                step.state.participation)
    next_b = body.step(alternate, torch.tensor([7, 7]), elapsed=0.0)
    assert not torch.equal(next_a.features, next_b.features)
    assert not torch.equal(next_a.logits, next_b.logits)


def test_value_credit_is_the_exact_one_step_causal_score_direction():
    body = _body()
    birth = body.initial_state(1)
    first = body.step(birth, torch.tensor([31]), elapsed=0.0)
    prepared = body.step(first.state, torch.tensor([88]), elapsed=0.0)

    # Keep the past computation graph only for this numerical audit.  The
    # online implementation uses the reconstructed receipt and retains none
    # of it.  The agreement is the executable admission of the GDN value path.
    score = torch.log_softmax(prepared.logits, dim=-1)[0, 88]
    exact_autograd, = torch.autograd.grad(score, body.event_value.weight)
    innovation = -torch.softmax(prepared.logits.detach(), dim=-1)
    innovation[0, 88] += 1
    local_error = innovation @ body.readout.weight.detach()
    source = body.byte_embedding(prepared.credit_token.clamp_min(0)).detach()
    reconstructed = (local_error * prepared.value_eligibility.detach().unsqueeze(-1)).T @ source
    torch.testing.assert_close(reconstructed, exact_autograd, rtol=1e-4, atol=1e-5)

    before = body.event_value.weight.detach().clone()
    fixed_before = tuple(parameter.detach().clone() for parameter in (
        body.causal_address.weight, body.event_likelihood.weight,
        body.event_likelihood.bias, body.dynamics_value.weight,
        body.activity_carrier.weight,
    ))
    body.update_value_from_arrival(prepared, torch.tensor([88]), step_size=0.05)
    torch.testing.assert_close(body.event_value.weight, before + 0.05 * reconstructed)
    # The Lean value-credit factorization holds its address, likelihood, and
    # H/S dynamics fixed.  The runtime must preserve that boundary exactly.
    for actual, expected in zip((
        body.causal_address.weight, body.event_likelihood.weight,
        body.event_likelihood.bias, body.dynamics_value.weight,
        body.activity_carrier.weight,
    ), fixed_before):
        torch.testing.assert_close(actual, expected)

    # Replaying the same two causal events under the infinitesimally updated
    # value map improves their later score.  This is a local derivative check,
    # never a language-capability conclusion.
    body = _body()
    birth = body.initial_state(1)
    first = body.step(birth, torch.tensor([31]), elapsed=0.0)
    prepared = body.step(first.state, torch.tensor([88]), elapsed=0.0)
    before_nll = -torch.log_softmax(prepared.logits, dim=-1)[0, 88]
    body.update_value_from_arrival(prepared, torch.tensor([88]), step_size=1e-3)
    replay_first = body.step(birth, torch.tensor([31]), elapsed=0.0)
    replay = body.step(replay_first.state, torch.tensor([88]), elapsed=0.0)
    after_nll = -torch.log_softmax(replay.logits, dim=-1)[0, 88]
    assert after_nll < before_nll
