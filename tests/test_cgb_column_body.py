import copy

import pytest
import torch

from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody, ColumnBodyState


def _body() -> CGBColumnLanguageBody:
    torch.manual_seed(7)
    return CGBColumnLanguageBody(6, (3, 4), active_columns=3)


def _uniform_legal_contact_mass(body: CGBColumnLanguageBody, state: ColumnBodyState) -> torch.Tensor:
    """Fill every contact bundle with total S mass 1/2, not impossible copies."""
    return torch.full_like(
        state.participation, 0.5 / (2 * body.contact_rank),
    )


def _full_legal_contact_mass(body: CGBColumnLanguageBody, state: ColumnBodyState) -> torch.Tensor:
    """Fill each bundle to its shared unit capacity without duplicating mass."""
    return torch.full_like(
        state.participation, 1.0 / (2 * body.contact_rank),
    )


def _isolate_s_to_h(body: CGBColumnLanguageBody, value: float) -> None:
    """Leave one shared-field block as the only H response path for an audit."""
    with torch.no_grad():
        body.reciprocal_contact.zero_()
        body.reciprocal_contact[body._contact_h_output, body._contact_s_input].fill_(value)


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


def test_each_live_contact_bundle_uses_one_shared_finite_s_reservoir():
    """Typed/rank S factors compete within the same physical contact."""
    body = _body()
    state = body.initial_state(1)
    for token in (39, 41, 43, 47):
        state = body.step(state, torch.tensor([token]), elapsed=0.0).state

    # A physical directed contact has source-type × rank factors.  Its free
    # partner is implicit, so every reachable bundle stays below unit mass.
    bundle_mass = state.participation.flatten(start_dim=-2).sum(dim=-1)
    assert torch.all((0.0 <= bundle_mass) & (bundle_mass <= 1.0))
    mismatch = body.reciprocal_participation_mismatch(state.participation)
    assert torch.allclose(mismatch, torch.zeros_like(mismatch), atol=1e-7)


def test_shared_s_pool_blocks_new_admission_when_the_existing_contact_is_full():
    occupied = torch.tensor([[[1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0]]])
    admit = torch.ones(1, 1)
    release = torch.zeros_like(occupied)
    after = CGBColumnLanguageBody._shared_contact_pool_exchange(
        occupied, target, admit, release,
    )
    torch.testing.assert_close(after, occupied)


def test_external_diagnostics_observe_real_turnover_without_changing_the_body():
    body = _body()
    _isolate_s_to_h(body, 0.1)
    born = body.initial_state(1)
    state = ColumnBodyState(
        born.activity, born.natural, born.precision, born.causal_token,
        _uniform_legal_contact_mass(body, born),
    )
    token = torch.tensor([31])
    plain = body.step(state, token, elapsed=0.0)
    totals = torch.zeros(6)
    observed = body.step(state, token, elapsed=0.0, diagnostic_totals=totals)
    for name in state.__dataclass_fields__:
        torch.testing.assert_close(getattr(plain.state, name), getattr(observed.state, name),
                                   rtol=0.0, atol=0.0)
    torch.testing.assert_close(plain.logits, observed.logits, rtol=0.0, atol=0.0)
    assert totals[0] > 0 and totals[1] > 0
    # Equal incoming and outgoing mass is real turnover, not zero activity.
    torch.testing.assert_close(totals[0], totals[1])
    net_mass = (observed.state.participation - state.participation).sum() * 0.5
    torch.testing.assert_close(net_mass, totals[0] - totals[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(totals[2], observed.contact_response_gate.sum().float())
    possible = body._local_neighbourhood(observed.route).sum() * body.d_model
    torch.testing.assert_close(totals[3], possible.float())
    assert 0 <= totals[2] <= totals[3]
    assert totals.grad_fn is None


def test_local_vjp_recomputation_cannot_count_a_second_forward_event():
    body = _body()
    state = body.initial_state(1)
    state = body.step(state, torch.tensor([31]), elapsed=0.0).state
    totals = torch.zeros(6)
    current = body.step(state, torch.tensor([73]), elapsed=0.0, diagnostic_totals=totals)
    counted_once = totals.clone()
    body._s_contact_direction_from_tensors(
        state.participation, state.precision, current.contact_pre,
        current.contact_response_gate, current.route,
        torch.zeros(1, dtype=torch.long), torch.ones_like(state.participation),
    )
    torch.testing.assert_close(totals, counted_once, rtol=0.0, atol=0.0)


def test_shared_contact_uses_the_same_response_gate_at_both_endpoints():
    body = CGBColumnLanguageBody(6, (3, 4), active_columns=3, contact_rank=3).double()
    state = body.initial_state(1)
    occupied = _uniform_legal_contact_mass(body, state)
    response = torch.zeros(1, 3, 4, body.local_contact_output_dim, dtype=torch.float64)
    route = torch.zeros(1, 3, 4, dtype=torch.float64)
    route[0, 1, 1] = 1.0
    source = torch.zeros(1, dtype=torch.long)

    for left_live, right_live in ((0, 0), (0, 3), (1, 3), (5, 3)):
        gate = torch.zeros_like(state.activity, dtype=torch.bool)
        gate[0, 1, 1, :left_live] = True
        gate[0, 1, 2, :right_live] = True
        after = body._reciprocal_contact_update(occupied, response, gate, route, source)
        # Zero contact outputs give drive=retain=1/2.  The east/west pair
        # must use the same edge gate even if just one endpoint responds.
        before_edge = occupied[0, 1, 1, 1]
        target = torch.zeros_like(before_edge)
        target[0] = 1.0 / body.contact_rank
        edge_gate = max(left_live, right_live) / body.d_model
        expected = before_edge + edge_gate * (
            0.5 * target * (1.0 - before_edge.sum()) - 0.5 * before_edge
        )
        torch.testing.assert_close(after[0, 1, 1, 1], expected)
        torch.testing.assert_close(after[0, 1, 2, 3], expected)
        torch.testing.assert_close(
            body.reciprocal_participation_mismatch(after), torch.zeros_like(after),
        )
        if left_live == right_live == 0:
            torch.testing.assert_close(after, occupied, rtol=0.0, atol=0.0)
        else:
            assert not torch.equal(after[0, 1, 1, 1], before_edge)


def test_vector_release_keeps_the_shared_pool_and_implicit_complement_legal():
    base = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float64)
    target = base.clone()
    rates = torch.tensor([[[0.0, 0.3, 0.8, 1.0]]], dtype=torch.float64)
    for mass in (0.0, 0.5, 1.0):
        for gate in (0.0, 0.35, 1.0):
            occupied = mass * base
            admit = torch.full((1, 1), gate * 0.7, dtype=torch.float64)
            release = gate * rates
            after = CGBColumnLanguageBody._shared_contact_pool_exchange(
                occupied, target, admit, release,
            )
            free = 1.0 - occupied.sum(dim=-1)
            returned = (release * occupied).sum(dim=-1)
            expected_free = (1.0 - admit) * free + returned
            assert torch.all(after >= 0.0)
            assert torch.all(after.sum(dim=-1) <= 1.0)
            torch.testing.assert_close(1.0 - after.sum(dim=-1), expected_free)
            if gate == 0.0:
                torch.testing.assert_close(after, occupied, rtol=0.0, atol=0.0)
            if mass == 1.0:
                torch.testing.assert_close(after, (1.0 - release) * occupied)


def test_typed_s_equals_its_actual_admission_and_rank_retention_expansion():
    occupied = torch.zeros(1, 1, 6, dtype=torch.float64)
    target = torch.tensor([[[0.2, 0.3, 0.5, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    rates = torch.tensor([[[0.1, 0.4, 0.8, 0.1, 0.4, 0.8]]], dtype=torch.float64)
    # Only this identity test materializes the finite expansion; the runtime
    # stores the current S bundle, never a sequence of these terms.
    terms = []
    for gate, inflow in ((0.2, 0.5), (0.8, 0.4), (0.4, 0.0), (0.0, 0.8), (0.6, 0.3)):
        admit = torch.full((1, 1), gate * inflow, dtype=torch.float64)
        release = gate * rates
        retained = 1.0 - release
        admitted = admit.unsqueeze(-1) * target * (1.0 - occupied.sum(dim=-1, keepdim=True))
        terms = [retained * term for term in terms] + [admitted]
        occupied = CGBColumnLanguageBody._shared_contact_pool_exchange(
            occupied, target, admit, release,
        )
        torch.testing.assert_close(occupied, torch.stack(terms).sum(dim=0))
        # Absent self-source admission cannot create a self-source trace.
        torch.testing.assert_close(occupied[..., 3:], torch.zeros_like(occupied[..., 3:]))
        assert not occupied.requires_grad


def test_equal_future_admission_retains_a_readable_source_history_difference():
    left = torch.tensor([[[0.2, 0.1, 0.0, 0.0]]], dtype=torch.float64)
    right = torch.tensor([[[0.0, 0.0, 0.2, 0.1]]], dtype=torch.float64)
    initial_difference = left - right
    product = torch.ones_like(left)
    target = torch.tensor([[[0.4, 0.6, 0.0, 0.0]]], dtype=torch.float64)
    # Each rank has the same release for both source types.  These two
    # histories have equal per-rank mass, so future admitted amounts agree.
    rates = torch.tensor([[[0.3, 0.1, 0.3, 0.1]]], dtype=torch.float64)
    for gate, inflow in ((0.2, 0.5), (0.0, 0.8), (0.7, 0.3), (0.5, 0.0)):
        admit = torch.full((1, 1), gate * inflow, dtype=torch.float64)
        release = gate * rates
        torch.testing.assert_close(left.sum(dim=-1), right.sum(dim=-1))
        left = CGBColumnLanguageBody._shared_contact_pool_exchange(left, target, admit, release)
        right = CGBColumnLanguageBody._shared_contact_pool_exchange(right, target, admit, release)
        product = product * (1.0 - release)
        torch.testing.assert_close(left - right, product * initial_difference)
        source_difference = left[..., :2] - left[..., 2:]
        expected = product[..., :2] * (initial_difference[..., :2] - initial_difference[..., 2:])
        right_source_difference = right[..., :2] - right[..., 2:]
        torch.testing.assert_close(source_difference - right_source_difference, expected)
    # Positive C demand modulates the reading without merging this difference.
    demand = 1.0 / (1.0 + 2.0)
    assert torch.all((demand * (left - right)).abs() > 0.0)


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


def test_recurrent_h_s_dynamics_have_one_shared_learned_contact_field():
    body = _body()
    names = set(dict(body.named_parameters()))
    assert "reciprocal_contact" in names
    assert not names.intersection({
        "activity_carrier.weight", "neighbour_mix.weight", "contact_to_activity.weight",
        "activity_gain.weight", "activity_gain.bias", "activity_release.weight",
        "activity_release.bias", "edge_flow.weight", "edge_flow.bias",
        "edge_participation.weight", "edge_participation.bias",
        "participation_retain.weight", "participation_retain.bias",
        "participation_scale_raw",
    })


def test_existing_s_rank_has_independent_learned_local_release_coordinates():
    """Rank is an S basis, with release emitted by the one shared contact field.

    The former global geometric ratio was removed: it was an imposed clock.
    Each rank now has a distinct row in the same reciprocal local field, so
    broad re-entry can select distinct local timescales without a fourth state
    page or a manually tuned global decay schedule.
    """
    body = CGBColumnLanguageBody(6, (3, 4), active_columns=3, contact_rank=3)
    assert body.reciprocal_contact.shape == (
        body.d_model + 2 + 8 * body.contact_rank,
        4 * body.d_model + 1 + 8 * body.contact_rank,
    )
    allocation_stop = body.d_model + 2 + 4 * body.contact_rank
    release_rows = body.reciprocal_contact[allocation_stop:]
    probe = release_rows.square().sum()
    probe.backward()
    assert body.reciprocal_contact.grad is not None
    assert body.reciprocal_contact.grad[allocation_stop:].abs().sum() > 0
    state = body.initial_state(1)
    assert set(state.__dataclass_fields__) == {
        "activity", "natural", "precision", "causal_token", "participation",
    }

def test_old_contact_state_changes_next_local_activity_but_not_its_prepared_prediction():
    body = _body()
    _isolate_s_to_h(body, 0.1)
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        causal_token=baseline.causal_token,
        participation=_uniform_legal_contact_mass(body, baseline),
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
    _isolate_s_to_h(body, 0.1)
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        causal_token=baseline.causal_token,
        participation=_uniform_legal_contact_mass(body, baseline),
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


def test_current_s_write_is_blind_to_the_arrived_language_innovation():
    """A result-sensitive S learner needs a transient boundary input.

    Keep H/C/S, the arrived byte, and all local S-rule parameters identical,
    but issue two different language predictions by changing only the affine
    boundary head.  The resulting observed-minus-predicted innovations differ,
    while the current H/C/S transition is exactly identical.  This is the
    executable counterexample behind Lean G9: the present reciprocal exchange
    is a causal trace generator, not yet an error-conditioned plastic rule.

    The test does *not* introduce a feedback state or prescribe a teaching
    matrix.  It fixes the minimal missing interface before an admissible
    three-factor candidate can be measured against the future-score theorem.
    """
    body = _body()
    alternate_head = copy.deepcopy(body)
    with torch.no_grad():
        alternate_head.readout.weight.zero_()
        alternate_head.readout.bias.fill_(-6.0)
        alternate_head.readout.bias[17] = 6.0

    state = body.initial_state(1)
    token = torch.tensor([41], dtype=torch.long)
    target = torch.tensor([73], dtype=torch.long)
    first = body.step(state, token, elapsed=0.0)
    alternate = alternate_head.step(state, token, elapsed=0.0)

    innovation = body._language_innovation(first.logits, target)
    alternate_innovation = alternate_head._language_innovation(alternate.logits, target)
    assert not torch.equal(innovation, alternate_innovation)
    assert not innovation.requires_grad
    assert not alternate_innovation.requires_grad

    # `_reciprocal_contact_update` reads only local H/C, post-arrival route,
    # and source type.  It cannot condition S on this boundary difference.
    for actual, expected in zip(
        (first.state.activity, first.state.natural, first.state.precision,
         first.state.causal_token, first.state.participation),
        (alternate.state.activity, alternate.state.natural, alternate.state.precision,
         alternate.state.causal_token, alternate.state.participation),
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
def test_shared_contact_field_has_a_finite_path_to_a_later_language_score():
    """The unified recurrent field, including its S-to-H block, reaches language."""
    torch.manual_seed(7)
    body = CGBColumnLanguageBody(6, (3, 4), active_columns=3, contact_rank=3)
    _isolate_s_to_h(body, 0.1)
    state = body.initial_state(1)
    for _ in range(4):
        prepared = body.step(state, torch.tensor([41]), elapsed=0.0)
        state = prepared.state
    score = torch.log_softmax(prepared.logits, dim=-1)[0, 41]
    field_score, = torch.autograd.grad(score, body.reciprocal_contact)
    s_to_h = field_score[body._contact_h_output, body._contact_s_input]
    assert torch.isfinite(s_to_h).all()
    assert s_to_h.abs().sum() > 1e-7

def test_muting_contact_readout_preserves_the_s_write_but_removes_its_h_effect():
    body = _body()
    baseline = body.initial_state(1)
    charged = ColumnBodyState(
        activity=baseline.activity,
        natural=baseline.natural,
        precision=baseline.precision,
        causal_token=baseline.causal_token,
        participation=_uniform_legal_contact_mass(body, baseline),
    )
    body.mute_contact_readout_()
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
        participation=_uniform_legal_contact_mass(body, initial),
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


def test_route_credit_is_the_exact_one_event_score_direction_on_its_fixed_support():
    """The learned query and keys use the admitted sparse-route receipt.

    Autograd remains only a numerical audit of the one issued score.  The
    runtime update reconstructs the centered softmax factor from the returned
    receipt; it retains neither this graph nor a historical route array.
    """
    body = _body()
    first = body.step(body.initial_state(1), torch.tensor([31]), elapsed=0.0)
    # The receipt audits the newly issued score only.  Earlier H/C/S writes
    # are physical state, not a retained reverse-time training graph.
    prepared = body.step(body.detach_state(first.state), torch.tensor([88]), elapsed=0.0)
    target = torch.tensor([88], dtype=torch.long)
    score = torch.log_softmax(prepared.logits, dim=-1)[0, target.item()]
    exact_address, exact_keys = torch.autograd.grad(
        score, (body.causal_address.weight, body.column_keys),
    )
    before_address = body.causal_address.weight.detach().clone()
    before_keys = body.column_keys.detach().clone()
    body.update_route_from_arrival(prepared, target, step_size=1.0)
    torch.testing.assert_close(body.causal_address.weight - before_address, exact_address,
                               rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(body.column_keys - before_keys, exact_keys,
                               rtol=1e-4, atol=1e-5)


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
        public.credit_token, public.value_eligibility, public.contact_response_gate,
    )
    for from_kernel, from_public in zip(actual, expected):
        torch.testing.assert_close(from_kernel, from_public)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shared_edge_stream_matches_cpu_and_cuda_graph_without_capture_events():
    import numpy as np

    from scripts.run_cgb_column_language_admission import (
        CudaGraphValueCredit, _eager_value_credit_phase,
    )

    torch.manual_seed(7)
    cpu = CGBColumnLanguageBody(6, (3, 4), active_columns=3, contact_rank=3)
    gpu = copy.deepcopy(cpu).cuda()
    initial = gpu.initial_state(1)
    before_weights = {name: value.detach().clone() for name, value in gpu.named_parameters()}
    runner = CudaGraphValueCredit(gpu, initial, 0.01, 0.001, 0.002, 0.002, 2)
    for name in initial.__dataclass_fields__:
        torch.testing.assert_close(getattr(runner.state(), name), getattr(initial, name), rtol=0.0, atol=0.0)
    for name, value in gpu.named_parameters():
        torch.testing.assert_close(value, before_weights[name], rtol=0.0, atol=0.0)

    receipt_shape = runner.previous_contact_pre.shape
    state_bytes = gpu.persistent_state_bytes(initial)
    # The special token advances actual empty time without a lexical write.
    events = np.asarray([35, 77, 0, 18, 111], dtype=np.int64)
    expected = _eager_value_credit_phase(
        cpu, cpu.initial_state(1), events, 0.01, 0.001, 0.002, 0.002, 100,
    )
    actual = runner.phase(events, 100)
    assert actual["scored_bytes"] == expected["scored_bytes"] == 4
    assert actual["special_tokens_elapsed_without_lexical_write"] == 1
    assert actual["mean_nll_nats"] == pytest.approx(expected["mean_nll_nats"], abs=2e-5)
    final = runner.state()
    for name in final.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(final, name).cpu(), getattr(expected["_state"], name), rtol=2e-5, atol=2e-6,
        )
        assert getattr(final, name).grad_fn is None
    for (name, left), (other_name, right) in zip(cpu.named_parameters(), gpu.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(left, right.cpu(), rtol=2e-5, atol=2e-6)
        assert left.grad is None and right.grad is None
    assert runner.previous_contact_pre.shape == receipt_shape
    assert gpu.persistent_state_bytes(final) == state_bytes


def test_contact_gate_times_full_local_pre_is_the_exact_one_step_contact_score_direction():
    """The local contract matches the exact two-event score derivative.

    The reference keeps a two-event graph only for this test.  The candidate
    itself is formed from the event's transient saturation gate, old S, and
    the next language boundary innovation; it neither stores a new state page
    nor writes a parameter.
    """
    body = _body()
    _isolate_s_to_h(body, 0.1)
    born = body.initial_state(1)
    state = ColumnBodyState(
        born.activity,
        born.natural,
        born.precision,
        born.causal_token,
            _uniform_legal_contact_mass(body, born),
    )
    current = torch.tensor([31], dtype=torch.long)
    following = torch.tensor([88], dtype=torch.long)
    prepared = body.step(state, current, elapsed=0.0)
    next_prepared = body.step(prepared.state, following, elapsed=0.0)
    score = torch.log_softmax(next_prepared.logits, dim=-1)[0, following.item()]
    exact, = torch.autograd.grad(score, body.reciprocal_contact)

    reconstructed = body.one_step_contact_score_direction(
        state, prepared, following, elapsed=0.0,
    )
    torch.testing.assert_close(reconstructed, exact[body._contact_h_output], rtol=1e-4, atol=1e-5)
    before = body.reciprocal_contact.detach().clone()
    body.update_reciprocal_h_from_following(prepared, next_prepared, following, step_size=0.05)
    torch.testing.assert_close(
        body.reciprocal_contact[body._contact_h_output],
        before[body._contact_h_output] + 0.05 * reconstructed,
        rtol=1e-4, atol=1e-5,
    )
    torch.testing.assert_close(
        body.reciprocal_contact[body.d_model:], before[body.d_model:], rtol=0.0, atol=0.0,
    )
    assert prepared.contact_response_gate.dtype == torch.bool
    assert not reconstructed.requires_grad


def test_response_gated_s_write_keeps_the_local_eligibility_distinction():
    """A responsive contact writes S; a saturated one holds its same pool.

    The prior body failed this representation test: the same S state was
    emitted whether a contact had a forward H response or was saturated.  The
    shared-pool gate now writes that already-derived response into S itself,
    without adding an eligibility tensor to persistent state.
    """
    low = _body()
    high = copy.deepcopy(low)
    with torch.no_grad():
        low.reciprocal_contact.zero_()
        high.reciprocal_contact.zero_()
        high.reciprocal_contact[high._contact_h_output, high._contact_s_input].fill_(4.0)
        low.dynamics_value.weight.zero_()
        high.dynamics_value.weight.zero_()

    born = low.initial_state(1)
    state = ColumnBodyState(
        born.activity,
        born.natural,
        born.precision,
        born.causal_token,
        _full_legal_contact_mass(low, born),
    )
    token = torch.tensor([31], dtype=torch.long)
    low_step = low.step(state, token, elapsed=0.0)
    high_step = high.step(state, token, elapsed=0.0)

    assert low_step.contact_response_gate.any()
    assert not high_step.contact_response_gate.any()
    assert not torch.equal(low_step.state.participation, high_step.state.participation)
    # The saturated contact's gate closes the entire local exchange, including
    # release, exactly as the Lean shared-pool gate contract requires.
    torch.testing.assert_close(high_step.state.participation, state.participation,
                               rtol=0.0, atol=0.0)


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
        body.reciprocal_contact,
    ))
    body.update_value_from_arrival(prepared, torch.tensor([88]), step_size=0.05)
    torch.testing.assert_close(body.event_value.weight, before + 0.05 * reconstructed)
    # The Lean value-credit factorization holds its address, likelihood, and
    # H/S dynamics fixed.  The runtime must preserve that boundary exactly.
    for actual, expected in zip((
        body.causal_address.weight, body.event_likelihood.weight,
        body.event_likelihood.bias, body.dynamics_value.weight,
        body.reciprocal_contact,
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


def test_likelihood_credit_is_the_exact_one_step_posterior_score_direction():
    """The confidence logit uses C's quotient sensitivity, not only its write.

    This local autograd comparison exists solely to audit the Lean-refined
    executable formula.  The online path reconstructs it from the real C
    state and releases it immediately; it retains no computation graph.
    """
    body = _body()
    birth = body.initial_state(1)
    first = body.step(birth, torch.tensor([31]), elapsed=0.0)
    prepared = body.step(first.state, torch.tensor([88]), elapsed=0.0)
    score = torch.log_softmax(prepared.logits, dim=-1)[0, 88]
    exact_weight, exact_bias = torch.autograd.grad(
        score, (body.event_likelihood.weight, body.event_likelihood.bias),
    )

    innovation = body._language_innovation(prepared.logits.detach(), torch.tensor([88]))
    language_error = innovation @ body.readout.weight.detach()
    eligibility = body.likelihood_eligibility(first.state)
    scalar_score = (language_error * eligibility).sum(dim=-1)
    source = body.byte_embedding(first.state.causal_token).detach()
    reconstructed_weight = scalar_score.unsqueeze(0) @ source
    reconstructed_bias = scalar_score.sum().view_as(body.event_likelihood.bias)
    torch.testing.assert_close(reconstructed_weight, exact_weight, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(reconstructed_bias, exact_bias, rtol=1e-4, atol=1e-5)
    before_weight = body.event_likelihood.weight.detach().clone()
    before_bias = body.event_likelihood.bias.detach().clone()
    body.update_likelihood_from_arrival(
        first.state, prepared, torch.tensor([88]), step_size=0.05,
    )
    torch.testing.assert_close(body.event_likelihood.weight,
                               before_weight + 0.05 * reconstructed_weight)
    torch.testing.assert_close(body.event_likelihood.bias,
                               before_bias + 0.05 * reconstructed_bias)


def test_dynamics_value_credit_reconstructs_the_exact_one_transition_direction():
    """One consumed contact receipt reproduces the bounded-flow score direction.

    Autograd is retained only inside this numerical audit.  The reconstructed
    path uses the prior C token, sparse contact gate and conservative flux;
    there is no retained computational graph or history-sized trace.
    """
    body = _body()
    birth = body.initial_state(1)
    first = body.step(birth, torch.tensor([31]), elapsed=0.0)
    prepared = body.step(first.state, torch.tensor([88]), elapsed=0.0)
    score = torch.log_softmax(prepared.logits, dim=-1)[0, 88]
    exact_autograd, = torch.autograd.grad(score, body.dynamics_value.weight)

    index, _, gate, unique = body._pack_local_receipt(
        first.contact_pre, first.contact_response_gate, first.route,
    )
    reconstructed = body._dynamics_value_direction_from_tensors(
        prepared.logits.detach(), prepared.read_route.detach(), index, gate, unique,
        first.edge_flux.detach(), first.state.causal_token, torch.tensor([88]),
        torch.ones(1, dtype=torch.bool),
    )
    torch.testing.assert_close(reconstructed, exact_autograd, rtol=1e-4, atol=1e-5)


def test_shared_event_embedding_sums_all_admitted_one_transition_returns():
    """A byte embedding receives the sum of C-value, C-confidence and H returns."""
    body = _body()
    birth = body.initial_state(1)
    first = body.step(birth, torch.tensor([31]), elapsed=0.0)
    prepared = body.step(first.state, torch.tensor([88]), elapsed=0.0)
    score = torch.log_softmax(prepared.logits, dim=-1)[0, 88]
    exact_autograd, = torch.autograd.grad(score, body.byte_embedding.weight)

    index, _, gate, unique = body._pack_local_receipt(
        first.contact_pre, first.contact_response_gate, first.route,
    )
    reconstructed = body._event_embedding_direction_from_tensors(
        prepared.logits.detach(), first.state.natural, first.state.precision,
        first.state.causal_token, first.state.participation, prepared.read_route.detach(),
        index, gate, unique, first.edge_flux.detach(), torch.tensor([88]),
        torch.ones(1, dtype=torch.bool),
    )
    torch.testing.assert_close(reconstructed, exact_autograd, rtol=1e-4, atol=1e-5)


def test_common_event_map_step_matches_all_one_transition_gradients():
    """The shared C-event scale updates each admitted map at its exact direction."""
    body = _body()
    birth = body.initial_state(1)
    first = body.step(birth, torch.tensor([31]), elapsed=0.0)
    prepared = body.step(first.state, torch.tensor([88]), elapsed=0.0)
    score = torch.log_softmax(prepared.logits, dim=-1)[0, 88]
    parameters = (
        body.event_value.weight, body.event_likelihood.weight,
        body.event_likelihood.bias, body.dynamics_value.weight, body.byte_embedding.weight,
    )
    exact = torch.autograd.grad(score, parameters)
    before = tuple(parameter.detach().clone() for parameter in parameters)
    index, _, gate, unique = body._pack_local_receipt(
        first.contact_pre, first.contact_response_gate, first.route,
    )
    body._update_event_maps_from_tensors(
        prepared.logits.detach(), first.state.natural, first.state.precision,
        first.state.causal_token, first.state.participation, prepared.read_route.detach(),
        index, gate, unique, first.edge_flux.detach(), torch.tensor([88]), 0.05,
        torch.ones(1, dtype=torch.bool),
    )
    for actual, initial, direction in zip(parameters, before, exact):
        torch.testing.assert_close(actual, initial + 0.05 * direction, rtol=1e-4, atol=1e-5)


def test_elapsed_prior_precedes_prediction_and_value_credit_receipt():
    """The known elapsed interval prepares C before the unknown byte is scored.

    This is a causal-order test, not a language-learning benchmark.  It keeps
    the runtime's prepared route/value-credit receipt aligned with Lean
    `GDNLife.prepare`: `Δt` is known at the boundary, so its analytic C prior
    must be visible before the arriving byte is used as a target or written.
    """
    body = _body()
    first = body.step(body.initial_state(1), torch.tensor([31]), elapsed=0.0)
    state = body.detach_state(first.state)
    with torch.no_grad():
        body.process_variance_raw.fill_(0.0)

    released = body.advance_time(state, 3.0)
    prepared = body.step(state, torch.tensor([88]), elapsed=3.0)

    torch.testing.assert_close(prepared.read_route, body._state_route(released))
    torch.testing.assert_close(
        prepared.read_values,
        released.natural / released.precision + released.activity,
    )
    torch.testing.assert_close(prepared.value_eligibility, body.value_eligibility(released))
    assert not torch.allclose(
        prepared.value_eligibility,
        body.value_eligibility(state),
    )


def test_prepare_predict_assimilate_is_the_public_single_event_transition():
    """The explicit G2 phases compose to the unchanged public body step."""
    body = _body()
    first = body.step(body.initial_state(1), torch.tensor([31]), elapsed=0.0)
    state = body.detach_state(first.state)
    token = torch.tensor([88])
    elapsed = 0.75
    prepared = body.step(state, token, elapsed=elapsed)

    dt = body._as_elapsed(elapsed, 1, state.precision).reshape(1)
    natural, precision = body._prepare_carrier_tensors(
        state.natural, state.precision, dt,
    )
    features, route, values, credit, eligibility = body._predict_tensors(
        state.activity, natural, precision, state.causal_token, state.participation,
    )
    source = body._source_index("external", 1, state.activity)
    (activity, natural, precision, participation, write_route, flux,
     gate, contact_pre) = body._assimilate_tensors(
        state.activity, natural, precision, state.participation, token, source,
    )

    torch.testing.assert_close(prepared.features, features)
    torch.testing.assert_close(prepared.read_route, route)
    torch.testing.assert_close(prepared.read_values, values)
    torch.testing.assert_close(prepared.credit_token, credit)
    torch.testing.assert_close(prepared.value_eligibility, eligibility)
    torch.testing.assert_close(prepared.route, write_route)
    torch.testing.assert_close(prepared.edge_flux, flux)
    torch.testing.assert_close(prepared.contact_response_gate, gate)
    torch.testing.assert_close(prepared.contact_pre, contact_pre)
    torch.testing.assert_close(prepared.state.activity, activity)
    torch.testing.assert_close(prepared.state.natural, natural)
    torch.testing.assert_close(prepared.state.precision, precision)
    torch.testing.assert_close(prepared.state.participation, participation)
    torch.testing.assert_close(prepared.state.causal_token, token)
