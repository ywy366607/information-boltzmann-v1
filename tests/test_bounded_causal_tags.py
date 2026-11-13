"""CPU algebra/interface checks, not a learned capability experiment."""
import copy

import numpy as np
import pytest

from scripts.ib_local.bounded_causal_tag_reference import (
    Code, ErrorTerm, Layout, NumericalBudget, Records, begin_capture,
    capture_activation, capture_credit, commit_record, insert_source,
    project_source, transport,
)


def small_layout():
    return Layout(layers=2, width=2, heads=1, head_dim=2, conv=2,
                  tokens=2, activation_tokens=3, ports=(0, 1), capacity=2)


def expanded(code):
    # Dense reference is permitted only in tiny tests, never retained by Code.
    return code.U.astype(np.float64).T @ code.V.astype(np.float64)


def complete_capture(code, records, layout, port=0, event=0):
    code = begin_capture(code, records, layout, port, event)
    code = capture_activation(code, records, layout, port, key=True)
    return capture_activation(code, records, layout, port)


def test_layout_roundtrip_views_and_restriction():
    layout = small_layout()
    rng = np.random.default_rng(1)
    values = {n: rng.normal(size=layout.shapes[n]) for n in layout.fields}
    packed = layout.pack(values)
    for name, value in layout.unpack(packed).items():
        np.testing.assert_array_equal(value, values[name])
        assert np.shares_memory(value, packed)
        assert np.linalg.norm(value) <= np.linalg.norm(packed)
    np.testing.assert_array_equal(layout.pack(layout.unpack(packed)), packed)


def test_actual_layout_is_metadata_only_and_capacity_is_per_port():
    layout = Layout(capacity=256, activation_tokens=128)
    assert math_size(layout.cache_shapes) == 274432
    assert math_size(layout.record_shapes) == 342016
    assert layout.size == 1024 * 128 + 6586368 + (2 * 256 + 2) * 342016
    budget = layout.budget(2, 16400)
    assert budget['effect_bytes'] == 2 * layout.size * 4
    assert budget['owner_bytes'] == 2 * 16400 * 4


def math_size(shapes):
    return sum(np.prod(shape) for shape in shapes.values())


def test_nonorthogonal_sweep_is_not_least_squares():
    U = np.array([[1., 0.], [1., 1.]])
    a, residual = project_source(U, [0., 1.])
    np.testing.assert_array_equal(a, [0., .5])
    np.testing.assert_array_equal(residual, [-.5, .5])
    np.testing.assert_array_equal(a @ U + residual, [0., 1.])
    assert np.linalg.norm(residual) > 0  # the source is in the span nevertheless
    np.testing.assert_array_equal(project_source(np.zeros((2, 2)), [1., 2.])[0], [0., 0.])


def test_empty_columns_absorb_and_ownership_does_not_cancel():
    layout = small_layout()
    c = Code.empty(2, 2, layout)
    b = np.zeros(layout.size); b[0] = 1
    c, first = insert_source(c, b, [1., 0.], layout)
    assert first.drop == 0 and first.cost == 0
    c, second = insert_source(c, b, [0., -1.], layout)
    assert second.drop is None and second.cost == 0
    assert c.radius == 0
    np.testing.assert_array_equal(expanded(c)[0], [1., -1.])
    assert expanded(c).sum() == 0      # total activity would lose the two owners


@pytest.mark.parametrize('old_owner,expected_drop', [(1., None), (.25, 0), (10., None)])
def test_strict_discard_scan_and_same_discard_field_cost(old_owner, expected_drop):
    layout = small_layout(); c = Code.empty(1, 2, layout)
    c.U[0, 0], c.V[0] = 1., old_owner
    b = np.zeros(layout.size); b[-1] = 1.
    full = expanded(c) + b[:, None] * [1., 1.]
    out, bill = insert_source(c, b, [1., 1.], layout)
    assert bill.drop == expected_drop
    np.testing.assert_allclose(full - expanded(out),
                               bill.discarded_effect[:, None] * bill.discarded_owner)
    expected = min(1., old_owner)
    assert bill.cost == expected and out.radius == expected
    for i, n in enumerate(layout.fields):
        actual = np.linalg.norm((full - expanded(out))[layout.slices[n]], axis=0).max()
        assert actual <= bill.field_costs[i] + 1e-15


def test_tiny_effect_with_large_ownership_is_not_cheap():
    layout = small_layout(); c = Code.empty(1, 1, layout)
    c.U[0, 0], c.V[0, 0] = 1e-9, 1e12
    b = np.zeros(layout.size); b[1] = 1
    out, bill = insert_source(c, b, [1.], layout)
    assert bill.drop is None and bill.cost == 1.
    assert out.V[0, 0] == 1e12


@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.float16])
def test_seeded_insertion_field_and_global_error_bills(dtype):
    layout = small_layout(); c = Code.empty(2, 3, layout, dtype)
    full = np.zeros((layout.size, 3)); rng = np.random.default_rng(17)
    for _ in range(24):
        b, owner = rng.normal(size=layout.size), rng.normal(size=3)
        before = expanded(c)
        c, bill = insert_source(c, b, owner, layout)
        full += b[:, None] * owner
        local = before + b[:, None] * owner - expanded(c)
        assert np.linalg.norm(local, axis=0).max() <= bill.cost + bill.representation_error + bill.projection_residual + 1e-10
        assert np.linalg.norm(full - expanded(c), axis=0).max() <= c.radius + 1e-10
        for i, n in enumerate(layout.fields):
            assert np.linalg.norm((full - expanded(c))[layout.slices[n]], axis=0).max() <= c.fields[i] + 1e-10


def test_transport_is_source_free_then_each_source_is_added_once():
    layout = small_layout(); c = Code.empty(2, 2, layout)
    b = np.zeros(layout.size); b[0] = 2
    c, _ = insert_source(c, b, [1., 2.], layout)
    full = expanded(c)
    c = transport(c, lambda u: .5 * u, .5, layout)
    full *= .5
    for index in (1, 2):
        b = np.zeros(layout.size); b[index] = 1
        c, _ = insert_source(c, b, [2., -1.], layout)
        full += b[:, None] * [2., -1.]
    assert np.linalg.norm(full - expanded(c), axis=0).max() <= c.radius + 1e-14
    assert np.max(expanded(c)) <= 2  # not one new source per basis column


def test_factor_storage_rounding_is_in_the_bill():
    layout = small_layout(); c = Code.empty(2, 2, layout, np.float16)
    b = np.zeros(layout.size); b[0] = 1.0003
    c, bill = insert_source(c, b, [123.456, -.12345], layout)
    assert bill.representation_error > 0 and c.bills[1] > 0
    assert np.linalg.norm(b[:, None] * [123.456, -.12345] - expanded(c), axis=0).max() <= c.radius + 1e-10


def test_record_opening_time_preinlet_key_and_capsule_overwrite():
    layout = small_layout(); c = Code.empty(2, 1, layout); records = Records.empty(layout)
    c.U[0] = np.arange(layout.size)
    opening = c.U[:, layout.slices['native.0.s']].copy()
    c = begin_capture(c, records, layout, 0, 11)
    c = capture_activation(c, records, layout, 0, key=True)
    key = c.U[:, layout.slices['pending.0.key']].copy()
    c.U[:, layout.slices['activation']] += 100  # actual post-inlet input differs
    c.U[:, layout.slices['native.0.s']] += 900  # ending cache is not the opening cache
    c = capture_activation(c, records, layout, 0)
    inputs = c.U[:, layout.slices['pending.0.inputs']].copy()
    c = commit_record(c, records, layout, 0, 0)
    np.testing.assert_array_equal(c.U[:, layout.slices['record.0.0.s']], opening)
    np.testing.assert_array_equal(c.U[:, layout.slices['record.0.0.key']], key)
    np.testing.assert_array_equal(c.U[:, layout.slices['record.0.0.inputs']], inputs)
    assert records.opening_events[0] == 11
    first = records.receipt(0, 0)
    old_key = c.U[:, layout.slices['record.0.0.key']].copy()
    c.U[:, layout.slices['activation']] += 10
    c = complete_capture(c, records, layout, event=12)
    fresh_key = c.U[:, layout.slices['pending.0.key']].copy()
    fresh_input = c.U[:, layout.slices['pending.0.inputs']].copy()
    c = commit_record(c, records, layout, 0, 0)
    np.testing.assert_array_equal(c.U[:, layout.slices['record.0.0.key']], (old_key + fresh_key) / 2)
    np.testing.assert_array_equal(c.U[:, layout.slices['record.0.0.inputs']], fresh_input)
    assert records.counts[0, 0] == 2
    with pytest.raises(ValueError, match='stale'):
        records.validate(first)
    second = records.receipt(0, 0)
    c = complete_capture(c, records, layout, event=13)
    fresh_key = c.U[:, layout.slices['pending.0.key']].copy()
    c = commit_record(c, records, layout, 0, 0, replace=True)
    np.testing.assert_array_equal(c.U[:, layout.slices['record.0.0.key']], fresh_key)
    assert records.counts[0, 0] == 1 and records.versions[0, 0] == 3
    with pytest.raises(ValueError, match='stale'):
        records.validate(second)


def test_incomplete_capture_cannot_commit_previous_pending_fields():
    layout = small_layout(); c = Code.empty(2, 1, layout); r = Records.empty(layout)
    c = begin_capture(c, r, layout, 0, 0)
    with pytest.raises(ValueError, match='incomplete'):
        commit_record(c, r, layout, 0, 0)


def test_pending_is_recompressed_with_the_shared_basis():
    layout = small_layout(); c = Code.empty(1, 1, layout); r = Records.empty(layout)
    c.U[0, layout.slices['native.0.s']] = .1; c.V[0] = .01
    c = complete_capture(c, r, layout)
    full_pending = expanded(c)[layout.slices['pending.0.s']].copy()
    b = np.zeros(layout.size); b[0] = 1
    c, bill = insert_source(c, b, [1.], layout)
    assert bill.drop == 0
    c = commit_record(c, r, layout, 0, 0)
    actual = expanded(c)[layout.slices['record.0.0.s']]
    # Pending is NOT an independent uncompressed array paired with new V.
    np.testing.assert_array_equal(actual, 0)
    error = np.linalg.norm(full_pending - actual, axis=0).max()
    assert error <= c.fields[layout.indices['record.0.0.s']]


def test_unknown_arithmetic_is_not_certified_and_threshold_is_strict():
    layout = small_layout(); c = Code.empty(1, 1, layout)
    c.U[0, 0] = c.V[0, 0] = 1
    m = np.zeros(layout.size); m[0] = 1
    result = capture_credit(c, m)
    assert result['unavailable'] and not result['margin_pass'].any()
    c.bills[0] = 1
    result = capture_credit(c, m, NumericalBudget(contraction=ErrorTerm(0.)))
    np.testing.assert_array_equal(result['margin'], 0)
    assert not result['margin_pass'].any() and not result['certified']
    c.bills[0] = .5
    result = capture_credit(c, m, NumericalBudget(contraction=ErrorTerm(0., 'empirical')))
    assert result['margin_pass'].all() and result['conditional_credit'][0] == 1
    assert result['certified_credit'][0] == 0 and not result['certified']


def test_same_reference_recent_subtraction_and_mismatch_bill():
    layout = small_layout(); c = Code.empty(1, 1, layout)
    c.U[0, 0], c.V[0, 0], c.bills[0] = 1., 2., .1
    m = np.zeros(layout.size); m[0] = 1
    budget = NumericalBudget(contraction=ErrorTerm(0.))
    total = capture_credit(c, m, budget)
    recent = np.array([.5])
    matched = capture_credit(c, m, budget, recent, same_reference=True)
    np.testing.assert_array_equal(matched['error'], total['error'])
    np.testing.assert_array_equal(matched['estimate'], total['estimate'] - recent)
    assert capture_credit(c, m, budget, recent)['unavailable']
    mismatch = capture_credit(c, m, NumericalBudget(contraction=ErrorTerm(0.), recent_mismatch=ErrorTerm(.2)), recent)
    np.testing.assert_allclose(mismatch['error'], total['error'] + .2)


def test_modulation_and_readout_errors_join_the_compression_bill():
    layout = small_layout(); c = Code.empty(1, 2, layout)
    c.U[0, 0], c.V[0], c.bills[0] = 2., [1., -2.], .1
    m = np.zeros(layout.size); m[0] = 3
    result = capture_credit(c, m, NumericalBudget(state=ErrorTerm(.2), modulation=ErrorTerm(.3), contraction=ErrorTerm(.4)))
    np.testing.assert_allclose(result['error'], [.9 + .6 + .4, .9 + 1.2 + .4])


def test_field_credit_does_not_charge_unread_record_error():
    layout = small_layout(); c = Code.empty(1, 1, layout)
    c.U[0, 0] = c.V[0, 0] = 1
    c.bills[0] = 100
    c.fields[layout.indices['record.0.0.s']] = 100
    m = np.zeros(layout.size); m[0] = 1
    budget = NumericalBudget(contraction=ErrorTerm(0.))
    assert not capture_credit(c, m, budget)['margin_pass'].any()
    result = capture_credit(c, m, budget, layout=layout)
    assert result['error'][0] == 0 and result['margin_pass'].all()
    assert not result['certified']


def test_persistent_bytes_and_continuation_do_not_grow_with_events():
    layout = small_layout(); c = Code.empty(2, 3, layout); r = Records.empty(layout)
    initial = c.nbytes + r.nbytes
    budget = layout.budget(2, 3, np.float64)
    assert initial == budget['effect_bytes'] + budget['owner_bytes'] + budget['ledger_bytes'] + budget['metadata_bytes']
    rng = np.random.default_rng(9)
    for i in range(48):
        b = rng.normal(size=layout.size)
        c, _ = insert_source(c, b, [1., -1., 2.], layout)
        assert c.nbytes + r.nbytes == initial
    restored = copy.deepcopy((c, r))
    b = rng.normal(size=layout.size)
    a, ba = insert_source(c, b, [1., 2., 3.], layout)
    z, bz = insert_source(restored[0], b, [1., 2., 3.], layout)
    assert ba.drop == bz.drop
    np.testing.assert_array_equal(a.U, z.U)
    np.testing.assert_array_equal(a.V, z.V)
    np.testing.assert_array_equal(a.bills, z.bills)
