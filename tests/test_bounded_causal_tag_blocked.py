"""Blocked/dense equivalence and bounded scratch space; no model loads."""
import copy

import numpy as np
import pytest

from scripts.ib_local.bounded_causal_tag_reference import (
    Code, ErrorTerm, Layout, NumericalBudget, Records, begin_capture,
    capture_activation, capture_credit, commit_record, insert_source,
)
from scripts.ib_local.bounded_causal_tag_blocked import (
    ArraySource, FieldLeg, assign_fields_inplace, begin_capture_inplace,
    capture_activation_inplace, capture_credit_blocked, commit_record_inplace,
    field_row_gain, insert_source_inplace, workspace_budget,
)


def layout(width=2):
    return Layout(layers=2, width=width, heads=1, head_dim=2, conv=2,
                  tokens=2, activation_tokens=3, ports=(0, 1), capacity=2)


def expansion(c):
    return c.U.astype(np.float64).T @ c.V.astype(np.float64)


@pytest.mark.parametrize('block_size', [1, 3, 17, 4096])
@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.float16])
def test_blocked_matches_dense_nonambiguous_stream(block_size, dtype):
    l = layout(); dense = Code.empty(2, 3, l, dtype); blocked = dense.copy()
    rng = np.random.default_rng(83)
    full = np.zeros((l.size, 3))
    u_identity, v_identity = id(blocked.U), id(blocked.V)
    for step in range(12):
        b, owner = rng.normal(size=l.size), rng.normal(size=3)
        dense, reference = insert_source(dense, b, owner, l)
        bill = insert_source_inplace(blocked, ArraySource(b, ('held', step)), owner, l, block_size)
        assert bill['drop'] == reference.drop
        np.testing.assert_allclose(expansion(blocked), expansion(dense), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(bill['cost'], reference.cost, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(bill['field_costs'], reference.field_costs, atol=1e-10, rtol=1e-10)
        full += b[:, None] * owner
        assert np.linalg.norm(full - expansion(blocked), axis=0).max() <= blocked.radius + 1e-10
        for f, n in enumerate(l.fields):
            error = np.linalg.norm((full - expansion(blocked))[l.slices[n]], axis=0).max()
            assert error <= blocked.fields[f] + 1e-10
    assert id(blocked.U) == u_identity and id(blocked.V) == v_identity


@pytest.mark.parametrize('old_owner,drop', [(1., None), (.25, 0), (10., None)])
def test_exact_ties_and_column_exchange(old_owner, drop):
    l = layout(); c = Code.empty(1, 2, l)
    c.U[0, 0] = 1; c.V[0] = old_owner
    b = np.zeros(l.size); b[-1] = 1
    old = expansion(c)
    bill = insert_source_inplace(c, ArraySource(b), [1., 1.], l, 3)
    assert bill['drop'] == drop
    error = np.linalg.norm(old + b[:, None] - expansion(c), axis=0)
    assert error.max() <= c.radius


def test_nonorthogonal_absorption_and_owner_cancellation():
    l = layout(); c = Code.empty(2, 2, l)
    c.U[:, :2] = [[1., 0.], [1., 1.]]
    b = np.zeros(l.size); b[1] = 1
    dense, reference = insert_source(c.copy(), b, [1., -1.], l)
    result = insert_source_inplace(c, ArraySource(b), [1., -1.], l, 1)
    np.testing.assert_array_equal(result['coefficients'], reference.coefficients)
    np.testing.assert_array_equal(expansion(c), expansion(dense))
    assert expansion(c).sum() == 0 and np.linalg.norm(expansion(c)) > 0


def test_changed_source_version_is_rejected_before_basis_mutation():
    l = layout(); c = Code.empty(2, 1, l)
    class Changed:
        token = 0
        def __call__(self, lo, hi):
            self.token += 1
            return np.ones(hi - lo)
    before = c.copy()
    with pytest.raises(ValueError, match='time/version'):
        insert_source_inplace(c, Changed(), [1.], l, 4)
    np.testing.assert_array_equal(c.U, before.U)
    np.testing.assert_array_equal(c.V, before.V)
    np.testing.assert_array_equal(c.bills, before.bills)


def test_no_full_basis_copy_or_cast_and_provider_blocks_are_bounded():
    l = layout(1024); c = Code.empty(2, 3, l, np.float32)
    original_shape = c.U.shape
    class Guard(np.ndarray):
        def astype(self, *a, **k):
            if self.shape == original_shape:
                raise AssertionError('full basis conversion')
            return super().astype(*a, **k)
        def copy(self, *a, **k):
            if self.shape == original_shape:
                raise AssertionError('full basis copy')
            return super().copy(*a, **k)
    c.U = c.U.view(Guard)
    calls = []
    def provider(lo, hi):
        calls.append(hi - lo)  # test audit only, not candidate state
        return np.sin(np.arange(lo, hi) * .125)
    result = insert_source_inplace(c, provider, [1., -1., .25], l, 64)
    assert max(calls) <= 64 and result['stats']['max_block'] <= 64
    credit = capture_credit_blocked(c, provider, l, NumericalBudget(contraction=ErrorTerm(0.)), block_size=64)
    assert credit['stats']['max_block'] <= 64
    assert result['workspace']['block_vector_bytes'] == 12 * 64 * 8
    other = workspace_budget(2, 3, len(l.fields), 64)
    assert other == result['workspace']  # source D itself is not a scratch dimension


def test_sparse_rows_gain_and_alias_guard():
    l = layout(); c = Code.empty(2, 3, l)
    rng = np.random.default_rng(9); c.U[:] = rng.normal(size=c.U.shape); c.V[:] = rng.normal(size=c.V.shape)
    c.bills[0] = 1.; c.fields[:] = .1
    rows = {'pending.0.key': [FieldLeg('activation', offset=2)]}
    assert field_row_gain(l, rows) == pytest.approx(np.sqrt(2))
    full = expansion(c)
    expected = full.copy(); expected[l.slices['pending.0.key']] = full[l.slices['activation']][2:4]
    bill = assign_fields_inplace(c, l, rows, 1)
    np.testing.assert_allclose(expansion(c), expected)
    assert bill['gain'] < 2 and c.radius == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match='cross-target'):
        assign_fields_inplace(c, l, {'pending.0.key': [FieldLeg('pending.1.key')],
                                    'pending.1.key': [FieldLeg('activation', offset=2)]})


def test_open_capture_compression_commit_and_versions_match_dense():
    l = layout(); dense = Code.empty(2, 3, l, np.float32); blocked = dense.copy()
    rd, rb = Records.empty(l), Records.empty(l)
    rng = np.random.default_rng(71)
    for epoch in range(6):
        dense = begin_capture(dense, rd, l, 0, epoch)
        begin_capture_inplace(blocked, rb, l, 0, epoch, 3)
        dense = capture_activation(dense, rd, l, 0, key=True)
        capture_activation_inplace(blocked, rb, l, 0, key=True, block_size=3)
        b = np.zeros(l.size); b[:l.slices['native.1.v'].stop] = rng.normal(size=l.slices['native.1.v'].stop)
        owner = rng.normal(size=3)
        dense, _ = insert_source(dense, b, owner, l)
        insert_source_inplace(blocked, ArraySource(b), owner, l, 3)
        dense = capture_activation(dense, rd, l, 0)
        capture_activation_inplace(blocked, rb, l, 0, block_size=3)
        old = rb.receipt(0, 0)
        dense = commit_record(dense, rd, l, 0, 0, replace=epoch == 4)
        commit_record_inplace(blocked, rb, l, 0, 0, replace=epoch == 4, block_size=3)
        np.testing.assert_allclose(expansion(blocked), expansion(dense), atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(rb.counts, rd.counts)
        np.testing.assert_array_equal(rb.versions, rd.versions)
        assert blocked.radius <= dense.radius + 1e-9
        with pytest.raises(ValueError, match='stale'):
            rb.validate(old)


def test_blocked_credit_matches_dense_field_budget_and_unknown_arithmetic():
    l = layout(); c = Code.empty(2, 3, l, np.float32)
    rng = np.random.default_rng(87)
    for _ in range(5):
        insert_source_inplace(c, ArraySource(rng.normal(size=l.size)), rng.normal(size=3), l, 7)
    m = rng.normal(size=l.size)
    budget = NumericalBudget(state=ErrorTerm(.1), modulation=ErrorTerm(.2), contraction=ErrorTerm(.3, 'empirical'))
    expected = capture_credit(c, m, budget, layout=l)
    actual = capture_credit_blocked(c, ArraySource(m), l, budget, block_size=3)
    for name in ('estimate', 'error', 'margin'):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12)
    np.testing.assert_array_equal(actual['margin_pass'], expected['margin_pass'])
    assert not actual['certified']
    assert capture_credit_blocked(c, ArraySource(m), l)['unavailable']
    assert capture_credit_blocked(c, ArraySource(m), l, budget, recent=np.ones(3))['unavailable']
    matched = capture_credit_blocked(c, ArraySource(m), l, budget, recent=np.ones(3), same_reference=True)
    np.testing.assert_allclose(matched['estimate'], actual['estimate'] - 1)


def test_near_ties_expose_costs_without_an_epsilon_selection_rule():
    l = layout(); source = np.zeros(l.size); source[1] = 1
    for weight in (np.nextafter(1., 0.), 1., np.nextafter(1., 2.)):
        c = Code.empty(1, 1, l); c.U[0, 0] = 1.; c.V[0] = weight
        bill = insert_source_inplace(c, ArraySource(source), [1.], l, 1)
        assert bill['drop'] == (0 if weight < 1 else None)
        assert bill['cost'] == min(bill['candidate_costs'])


def test_incomplete_capture_and_unaffected_records_keep_identity():
    l = layout(); c = Code.empty(2, 1, l); r = Records.empty(l)
    untouched = c.U[:, l.slices['record.1.1.s']].copy()
    begin_capture_inplace(c, r, l, 0, 0)
    with pytest.raises(ValueError, match='incomplete'):
        commit_record_inplace(c, r, l, 0, 0)
    capture_activation_inplace(c, r, l, 0, key=True)
    capture_activation_inplace(c, r, l, 0)
    commit_record_inplace(c, r, l, 0, 0)
    np.testing.assert_array_equal(c.U[:, l.slices['record.1.1.s']], untouched)
    assert r.versions[1, 1] == 0


def test_composed_readout_gradient_matches_scalar_finite_difference():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scratch/bounded_causal_tags_20261108/run_blocked_field_admission.py'
    spec = importlib.util.spec_from_file_location('blocked_field_audit', path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    l = layout(); r = Records.empty(l); r.counts[0] = 1
    rng = np.random.default_rng(23)
    z, dz = rng.normal(size=l.size), rng.normal(size=l.size)
    m, _ = audit.readout(z, l, r)
    expected = float(m(0, l.size) @ dz)
    h = 1e-6
    plus = audit.readout(z + h * dz, l, r)[1]['value'] @ np.array([.7, -.3])
    minus = audit.readout(z - h * dz, l, r)[1]['value'] @ np.array([.7, -.3])
    np.testing.assert_allclose(expected, (plus - minus) / (2 * h), rtol=1e-7, atol=1e-9)
