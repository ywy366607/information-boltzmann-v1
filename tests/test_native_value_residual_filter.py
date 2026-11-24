import torch
from types import SimpleNamespace

from scripts.native_value_residual_filter import coefficients, prediction_delta, spectral_report


def test_two_fits_equal_original_fit_on_actual_remaining_residual():
    torch.manual_seed(221122)
    x = torch.randn(9, 5, dtype=torch.double)
    gram = (1+x@x.T).square()
    ridge = .1
    factor = torch.linalg.cholesky(gram+ridge*torch.eye(len(x), dtype=x.dtype))
    r = torch.randn(9, dtype=x.dtype)
    first = coefficients(factor, r, ridge, 1)
    second = coefficients(factor, r, ridge, 2)
    explicit = first+torch.cholesky_solve(r[:, None]-gram@first, factor)
    torch.testing.assert_close(second, explicit)
    eigenvalues, vectors = torch.linalg.eigh(gram)
    expected = vectors@((ridge/(eigenvalues+ridge)).square()*(vectors.T@r))
    torch.testing.assert_close(r-gram@second[:, 0], expected, atol=1e-12, rtol=1e-10)
    report = spectral_report(gram, r, ridge)
    assert report['two_step_remaining_fraction'] < report['one_step_remaining_fraction']


def test_single_step_and_duplicate_conditions_preserve_solver_semantics():
    x = torch.tensor([[1., 2.], [1., 2.], [-1., 0.]], dtype=torch.double)
    gram = (1+x@x.T).square()
    factor = torch.linalg.cholesky(gram+.1*torch.eye(3, dtype=x.dtype))
    r = torch.tensor([1., -1., .4], dtype=x.dtype)
    torch.testing.assert_close(coefficients(factor, r, .1, 1),
                               torch.cholesky_solve(r[:, None], factor))
    second = coefficients(factor, r, .1, 2)
    fitted = gram@second[:, 0]
    torch.testing.assert_close(fitted[0], fitted[1])


def test_active_packet_increment_matches_second_fit_and_preserves_cache():
    from scripts.ib_local.native_amortized_decision import features, _kernel
    torch.manual_seed(2201)
    encoder = torch.nn.Linear(4, 3, bias=False).double()
    e = SimpleNamespace(model=SimpleNamespace(encoder=encoder), relation_ridge=.1)
    packets = [dict(raw=torch.randn(2, 4, dtype=torch.double), scalars=[.8, float(i)])
               for i in range(5)]
    x = features(e, packets)
    d = torch.tensor([1., 1., 1., .7, .7], dtype=x.dtype)
    gram = _kernel(x, x)*d[:, None]*d[None]
    factor = torch.linalg.cholesky(gram+.1*torch.eye(len(x), dtype=x.dtype))
    r = torch.randn(len(x), dtype=x.dtype)
    alpha1 = coefficients(factor, r, .1, 1)
    e._native_fit_cache = dict(x=x, scale=d, factor=factor, coefficient=alpha1)
    before = {k:v.clone() for k,v in e._native_fit_cache.items()}
    delta = prediction_delta(e, packets)
    expected = (_kernel(x, x)*d[None])@(coefficients(factor, r, .1, 2)-alpha1)
    torch.testing.assert_close(delta, expected[:, 0])
    assert all(torch.equal(value, e._native_fit_cache[key]) for key,value in before.items())


def test_root_scope_preserves_nonroot_reads_and_restores_ports(monkeypatch):
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local import persistent_rhythm as rhythm
    from scripts.ib_local import native_amortized_decision as amort
    from scripts.ib_local.competing_action_teachers import action_key
    from scripts import native_value_residual_filter as refinement
    e = SimpleNamespace(persistent_rhythm_schema=1, _native_fit_cache={'ready': True})
    source, action = torch.zeros(1, 2), torch.ones(1, 2)
    e._competing_action_roots = {action_key(source, action): [1]}
    def read(engine, states, sources, *, carried=None):
        return torch.tensor([.3]), [['original cases']], 1
    def value(engine, state, s, a, mother, distribution, momentum):
        return rhythm.read_plans(engine, [state], [s], carried=state)[0]
    monkeypatch.setattr(rhythm, 'read_plans', read)
    monkeypatch.setattr(planner, 'value', value)
    monkeypatch.setattr(amort, 'recipe', lambda *args, **kwargs: {})
    monkeypatch.setattr(refinement, 'prediction_delta', lambda *args: torch.tensor([.2]))
    with refinement.refined_root_values(e) as stats:
        direct, cases, count = rhythm.read_plans(e, [object()], [source])
        torch.testing.assert_close(direct, torch.tensor([.3]))
        torch.testing.assert_close(planner.value(e, object(), source, action, None, None, None), torch.tensor([.5]))
        torch.testing.assert_close(planner.value(e, object(), source, source, None, None, None), torch.tensor([.3]))
        assert stats['root_calls'] == 1 and stats['nodes'] == 1
        assert cases == [['original cases']] and count == 1
    assert rhythm.read_plans is read and planner.value is value


def test_refined_root_completes_original_learning_and_settlement(tmp_path):
    from test_persistent_rhythm import body
    from scripts.ib_local import autonomous_cycle
    from scripts.native_value_residual_filter import refined_root_values
    torch.set_num_threads(1)
    torch.manual_seed(201120)
    e, archive = body(tmp_path, root_mode='observe')
    updates = []
    step = e.optimizer.step
    def counted(*args, **kwargs):
        result = step(*args, **kwargs)
        updates.append(1)
        return result
    e.optimizer.step = counted
    try:
        rows = []
        with refined_root_values(e) as stats:
            for _ in range(96):
                row = autonomous_cycle.dispatch(e)
                if row['autonomous_event']['kind'] == 'external':
                    rows.append(row)
                if len(rows) == 2:
                    break
        assert len(rows) == 2 and len(updates) >= 8
        assert stats['root_calls'] > 0 and stats['nodes'] > 0
        for row in rows:
            assert row['causal_use']['finite_U']['verified']
            assert row['causal_use']['native_amortization']['new_world_label']
            assert row['causal_use']['native_amortization']['action_planner']['teacher_count'] == 2
            assert row['policy_replay_max_error'] < 1e-3
        assert e._rhythm_runtime is None
        assert e._competing_action_roots is None
        assert archive.db.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        archive.close()
