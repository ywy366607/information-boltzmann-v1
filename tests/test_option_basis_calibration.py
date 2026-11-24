import torch

from scripts.option_basis_calibration import evaluate, fit


def problem():
    x = torch.tensor([[1., .2], [.3, 1.], [.9, .5], [.1, .6]], dtype=torch.double)
    scale = torch.tensor([1., 2., .7, 1.2], dtype=torch.double)
    options = torch.tensor([0, 1, 0, 2])
    gram = (1+x@x.T).square()*scale[:, None]*scale[None]
    factor = torch.linalg.cholesky(gram+.1*torch.eye(4, dtype=torch.double))
    residual = torch.tensor([.3, -.7, .9, .1], dtype=torch.double)
    return factor, scale, options, residual, gram


def test_schur_stationarity_and_regularized_minimum():
    factor, scale, options, residual, gram = problem()
    got = fit(factor, scale, options, residual)
    h = scale[:, None]*(options[:, None] == got['classes'][None]).double()
    y = scale*residual
    a, b = got['alpha'], got['beta']
    assert got['equation_error'] < 1e-12
    assert got['orthogonality_error'] < 1e-12

    def objective(a, b):
        return (y-gram@a-h@b).square().sum()+.1*(a@gram@a)

    da = torch.tensor([.2, -.1, .7, -.3], dtype=torch.double)
    db = torch.tensor([.1, .4, -.2], dtype=torch.double)
    increase = (gram@da+h@db).square().sum()+.1*(da@gram@da)
    torch.testing.assert_close(objective(a+da, b+db)-objective(a, b), increase)
    assert increase > 0


def test_option_prior_shift_invariance_on_same_training_labels():
    factor, scale, options, residual, _ = problem()
    base = torch.tensor([.2, -.4, .2, .8], dtype=torch.double)
    shift = torch.tensor([5., -2., 8.], dtype=torch.double)
    first = fit(factor, scale, options, residual)
    second = fit(factor, scale, options, residual-shift[options])
    torch.testing.assert_close(first['alpha'], second['alpha'])
    torch.testing.assert_close(second['beta'], first['beta']-shift)
    cross = factor@factor.T-.1*torch.eye(4, dtype=torch.double)
    cross = cross/scale[:, None]
    torch.testing.assert_close(evaluate(first, base, options, cross),
                               evaluate(second, base+shift[options], options, cross))


def test_unobserved_option_keeps_prior_plus_content_only():
    factor, scale, _, residual, _ = problem()
    got = fit(factor, scale, torch.zeros(4, dtype=torch.long), residual)
    prior = torch.tensor([.3, .7], dtype=torch.double)
    answer = evaluate(got, prior, torch.tensor([0, 2]), torch.zeros(2, 4))
    torch.testing.assert_close(answer[0], prior[0]+got['beta'][0])
    torch.testing.assert_close(answer[1], prior[1])


def test_private_calibration_survives_real_learning_and_restores_scope(tmp_path):
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local import native_amortized_decision as amort
    from scripts.option_basis_calibration import calibrated_predictions
    from test_relation_retention_use import configured

    torch.set_num_threads(1)
    engine, archive = configured(tmp_path)
    original = planner.predict
    try:
        engine.gain_units_schema = 1
        ev.configure(engine, schema=5, word_execution='native_anchor')
        engine.native_word_runtime = 'batch'
        engine.model._preview_batch_enabled = True
        amort.configure(engine, dict(schema=1, seed=201120, planner='action_batch',
                                     correction='decision_budget'))
        with calibrated_predictions(engine) as stats:
            first = engine.step(torch.randn(8, 4))
            revision = engine.latent_action_bank['revision']
            packets = [r['packet'] for r in engine.latent_action_bank['records'][:2]]
            baseline, novelty = original(engine, packets)
            changed, new_novelty = planner.predict(engine, packets)
            torch.testing.assert_close(changed[:, :-1], baseline[:, :-1], rtol=0, atol=0)
            torch.testing.assert_close(new_novelty, novelty, rtol=0, atol=0)
            fitted_count = stats['fits']
            planner.predict(engine, packets)
            assert stats['fits'] == fitted_count
            second = engine.step(torch.randn(8, 4))
            assert engine.latent_action_bank['revision'] > revision
            planner.predict(engine, packets)
            assert stats['fits'] > fitted_count and stats['queries'] >= stats['fits']
            for row in (first, second):
                receipt = row['causal_use']['native_amortization']['action_planner']
                assert receipt['new_world_label'] and receipt['native_labels'] > 0
        assert planner.predict is original
    finally:
        archive.close()


def test_private_scope_restores_predictor_on_exception():
    import pytest
    from scripts.ib_local import latent_action_planner as planner
    from scripts.option_basis_calibration import calibrated_predictions

    original = planner.predict
    with pytest.raises(RuntimeError, match='scope'):
        with calibrated_predictions(object()):
            raise RuntimeError('scope')
    assert planner.predict is original
