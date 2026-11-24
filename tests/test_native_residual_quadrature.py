"""Finite integration contracts and production whole-loop wiring."""
import itertools

import torch

from scripts.ib_local.native_residual_quadrature import partition, plan, readout, configure


def test_strata_cover_same_points_mass_and_global_rng():
    proxy = torch.tensor([0., 0., 0., .1, .2, 2., 3., 4., 5.])
    state = torch.get_rng_state().clone()
    index, weights, groups = plan(proxy, 4, seed=191120)
    assert torch.equal(torch.get_rng_state(), state)
    assert sorted(torch.cat(groups).tolist()) == list(range(len(proxy)))
    assert len(index) == 4 and weights.sum() == 1
    for g, i in zip(groups, index):
        assert bool((g == i).any())
    again = plan(proxy, 4, seed=191120)
    assert torch.equal(index, again[0])
    for k in range(1, 4):
        previous = partition(proxy, k)
        following = partition(proxy, k+1)
        sse = lambda gs: sum(((proxy[g]-proxy[g].mean())**2).sum() for g in gs)
        assert sse(following) <= sse(previous)+1e-6


def test_residual_finite_average_is_exact_with_unequal_strata():
    # Numerical integration unit check: enumerate ALL local draws to bind
    # NativeResidualQuadrature.residual_average_exact to production readout.
    proxy = torch.tensor([-4., 0., 1., 1., 8., 9.], dtype=torch.float64)
    native = torch.tensor([2., -3., 4., 11., 6., -7.], dtype=torch.float64)
    groups = partition(proxy, 3)
    weights = proxy.new_tensor([len(g)/len(proxy) for g in groups])
    estimates = [readout(proxy, torch.tensor(indices), weights, native[list(indices)])
                 for indices in itertools.product(*(g.tolist() for g in groups))]
    torch.testing.assert_close(torch.stack(estimates).mean(), native.mean())
    for seed in (17, 191120):
        index, weights, _ = plan(proxy, 3, seed=seed)
        torch.testing.assert_close(readout(proxy, index, weights, proxy[index]+2), proxy.mean()+2)


def test_flat_proxy_and_one_point_keep_original_finite_budget():
    for n in (1, 3, 19):
        proxy = torch.ones(n)
        index, weights, groups = plan(proxy, 4, seed=191120)
        assert len(index) == min(n, 4)
        assert torch.isfinite(readout(proxy, index, weights, proxy[index]))
        assert sorted(torch.cat(groups).tolist()) == list(range(n))


def test_native_whole_loop_uses_same_joint_law_and_finite_U(tmp_path):
    from test_relation_retention_use import configured
    from scripts.ib_local import epistemic_value_reference as ev
    e, archive = configured(tmp_path)
    try:
        e.gain_units_schema = 1
        ev.configure(e, schema=5, word_execution='native_anchor')
        e.native_word_runtime = 'batch'
        e.model._preview_batch_enabled = True
        configure(e, dict(schema=1, mode='residual', seed=191120))
        e.step(torch.randn(8, 4))
        row = e.step(torch.randn(8, 4))
        trace = e.successor_distribution_last
        assert trace['schema'] == 2 and trace['joint']['samples'] == 4096
        assert trace['branches'] <= e.joint_proposal_budget
        assert abs(sum(trace['joint']['address_mass'].values())-1) < 1e-7
        assert row['causal_use']['finite_U']['verified']
        assert row['causal_use']['reward_version'] == 'finite_relation_use_v5'
        assert row['causal_program']['successor_learning']['target_excluded']
        assert row['causal_program']['mother_joint_prequential'] is None
    finally:
        archive.close()
