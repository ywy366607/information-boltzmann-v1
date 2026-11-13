"""Analytic estimator identities; no synthetic capability claims."""
import numpy as np
import pytest

from scripts.ib_local.closed_loop_information import (
    partitions, geometric_cut, analyze_trace, mean_interval)


def test_all_unordered_partitions_once():
    assert len(list(partitions(range(6)))) == 31
    assert len(list(partitions(range(5)))) == 15


def test_independent_lag_dynamics_zero_even_with_correlated_noise():
    A=np.diag([.6,.3])
    noise=np.array([[1.,.4],[.4,1.]])
    result,_=geometric_cut(A,np.eye(2),noise,[[0],[1]])
    assert abs(result['phi_g_nats']) < 1e-9
    assert result['converged']


def test_cut_loss_positive_and_bounded_by_no_history_model():
    A=np.array([[.3,.5],[.4,.2]])
    noise=np.eye(2)
    sx=np.eye(2)
    result,cut=geometric_cut(A,sx,noise,[[0],[1]])
    upper=.5*np.linalg.slogdet(noise+A@sx@A.T)[1]
    assert 0 < result['phi_g_nats'] <= upper
    assert cut[0,1] == cut[1,0] == 0
    assert result['max_gradient'] < 1e-5


def test_zero_temporal_information_does_not_count_shared_innovation_as_phi():
    result,_=geometric_cut(np.zeros((2,2)),np.eye(2),np.array([[1.,.8],[.8,1.]]),[[0],[1]])
    assert abs(result['phi_g_nats']) < 1e-10


def test_signed_bootstrap_does_not_clip_negative_results():
    result=mean_interval(np.full(64,-.7))
    assert result['mean']==pytest.approx(-.7)
    assert result['ci95'][1] < 0


def test_chronological_test_is_not_used_to_fit_phi_or_select_partition():
    rng=np.random.default_rng(91)
    # Estimator interface data, not a learned synthetic-task experiment.
    data=rng.normal(size=(128,4))
    groups=dict(a=[0],b=[1],c=[2],d=[3])
    first=analyze_trace(data,groups)
    data[-24:]*=20
    second=analyze_trace(data,groups)
    assert first['minimum_projected_phi_g']==second['minimum_projected_phi_g']
    assert first['validation_selected_partition']==second['validation_selected_partition']
    assert first['heldout_predictive_cut_penalty'] != second['heldout_predictive_cut_penalty']


def test_too_short_or_nonfinite_not_silently_zero():
    with pytest.raises(ValueError):
        analyze_trace(np.zeros((64,4)),dict(a=[0,1],b=[2,3]))
    with pytest.raises(ValueError):
        analyze_trace(np.full((128,4),np.nan),dict(a=[0,1],b=[2,3]))


def test_method_observer_restores_class_lookup_even_on_exception():
    from scripts.ib_local.audit_closed_loop_information import patch_method
    class Item:
        def read(self):
            return 'original'
    obj=Item()
    with pytest.raises(RuntimeError):
        with patch_method(obj,'read',lambda:'intervention'):
            assert obj.read()=='intervention'
            raise RuntimeError('abort branch')
    assert obj.read()=='original'
    assert 'read' not in vars(obj)


def test_surrogate_preserves_both_coordinates_and_within_block_order():
    from scripts.ib_local.analyze_closed_loop_information import block_surrogate
    trace=np.arange(64*4).reshape(64,4)
    shuffled=block_surrogate(trace,dict(a=[0,1],b=[2,3]),np.random.default_rng(3))
    for columns in ([0,1],[2,3]):
        before=trace[:,columns].reshape(8,8,2)
        after=shuffled[:,columns].reshape(8,8,2)
        assert sorted(tuple(b.flatten()) for b in before)==sorted(tuple(b.flatten()) for b in after)
    assert not np.array_equal(trace,shuffled)
