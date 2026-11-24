"""Algebra/data-contract tests, not synthetic capability experiments."""
import numpy as np

from scripts.audit_full_loop_topology import operators, persistence, phase_shift, solve_regression, dirac_signal_dynamics


def test_observed_faces_are_not_filled_from_pairwise_edges():
    hollow=np.asarray([[1,1,0],[0,1,1],[1,0,1]],bool)
    full=np.asarray([[1,1,1]],bool)
    assert persistence(hollow)['dimensions'][1]['essential_at_zero_support']==1
    assert persistence(full)['dimensions'][1]['maximum_lifetime']==0


def test_tetrahedron_boundary_and_filling_have_distinct_h2():
    sphere=~np.eye(4,dtype=bool)
    assert persistence(sphere)['dimensions'][2]['essential_at_zero_support']==1
    assert persistence(np.ones((1,4),bool))['dimensions'][2]['maximum_lifetime']==0


def test_real_incidence_chain_and_dirac_square():
    data=np.asarray([[1,1,1,1],[1,0,1,1],[0,1,1,1]],bool)
    groups,boundaries,d,error=operators(data,.1)
    assert error==0
    offset=np.cumsum([0]+[len(g) for g in groups])
    square=(d@d).toarray()
    for a,(start,end) in enumerate(zip(offset[:-1],offset[1:])):
        for b,(other,last) in enumerate(zip(offset[:-1],offset[1:])):
            if a!=b:assert np.max(abs(square[start:end,other:last]),initial=0)==0


def test_surrogate_preserves_within_phase_and_local_marginals():
    data=np.arange(128*20).reshape(128,20)
    result=phase_shift(data,np.random.default_rng(3))
    for start in (0,64):
        for phase in range(5):
            sl=slice(phase*4,phase*4+4)
            assert np.array_equal(np.sort(result[start:start+64,sl],axis=0),np.sort(data[start:start+64,sl],axis=0))
            assert np.array_equal(np.diff(result[start:start+64,sl],axis=1),np.ones((64,3)))


def test_regression_selection_does_not_use_test_targets():
    x=np.arange(30,dtype=float)[:,None]
    y=3*x[:,0]+2
    prediction,alpha=solve_regression(x[:15],y[:15],x[15:20],y[15:20],x[20:])
    assert np.mean((prediction-y[20:])**2)<1e-3
    assert alpha==.001


def test_dirac_regression_is_invariant_to_cell_orientation():
    # Algebraic coordinate invariance, not a trained-memory capability task.
    import scipy.sparse as sp
    rng=np.random.default_rng(3)
    raw=rng.standard_normal((512,3))
    d=sp.csr_matrix([[0.,0.,-1.],[0.,0.,1.],[-1.,1.,0.]])
    flip=np.array([1.,1.,-1.])
    actual=dirac_signal_dynamics(raw,d,[2,1])
    changed=dirac_signal_dynamics(raw*flip,sp.diags(flip)@d@sp.diags(flip),[2,1])
    for a,b in zip(actual['orders'],changed['orders']):
        assert abs(a['dirac_test_mse']-b['dirac_test_mse'])<1e-12
        assert abs(a['hodge_only_test_mse']-b['hodge_only_test_mse'])<1e-12


def test_port_units_do_not_change_standardized_dirac_results():
    import scipy.sparse as sp
    rng=np.random.default_rng(18)
    raw=rng.standard_normal((512,3))
    d=sp.csr_matrix([[0.,0.,-1.],[0.,0.,1.],[-1.,1.,0.]])
    first=dirac_signal_dynamics(raw,d,[2,1])
    second=dirac_signal_dynamics(raw*np.array([1e-5,3.,2000.]),d,[2,1])
    for a,b in zip(first['orders'],second['orders']):
        assert abs(a['dirac_test_mse']-b['dirac_test_mse'])<1e-12


def test_activity_surrogate_preserves_all_within_phase_joint_frequencies():
    from scripts.audit_full_loop_topology import activity
    from itertools import combinations
    active=activity(np.random.default_rng(4).normal(size=(128,20)))
    shifted=phase_shift(active,np.random.default_rng(29))
    for start in (0,64):
        for phase in range(5):
            for size in range(1,5):
                for subset in combinations(range(4*phase,4*phase+4),size):
                    assert active[start:start+64,subset].all(1).sum()==shifted[start:start+64,subset].all(1).sum()
