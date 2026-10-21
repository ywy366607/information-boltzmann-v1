import numpy as np
import pytest
from information_boltzmann.cgb.tree_geometry import chain_geometry,forest_branch_probability


def test_degree_two_contraction_preserves_cable_and_chain_lengths():
    x=np.array([[0,0,0],[1,0,0],[2,0,0],[3,1,0],[3,-1,0]])
    g=chain_geometry(x,[-1,0,1,2,2])
    assert g['branches']==1 and g['tips']==2 and g['roots']==1
    np.testing.assert_allclose(sorted(g['chains']),[np.sqrt(2),np.sqrt(2),2])
    np.testing.assert_allclose(g['length'],2+2*np.sqrt(2))
    # Same physical tree with a different node order.
    order=np.array([4,2,0,3,1]);inverse=np.argsort(order)
    parent=np.array([-1,0,1,2,2]);p=parent[order].copy();p[p>=0]=inverse[p[p>=0]]
    other=chain_geometry(x[order],p)
    np.testing.assert_allclose(sorted(other['chains']),sorted(g['chains']))


def test_conditional_branch_probability_is_normalized_and_prevents_early_extinction():
    for b in range(1,10):
        assert forest_branch_probability(b,1)==1
        for m in range(1,10):
            stop=(m-1)*(b+m)/(m*(2*b+m-1))
            np.testing.assert_allclose(forest_branch_probability(b,m)+stop,1)
    assert forest_branch_probability(0,1)==0


def test_tree_cycle_rejected():
    with pytest.raises(ValueError): chain_geometry(np.zeros((3,3)),[1,2,0])
    with pytest.raises(ValueError): chain_geometry(np.zeros((3,3)),[1,0,0])


def test_finite_tree_branch_stock_is_not_a_cumulative_workspace_cap():
    import json
    from information_boltzmann.cgb.tree_development import ConditionalTreeDevelopment
    from information_boltzmann.cgb.development import SpatialDevelopment
    plan=dict(soma_mean=np.zeros((2,3)),centroid_offset=np.ones((2,3))*3,
        covariance=np.broadcast_to(np.eye(3)*9,(2,3,3)),length=np.full(2,120.),branch_density=np.full(2,.175))
    program=dict(known=np.ones(2,bool),gamma_shape=np.ones(2),primary_length=np.ones(2)*5,
                 root_count=np.ones(2,int),output_share=np.full(2,.5))
    a=ConditionalTreeDevelopment(np.eye(2),np.arange(2),plan,program)
    for _ in range(4): a.step()
    b=SpatialDevelopment.from_state_dict(json.loads(json.dumps(a.state_dict(),default=lambda v:v.tolist())))
    for _ in range(600):
        a.step();b.step()
        if not a.cone_active.any(): break
    assert not a.cone_active.any() and np.all(a.remaining_branches==0)
    assert np.bincount(a.cone_cell).min()==41  # 20 splits -> 41 tree chains.
    for k,v in a.segments().items(): np.testing.assert_array_equal(v,b.segments()[k])
    np.testing.assert_allclose(a.budget.reserved/a.params.material_per_um,[120,120],atol=1e-9)
    np.testing.assert_allclose(a.budget.balance(np.zeros(2)),0,atol=1e-12)
