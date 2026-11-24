"""Numerical observation contracts, not synthetic capability experiments."""
import numpy as np
import torch

from scripts.native_topology_observer import matrix_observation
from scripts.ib_local.adaptive_hamiltonian import potential


def values():
    rng=np.random.default_rng(161110)
    s=rng.normal(size=(2,8,4,4))*.1
    p=rng.normal(size=s.shape)*.1
    theta=rng.normal(size=(2,12))*.1
    context=np.zeros((2,1,8,1,1));scale=np.ones_like(context)
    impedance=np.ones_like(context)*.8
    return s,p,theta,context,scale,impedance


def test_observer_edges_and_energy_match_actual_potential():
    s,p,t,c,w,z=values()
    result,_=matrix_observation(s,p,t,c,w,z)
    actual=potential(torch.tensor(s)[:,None],torch.tensor(t),8)
    np.testing.assert_allclose(result['edge_potential'].sum(1),actual[:,0],rtol=1e-12)
    assert result['edge_flux'].shape==(2,28)
    assert result['complex_overlap_real'].shape==(2,8,8)
    assert np.isnan(result['phase_delta']).all()


def test_phase_coordinate_follows_exact_native_uncoupled_rotation():
    s,p,t,c,w,z=values();t[:,9]=-1000
    result,a=matrix_observation(s,p,t,c,w,z)
    dt=.01;angle=result['omega'][...,None,None]*dt
    zz=z.reshape(2,8,1,1)
    ns=np.cos(angle)*s+np.sin(angle)*p/zz
    np_=np.cos(angle)*p-np.sin(angle)*zz*s
    after,b=matrix_observation(ns,np_,t,c,w,z,previous=a)
    np.testing.assert_allclose(b,np.exp(-1j*angle)*a,atol=2e-16)
    np.testing.assert_allclose(after['phase_delta'],-angle[...,0,0],atol=2e-15)
    np.testing.assert_allclose(after['temporal_overlap'],1.,atol=3e-16)


def test_graph_harmonic_has_zero_incidence_and_no_fabricated_faces():
    from scripts.audit_higher_order_dynamics import incidence
    s,p,t,c,w,z=values();r,_=matrix_observation(s,p,t,c,w,z)
    i,j=np.triu_indices(8,1);b=incidence(8,i,j)
    np.testing.assert_allclose(r['edge_graph_harmonic']@b.T,0.,atol=1e-15)
    assert not any('face' in name for name in r)


def test_all_32_layers_are_observed_without_random_draws():
    s,p,t,c,w,z=values()
    args=[np.tile(a,(16,)+(1,)*(a.ndim-1)) for a in (s,p,t,c,w,z)]
    old=np.random.get_state()
    r,_=matrix_observation(*args)
    new=np.random.get_state()
    assert r['node_energy'].shape==(32,8)
    assert all(np.array_equal(a,b) for a,b in zip(old,new))


def test_admission_hash_checks_actual_module_tensors_and_configuration():
    from scripts.run_native_topology import canonical_carried_value
    from scripts.audit_spectrum_runtime import digest
    from scripts.ib_local.language_navigation_kanli_v4p import ActionEvidence
    a,b=ActionEvidence(.95),ActionEvidence(.95)
    assert digest(canonical_carried_value(a))==digest(canonical_carried_value(b))
    b.seen[0]+=1
    assert digest(canonical_carried_value(a))!=digest(canonical_carried_value(b))
    b.seen[0]-=1;b.decay=.94
    assert digest(canonical_carried_value(a))!=digest(canonical_carried_value(b))


def test_lossless_tensor_store_preserves_broadcasts_and_deduplicates(tmp_path):
    from scripts.native_topology_observer import TensorStore,load_tensor
    value=torch.arange(128,dtype=torch.float32).reshape(1,128)
    path=tmp_path/'tensors.bin';store=TensorStore(path)
    first=store.put(value.expand(8,-1));second=store.put(value.expand(3,-1));store.close()
    assert first['tensor_ref']['offset']==second['tensor_ref']['offset']
    np.testing.assert_array_equal(load_tensor(path,first),value.expand(8,-1).numpy())
    np.testing.assert_array_equal(load_tensor(path,second),value.expand(3,-1).numpy())
