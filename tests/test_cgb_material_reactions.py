"""Stoichiometry, implicit solution and dispersion algebra; no capability run."""
import numpy as np
import pytest

from information_boltzmann.cgb.material_reactions import (
    material_slot_step, redistribution_eigenvalues,
    subthreshold_slot_gain,
)


def test_stationary_structure_still_pays_gross_turnover_loss():
    # At B=1, g=.5 both reaction fluxes=.5; half the dismantled mass is lost.
    step=material_slot_step([1.],[.5],np.array([0]),1.,1.,1.,.2,
                            recycle=.5,supply=.25)
    np.testing.assert_allclose(step.structure,[.5],atol=1e-14)
    np.testing.assert_allclose(step.free,[1.],atol=1e-14)
    np.testing.assert_allclose(step.formed,[.1],atol=1e-14)
    np.testing.assert_allclose(step.dismantled,[.1],atol=1e-14)
    np.testing.assert_allclose(step.discarded,[.05],atol=1e-14)
    np.testing.assert_allclose(step.imported,[.05],atol=1e-14)
    np.testing.assert_allclose(step.balance_error,0,atol=1e-14)


def test_closed_pools_stay_positive_and_conserve_local_material_with_competition():
    b=np.array([.1,.7]);g=np.array([.1,.3,.05]);post=np.array([0,0,1])
    initial=b+2*np.bincount(post,weights=g,minlength=2)
    for _ in range(40):
        s=material_slot_step(b,g,post,1.,[2.,.4,8.],[.2,.3,.5],1.,cost=2)
        assert np.all(s.free>=0) and np.all(s.structure>=0) and np.all(s.structure<=1)
        np.testing.assert_allclose(s.balance_error,0,atol=2e-14)
        np.testing.assert_allclose(s.free+2*np.bincount(post,weights=s.structure,minlength=2),initial,atol=2e-13)
        b,g=s.free,s.structure


def test_edge_order_and_separate_donors_do_not_change_reaction():
    inputs=([.2,.4],[.1,.3,.2],np.array([0,1,0]),[1.,2.,3.],[2.,3.,4.],[.1,.2,.3])
    original=material_slot_step(*inputs,.3,recycle=.4,supply=[.02,.01],clearance=.1)
    order=np.array([2,0,1]);b,g,p,cap,on,off=map(np.asarray,inputs)
    shuffled=material_slot_step(b,g[order],p[order],cap[order],on[order],off[order],.3,
                               recycle=.4,supply=[.02,.01],clearance=.1)
    np.testing.assert_allclose(original.free,shuffled.free,atol=1e-14)
    np.testing.assert_allclose(original.structure[order],shuffled.structure,atol=1e-14)
    for donor in (0,1):
        mask=p==donor
        single=material_slot_step([b[donor]],g[mask],np.zeros(mask.sum(),int),cap[mask],on[mask],off[mask],.3,
                                  recycle=.4,supply=[.02,.01][donor],clearance=.1)
        np.testing.assert_allclose(original.structure[mask],single.structure,atol=1e-14)


def test_implicit_solution_converges_to_closed_association_analytic_solution():
    exact=.5*(1-np.exp(-.5))/(1-.5*np.exp(-.5))
    errors=[]
    for dt in (.25,.125,.0625):
        b,g=np.array([.5]),np.array([0.])
        for _ in range(round(1/dt)):
            step=material_slot_step(b,g,np.array([0]),1.,1.,0.,dt)
            b,g=step.free,step.structure
        errors.append(abs(g[0]-exact))
    assert errors[0]/errors[1]>1.7 and errors[1]/errors[2]>1.8


def test_zero_stock_cannot_synthesize_and_empty_compartment_follows_clearance():
    step=material_slot_step([0.,1.],[0.],np.array([0]),1.,20.,0.,2.,clearance=.5)
    np.testing.assert_allclose(step.structure,0,atol=1e-14)
    np.testing.assert_allclose(step.free,[0.,.5],atol=1e-14)


def test_large_rates_keep_slot_and_material_bounds():
    step=material_slot_step([.01],[.5],np.array([0]),1.,1e5,1e4,10.,recycle=.3)
    assert 0<=step.structure[0]<=1 and step.free[0]>=0
    np.testing.assert_allclose(step.balance_error,0,atol=1e-12)


@pytest.mark.parametrize('dt,cost,recycle',[(0,1,1),(1,0,1),(1,1,1.1)])
def test_invalid_reaction_parameters_rejected(dt,cost,recycle):
    with pytest.raises(ValueError):
        material_slot_step([1.],[.5],np.array([0]),1.,1.,1.,dt,cost=cost,recycle=recycle)


def test_conserved_linear_modes_and_derived_instability_band():
    # f_a=.5, f_b=1: local fixed-mass relaxation has eigenvalue -.5.
    # D_a=.01, D_b=1 gives 0<q<49 as the unstable redistribution band.
    q=np.array([0.,.01,1.,48.,50.,100.])
    spectrum=redistribution_eigenvalues(.5,1.,.01,1.,q)
    np.testing.assert_allclose(np.sort(spectrum[0]),[-.5,0.],atol=1e-14)
    assert np.all(spectrum[1:4].real.max(axis=1)>0)
    assert np.all(spectrum[4:].real.max(axis=1)<0)
    equal=redistribution_eigenvalues(.5,1.,1.,1.,q[1:])
    assert np.all(equal.real.max(axis=1)<0)


@pytest.mark.parametrize('nmda,reversal',[(.1,5.),(.4,5.),(.1,-.3)])
def test_inherited_membrane_calcium_tag_gain_matches_finite_differences(nmda,reversal):
    from information_boltzmann.cgb.medium import CellClass,SynapseParams
    kw=dict(capacity=1.,unit_conductance=.1,occupancy=.5,nmda_occupancy=nmda,
            cell=CellClass(),syn=SynapseParams(aK=.1),reversal=reversal,
            modulator=.1,turnover=.001)
    n,b,h=.1,.5,1e-6
    center=subthreshold_slot_gain(n,b,**kw)
    plus=subthreshold_slot_gain(n+h,b,**kw)
    minus=subthreshold_slot_gain(n-h,b,**kw)
    numerical=(plus['net_structure_rate']-minus['net_structure_rate'])/(2*h)
    np.testing.assert_allclose(center['f_structure'],numerical,rtol=2e-6,atol=1e-10)
    db=(subthreshold_slot_gain(n,b+h,**kw)['net_structure_rate']-
        subthreshold_slot_gain(n,b-h,**kw)['net_structure_rate'])/(2*h)
    np.testing.assert_allclose(center['f_mobile'],db,rtol=1e-7,atol=1e-10)
    assert np.sign(center['membrane_structure_slope'])==np.sign(reversal-center['membrane'])


def test_subthreshold_reduction_refuses_spikes_and_unheld_sliding_threshold():
    from information_boltzmann.cgb.medium import CellClass,SynapseParams
    kw=dict(capacity=1.,unit_conductance=.1,occupancy=.5,nmda_occupancy=.1,
            cell=CellClass(),syn=SynapseParams(),reversal=5.)
    with pytest.raises(ValueError,match='threshold crossings'):
        subthreshold_slot_gain(.9,.5,**kw)
    kw['syn']=SynapseParams(slide_A0=5)
    with pytest.raises(ValueError,match='sliding threshold'):
        subthreshold_slot_gain(.1,.5,**kw)
