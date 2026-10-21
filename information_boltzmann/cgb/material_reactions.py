"""E6 finite-slot chemistry coupled to finite material by stoichiometry.

This derived subsystem inherits E5/E6 synthesis/disassembly catalysts and E9
turnover. It does not prescribe a growth direction, branch count or behavior.
Rates are held over one implicit Euler step; accuracy is first order, not exact
continuous-time integration. Shared pools assume a well-mixed compartment.

Per edge: formation = k_on * B_post * (capacity-g), breakdown = k_off*g.
Per donor: B' = supply-clearance*B-cost*sum(formation-recycle*breakdown).
Unrecovered breakdown enters an explicit sink. Gross fluxes are retained even
when formation and breakdown leave the net structure unchanged.
Structure counts identical occupied units. Their mapping to electrical
conductance is separate and is not derived by this material subsystem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class ReactionStep:
    free: np.ndarray
    structure: np.ndarray
    formed: np.ndarray
    dismantled: np.ndarray
    imported: np.ndarray
    cleared: np.ndarray
    discarded: np.ndarray
    balance_error: np.ndarray


def material_slot_step(free, structure, post, capacity, on_rate, off_rate, dt,
                       *, cost=1., recycle=1., supply=0., clearance=0.):
    """Unique nonnegative implicit solution, using bounded scalar bisections.

    All compartments solve simultaneously; no edge-order priority or clipping
    of physical pools. 64 bisections are a numerical root solver, not a physical
    rate or history horizon. State/workspace O(donors+edges), independent of age.
    No coupling to existing saved individuals occurs in this standalone API.
    """
    b0=np.asarray(free,dtype=float)
    g0=np.asarray(structure,dtype=float)
    post=np.asarray(post)
    if b0.ndim!=1 or g0.ndim!=1 or post.shape!=g0.shape:
        raise ValueError('One scalar pool per donor and donor index per edge required')
    if not np.issubdtype(post.dtype,np.integer):
        raise ValueError('Donor indices must be integers')
    post=post.astype(np.int64,copy=False)
    if np.any(post<0) or np.any(post>=len(b0)):
        raise ValueError('Unknown material donor')
    cap,on,off=(np.broadcast_to(np.asarray(v,float),g0.shape) for v in (capacity,on_rate,off_rate))
    feed,clear=(np.broadcast_to(np.asarray(v,float),b0.shape) for v in (supply,clearance))
    arrays=(b0,g0,cap,on,off,feed,clear)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Finite pools, capacities and rates required')
    if any(np.any(a<0) for a in arrays) or np.any(g0>cap):
        raise ValueError('Nonnegative pools/rates and occupied slots within capacity required')
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(cost) or cost<=0:
        raise ValueError('Positive finite elapsed time and material cost required')
    if not np.isfinite(recycle) or not 0<=recycle<=1:
        raise ValueError('Recycling fraction must be in [0,1]')
    n=len(b0)
    def total(values):
        return np.bincount(post,weights=values,minlength=n)
    incoming=feed*dt
    old_mass=cost*total(g0)
    low=np.zeros_like(b0)
    high=b0+incoming+old_mass
    coefficient=1+(1-recycle)*dt*off
    def occupancy(b):
        k=dt*on*b[post]
        return (g0+k*cap)/(1+dt*off+k)
    # F(B)=(1+clear*dt)B+cost*sum(c*g(B)-g0)-B0-feed*dt.
    # F(0)<=0, F(high)>=0, and F' >= 1: one root per donor.
    for _ in range(64):
        mid=(low+high)*.5
        value=(1+clear*dt)*mid+cost*total(coefficient*occupancy(mid)-g0)-b0-incoming
        below=value<0
        low=np.where(below,mid,low)
        high=np.where(below,high,mid)
    b=(low+high)*.5
    g=occupancy(b)
    formed=dt*on*b[post]*(cap-g)
    dismantled=dt*off*g
    cleared=dt*clear*b
    discarded=(1-recycle)*cost*total(dismantled)
    error=b+cost*total(g)+cleared+discarded-b0-old_mass-incoming
    return ReactionStep(b,g,formed,dismantled,incoming,cleared,discarded,error)


def redistribution_eigenvalues(f_active, f_mobile, d_active, d_mobile, q):
    """Linear modes of a conserved active/mobile pool, q=spatial wavenumber².

    a_t=D_a Laplacian(a)+f(a,b); b_t=D_b Laplacian(b)-f(a,b).
    This diagnoses mass redistribution, NOT completed branching morphogenesis.
    Inputs are a local kinetic Jacobian at a stated homogeneous equilibrium.
    """
    if min(d_active,d_mobile)<=0:
        raise ValueError('Positive diffusivities required')
    q=np.asarray(q,dtype=float)
    if not np.isfinite(q).all() or np.any(q<0):
        raise ValueError('Finite nonnegative squared wavenumbers required')
    matrix=np.empty(q.shape+(2,2))
    matrix[...,0,0]=f_active-d_active*q
    matrix[...,0,1]=f_mobile
    matrix[...,1,0]=-f_active
    matrix[...,1,1]=-f_mobile-d_mobile*q
    return np.linalg.eigvals(matrix)


def subthreshold_slot_gain(n, b, *, capacity, unit_conductance,
                          occupancy, nmda_occupancy, cell, syn,
                          reversal, external_current=0., modulator=None,
                          reference_substrate=1., turnover=0.):
    """Derive local slot-rate derivatives from inherited E1/E4/E5/E6.

    This is a subthreshold, fixed-occupancy, well-mixed quasi-steady limit.
    No spike/backprop or spatially coupled membrane state is eliminated here.
    Unit conductance is explicit; it is NOT inferred from a material amount.
    Returned slopes can enter a stated reduced reaction model, not substitute
    for the full spatial/spiking Jacobian outside this limit.
    """
    m=syn.m0 if modulator is None else modulator
    if not 0<=n<=capacity or b<0 or capacity<=0 or cell.gL<=0:
        raise ValueError('Positive capacity/leak and feasible pools required')
    if min(unit_conductance,occupancy,nmda_occupancy,m,turnover)<0 or reference_substrate<=0:
        raise ValueError('Nonnegative rates/occupancies and positive reference stock required')
    if syn.slide_A0>0:
        raise ValueError('Hold the actual sliding threshold fixed explicitly before this reduction')
    conductance=unit_conductance*n*occupancy
    denominator=cell.gL+conductance
    u=(cell.gL*cell.EL+conductance*reversal+external_current)/denominator
    if u>=cell.theta:
        raise ValueError('This limit excludes threshold crossings; retain the spiking system')
    u_n=unit_conductance*occupancy*(reversal-u)/denominator
    unblock=float(expit((u-syn.u_half)/syn.u_slope))
    calcium=syn.tau_c*syn.kN*nmda_occupancy*unblock
    calcium_n=syn.tau_c*syn.kN*nmda_occupancy*unblock*(1-unblock)/syn.u_slope*u_n
    def enzyme(on,half,power,tau):
        numerator=calcium**power
        h=numerator/(numerator+half**power)
        hp=power*calcium**(power-1)*half**power/(numerator+half**power)**2
        value=on*tau*h/(1+on*tau*h)
        derivative=on*tau*hp/(1+on*tau*h)**2
        return value,derivative
    kinase,k_c=enzyme(syn.aK,syn.thK,syn.nK,syn.tauK)
    phosphatase,p_c=enzyme(syn.aP,syn.thP,syn.nP,syn.tauP)
    tag=(kinase-phosphatase)/(m+1/syn.tau_z)
    tag_n=(k_c-p_c)*calcium_n/(m+1/syn.tau_z)
    if abs(tag)<1e-12:
        raise ValueError('At a zero tag use one-sided or full dynamics, not this differentiable branch')
    on=syn.eta*m*max(tag,0)/(reference_substrate*capacity)
    off=syn.eta*m*max(-tag,0)/capacity+turnover
    on_n=syn.eta*m*tag_n/(reference_substrate*capacity) if tag>0 else 0.
    off_n=-syn.eta*m*tag_n/capacity if tag<0 else 0.
    f=on*b*(capacity-n)-off*n
    f_n=on_n*b*(capacity-n)-on*b-off_n*n-off
    f_b=on*(capacity-n)
    return {'membrane':u,'calcium':calcium,'kinase':kinase,'phosphatase':phosphatase,
            'tag':tag,'membrane_structure_slope':u_n,'calcium_structure_slope':calcium_n,
            'tag_structure_slope':tag_n,'formation_rate':on,'breakdown_rate':off,
            'net_structure_rate':f,'f_structure':f_n,'f_mobile':f_b}
