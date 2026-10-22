"""Constitutive sensory matching from the existing material generator.

No forecast head, new state, target, gradient or rate update is installed.
The neutral instantaneous first moment includes domain recovery; accepted
birth charge has exactly zero mean at a neutral parent, not zero noise.
Matching a possible sensation does not prove prediction of a future one.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .mobile_exchange import MobileExchange
from .identity_material import IdentityMobileExchange
from .boundary_material import BoundaryIdentityMobileExchange
from .growing_material import GrowingIdentityMobileExchange


def _readonly(value,dtype=float):
    value=np.asarray(value,dtype=dtype).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class SensorJump:
    port: int
    delta_g_neutral: float
    attempt: float
    body_quantum: float
    world_quantum: float
    original_world_voltage: float
    kind: str
    sign: int

    def rate(self,world_voltage):
        dg=self.delta_g_neutral+self.world_quantum*(world_voltage-self.original_world_voltage)
        return float(self.attempt*expit(-dg))


@dataclass(frozen=True)
class NativePortConstitution:
    time_ms: float
    port_nodes: np.ndarray
    body_passive_current: np.ndarray
    body_reaction_current: np.ndarray
    body_recovery_current: np.ndarray
    fixed_carrier_admittance: np.ndarray
    phase_admittance_diagonal_bounds: np.ndarray
    sensor_conductance: np.ndarray
    sensor_jumps: tuple[SensorJump,...]
    phase_receipts: tuple[dict,...]

    @property
    def body_current(self):
        return self.body_passive_current+self.body_reaction_current+self.body_recovery_current

    @property
    def matching_sensory_current(self):
        return -self.body_current

    def sensory_current(self,world_voltage):
        voltage=np.asarray(world_voltage,float)
        if voltage.shape!=self.sensor_conductance.shape or not np.isfinite(voltage).all():
            raise ValueError('One finite candidate world-capacitor voltage per port required')
        result=self.sensor_conductance*voltage
        for event in self.sensor_jumps:
            result[event.port]+=event.body_quantum*event.rate(voltage[event.port])
        return result

    def match_sensation(self):
        roots=[];status=[];residual=[]
        target=self.matching_sensory_current
        for port,g in enumerate(self.sensor_conductance):
            jumps=[e for e in self.sensor_jumps if e.port==port]
            def f(voltage):
                return float(g*voltage+sum(e.body_quantum*e.rate(voltage) for e in jumps)-target[port])
            limit=sum(abs(e.body_quantum)*e.attempt for e in jumps)
            if g>0:
                radius=(abs(target[port])+limit)/g+1.
            else:
                lo=sum(min(0.,e.body_quantum)*e.attempt for e in jumps)
                hi=sum(max(0.,e.body_quantum)*e.attempt for e in jumps)
                if not jumps or not limit:
                    roots.append(None);residual.append(None)
                    status.append('non_unique' if target[port]==0 else 'unreachable')
                    continue
                if not lo<target[port]<hi:
                    roots.append(None);residual.append(None);status.append('unreachable_finite_voltage')
                    continue
                # Active sensing alone can identify a finite voltage. A zero
                # passive conductance is not automatically unobservability.
                radius=max(1.,max(abs(e.delta_g_neutral/e.world_quantum-e.original_world_voltage) for e in jumps)+1.)
                while f(-radius)>=0 or f(radius)<=0:
                    radius*=2
                    if not np.isfinite(radius):
                        raise FloatingPointError('Finite active-sensor root was not resolved')
            if not np.isfinite(radius):raise FloatingPointError('Nonfinite derived sensory bracket')
            root=float(brentq(f,-radius,radius,xtol=1e-12,rtol=4*np.finfo(float).eps))
            roots.append(root);residual.append(f(root));status.append('unique_finite')
        return dict(matching_world_voltage=roots,sensor_status=status,current_residual=residual,
            scope='Current native constitution matches this world-capacitor state; not a future-byte likelihood or a finite-time forecast theorem')

    def state_dict(self):
        return dict(time_ms=self.time_ms,port_nodes=self.port_nodes.tolist(),
            body_passive_current=self.body_passive_current.tolist(),body_reaction_current=self.body_reaction_current.tolist(),
            body_recovery_current=self.body_recovery_current.tolist(),body_current=self.body_current.tolist(),
            matching_sensory_current=self.matching_sensory_current.tolist(),
            fixed_carrier_admittance=self.fixed_carrier_admittance.tolist(),
            phase_admittance_diagonal_bounds=self.phase_admittance_diagonal_bounds.tolist(),
            sensor_conductance=self.sensor_conductance.tolist(),
            sensor_jumps=[vars(e) for e in self.sensor_jumps],phase_receipts=list(self.phase_receipts),
            **self.match_sensation(),
            role='Read-only instantaneous native-generator interface; zero new physical state or learning rule')


def native_port_constitution(model):
    """Evaluate the *current* quadratic-storage material law at neutral B.

    Known concrete laws are admitted explicitly so adding another charge
    energy/coupler cannot silently reuse the separable-capacitor derivation.
    This finite O(N+E) instrument does not advance clocks or draw RNGs.
    """
    if type(model) not in (MobileExchange,IdentityMobileExchange,BoundaryIdentityMobileExchange,GrowingIdentityMobileExchange):
        raise TypeError('This constitutive charge law must be explicitly verified')
    ports=model.port_nodes
    if not len(ports) or len(set(ports.tolist()))!=len(ports):
        raise ValueError('Distinct material ports are required for this one-to-one sensory inversion')
    model._validate()
    count=len(ports);lookup={int(node):p for p,node in enumerate(ports)}
    actual=np.r_[model.charge/model.capacitance,model.world_charge/model.world_capacitance]
    neutral=actual.copy();neutral[ports]=0.
    current=np.zeros(model.n);admittance=np.zeros((count,count))
    g=model.bridge_conductance()
    for edge,(i,j) in enumerate(zip(model.pre,model.post)):
        flow=g[edge]*(neutral[i]-neutral[j]);current[i]-=flow;current[j]+=flow
        b=np.zeros(count)
        if int(i) in lookup:b[lookup[int(i)]]+=1.
        if int(j) in lookup:b[lookup[int(j)]]-=1.
        admittance+=g[edge]*np.outer(b,b)
    passive=current[ports]
    reaction=np.zeros(count);recovery=np.zeros(count);phase_bounds=np.zeros((count,2));receipts=[];sensors=[]
    degree=model.degrees()
    for event in model.events(degree=degree):
        changes=model._charge_changes(event)
        if not changes:continue
        nu=np.zeros(count)
        for node,dq in changes:
            if node in lookup:nu[lookup[node]]+=dq
        if not np.any(nu):continue
        dg=model.event_delta(event,degree)[1]+sum((neutral[node]-actual[node])*dq for node,dq in changes)
        attempt=model.attempt_rate(event,degree)
        if event.interface>=0:
            p=event.interface
            world=sum(dq for node,dq in changes if node==model.n+p)
            if (np.count_nonzero(nu)!=1 or nu[p]==0 or world!=-nu[p]
                    or any(node>=model.n and node!=model.n+p for node,dq in changes)):
                raise ValueError('The actual sensory role mapping differs from the derived one-to-one interface')
            sensors.append(SensorJump(p,float(dg),float(attempt),float(nu[p]),float(world),
                                      float(actual[model.n+p]),event.kind,event.sign))
        else:
            s=float(expit(-dg));reaction+=nu*attempt*s
            admittance+=attempt*s*float(expit(dg))*np.outer(nu,nu)
    if getattr(model,'domain_reactions_enabled',False):
        for parent in model.birth_parents(degree):
            if int(parent) not in lookup:continue
            p=lookup[int(parent)]
            attempt=model._domain_attempt_rate(degree[parent])
            phase_bounds[p,1]+=.5*model.capacitance[parent]*attempt
            receipts.append(dict(kind='birth',parent=int(parent),mean_neutral_port_charge_increment=0.,
                accepted_rate_upper_bound=float(attempt),birth_charge_noise_zero=False))
        for parent,child,edge in model.death_channels(degree):
            if child in lookup:raise ValueError('An actual sensory port may not be removed by this phase law')
            if parent not in lookup:continue
            p=lookup[parent];q=model.charge[child]
            ratio=model.death_log_ratio(parent,child,degree)+(q*q-(model.charge[parent]+q)**2)/(4*model.capacitance[parent])
            attempt=model._domain_attempt_rate(degree[parent]-1)
            rate=float(attempt*math.exp(min(0.,-ratio)))
            recovery[p]+=rate*q
            local=.5*rate*q*q
            if ratio>0:phase_bounds[p]+=local
            elif ratio==0:phase_bounds[p,1]+=local
            receipts.append(dict(kind='recovery',parent=parent,child=child,edge=edge,
                neutral_birth_log_ratio=float(ratio),accepted_rate=rate,port_charge_increment=float(q)))
    if not all(np.isfinite(v).all() for v in (passive,reaction,recovery,admittance,phase_bounds)):
        raise FloatingPointError('Nonfinite native constitutive current')
    return NativePortConstitution(model.t,_readonly(ports,np.int64),_readonly(passive),_readonly(reaction),
        _readonly(recovery),_readonly(admittance),_readonly(phase_bounds),
        _readonly(model.effective_port_conductance()),tuple(sensors),tuple(receipts))
