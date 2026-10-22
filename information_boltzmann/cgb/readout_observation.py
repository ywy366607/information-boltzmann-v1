"""Frozen external likelihood instruments; never update the material.

These fit observation conventions to historical pre-arrival measurements.
They are not a predictive circuit or a new learning rule of the organism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def bridge_pool_port_current(model):
    """Conditional OU+bridge/pool current; not the domain-phase generator.

    It exposes the current constitutive state to an observer. Birth marks
    are not silently approximated as zero full-generator drift. No external
    future source voltage is read, and no derived current feeds the body.
    """
    voltage=model.charge/model.capacitance
    current=np.zeros(model.n)
    flow=model.bridge_conductance()*(voltage[model.pre]-voltage[model.post])
    np.add.at(current,model.pre,-flow);np.add.at(current,model.post,flow)
    np.add.at(current,model.port_nodes,model.effective_port_conductance()*(
        model.world_charge/model.world_capacitance-voltage[model.port_nodes]))
    degree=model.degrees()
    for event in model.events(degree=degree):
        changes=model._charge_changes(event)
        if not changes:continue
        rate=model.event_rate(event,degree)
        for node,delta in changes:
            if node<model.n:current[node]+=rate*delta
    return current[model.port_nodes]


def before_arrival_features(model, session, last_actual_byte):
    if len(model.port_nodes)!=8 or not isinstance(last_actual_byte,(int,np.integer)) or not 0<=last_actual_byte<256:
        raise ValueError('Eight physical byte ports and a real previous byte are required')
    brain=model.charge[model.port_nodes]/model.capacitance[model.port_nodes]
    source=np.asarray(session['source_only_voltage'],float)
    previous=2*((int(last_actual_byte)>>np.arange(8))&1)-1
    if source.shape!=(8,) or not np.isfinite(source).all():
        raise ValueError('A finite causal source-only reference is required')
    return dict(brain=brain.copy(),source=source.copy(),previous=previous.astype(float),
                brain_source=np.r_[brain,source],source_previous=np.r_[source,previous],
                brain_current=np.r_[brain,bridge_pool_port_current(model)])


def objective(theta,x,bits,prior,ridge,diagonal=False):
    dimensions=x.shape[1]
    weights=theta[:dimensions if diagonal else 8*dimensions]
    bias=theta[len(weights):]
    if diagonal:
        logits=x*weights+prior+bias
    else:
        weights=weights.reshape(dimensions,8)
        logits=x@weights+prior+bias
    error=expit(logits)-bits
    loss=float(np.mean(np.sum(np.logaddexp(0.,logits)-bits*logits,axis=1))+.5*ridge*np.dot(theta,theta))
    dw=np.mean(error*x,axis=0) if diagonal else (x.T@error/len(x)).ravel()
    gradient=np.r_[dw,error.mean(axis=0)]+ridge*theta
    return loss,gradient


@dataclass
class FrozenBitReadout:
    mean: np.ndarray
    scale: np.ndarray
    prior: np.ndarray
    theta: np.ndarray
    diagonal: bool
    fit_info: dict

    def logits(self,features):
        value=np.asarray(features,float)
        if value.shape!=(len(self.mean),) or not np.isfinite(value).all():
            raise ValueError('One finite pre-arrival feature vector required')
        x=(value-self.mean)/self.scale
        d=len(x)
        if self.diagonal:
            return x*self.theta[:d]+self.prior+self.theta[d:]
        return x@self.theta[:8*d].reshape(d,8)+self.prior+self.theta[8*d:]

    def state_dict(self):
        return dict(schema=1,mean=self.mean.tolist(),scale=self.scale.tolist(),prior=self.prior.tolist(),
                    theta=self.theta.tolist(),diagonal=self.diagonal,fit_info=self.fit_info,
                    role='external frozen likelihood instrument; no feedback to material')

    @classmethod
    def from_state_dict(cls,state):
        if state.get('schema')!=1 or not isinstance(state.get('diagonal'),bool):
            raise ValueError('Explicit supported observation schema required')
        arrays={key:np.asarray(state[key],float) for key in ('mean','scale','prior','theta')}
        d=len(arrays['mean'])
        if (not d or arrays['mean'].shape!=(d,) or arrays['scale'].shape!=(d,)
                or arrays['prior'].shape!=(8,) or np.any(arrays['scale']<=0)
                or any(not np.isfinite(x).all() for x in arrays.values())
                or arrays['theta'].shape!=((d if state['diagonal'] else 8*d)+8,)
                or state['diagonal'] and d!=8):
            raise ValueError('Finite complete observation arrays required')
        return cls(**arrays,diagonal=state['diagonal'],fit_info=state['fit_info'])


def fit_readout(features,targets,*,diagonal=False,ridge=.01):
    x=np.asarray(features,float);targets=np.asarray(targets)
    if (x.ndim!=2 or not len(x) or targets.shape!=(len(x),) or targets.dtype.kind not in 'iu'
            or np.any((targets<0)|(targets>255)) or not np.isfinite(x).all()
            or diagonal and x.shape[1]!=8 or not np.isfinite(ridge) or ridge<=0):
        raise ValueError('Real pre-arrival rows/bytes and a positive declared ridge required')
    mean=x.mean(axis=0);scale=x.std(axis=0)
    constant=np.all(x==x[0],axis=0)
    mean[constant]=x[0,constant]
    scale[constant]=1.  # Exact constant columns are zero; no noise is added.
    x=(x-mean)/scale
    bits=((targets[:,None]>>np.arange(8))&1).astype(float)
    probability=(bits.sum(axis=0)+.5)/(len(bits)+1.)
    prior=np.log(probability)-np.log1p(-probability)
    size=(x.shape[1] if diagonal else 8*x.shape[1])+8
    initial=np.zeros(size)
    result=minimize(objective,initial,args=(x,bits,prior,ridge,diagonal),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=2000,ftol=1e-13,gtol=1e-8,maxls=40))
    loss,grad=objective(result.x,x,bits,prior,ridge,diagonal)
    maximum=float(np.abs(grad).max())
    if not result.success or maximum>1e-6:
        raise RuntimeError('Observation fit is not converged: '+str(result.message)+', gradient '+str(maximum))
    info=dict(samples=len(x),features=x.shape[1],parameters=size,ridge=ridge,iterations=int(result.nit),
        final_objective=loss,gradient_inf=maximum,converged=True,constant_features=np.flatnonzero(constant).tolist(),
        source='training rows only; centering/scaling/prior and coefficients frozen before validation',
        trace=[dict(stage='initial',objective=objective(initial,x,bits,prior,ridge,diagonal)[0]),
               dict(stage='final',objective=loss)])
    return FrozenBitReadout(mean,scale,prior,result.x,diagonal,info)
