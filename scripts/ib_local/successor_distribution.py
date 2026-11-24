"""The shared DECISION distribution, carried through the native finite loop.

Each output coordinate has the same Student-t marginal used by log_score.
Fixed antithetic quantiles define a bounded product-marginal cubature; its
budget is the existing proposal budget. Every action sees the same standardized
directions. No world label or live random stream chooses a quadrature point.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _student_directions(df, count, width):
    import numpy as np
    from scipy.special import ndtr, stdtrit
    from scripts.ib_local.identity_preview import _cpu_direction
    half=count//2
    normal=_cpu_direction('decision:successor-distribution:v1',count,width)[:half].numpy()
    probability=np.clip(ndtr(normal),np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    positive=torch.from_numpy(stdtrit(df,probability).copy())
    return torch.cat((positive,-positive),0)


def quadrature(distribution, budget):
    """B x K x latent points plus K weights, independent of presentation order."""
    mean,scale=distribution['mean'],distribution['scale2']
    if mean.ndim!=2 or scale.shape!=mean.shape or not bool(torch.isfinite(mean).all()
            and torch.isfinite(scale).all() and (scale>=0).all()):
        raise ValueError('finite matched successor location and nonnegative scale required')
    df=float(torch.as_tensor(distribution['df']).detach())
    if not df>0:
        raise ValueError('positive predictive degrees of freedom required')
    if not bool((scale>0).any()):
        return mean[:,None],mean.new_ones(1)
    count=2*(int(budget)//2)
    if count<2:
        raise ValueError('successor uncertainty requires a paired proposal budget')
    directions=_student_directions(df,count,mean.shape[-1]).to(mean)
    points=mean[:,None]+scale.sqrt()[:,None]*directions[None]
    return points,mean.new_full((count,),1/count)


def training_distribution(model, state, mediated, direct):
    """The SAME same-output reconciliation used by execute_programs()."""
    from scripts.ib_local.joint_relation_update import overlap_consensus
    if getattr(model,'joint_relation_update_schema',None)!=1 or state.program_weights is None:
        raise ValueError('successor learning requires the native coordinated program')
    if getattr(getattr(model,'_epistemic_value_engine',None),'epistemic_value_schema',0)>=5:
        from scripts.ib_local.joint_successor import coordinate
        return coordinate((mediated,direct),state.program_weights.to(mediated['mean']))
    return overlap_consensus((mediated,direct),state.program_weights.to(mediated['mean']))


def learning_state(state):
    """Freeze only the actual finite post-U facts; no historical body copy."""
    from dataclasses import replace
    from scripts.ib_local.causal_use_transport import clone_fact
    return replace(state,constraints=tuple(clone_fact(f) for f in state.constraints),
        extra=tuple(clone_fact(f) for f in state.extra),organization=None,
        program_weights=None if state.program_weights is None else state.program_weights.detach().clone())


@torch.no_grad()
def value(engine, state, source, action, mother, distribution, momentum):
    if getattr(engine, 'native_amortization', {}).get('schema') == 1:
        from scripts.ib_local.native_amortized_decision import value as amortized_value
        return amortized_value(engine, state, source, action, mother, distribution, momentum)
    if getattr(engine, 'successor_quadrature', {}).get('mode') == 'residual':
        from scripts.ib_local.native_residual_quadrature import value as residual_value
        return residual_value(engine, state, source, action, mother, distribution, momentum)
    from scripts.ib_local.prospective_decision import value as native_value
    geometry_trace=None
    if engine.epistemic_value_schema>=5:
        from scripts.ib_local.joint_successor import quadrature as joint_quadrature
        if len(source)!=1:raise ValueError('native joint successor requires one original typed branch')
        points,weights,geometry_trace=joint_quadrature(engine,state,source,action,mother,distribution)
    else:
        points,weights=quadrature(distribution,engine.joint_proposal_budget)
    b,k,d=points.shape
    def expand(tensor):
        return tensor[:,None].expand(b,k,*tensor.shape[1:]).reshape(b*k,*tensor.shape[1:])
    values=native_value(engine,state,expand(source),expand(action),expand(mother),
        points.reshape(b*k,d),expand(momentum)).reshape(b,k)
    # Centering makes a constant downstream value exact for every branch mass.
    reference=values[:,:1]
    expected=reference[:,0]+((values-reference)*weights).sum(-1)
    engine.successor_distribution_last=dict(schema=1,budget=int(engine.joint_proposal_budget),
        branches=k,weights=weights.tolist(),values=values.tolist(),expected=expected.tolist(),
        df=float(torch.as_tensor(distribution['df']).detach()),
        scale_mean=float(distribution['scale2'].mean()),
        semantics='fixed antithetic Student-t product-marginal cubature before native addressing')
    if geometry_trace is not None:
        engine.successor_distribution_last.update(joint=geometry_trace,
            semantics='coherent Student-t, original address-cell mass, bounded native executions')
    return expected
