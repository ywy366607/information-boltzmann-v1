"""Learn head interaction through the SAME first real-response objective.

Planning evolves a detached live boundary once. Its first language training
forward re-executes that exact local step with derivatives, including actual
self-hearing. No historical graph, fabricated target or extra CE is added.
"""
from __future__ import annotations

import torch

from scripts.ib_local.bfss_gdn_capture import matrix_view
from scripts.ib_local.navigation_bfss_dynamics import MomentumBody
from scripts.ib_local.navigation_causal_program import CausalProgramEngine
from scripts.ib_local.learned_head_coupling import LearnedHeadCoupling


class LearnedCouplingBody(MomentumBody):
    def objective(self,*args,**kwargs):
        receipt=getattr(self,'_learned_coupling_prefix',None)
        if receipt is not None:
            self._learned_coupling_prefix=None
            before,momenta,settings,expected=receipt
            state,_=self.head_coupling.quiet(self.body.base,before,momenta,
                                          **settings,create_graph=True)
            if not torch.allclose(state.detach(),args[2],atol=2e-6,rtol=2e-5):
                raise RuntimeError('learned internal step differs between planning and actual replay')
            self._coupling_replay_error=float((state.detach()-expected).abs().max())
            args=(*args[:2],state,*args[3:])
        return super().objective(*args,**kwargs)


class LearnedCouplingEngine(CausalProgramEngine):
    def step(self,directions,age_ablate=False):
        settings=dict(dt=self.bfss_dt,frequency=self.bfss_frequency,steps=self.bfss_steps)
        before=self.recurrent.detach().clone()
        ps=[p.detach().clone() for p in self.bfss_momenta]
        weights=[p.detach().clone() for p in self.model.head_coupling.parameters()]
        self.model._head_phase_gradients=[]
        with torch.no_grad():
            self.recurrent,self.bfss_momenta=self.model.head_coupling.quiet(
                self.model.body.base,before,ps,**settings)
        quiet_distance=float((self.recurrent-before).norm())
        self.model._learned_coupling_prefix=(before,ps,settings,self.recurrent.detach())
        enabled=self.bfss_enabled
        self.bfss_enabled=False  # Parent still performs all real S/P writes.
        try:
            row=super().step(directions,age_ablate)
        finally:
            self.bfss_enabled=enabled
            self.model._learned_coupling_prefix=None
        changes=[float((p.detach()-old).norm()) for p,old in
                 zip(self.model.head_coupling.parameters(),weights)]
        phases=self.model._head_phase_gradients
        per_layer=[max((v[i] for v in phases),default=0.)
                   for i in range(len(self.model.head_coupling.potentials))]
        row['learned_head_coupling']=dict(kind=self.learned_head_kind,
            replay_max_error=self.model._coupling_replay_error,
            parameter_change_norm=sum(x*x for x in changes)**.5,
            gradient_norm=sum(x*x for x in per_layer)**.5,
            per_layer_gradient_norm=per_layer,phase_gradient_norms=phases)
        row['bfss_dynamics'].update(enabled=True,internal_steps=self.bfss_steps,
            state_change=quiet_distance,
            interaction=self.learned_head_kind)
        return row

    def _advance(self,*args,**kwargs):
        # The original optimizer step is unchanged; capture the actual gradient
        # immediately before Adam, after original clipping and delayed actor.
        optimizer=self.optimizer
        original=optimizer.step
        def step(*a,**kw):
            layers=[]
            for module in self.model.head_coupling.potentials:
                norms=[p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
                layers.append(float(torch.stack(norms).norm()) if norms else 0.)
            self.model._head_phase_gradients.append(layers)
            return original(*a,**kw)
        optimizer.step=step
        try:
            return super()._advance(*args,**kwargs)
        finally:
            optimizer.step=original


def configure_learned_coupling(engine,kind,*,calibrate=False):
    if type(engine) not in (CausalProgramEngine,LearnedCouplingEngine):
        raise TypeError('requires the full continuing causal-program mother body')
    model=engine.model
    if not hasattr(model,'head_coupling'):
        device=model.encoder.weight.device
        with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
            torch.manual_seed(111117)
            model.head_coupling=LearnedHeadCoupling(model.body.base,kind,engine.bfss_coupling).to(device)
        if calibrate:
            for i,potential in enumerate(model.head_coupling.potentials):
                potential.calibrate(matrix_view(model.body.base.layers[i],engine.recurrent[i]))
        added=list(model.head_coupling.parameters())
        engine.optimizer.add_param_group(dict(params=added,lr=engine.optimizer.param_groups[0]['lr']))
        engine.params.extend(added)
        engine.groups=(*engine.groups,added)
    model.__class__=LearnedCouplingBody
    engine.__class__=LearnedCouplingEngine
    engine.learned_head_kind=kind
    engine.learned_head_schema=1
