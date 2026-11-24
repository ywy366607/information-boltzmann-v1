"""Learned common role coordinates, with bounded per-relation adaptation.

The neural prior, encoder and these coordinates learn across real episodes.
An eight-fact local solve only estimates four/five input coefficients per
output, instead of defining a new unconstrained 128/192-input mechanism.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SharedRelationBasis(nn.Module):
    def __init__(self, latent, *, device=None, dtype=None):
        super().__init__()
        kw=dict(device=device,dtype=dtype)
        self.condition=nn.Linear(latent,1,**kw)
        self.action=nn.Linear(latent,2,bias=False,**kw)
        self.gate=nn.Linear(latent,2,**kw)
        self.mother=nn.Linear(latent,1,bias=False,**kw)
        self.log_noise=nn.Parameter(torch.zeros(3,latent,**kw))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, source, action, mother=None):
        if not torch.is_grad_enabled():
            from scripts.ib_local.preview_runtime import _scope, tensor_key
            active = _scope.get()
            if active is not None and getattr(active[0], 'relation_basis', None) is self:
                cache = active[1]
                key = ('shared_relation_basis_cg', tensor_key(self.condition.weight),
                       tensor_key(self.gate.weight))
                entry = cache.get(key)
                if entry is None:
                    cg_w = torch.cat((self.condition.weight, self.gate.weight), 0)
                    cg_b = torch.cat((self.condition.bias, self.gate.bias), 0)
                    cache[key] = ((cg_w, cg_b), (self.condition.weight, self.gate.weight))
                else:
                    cg_w, cg_b = entry[0]
                cg = F.linear(source, cg_w, cg_b)
                c = cg[..., :1].tanh()
                u = self.action(action-source).tanh()*(2*cg[..., 1:].sigmoid())
                if mother is None:
                    return torch.cat((torch.ones_like(c), c, u), -1)
                return torch.cat((torch.ones_like(c), c, u, self.mother(mother-source).tanh()), -1)
        c=self.condition(source).tanh()
        u=self.action(action-source).tanh()*(2*self.gate(source).sigmoid())
        values=[torch.ones_like(c),c,u]
        if mother is not None:
            values.append(self.mother(mother-source).tanh())
        return torch.cat(values,-1)

    def noise(self, *, mother=False, transport=False):
        index=2 if transport else (0 if mother else 1)
        if not torch.is_grad_enabled():
            from scripts.ib_local.preview_runtime import _scope, tensor_key
            active = _scope.get()
            if active is not None and getattr(active[0], 'relation_basis', None) is self:
                cache = active[1]
                key = ('shared_relation_basis_noise', index, tensor_key(self.log_noise))
                entry = cache.get(key)
                if entry is not None:
                    return entry[0]
                out = F.softplus(self.log_noise[index])+torch.finfo(self.log_noise.dtype).eps
                if bool((out > 0).all()):
                    out._verified_positive = True
                cache[key] = (out, (self.log_noise,))
                return out
        return F.softplus(self.log_noise[index])+torch.finfo(self.log_noise.dtype).eps

    @staticmethod
    @torch.no_grad()
    def _pca(layer, observations, *, center):
        mean=observations.mean(0)
        centered=observations-mean
        covariance=centered.T@centered/max(len(observations)-1,1)
        eigenvalue,eigenvector=torch.linalg.eigh(covariance.double())
        count=layer.out_features
        directions=eigenvector[:,-count:].T.flip(0).to(observations)
        scale=eigenvalue[-count:].flip(0).to(observations).clamp_min(
            torch.finfo(observations.dtype).eps).sqrt()
        layer.weight.copy_(directions/scale[:,None])
        if layer.bias is not None:
            layer.bias.copy_(-layer.weight@mean if center else torch.zeros_like(layer.bias))

    @torch.no_grad()
    def initialize(self, model):
        m,enc=model.memory,model.encoder
        valid=m.occupied & (m.count==1) & m.mother_valid & (m.mother_stamp==m.last_write)
        slots=torch.nonzero(valid).flatten()
        if len(slots)<8:
            raise ValueError('eight actual labelled facts required for shared-coordinate migration')
        s,a,y=enc(m.source[slots]),enc(m.action[slots]),enc(m.mother_next[slots])
        self._pca(self.condition,s,center=True)
        self._pca(self.action,a-s,center=False)
        self._pca(self.mother,y-s,center=False)
        base=model.long_predictor(torch.cat((s,a),-1))
        mother_base=base+model.mother_from_boundary(base)
        noise=[(y-mother_base).square().mean(0)]
        decision=m.internal_valid[slots] & (m.internal_stamp[slots]==m.last_write[slots])
        if not bool(decision.any()):
            raise ValueError('actual completed decision facts required for transport scale')
        end=enc(m.internal_next[slots[decision]])
        noise.extend(((end-base[decision]).square().mean(0),
                      (end-y[decision]).square().mean(0)))
        value=torch.stack(noise).clamp_min(torch.finfo(s.dtype).eps)
        # Stable inverse softplus: log(expm1(v)) = v+log(1-exp(-v)).
        self.log_noise.copy_(value+torch.log(-torch.expm1(-value)))
        return dict(facts=len(slots),decision_facts=int(decision.sum()),
                    initial_noise=value.mean(-1).tolist())


def configure_shared_relation_basis(engine, *, initialize=False):
    if not hasattr(engine.model,'causal_relations'):
        raise ValueError('requires the inherited complete causal-program body')
    if engine.model.causal_relations.evidence_budget<8:
        raise ValueError('shared four/five-coordinate mechanism requires the original eight-fact budget')
    if hasattr(engine.model,'relation_basis'):
        return None
    w=engine.model.encoder.weight
    # Initialization never consumes the continuing individual's sample RNG.
    with torch.random.fork_rng(devices=[w.device] if w.is_cuda else []):
        module=SharedRelationBasis(w.shape[0],device=w.device,dtype=w.dtype)
    receipt=module.initialize(engine.model) if initialize else None
    engine.model.relation_basis=module
    engine.optimizer.add_param_group(dict(params=list(module.parameters()),
        lr=engine.optimizer.param_groups[0]['lr']))
    engine.params=[p for p in engine.model.parameters() if p.requires_grad]
    engine.groups=(*engine.groups[:2],[p for n,p in engine.model.named_parameters()
                                     if not n.startswith('body.')])
    engine.shared_relation_basis_schema=1
    return receipt
