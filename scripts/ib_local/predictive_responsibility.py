"""One proper mixture likelihood, with outcome-conditioned responsibility.

Components are predictions made BEFORE the observed outcome. The posterior is
an explanation inside this predictive model, not proof of an external cause.
The same NLL differentiates both selection and the complete component scores.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class PredictiveMixture:
    prior_scores: torch.Tensor        # [positions,components], unconstrained logs
    component_logp: torch.Tensor      # [positions,components,vocabulary]
    logp: torch.Tensor                # [positions,vocabulary]
    identities: list                 # None, (layer,slot,version[,view])

    @property
    def log_prior(self):
        return self.prior_scores.log_softmax(-1)

    def posterior(self, targets):
        labels=targets.reshape(-1)
        if len(labels)!=len(self.logp):
            raise ValueError('one observed target per predicted position')
        likelihood=self.component_logp.gather(-1,
            labels[:,None,None].expand(-1,self.component_logp.shape[1],1)).squeeze(-1)
        # Normalize directly in log space; no raw-likelihood underflow.
        return (self.log_prior+likelihood).softmax(-1)

    @torch.no_grad()
    def report(self,targets,mask=None):
        selected=(torch.ones(len(self.logp),dtype=torch.bool,device=self.logp.device)
                  if mask is None else mask.reshape(-1).bool().to(self.logp.device))
        if not bool(selected.any()):
            raise ValueError('no observed scoring positions')
        prior=self.log_prior.exp()[selected]
        posterior=self.posterior(targets)[selected]
        result = dict(components=len(self.identities),supervised_positions=int(selected.sum()),
            identities=self.identities,prior_mass=prior.mean(0).cpu().tolist(),
            posterior_mass=posterior.mean(0).cpu().tolist(),
            responsibility_shift=(posterior-prior).mean(0).cpu().tolist(),
            posterior_null=float(posterior[:,0].mean()),
            posterior_change_l1=float((posterior-prior).abs().sum(-1).mean()),
            predictive_probability_sum_error=float((self.logp.exp().sum(-1)-1).abs().max()))
        if any(isinstance(identity,tuple) and len(identity)==4 for identity in self.identities):
            result['layer_views'] = []
            for layer in sorted({i[0] for i in self.identities if i is not None}):
                for view in ('state', 'transition'):
                    indices=[j for j,i in enumerate(self.identities)
                             if i is not None and i[0]==layer and i[3]==view]
                    result['layer_views'].append(dict(layer=layer+1,view=view,
                        prior_mass=float(prior[:,indices].sum(-1).mean()),
                        posterior_mass=float(posterior[:,indices].sum(-1).mean())))
        return result


def predictive_mixture(prior_scores,component_logits,identities=None):
    """Softmax each expert, THEN mix probabilities (never mix hidden vectors)."""
    if component_logits.ndim!=3 or prior_scores.shape!=component_logits.shape[:2]:
        raise ValueError('expected [positions,components] and [positions,components,vocabulary]')
    if not torch.isfinite(prior_scores).all() or not torch.isfinite(component_logits).all():
        raise FloatingPointError('nonfinite predictive mixture input')
    scores=prior_scores.float()
    expert=component_logits.float().log_softmax(-1)
    logp=torch.logsumexp(scores.log_softmax(-1)[...,None]+expert,dim=1)
    return PredictiveMixture(scores,expert,logp,
        list(range(prior_scores.shape[1])) if identities is None else identities)


def native_predictive_mixture(model,bank,ids,state,table_logp,actual_logits):
    """Compare the SAME body using null or one particular local experience.

    Actual recurrence follows the existing soft recall transition. Conditional
    scores are finite counterfactual suffix executions from that same incoming
    state: null uses no extra local capsule, each expert uses exactly one.
    Original lexical and navigation memory remain in every conditional body.
    No hypothetical branch state is written back to the continuing individual.
    """
    if ids.shape[1]>bank.config['replay_tokens']:
        raise ValueError('predictive mixture requires a bounded128-position call')
    count=len(bank.memories);body=model.body;base=model.base
    if not any(bool(m.memory.occupied.any()) for m in bank.memories):
        scores=actual_logits.new_zeros(ids.shape[1],1)
        bank.forecast=predictive_mixture(scores,actual_logits[0,:,None],[None])
        return actual_logits  # exactly preserve the empty-reservoir migration

    def incoming(h,index):
        if index==body.inject_layer:
            if body.use_table:
                f=table_logp-table_logp.mean(-1,keepdim=True)
                key=body.key_norm(body.table_key(f))
                gate=torch.sigmoid((body.query_norm(h)*key).sum(-1,keepdim=True)/base.d**.5)
                h=h+gate*body.table_value(f)
            flat=h.reshape(-1,base.d)
            read=model.memory.read(flat,model.encoder)
            residual=model.to_hidden(read.value)*model.read_gain.tanh()
            h=h+residual.reshape_as(h)
        return h

    def decode(h):
        h=base.final_norm(h)
        logits=base.decoder(h)
        if body.use_table:
            logits=logits+F.softplus(body.mix(h))*table_logp
        return logits

    # Shared null prefix. Direct calls bypass the CURRENT local/global hooks;
    # incoming() implements the original global/lexical port explicitly.
    h=base.source.embedding(ids)
    prefixes=[]
    for index,layer in enumerate(base.layers):
        h=layer.forward(incoming(h,index),state[index])[0]
        prefixes.append(h)
    components=[decode(h).transpose(0,1)]  # [T,1,V]
    priors=[];null_priors=[];identities=[None]
    for index,wrapper in enumerate(bank.memories):
        pr=wrapper.position_read
        if pr is None:
            null_priors.append(actual_logits.new_ones(ids.shape[1]))
            continue
        slots=pr['slots'];mass=wrapper.last_read.weights.to(actual_logits.device)
        null_priors.append(pr['null_weight'].to(actual_logits.device))
        if not len(slots):continue
        dual = 'prediction_values' in pr
        conditional=pr['prediction_values' if dual else 'conditional_values'].to(actual_logits.device)
        candidate_slots = pr['prediction_slots'] if dual else slots.tolist()
        if dual:
            mass = pr['prediction_weights'].to(actual_logits.device)
        injection=wrapper.inlet.projection(conditional).to(prefixes[index].dtype)
        # [records,T,d], a shared model applied to finite alternative inputs.
        branch=prefixes[index]+injection
        for following in range(index+1,count):
            boundary=state[following].expand(len(candidate_slots),*state[following].shape[1:])
            branch=base.layers[following].forward(incoming(branch,following),boundary)[0]
        components.append(decode(branch).transpose(0,1))
        priors.append(mass/count)
        if dual:
            identities.extend((index,int(slot),int(wrapper.memory.read_credit_version[slot]),view)
                              for slot,view in zip(candidate_slots,pr['prediction_views']))
        else:
            identities.extend((index,int(slot),int(wrapper.memory.read_credit_version[slot]))
                              for slot in candidate_slots)
    null=torch.stack(null_priors).mean(0)[:,None]
    probabilities=torch.cat((null,*priors),dim=-1)
    scores=probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    bank.forecast=predictive_mixture(scores,torch.cat(components,dim=1),identities)
    return bank.forecast.logp[None]
