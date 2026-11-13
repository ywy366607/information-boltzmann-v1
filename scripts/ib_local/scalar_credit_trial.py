"""One bounded native-forward learning trial of the registered scalar trace.

Only eight native A_log coordinates learn; original Adam histories are reused.
No retrieval injection, global reset, extra language head, or historical graph.
"""
import math
from copy import deepcopy

import numpy as np
import torch

from scripts.ib_local.native_s_learner import NativeSEligibilityLearner
from scripts.ib_local.irreversible_credit_stream import PretrainedStreamBackend
from scripts.ib_local.native_s_learner import NativeSEligibilityBackend
from scripts.ib_local.pretrained_gdn_learning import language_loss, PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn import clone_cache
from scripts.ib_local.scalar_gate_eligibility import ScalarGateEligibility


def upgrade_legacy_value_map_optimizer(stored, memories):
    """Insert empty Adam slots for value maps added after the checkpoint.

    The legacy encoder and inlet keep their exact per-parameter moments. This
    compatibility conversion never trains the value maps in this trial.
    """
    result=deepcopy(stored)
    groups=result['param_groups']
    if len(groups) != 3:
        raise ValueError('expected three original optimizer groups')
    names=[name for memory in memories.values() for name,_ in memory.named_parameters()
           if name not in ('null_key','null_bias')]
    old_names=[name for name in names if not name.startswith('value_map.')]
    old_ids=groups[1]['params']
    if len(old_ids)==len(names):
        return result
    if len(old_ids)!=len(old_names) or len(names)-len(old_names)!=2*len(memories):
        raise ValueError('unrecognized legacy optimizer layout')
    old_iter=iter(old_ids)
    next_id=max((i for group in groups for i in group['params']),default=-1)+1
    mapped=[]
    for name in names:
        if name.startswith('value_map.'):
            mapped.append(next_id)
            next_id+=1
        else:
            mapped.append(next(old_iter))
    groups[1]['params']=mapped
    return result


def select_scalar_credit(old, mode):
    """Control only the same eight strictly-old source coordinates."""
    if mode == 'raw':
        return {name: value.copy() for name,value in old.items()}
    if mode == 'cut_old':
        return {name: np.zeros_like(value) for name,value in old.items()}
    if mode == 'mismatch_old':
        return {name: np.roll(value,1) for name,value in old.items()}
    raise ValueError('unknown strictly-old credit control')


def shadow_adam_comparison(optimizer, selected_masters, current, old):
    """Measure the actual one-step FP32 displacement without mutating its source."""
    source=[master for master,_ in selected_masters.values()]
    group=next(g for g in optimizer.param_groups if any(p is source[0] for p in g['params']))
    if not all(any(p is x for p in group['params']) for x in source):
        raise ValueError('selected masters must share the original Adam group')
    outputs={}
    for mode in ('raw','cut_old'):
        clones=[torch.nn.Parameter(p.detach().clone()) for p in source]
        config={k:deepcopy(v) for k,v in group.items() if k!='params'}
        trial=torch.optim.AdamW([dict(params=clones,**config)])
        for clone,original in zip(clones,source):
            trial.state[clone]=deepcopy(optimizer.state.get(original,{}))
        extra=select_scalar_credit(old,mode)
        for (name,_),clone,g in zip(selected_masters.items(),clones,current):
            clone.grad=g.detach().float().clone()+torch.from_numpy(extra[name].copy()).to(clone.device)
        torch.nn.utils.clip_grad_norm_(clones,1.)
        trial.step()
        outputs[mode]={name:(clone.detach()-original.detach()).cpu().tolist()
                       for (name,_),clone,original in zip(selected_masters.items(),clones,source)}
    return outputs


class ScalarCreditLearner(NativeSEligibilityLearner):
    def initialize_trial(self, mode):
        if mode not in ('raw','cut_old','mismatch_old'):
            raise ValueError('registered scalar arm required')
        self.scalar_mode=mode
        self.enable_native_s_eligibility()
        self.scalar=ScalarGateEligibility(self.eligibility.specs)
        self._bind_scalar_parameters()

    def _bind_scalar_parameters(self):
        self.selected_masters={}
        self.model.requires_grad_(False)
        for param in self.extras+self.null_parameters:
            param.requires_grad_(False)
        names={f'layer{layer}.A_log' for layer in self.scalar.specs}
        for master,(name,param) in zip(self.masters,self.native):
            if name in names:
                param.requires_grad_(True)
                self.selected_masters[name]=(master,param)
        if len(self.selected_masters)!=2:
            raise ValueError('two A_log vectors must own eight scalar coordinates')
        self.model.eval()

    def state_dict(self):
        state=super().state_dict()
        state['scalar_credit_trial']=dict(schema=1,mode=self.scalar_mode,
                                           eligibility=self.scalar.state_dict())
        return state

    def load_state_dict(self,state,**kwargs):
        if len(state['optimizer']['param_groups'])==3:
            state=dict(state)
            state['optimizer']=upgrade_legacy_value_map_optimizer(state['optimizer'],self.memories)
        super().load_state_dict(state,**kwargs)
        trial=state.get('scalar_credit_trial')
        if trial is not None:
            if trial.get('schema')!=1 or self.eligibility is None:
                raise ValueError('incompatible scalar trial state')
            if trial.get('mode') not in ('raw','cut_old','mismatch_old'):
                raise ValueError('invalid stored scalar mode')
            self.scalar_mode=trial['mode']
            self.scalar=ScalarGateEligibility(self.eligibility.specs)
            self.scalar.load_state_dict(trial['eligibility'])
            self._bind_scalar_parameters()

    @torch.no_grad()
    def predict(self,x,cache=None,**kwargs):
        output=self.model(x,past_key_values=clone_cache(self.cache if cache is None else cache),use_cache=True)
        return output.logits,output.past_key_values

    @torch.no_grad()
    def observe(self,x,*,return_logits=False,cache_override=None,preserve_live_cache=False,**kwargs):
        cache=self.cache if cache_override is None else clone_cache(cache_override)
        output=self.model(x,past_key_values=cache,use_cache=True)
        if not preserve_live_cache:
            self.cache=clone_cache(output.past_key_values)
        self.tokens+=x.numel()
        return ({},output.logits) if return_logits else {}

    def step(self,x,target,*,loss_mask=None,cache_override=None,preserve_live_cache=False,**kwargs):
        pending=self.eligibility.pending
        if pending is None or not self.eligibility.is_complete():
            raise RuntimeError('learning requires a complete factual capture')
        if self._native_window_version!=self.updates:
            raise RuntimeError('parameters changed during factual capture')
        if cache_override is None or cache_override._seen_tokens!=pending['start_seen']:
            raise RuntimeError('teaching boundary does not match factual source window')
        if self.scalar.seen is not None and self.scalar.seen!=pending['start_seen']:
            raise RuntimeError('scalar history and teaching boundary diverged')
        self.optimizer.zero_grad(set_to_none=True)
        for _,param in self.native:
            param.grad=None
        cache=clone_cache(self.cache if cache_override is None else cache_override)
        leaves=[]
        for layer in self.layers:
            leaf=cache.states[layer]['recurrent_state'].detach().clone().requires_grad_(True)
            cache.states[layer]['recurrent_state']=leaf
            leaves.append(leaf)
        output=self.model(x,past_key_values=cache,use_cache=True)
        _,ce=language_loss(output.logits,target,loss_mask)
        parameters=[p for _,p in self.selected_masters.values()]
        gradients=torch.autograd.grad(10*ce,leaves+parameters)
        teaching={l:g.detach().float().cpu().numpy()[0] for l,g in zip(self.layers,gradients)}
        old=self.scalar.credit(teaching)
        extra=select_scalar_credit(old,self.scalar_mode)
        shadow=(shadow_adam_comparison(self.optimizer,self.selected_masters,gradients[len(leaves):],old)
                if self.scalar_mode=='raw' else None)
        before={n:p.detach().clone() for n,(_,p) in self.selected_masters.items()}
        for master in self.masters:
            master.grad=None
        for (name,(master,param)),current in zip(self.selected_masters.items(),gradients[len(leaves):]):
            master.grad=current.detach().float().clone()+torch.from_numpy(extra[name].copy()).to(master.device)
        preclip=float(torch.nn.utils.clip_grad_norm_([m for m,_ in self.selected_masters.values()],1.))
        if not math.isfinite(preclip):
            raise FloatingPointError('nonfinite scalar gradient')
        self.optimizer.step()
        with torch.no_grad():
            for name,(master,param) in self.selected_masters.items():
                param.copy_(master.to(param.dtype))
        self.updates+=1;self.tokens+=x.numel()
        if not preserve_live_cache:
            self.cache=clone_cache(output.past_key_values)
        return dict(ce=float(ce.detach()),updates=self.updates,tokens=self.tokens,
            committed_slots={},native_scalar=dict(mode=self.scalar_mode,
                old={n:v.tolist() for n,v in old.items()},
                added={n:v.tolist() for n,v in extra.items()},
                current={n:g.detach().cpu().tolist() for n,g in zip(self.selected_masters,gradients[len(leaves):])},
                parameter_delta={n:(p.detach()-before[n]).cpu().tolist()
                                 for n,(_,p) in self.selected_masters.items()},
                shadow_fp32_delta=shadow,
                preclip_norm=preclip,strict_old=True,trace_bytes=self.scalar.nbytes,
                teaching_start=pending['start_seen'],
                strict_old_source_exclusive_stop=(None if self.scalar.seen is None else self.scalar.seen-128)))


class ScalarCreditBackend(NativeSEligibilityBackend):
    def commit(self,inputs,targets,mask,boundary):
        result=PretrainedStreamBackend.commit(self,inputs,targets,mask,boundary)
        learner=self.learner
        pending=learner.eligibility.pending
        if pending is not None:
            if not learner.eligibility.is_complete():
                raise RuntimeError('actual scalar capture incomplete')
            capture_bytes=learner.eligibility.nbytes
            result['scalar_advance']=learner.scalar.advance(
                pending['initial_s'],pending['chunks'],pending['start_seen'])
            result['scalar_advance']['capture_bytes_at_commit']=capture_bytes
            learner.eligibility.pending=None
            learner._native_actual_count=0
        else:
            result['scalar_advance']={'inherited_partial_not_backfilled':True}
        return result
