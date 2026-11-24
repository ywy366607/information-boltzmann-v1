"""Calibrate the qualified read increment itself and deliver it to choice.

No new neural parameters. Moments advance only on participating, compatible
gain reads; neither unrelated observations nor replay count as fresh evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import torch

from scripts.ib_local.navigation_causal_wood import CausalWoodEngine, configure_causal_wood
from scripts.ib_local.navigation_value_scale import critic_output
from scripts.ib_local.language_action import peek
from scripts.ib_local.language_navigation_kanli_v4c import choice_replay_error


def gain_state(decay):
    return dict(cross=np.zeros(3),second=np.zeros(3),error=np.zeros(3),
                count=np.zeros(3,dtype=np.int64),decay=float(decay))


def gain_forecast(state, baseline, delta, eligible):
    c,v,n=state['cross'],state['second'],state['count']
    alpha=np.clip(np.divide(c,v,out=np.zeros_like(c),where=v>0),0,1)
    alpha=np.where((n>1)&eligible,alpha,0.)
    bias=1-state['decay']**n
    rms=np.sqrt(np.divide(state['error'],bias,out=np.zeros_like(c),where=bias>0))
    correction=alpha*delta
    offset=np.divide(correction,rms,out=np.zeros_like(c),where=rms>0)
    if not all(np.isfinite(x).all() for x in (alpha,rms,offset)):
        raise FloatingPointError('nonfinite conditional read calibration')
    return dict(mean=baseline+correction,alpha=alpha,rms=rms,offset=offset,
                baseline=np.array(baseline,copy=True),delta=np.array(delta,copy=True),
                eligible=np.array(eligible,copy=True))


def observe_gain(state,option,target,forecast):
    if not forecast['eligible'][option]:return
    if state['count'][option]==np.iinfo(np.int64).max:raise OverflowError('read gain clock exhausted')
    b,d=forecast['baseline'][option],forecast['delta'][option]
    rate=1-state['decay']
    state['cross'][option]=state['decay']*state['cross'][option]+rate*(target-b)*d
    state['second'][option]=state['decay']*state['second'][option]+rate*d*d
    state['error'][option]=state['decay']*state['error'][option]+rate*(forecast['mean'][option]-target)**2
    state['count'][option]+=1


def replay_error(replay,record):
    errors=[choice_replay_error(replay['choice'],record['choice_logits'],record['habit_logits'])]
    errors.extend(abs(a-b) for a,b in zip(replay['stops'],record['stop_logits']))
    if 'byte_logp' in replay:
        errors.append(float((replay['byte_logp']-replay['byte_logp'].new_tensor(record['byte_logp'])).abs().max()))
    return max(errors)


def causal_prefix(engine,targets,own,record):
    """Differentiable one-byte replay of the bounded action, no old graph."""
    from scripts.ib_local.variable_expression import read_route_scope
    from scripts.ib_local.adaptive_hamiltonian import bridge_state
    route = (getattr(engine, '_selected_handover', None)
             if own and record.get('choice') == 1 else None)
    if getattr(engine,'policy_replay_backend','one_byte')=='batched_guarded':
        count=max(own,1)
        ids=torch.cat((targets.new_tensor([[engine.pending]]),targets[:,:max(own-1,0)]),1)
        values=record['probs'][:count] if own else [record['boundary_probability']]
        logp=torch.as_tensor(np.log(np.maximum(np.stack(values),1e-6)),device=engine.device)[None]
        from scripts.ib_local.variable_expression import budget_scope
        remaining=(1-torch.arange(count,device=engine.device)[None]/engine.max_utterance
                   if getattr(engine,'variable_expression_schema',None)==1 and own else None)
        with budget_scope(engine.model,remaining), read_route_scope(engine,route,0,count):
            out=engine.model.predict_observed(ids,bridge_state(engine.model,engine.recurrent.detach()),logp)
        expected=record.get('expected_byte_logp',record.get('byte_logp'))
        if own and expected is not None and len(expected)==own:
            actual=out[0][0].log_softmax(-1).gather(1,targets[0,:own,None]).flatten()
            error=float((actual.detach()-actual.new_tensor(expected)).abs().max())
            if error<=1e-3:
                engine._batched_replay_hits=getattr(engine,'_batched_replay_hits',0)+1
                return out[3],out[0]
            del out,actual
        elif not own:
            return out[3],out[0]
        # A discrete top-k boundary may move with GEMM rounding. Preserve the
        # original one-byte path, with the same strict replay tolerance.
        engine._batched_replay_fallbacks=getattr(engine,'_batched_replay_fallbacks',0)+1
    state=bridge_state(engine.model,engine.recurrent.detach())
    logits,features=[],[]
    for i in range(max(own,1)):
        byte=engine.pending if i==0 else int(targets[0,i-1])
        probability=record['probs'][i] if own else record['boundary_probability']
        ids=torch.tensor([[byte]],device=engine.device)
        logp=torch.as_tensor(np.log(np.maximum(probability,1e-6)),device=engine.device)[None,None]
        from scripts.ib_local.variable_expression import budget_scope
        remaining=(1-i/engine.max_utterance if getattr(engine,'variable_expression_schema',None)==1 and own else None)
        with budget_scope(engine.model,remaining), read_route_scope(engine,route,i,1):
            out=engine.model.predict_observed(ids,state,logp)
        logits.append(out[0]);state=out[1];features.append(out[3])
    return torch.cat(features),torch.cat(logits,dim=1)


@contextmanager
def observer_reads(engine):
    if engine.observer_mode=='full':
        yield
        return
    if engine.settlement_kind not in ('continuation_joint', 'world_stream',
                                      'coefficient_information_v1', 'targeted_use_v1',
                                      'finite_targeted_use_v2', 'finite_event_targeted_use_v3',
                                      'finite_relation_use_v4', 'finite_relation_use_v5'):
        raise ValueError('omitted witness reads require actual joint-continuation settlement')
    owned='witness_nll' in vars(engine);old=engine.witness_nll
    def unscored():
        engine._witness_readings.append((0.,0.))
        return 0.
    engine.witness_nll=unscored
    try:yield
    finally:
        if owned:engine.witness_nll=old
        else:del engine.witness_nll


class DirectReadGainEngine(CausalWoodEngine):
    def _utter(self):
        boundary=peek(self.table,self.pending).copy()
        utterance,record=super()._utter()
        record['boundary_probability']=boundary
        return utterance,record

    def _policy_logp(self,h,logits,targets,own,record):
        result=super()._policy_logp(h,logits,targets,own,record)
        error=replay_error(result[1],record)
        self._policy_replay_fallback=dict(used=False,batch_error=error)
        if error<=1e-3:return result
        # Near a top-k tie, a GEMM's rounded prefix can select a different
        # discrete read than the actual one-byte action. Re-evaluate that same
        # finite action path; never widen the parent's replay guard.
        features,prefix_logits=causal_prefix(self,targets,own,record)
        result=super()._policy_logp(features,prefix_logits,targets,own,record)
        self._policy_replay_fallback.update(used=True,exact_path_error=replay_error(result[1],record))
        return result

    @torch.no_grad()
    def _value(self,h,films):
        mode=self.episodic_wood_mode
        self.episodic_wood_mode='blocked'
        try:values,inputs=super()._value(h,films)
        finally:self.episodic_wood_mode=mode
        f,r=self._forecast,self._episodic_decision_read
        plain=f['episodic_plain_used'][:,3].cpu().numpy()
        delta=(f['raw'][:,3]-f['episodic_raw_plain'][:,3]).cpu().numpy()
        eligible=(r['support'][:,3].cpu().numpy()>0)&(delta!=0)
        self._gain_forecast=gain_forecast(self.read_gain_state,plain,delta,eligible)
        offset=values.new_tensor(self._gain_forecast['offset'])
        available=len(self._draft_actions) if self._draft_actions else 1
        offset[available:]=0
        offset[:available]-=offset[:available].mean()
        self._direct_read_offset=offset
        f['episodic_offset']=offset
        width=self.model.encoder.out_features
        def old_offset(other):
            d=values-other
            out=torch.zeros_like(d)
            out[:available]=self.precision()*(d[:available]-d[:available].mean())
            return out
        blind=inputs.clone();blind[:,width:2*width]=0
        base=inputs.clone();base[:,width:width+4]=f['base']
        no_match=inputs.clone();no_match[:,width+2]=f['base'][:,2]
        def value(x):return critic_output(self,x).squeeze(-1)+x[:,-2]
        direct=offset if mode=='real' else torch.zeros_like(offset)
        f['total_offset']=old_offset(value(blind))+direct
        f['residual_offset']=old_offset(value(base))+direct
        f['typed_match_offset']=old_offset(value(no_match))
        self.draft_diagnostics=dict(
            policy_logit_F_effect_max=float(f['total_offset'].abs().max()),
            policy_logit_residual_effect_max=float(f['residual_offset'].abs().max()),
            confidence=f['confidence'].tolist(),
            direct_read_logit_max=float(direct.abs().max()))
        return values,inputs

    def _choice_offset(self,values,available):
        old=super()._choice_offset(values,available)
        return old+self._direct_read_offset.masked_fill(~available,0.) if self.episodic_wood_mode=='real' else old

    def step(self,*a,**kw):
        with observer_reads(self):
            row=super().step(*a,**kw)
        k=self._pending_record['choice'];p=self._gain_forecast
        target=row['wood_target'][3]
        shuffled_raw=float(self._forecast['episodic_raw_shuffled'][k,3])
        plain_raw=float(self._forecast['episodic_raw_plain'][k,3])
        shuffled=p['baseline'][k]+p['alpha'][k]*(shuffled_raw-plain_raw)
        row['episodic_used_real_error'][3]=float((p['mean'][k]-target)**2)
        row['episodic_used_shuffled_error'][3]=float((shuffled-target)**2)
        delivered=self._forecast['episodic_plain_used'].clone()
        if self.episodic_wood_mode=='real':delivered[:,3]=delivered.new_tensor(p['mean'])
        row['causal_delivered_prediction']=delivered[k].tolist()
        row['causal_forecast_role']='delivered_gain_only' if self.episodic_wood_mode=='real' else 'shadow'
        row['wood_prediction']=delivered[k].tolist()
        row['wood_all_predictions']=delivered.tolist()
        row['wood_used_error']=(delivered[k]-delivered.new_tensor(row['wood_target'])).square().tolist()
        probability=float(delivered[k,2]);match=row['wood_target'][2]
        row.update(match_probability=probability,match_distribution=[1-probability,probability],
                   match_brier=(probability-match)**2,match_prediction_error=match-probability,
                   match_score=-(probability-match)**2)
        row.update(direct_gain_mean=p['mean'].tolist(),direct_gain_baseline=p['baseline'].tolist(),
            direct_gain_alpha=p['alpha'].tolist(),direct_gain_rms=p['rms'].tolist(),
            direct_gain_eligible=p['eligible'].tolist(),direct_gain_offset=self._direct_read_offset.tolist(),
            wood_gain_route='qualified increment -> own occurrence calibration -> choice',
            observer_mode=self.observer_mode)
        row['policy_replay_fallback']=self._policy_replay_fallback
        observe_gain(self.read_gain_state,k,target,p)
        row['direct_gain_counts_next']=self.read_gain_state['count'].tolist()
        if self.observer_mode!='full':
            for name in ('witness_pre','witness_post','witness_g','stored_table_g','current_table_g',
                         'witness_current_pre','witness_current_post'):
                row[name]=None
            row['witness_scored']=False
        else:row['witness_scored']=True
        return row


def configure_direct_read(engine,mode,observer_mode=None):
    configure_causal_wood(engine,mode)
    engine.__class__=DirectReadGainEngine
    if not hasattr(engine,'read_gain_state'):engine.read_gain_state=gain_state(engine.wood_decay)
    if observer_mode is None:observer_mode=getattr(engine,'observer_mode','full')
    if observer_mode not in ('full','joint_only'):raise ValueError(observer_mode)
    engine.observer_mode=observer_mode
    return engine
