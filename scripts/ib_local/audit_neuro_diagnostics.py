"""Paired real-text diagnostic battery; checkpoints and body rules unchanged.

Eight factorial interventions share exactly the same held-out bytes. A separately
trained original Delta checkpoint is a reference, not a matched training-budget
claim. Pure GDN extracted from today's same weights is another explicit control.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.audit_closed_loop_information import patch_method, sha, write_json
from scripts.ib_local.navigation_live_witness import frozen_table_logp
from scripts.ib_local.navigation_checkpoint import snapshot,restore


@contextmanager
def reset_each_token(layers):
    """Exact zero-state-per-byte intervention, vectorized as independent batches.

    Outer layers still see [1,T,d], including original table and memory hooks.
    Only the layer's own internal evaluation changes [1,T,d] to [T,1,d].
    """
    with ExitStack() as stack:
        for layer in layers:
            original=layer.forward
            def reset(x,state,original=original):
                if x.shape[0]!=1:
                    raise ValueError('one continuous source stream required')
                zero=state.new_zeros((x.shape[1],)+state.shape[1:])
                out,nxt,diag=original(x.transpose(0,1),zero)
                return out.transpose(0,1),nxt[-1:],diag
            stack.enter_context(patch_method(layer,'forward',reset))
        yield


class View:
    def __init__(self,model,kind,engine=None):
        self.model,self.kind,self.engine=model,kind,engine
        self.base=model.base
        self.layers=self.base.layers
        self.device=next(model.parameters()).device

    def initial(self):
        return self.base.initial_state(1,device=self.device)

    @contextmanager
    def configuration(self,memory=True,table=True,recurrence=True,shuffle=False):
        if self.kind=='plain':
            with ExitStack() as stack:
                if not recurrence: stack.enter_context(reset_each_token(self.layers))
                yield
            return
        original=self.model.body.use_table
        self.model.body.use_table=table
        action=successor=None
        if shuffle:
            mem=self.model.memory
            occupied=torch.nonzero(mem.occupied).flatten()
            rng=np.random.default_rng(62)
            order=occupied[torch.tensor(rng.permutation(len(occupied)),device=occupied.device)]
            action,successor=mem.action.clone(),mem.long_successor.clone()
            mem.action[occupied]=action[order]; mem.long_successor[occupied]=successor[order]
        self.memory=memory
        try:
            with ExitStack() as stack:
                if not recurrence: stack.enter_context(reset_each_token(self.layers))
                yield
        finally:
            self.model.body.use_table=original
            if action is not None:
                self.model.memory.action.copy_(action)
                self.model.memory.long_successor.copy_(successor)

    @torch.no_grad()
    def forward(self,ids,state,logp,collect=False):
        handles=[]; inputs={}; hidden={}; alpha={}
        if collect:
            for i,layer in enumerate(self.layers):
                def hook(layer,args,out,i=i):
                    inputs[i]=args[0].detach().clone()
                    hidden[i]=out[0].detach().clone()
                    alpha[i]=layer.retention(layer.norm(args[0])).detach().clone()
                handles.append(layer.register_forward_hook(hook))
        try:
            if self.kind=='plain':
                logits,nxt,diag=self.model.predict_logits(ids,state)
                features=None
            elif self.memory:
                logits,nxt,diag,features,_=self.model.predict_observed(ids,state,logp)
            else:
                logits,nxt,diag=self.model.body.predict(ids,state,logp)
                features=None
        finally:
            for h in handles:h.remove()
        return logits,nxt,dict(hidden=hidden,inputs=inputs,alpha=alpha,features=features,diag=diag)


class PlainModel:
    """Non-owning adapter to the independently trained GDN interface."""
    def __init__(self,base):self.base=base
    def parameters(self):return self.base.parameters()
    def predict_logits(self,*args):return self.base.predict_logits(*args)


def load_control(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    cfg=saved['config']['model']
    model=GatedDeltaNetLM(**cfg).to(device)
    model.load_state_dict(saved['model']);model.eval()
    return View(PlainModel(model),'plain'),dict(checkpoint=str(path),sha256=sha(path),
        config=saved['config'],step=saved.get('step'),
        scope='independently trained original Delta reference; training history differs')


def score_arm(view,segments,logps,warmup,condition,collect=False):
    m,t,r=condition
    losses=[]; hidden=[]; confidence=[]; correctness=[]; brier=[]
    with view.configuration(m,t,r):
        for raw,lp in zip(segments,logps):
            ids=torch.as_tensor(raw[:-1].astype(np.int64),device=view.device)[None]
            targets=torch.as_tensor(raw[1:].astype(np.int64),device=view.device)
            table=torch.as_tensor(lp,device=view.device)[None]
            state=view.initial(); scores=[]; reps=[];probs=[]
            for start in range(0,len(raw)-1,256):
                logits,state,record=view.forward(ids[:,start:start+256],state,table[:,start:start+256],collect)
                stop=min(start+256,len(raw)-1)
                offset=max(warmup-start,0)
                if start+offset<stop:
                    current=logits[0,offset:].float()
                    scores.append(F.cross_entropy(current,targets[start+offset:stop],reduction='none').cpu().numpy())
                    p=current.softmax(-1); y=targets[start+offset:stop]
                    probs.append((p.max(-1).values.cpu().numpy(),(p.argmax(-1)==y).cpu().numpy(),
                                  ((p*p).sum(-1)-2*p[torch.arange(len(y),device=y.device),y]+1).cpu().numpy()))
                    if collect:
                        reps.append(torch.stack([record['hidden'][i][0,offset:,::16] for i in range(len(view.layers))],1).cpu().numpy())
            losses.append(np.concatenate(scores))
            if collect:hidden.append(np.concatenate(reps))
            confidence.append(np.concatenate([p[0] for p in probs]));correctness.append(np.concatenate([p[1] for p in probs]));brier.append(np.concatenate([p[2] for p in probs]))
    return dict(losses=np.array(losses),hidden=np.array(hidden) if hidden else np.empty(0),
                confidence=np.array(confidence),correctness=np.array(correctness),brier=np.array(brier))


def verify_vectorized_reset(view,raw,lp):
    ids=torch.as_tensor(raw[:8].astype(np.int64),device=view.device)[None]
    logp=torch.as_tensor(lp[:8],device=view.device)[None]
    with view.configuration(True,True,False):
        fast,state,_=view.forward(ids,view.initial(),logp)
    slow=[]
    with view.configuration(True,True,True):
        for i in range(8):
            logits,last,_=view.forward(ids[:,i:i+1],view.initial(),logp[:,i:i+1])
            slow.append(logits)
    error=float((fast-torch.cat(slow,1)).abs().max())
    state_error=float((state-last).abs().max())
    if error>1e-4 or state_error>1e-4:
        raise AssertionError((error,state_error))
    return dict(max_logit_error=error,max_state_error=state_error)


def rms_direction(state,rng,matrix_sizes,layer=None,amplitude=.001):
    direction=torch.zeros_like(state)
    for i,n in enumerate(matrix_sizes):
        if layer is None or layer==i:
            value=torch.as_tensor(rng.normal(size=tuple(state[i,:,:n].shape)),device=state.device,dtype=state.dtype)
            direction[i,:,:n]=value
    norm=direction.norm().clamp_min(1e-20)
    magnitude=amplitude*state.norm().clamp_min(1)
    return direction/norm*magnitude


@torch.no_grad()
def dynamics(view,raw,lp,warmup,horizon=256,contexts=4):
    """Driven conditional stability with frozen weights, plus causal direction.

    Original full body, byte prior and causal recall remain in the CGB view.
    This is distinct from whole body-world learning-mediated feedback.
    """
    rng=np.random.default_rng(94)
    global_rates=[];local_rates=[];causal=[];ignition=[];retentions=[]
    widths=[l.matrix_size for l in view.layers]
    for c in range(min(contexts,len(raw))):
        ids=torch.as_tensor(raw[c][:-1].astype(np.int64),device=view.device)[None]
        logp=torch.as_tensor(lp[c],device=view.device)[None]
        with view.configuration(True,True,True):
            _,state,_=view.forward(ids[:,:warmup],view.initial(),logp[:,:warmup])
            seq=ids[:,warmup:warmup+horizon];prior=logp[:,warmup:warmup+horizon]
            reference,base_end,record=view.forward(seq,state,prior,True)
            for eps in (.001,.003):
                initial_delta=rms_direction(state,rng,widths,amplitude=eps)
                magnitude=float(initial_delta.norm())
                a,b=state.clone(),state+initial_delta
                rates=[]
                for start in range(0,horizon,16):
                    _,a,_=view.forward(seq[:,start:start+16],a,prior[:,start:start+16])
                    _,b,_=view.forward(seq[:,start:start+16],b,prior[:,start:start+16])
                    delta=b-a;norm=float(delta.norm())
                    rates.append(float(np.log(max(norm,1e-30)/magnitude)/16))
                    b=a+delta*(magnitude/max(norm,1e-30))
                global_rates.append(dict(context=c,epsilon=eps,mean_log_gain_per_byte=float(np.mean(rates)),blocks=rates))
            # Actual per-layer/head transitions with the observed lower-layer
            # input held fixed. All heads are evaluated in parallel.
            for i,layer in enumerate(view.layers):
                h=record['inputs'][i]
                alpha=record['alpha'][i].cpu().numpy()
                retentions.append(dict(context=c,layer=i,mean_log_alpha=np.log(alpha.clip(1e-30)).mean((0,1)).tolist()))
                H,D=layer.heads,layer.dk*layer.dv
                a=state[i].clone();b=a.expand(H,-1).clone()
                magnitude=[]
                for head in range(H):
                    sl=slice(head*D,(head+1)*D)
                    v=torch.as_tensor(rng.normal(size=D),device=view.device,dtype=a.dtype)
                    mag=.001*max(float(a[0,sl].norm()),1.)
                    b[head,sl]+=v/v.norm()*mag;magnitude.append(mag)
                values=[]
                for start in range(0,horizon,16):
                    _,a,_=layer(h[:,start:start+16],a)
                    _,b,_=layer(h[:,start:start+16].expand(H,-1,-1),b)
                    delta=b-a;next_b=a.expand(H,-1).clone();gains=[]
                    for head in range(H):
                        sl=slice(head*D,(head+1)*D);v=delta[head,sl];norm=float(v.norm())
                        gains.append(np.log(max(norm,1e-30)/magnitude[head])/16)
                        next_b[head,sl]+=v*(magnitude[head]/max(norm,1e-30))
                    b=next_b;values.append(gains)
                local_rates.append(dict(context=c,layer=i,mean_log_gain_per_byte=np.mean(values,0).tolist()))
            # Intervene in each actual layer's state; fixed input permits
            # direct upper->lower effects to be distinguished from common input.
            for i in range(len(view.layers)):
                delta=rms_direction(state,rng,widths,layer=i,amplitude=.01)
                logits,nxt,pert=view.forward(seq,state+delta,prior,True)
                hidden_gain=[float((pert['hidden'][j]-record['hidden'][j]).norm()/delta.norm()) for j in range(len(view.layers))]
                state_gain=[float((nxt[j]-base_end[j]).norm()/delta.norm()) for j in range(len(view.layers))]
                causal.append(dict(context=c,source_layer=i,hidden_response_gain=hidden_gain,state_response_gain=state_gain,
                    decoder_kl=float((reference.log_softmax(-1).exp()*(reference.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).mean())))
            # One same direction, five amplitudes; report graded/nonlinear
            # amplification rather than treating any persistent response as ignition.
            unit=rms_direction(state,rng,widths,layer=0,amplitude=1.)
            for amp in (.001,.003,.01,.03,.1):
                logits,_,pert=view.forward(seq,state+amp*unit,prior,True)
                energy=torch.stack([(pert['hidden'][j]-record['hidden'][j]).square().mean(-1).sqrt()[0] for j in range(len(view.layers))],-1)
                row=dict(context=c,relative_amplitude=amp,early_response=energy[:32].mean(0).cpu().tolist(),
                         late_response=energy[-32:].mean(0).cpu().tolist(),
                         decoder_kl=float((reference.log_softmax(-1).exp()*(reference.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).mean()))
                if view.kind!='plain':
                    old_h=record['features'][-1:];new_h=pert['features'][-1:]
                    old_policy,old_plan,_=view.engine.plan(old_h)
                    new_policy,new_plan,_=view.engine.plan(new_h)
                    row['existing_readout_changes']=dict(habit_policy_tv=float((old_policy.softmax(-1)-new_policy.softmax(-1)).abs().sum()/2),
                        short_prediction_norm=float((old_plan['short']-new_plan['short']).norm()),
                        recalled_action_norm=float((old_plan['action']-new_plan['action']).norm()))
                ignition.append(row)
    return dict(global_conditional_lyapunov=global_rates,local_head_conditional_lyapunov=local_rates,
                alpha_diagnostics=retentions,directed_state_interventions=causal,amplitude_response=ignition,
                scope='Frozen-weight driven body, conditional on actual real-text input. Not the Lyapunov exponent of the learning body-world hybrid.')


def macro_downward(engine,initial):
    """One actual complete learning/world cycle per branch, source restored after.

    Capture lower-layer state before the first optimizer update and after the
    full cycle. Identical utterance/world are checked rather than assumed.
    """
    outcomes={};states={};weights={};first={}
    for name in ('intact','upper_pulse'):
        restore(engine,initial)
        if name=='upper_pulse':
            layer=engine.model.base.layers[-1]
            width=layer.matrix_size
            rng=np.random.default_rng(208)
            value=engine.recurrent[-1,:,:width]
            delta=torch.as_tensor(rng.normal(size=tuple(value.shape)),device=value.device,dtype=value.dtype)
            with torch.no_grad():value.add_(delta/delta.norm()*.01*value.norm().clamp_min(1.))
        original=engine.model.objective
        def objective(*args,**kwargs):
            result=original(*args,**kwargs)
            if kwargs.get('score_mask') is not None:
                first[name]=result[1][:2].detach().clone()
            return result
        with patch_method(engine.model,'objective',objective):
            outcomes[name]=engine.step(torch.randn(64,32,device=engine.device))
        states[name]=engine.recurrent[:2].detach().clone()
        weights[name]=torch.cat([p.detach().flatten() for layer in engine.model.base.layers[:2]
                                for p in layer.parameters()]).clone()
    report=dict(same_utterance=outcomes['intact']['utterance_hex']==outcomes['upper_pulse']['utterance_hex'],
                same_world_position=outcomes['intact']['world']==outcomes['upper_pulse']['world'],
                before_first_update_lower_state_max_difference=float((first['intact']-first['upper_pulse']).abs().max()),
                after_cycle_lower_state_max_difference=float((states['intact']-states['upper_pulse']).abs().max()),
                after_cycle_lower_weight_max_difference=float((weights['intact']-weights['upper_pulse']).abs().max()),
                settlement_difference=outcomes['upper_pulse']['metal_g']-outcomes['intact']['metal_g'],
                note='If incoming/self language match, zero pre-update but nonzero post-update shows learning-mediated downward influence.')
    restore(engine,initial)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--control',type=Path,default=Path('results/cgb_feedback_pair_20261103_5000/decay_then_delta/last.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--segments',type=int,default=16)
    p.add_argument('--tokens',type=int,default=1024)
    p.add_argument('--warmup',type=int,default=256)
    p.add_argument('--offset',type=int,default=65536)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);gdn_fused.enable()
    engine,validation,offset,meta=build(args.checkpoint)
    initial=snapshot(engine)
    engine.model.eval()
    count=args.segments*(args.tokens+args.warmup+1)
    selected=np.asarray(validation[:2*(args.offset+count+1)])
    raw=(selected[selected>=offset]-offset).astype(np.uint8)[args.offset:args.offset+count]
    if len(raw)!=count:raise ValueError('held-out source too short')
    segments=raw.reshape(args.segments,-1)
    logps=[frozen_table_logp(engine.table,s[:-1]) for s in segments]
    view=View(engine.model,'cgb',engine)
    control,control_meta=load_control(args.control,view.device)
    verify=dict(cgb=verify_vectorized_reset(view,segments[0],logps[0]),
                control=verify_vectorized_reset(control,segments[0],logps[0]))
    config=dict(stage='corrected real-text neuro diagnostic battery',source=meta,control=control_meta,
                segments=args.segments,score_bytes_per_segment=args.tokens,warmup=args.warmup,
                validation_filtered_offset=args.offset,validation_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                contexts='same real validation bytes, cold then warm each segment; stored memories read-only',
                table_context='one vector read over the full prefix+scored sequence; history never reset per byte',
                reset_equivalence=verify,source_checkpoint_unchanged_required=True,
                existing_gain_evidence='results/published/cgb_water_reply_20261105.json; no repeated independent-gain training')
    sources={str(Path(m.__file__)) for n,m in list(sys.modules.items()) if n.startswith('scripts.ib_local.') and getattr(m,'__file__',None)}
    config['sources']={s:sha(s) for s in sorted(sources)}
    frozen=args.output/'executed_sources';frozen.mkdir()
    for s in sources:shutil.copyfile(s,frozen/Path(s).name)
    write_json(args.output/'config.json',config)
    start=time.perf_counter();data={};summary={}
    for m,t,r in itertools.product((0,1),repeat=3):
        name=f'M{m}T{t}R{r}'
        collected=True
        row=score_arm(view,segments,logps,args.warmup,(m,t,r),collected)
        for k,v in row.items():data[name+'_'+k]=v
        summary[name]=dict(mean_nll=float(row['losses'].mean()),segment_nll=row['losses'].mean(1).tolist())
        print('FACTORIAL',name,summary[name]['mean_nll'],round(time.perf_counter()-start,1),flush=True)
    for recurrence in (0,1):
        name=f'original_delta_R{recurrence}'
        row=score_arm(control,segments,logps,args.warmup,(0,0,recurrence),True)
        for k,v in row.items():data[name+'_'+k]=v
        summary[name]=dict(mean_nll=float(row['losses'].mean()),segment_nll=row['losses'].mean(1).tolist())
        print('CONTROL',name,summary[name]['mean_nll'],flush=True)
    with view.configuration(True,True,True,shuffle=True):
        # score_arm enters another config; it does not alter shuffled payloads.
        row=score_arm(view,segments,logps,args.warmup,(1,1,1),False)
        for k,v in row.items():data['shuffled_memory_'+k]=v
        summary['shuffled_memory']=dict(mean_nll=float(row['losses'].mean()),segment_nll=row['losses'].mean(1).tolist())
    np.savez_compressed(args.output/'observations.npz',input_bytes=segments[:,args.warmup:-1],**data)
    write_json(args.output/'scores.json',summary)
    if not args.smoke:
        dyn={}
        for name,item in [('cgb',view),('same_weight_gdn',View(PlainModel(view.base),'plain')),
                          ('original_delta',control)]:
            dyn[name]=dynamics(item,segments,logps,args.warmup)
            write_json(args.output/(name+'_dynamics.json'),dyn[name])
            print('DYNAMICS',name,'done',round(time.perf_counter()-start,1),flush=True)
        write_json(args.output/'macro_downward.json',macro_downward(engine,initial))
    assert sha(args.checkpoint)==meta['sha256'] and sha(args.control)==control_meta['sha256']
    write_json(args.output/'completion.json',dict(seconds=time.perf_counter()-start,
        checkpoints_unchanged=True,new_capability_training=False,
        frozen_forward_battery=True,isolated_complete_cycle_learning=not args.smoke,
        diagnostic_complete_cycles=0 if args.smoke else 2,
        frozen_battery_external_memories_readonly=True,
        diagnostic_branch_memories_update_and_restore=not args.smoke))
    print('COMPLETE',str(args.output),flush=True)


if __name__=='__main__':main()
