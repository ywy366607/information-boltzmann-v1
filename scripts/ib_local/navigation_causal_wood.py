"""Existing F learns to use action-addressed, channel-typed retained evidence.

The shared byte payload table supplies addresses, not causal values. A source's
old gain is an input to today's forecaster, never assigned as today's gain.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from scripts.ib_local.navigation_episodic_wood import EpisodicWoodEngine
from scripts.ib_local.run_causal_water_reply import WaterReplyEngine
from scripts.ib_local.navigation_settlement import complete_action_labels
from scripts.ib_local.navigation_value_scale import critic_output


FEATURES = 14


def pack_words(words, width, device):
    data=torch.zeros(len(words),width,dtype=torch.uint8,device=device)
    sizes=torch.tensor([len(w) for w in words],device=device)
    for i,w in enumerate(words):
        if w:data[i,:len(w)]=torch.tensor(list(w),dtype=torch.uint8,device=device)
    return data,sizes


def common_suffix(query, sizes, bank, bank_sizes):
    """Actual known utterance suffix, independent of generate/reuse labels."""
    pos=torch.arange(query.shape[1],device=query.device)
    q=query.gather(1,(sizes[:,None]-1-pos).clamp_min(0))
    k=bank.gather(1,(bank_sizes[:,None]-1-pos).clamp_min(0))
    valid=(pos[None,None]<sizes[:,None,None])&(pos[None,None]<bank_sizes[None,:,None])
    same=(q[:,None]==k[None])&valid
    return same.to(torch.int32).cumprod(-1,dtype=torch.int32).sum(-1)


def stratified_shuffle(values, options, ages, gain_valid, *, event_bytes, decay):
    """Same option/provenance and one evidence-timescale age band, no RNG."""
    age_band=(ages/(event_bytes/(1-decay))).floor().to(torch.long)
    group=torch.stack((options,age_band,gain_valid.to(torch.long)),dim=-1)
    _,ids,counts=torch.unique(group,dim=0,return_inverse=True,return_counts=True)
    order=ids.argsort(stable=True)
    start=counts.cumsum(0)-counts
    position=torch.arange(len(values),device=values.device)
    groups=ids[order]
    shift=(counts//2).clamp_min(1)
    other=start[groups]+(position-start[groups]-shift[groups])%counts[groups]
    index=torch.empty_like(order);index[order]=order[other]
    return values[index]


class CausalWoodEngine(EpisodicWoodEngine):
    def _utter(self):
        self._causal_words=[]
        return super()._utter()

    def _film(self, words, table):
        result=super()._film(words,table)
        self._causal_words.append(result[0])
        return result

    def _freeze_wood_bank(self):
        m=self.model.memory
        # Multi-utterance merges do not prove a single byte-address provenance.
        valid=complete_action_labels(m)&(m.count==1)
        valid&=(self.model.retrieval.word_count>0)|(m.wood_option==0)
        slots=torch.nonzero(valid).flatten()
        self._episodic_bank={name:getattr(m,name)[slots].detach().clone()
            for name in ('source','action','wood_target','wood_baseline','wood_option','feeling')}
        bank=self._episodic_bank
        bank['slots']=slots
        bank['age']=(m.clock-m.last_write[slots]).detach().clone()
        bank['words']=self.model.retrieval.words[slots].detach().clone()
        bank['sizes']=self.model.retrieval.word_count[slots].detach().clone()
        bank['valid']=torch.ones(len(slots),4,dtype=torch.bool,device=self.device)
        bank['valid'][:,3]=self.reward_label_valid(slots,wood=True)

    def _wood_replay_batch(self):
        result=super()._wood_replay_batch()
        if result is not None:
            slots=self._episodic_replay_slots
            m=self.model.memory
            known=m.causal_evidence_valid[slots]&(m.count[slots]==1)
            self._causal_replay_evidence=(m.causal_evidence[slots]*known[:,None]).detach().clone()
        return result

    def _training_batch(self, target):
        result=super()._training_batch(target)
        actual=self._causal_words[self._pending_record['choice']] if self._causal_words else b''
        words,sizes=pack_words([actual],self.model.retrieval.words.shape[1],self.device)
        self._causal_query_words,self._causal_query_sizes=words,sizes
        return result

    def _read_features(self, sources, actions, options, base, words, sizes):
        bank=self._episodic_bank
        b,n=len(options),len(bank['slots'])
        if not n:
            return sources.new_zeros(b,FEATURES),dict(weights=sources.new_zeros(b,0,4),
                slots=bank['slots'].new_empty(b,0),support=sources.new_zeros(b,4),
                shuffle=sources.new_zeros(b,FEATURES))
        enc=self.model.encoder
        qs,qa=enc(sources),enc(actions)
        ks,ka=enc(bank['source']),enc(bank['action'])
        score=(F.normalize(qs,dim=-1)@F.normalize(ks,dim=-1).T
            +F.normalize(qa-qs,dim=-1)@F.normalize(ka-ks,dim=-1).T)/2
        feeling=self.intero().expand(b,-1) if self._episodic_deciding else self._episodic_query_feeling[:b]
        score=score-(feeling[:,None]-bank['feeling'][None]).norm(dim=-1)/math.sqrt(feeling.shape[-1])
        suffix=common_suffix(words,sizes,bank['words'],bank['sizes'])
        valid=suffix>=4
        # Silence has no outgoing address; context distinguishes passive visits.
        silent=sizes==0
        valid=torch.where(silent[:,None],bank['sizes'][None]==0,valid)
        if not self._episodic_deciding:
            valid&=self._episodic_exclude[:b,None]!=bank['slots'][None]
        longest=suffix.masked_fill(~valid,-1).max(-1,keepdim=True).values
        valid&=(suffix==longest)|silent[:,None]
        step_bytes=self.response_bytes+self.holdout_bytes+self.max_utterance
        score=score/self.model.memory.temperature+bank['age'][None]/step_bytes*math.log(self.wood_decay)
        k=min(self.model.memory.topk,n)
        selected,ids=score.masked_fill(~valid,-1e4).topk(k,dim=-1)
        allowed=valid.gather(1,ids)[...,None]&bank['valid'][ids]
        weights=selected[...,None].expand(-1,-1,4).masked_fill(~allowed,-1e4).softmax(1)*allowed
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        old=bank['wood_target'];old_base=bank['wood_baseline']
        together=torch.cat((old,old_base),-1)
        support=allowed.sum(1).to(sources.dtype)/self.model.memory.topk
        age_weight=weights.sum(-1)/4
        log_age=(age_weight*torch.log1p(bank['age'][ids]/self.model.memory.long)).sum(1,keepdim=True)
        specificity=torch.where(silent[:,None],0.,longest.clamp_min(0).to(sources.dtype)/words.shape[1])
        def assemble(value):
            target=(weights*value[ids,:4]).sum(1)
            before=(weights*value[ids,4:]).sum(1)
            # Explicit support distinguishes an unavailable field from zero.
            return torch.cat((target,(base-before)*(support>0),support,log_age,specificity),-1)
        shuffled=None
        if self._episodic_deciding:
            shuffled=assemble(stratified_shuffle(together,bank['wood_option'],bank['age'],bank['valid'][:,3],
                event_bytes=step_bytes,decay=self.wood_decay))
        return assemble(together),dict(weights=weights,slots=bank['slots'][ids],support=support,shuffle=shuffled)

    def _residual(self,sources,actions,options):
        enc=self.model.encoder
        z,a=enc(sources),enc(actions)
        original=torch.cat((z,a-z,F.one_hot(options,3).to(z.dtype)),-1)
        if self._episodic_deciding:
            words,sizes=pack_words(self._causal_words+[b'']*(len(options)-len(self._causal_words)),
                self.model.retrieval.words.shape[1],self.device)
            base=self.model.wood_evidence.bases()
        else:
            words,sizes=self._causal_query_words,self._causal_query_sizes
            base=self._batch_baseline
        if self._episodic_deciding:
            features,packet=self._read_features(sources,actions,options,base,words,sizes)
        else:
            # Current example re-encodes only the bank observed before this
            # action. Historical examples use their actually observed read,
            # not a reconstruction from a future version of the memory.
            features,packet=self._read_features(sources[:1],actions[:1],options[:1],base[:1],words,sizes)
            if self._wood_replay is not None:
                features=torch.cat((features,self._causal_replay_evidence))
        head=self.model.kanli.outcome
        pre=head[0](original)
        inlet=self.model.kanli.causal_input
        def output(f):return head[2](head[1](pre+inlet(f)))[:,:4]
        prediction=output(features)
        if self._episodic_deciding:
            plain=output(torch.zeros_like(features))
            shuffled=output(packet['shuffle'])
            self._episodic_decision_read=dict(parametric=plain.detach().clone(),
                shuffled=shuffled.detach().clone(),correction=(prediction-plain).detach().clone(),
                slots=packet['slots'].detach().clone(),weights=packet['weights'][...,3].detach().clone(),
                channel_weights=packet['weights'].detach().clone(),support=packet['support'].detach().clone(),
                features=features.detach().clone(),query_sizes=sizes.detach().clone())
        return prediction

    @torch.no_grad()
    def _value(self,h,films):
        # Retain the existing calibrated outcome -> choice path. No direct
        # copying of historical gain to a policy logit, and no old NN subtraction.
        self._freeze_wood_bank();self._episodic_deciding=True
        try:values,inputs=super(EpisodicWoodEngine,self)._value(h,films)
        finally:self._episodic_deciding=False
        f,r=self._forecast,self._episodic_decision_read
        base,confidence=f['base'],f['confidence']
        plain=self._compose_used_prediction(base,r['parametric'],confidence)
        shuffled=self._compose_used_prediction(base,r['shuffled'],confidence)
        width=self.model.encoder.out_features
        blind=inputs.clone();blind[:,width:width+4]=plain
        blind_values=critic_output(self,blind).squeeze(-1)+blind[:,-2]
        available=len(self._draft_actions) if self._draft_actions else 1
        delta=torch.zeros_like(values)
        delta[:available]=self.precision()*((values-blind_values)[:available]-(values-blind_values)[:available].mean())
        f.update(episodic_offset=delta,episodic_plain_used=plain,episodic_shuffled_used=shuffled,
                 episodic_raw_shuffled=self._compose_prediction(base,r['shuffled']),
                 episodic_raw_plain=self._compose_prediction(base,r['parametric']))
        return (blind_values,blind) if self.episodic_wood_mode=='blocked' else (values,inputs)

    def step(self,*a,**kw):
        row=super().step(*a,**kw)
        choice=self._pending_record['choice'];r=self._episodic_decision_read
        row.update(causal_features=r['features'][choice].tolist(),
            causal_read_support=r['support'][choice].tolist(),
            causal_channel_read_weights=r['channel_weights'][choice].tolist(),
            causal_input_weight_norm=float(self.model.kanli.causal_input.weight.detach().norm()),
            causal_shuffled_prediction=self._forecast['episodic_raw_shuffled'][choice].tolist(),
            causal_plain_prediction=self._forecast['episodic_raw_plain'][choice].tolist())
        delivered=(self._forecast['episodic_plain_used'] if self.episodic_wood_mode=='blocked'
                   else self._forecast['used'])
        row['causal_delivered_prediction']=delivered[choice].tolist()
        row['causal_forecast_role']='shadow' if self.episodic_wood_mode=='blocked' else 'delivered'
        return row

    @torch.no_grad()
    def _store_label(self,result,choice,target):
        super()._store_label(result,choice,target)
        m,slot=self.model.memory,result['episode_slot']
        # The nonlinear F is not an exact evaluator of averaged read inputs.
        # Keep merged content for its legacy paths but not as an exact tag.
        m.causal_evidence_valid[slot]=not result['episode_merged']
        if not result['episode_merged']:
            m.causal_evidence[slot].copy_(self._episodic_decision_read['features'][choice])
        else:
            m.causal_evidence[slot].zero_()


def configure_causal_wood(engine,mode):
    if mode not in ('real','blocked'):raise ValueError(mode)
    engine.__class__=CausalWoodEngine
    if not hasattr(engine.model.kanli,'causal_input'):
        old=engine.model.kanli.outcome[0]
        # Construction must not consume the body's actor/teacher RNG stream.
        with torch.random.fork_rng(devices=[old.weight.device] if old.weight.is_cuda else []):
            inlet=nn.Linear(FEATURES,old.out_features,bias=False,device=old.weight.device)
        with torch.no_grad():inlet.weight.zero_()
        engine.model.kanli.causal_input=inlet
        engine.optimizer.add_param_group(dict(params=list(inlet.parameters()),lr=engine.optimizer.param_groups[0]['lr']))
        engine.params=[p for p in engine.model.parameters() if p.requires_grad]
        engine.groups=(*engine.groups[:2],[p for n,p in engine.model.named_parameters() if not n.startswith('body.')])
    if not hasattr(engine.model.memory,'causal_evidence'):
        memory=engine.model.memory
        memory.register_buffer('causal_evidence',torch.zeros(memory.capacity,FEATURES,device=engine.device))
        memory.register_buffer('causal_evidence_valid',torch.zeros(memory.capacity,dtype=torch.bool,device=engine.device))
    engine.episodic_wood_mode=mode;engine._episodic_deciding=False
    engine._episodic_exclude=torch.empty(0,dtype=torch.long,device=engine.device)
    engine._episodic_replay_slots=engine._episodic_exclude
    engine._causal_words=[]
    return engine
