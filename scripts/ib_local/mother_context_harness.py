"""Bounded, append-only caregiver context with observable provider KV reuse.

Keep the actual child's words and raw assistant output. No invented summary,
fixed reply or cache padding is introduced. Context advances only after the
world has delivered the real observation successfully.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import sqlite3
import time

from scripts.ib_local.mother_reply_world import MotherReplyWorld, StrictToddlerMother


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':')).encode('utf-8')).hexdigest()


def native_messages(transcript):
    return [dict(role='assistant' if who=='mother' else 'user',content=str(text))
            for who,text in transcript]


class ContextCachedMother(StrictToddlerMother):
    def __init__(self, *, segment_turns=16, anchor_turns=4, max_context_chars=12000, **kwargs):
        super().__init__(**kwargs)
        if not 1<=segment_turns<=64 or not 0<=anchor_turns<=8 or max_context_chars<2048:
            raise ValueError('invalid bounded caregiver context')
        self.segment_turns,self.anchor_turns=segment_turns,anchor_turns
        self.max_context_chars=max_context_chars
        self.context=[]
        self.segment_count=0
        self.turn_count=0
        self.rollovers=0
        self.pending=None
        self.response_cache=OrderedDict()
        self.meter=dict(api_calls=0,api_successes=0,api_failures=0,
                        local_cache_hits=0,shared_cache_hits=0,
                        prompt_tokens=0,cache_hit_tokens=0,cache_miss_tokens=0,
                        completion_tokens=0,api_seconds=0.,unknown_usage_calls=0)
        self.last_telemetry={}
        self.session=self.delegate.requests.Session()

    def prepare(self,transcript):
        if not transcript or transcript[-1][0]=='mother':
            raise ValueError('a request must end with the actual child turn')
        stable,tail=self.delegate.get_prompt_parts(transcript)
        system=stable+ ('\n\n'+tail if tail else '')
        base=copy.deepcopy(self.context)
        initial=not base
        if initial:
            base=native_messages(transcript[:-1])[-2*self.anchor_turns:] if self.anchor_turns else []
        elif len(transcript)>=3:
            # Match the last delivered child/answer pair, not a topic guess.
            previous=native_messages(transcript[-3:-1])
            committed=[dict(role=v['role'],content=v.get('delivered',v['content'])) for v in base[-2:]]
            if previous!=committed:
                raise ValueError('caregiver context does not match the actual continuing world')
        current=dict(role='user',content=str(transcript[-1][1]))
        size=lambda a:len(system)+sum(len(v['content']) for v in a)+len(current['content'])
        rollover=(self.segment_count>=self.segment_turns or size(base)>self.max_context_chars)
        count=self.segment_count
        if rollover:
            base=base[-2*self.anchor_turns:] if self.anchor_turns else []
            count=0
        # Drop complete oldest pairs only; never silently truncate literal words.
        while base and size(base)>self.max_context_chars:
            base=base[2:]
        if size(base)>self.max_context_chars:
            raise ValueError('single caregiver request exceeds context character budget')
        messages=[dict(role='system',content=system)]
        messages += [dict(role=v['role'],content=v['content']) for v in base]
        messages.append(current)
        payload=dict(model=self.delegate.model,messages=messages,stream=False,
                     max_tokens=self.delegate.max_tokens,temperature=self.delegate.temperature)
        if self.delegate.thinking_field:
            payload.update(self.delegate.thinking_off)
        receipt=dict(base=base,current=current,count=count,rollover=rollover,
                     transcript_key=fingerprint(transcript),request_key=fingerprint(payload))
        return payload,receipt

    def _shared_get(self,key):
        if self.shared_cache:
            with sqlite3.connect(self.shared_cache,timeout=30) as db:
                row=db.execute('SELECT reply FROM replies WHERE key=?',(key,)).fetchone()
            if row:
                value=json.loads(row[0])
                if not isinstance(value,dict) or value.get('schema')!=1:
                    raise ValueError('invalid context-harness response cache record')
                return value
        return None

    def _remember(self,key,value):
        if self.shared_cache:
            with sqlite3.connect(self.shared_cache,timeout=30) as db:
                db.execute('INSERT OR IGNORE INTO replies (key,reply) VALUES (?,?)',
                           (key,json.dumps(value,ensure_ascii=False)))
                db.execute('DELETE FROM replies WHERE id IN '
                           '(SELECT id FROM replies ORDER BY id DESC LIMIT -1 OFFSET ?)',
                           (self.shared_limit,))
                value=json.loads(db.execute('SELECT reply FROM replies WHERE key=?',(key,)).fetchone()[0])
        self.response_cache[key]=copy.deepcopy(value)
        self.response_cache.move_to_end(key)
        while len(self.response_cache)>self.max_cache:
            self.response_cache.popitem(last=False)
        return value

    def _request(self,payload):
        last=None
        for attempt in range(self.delegate.retries):
            started=time.perf_counter()
            self.meter['api_calls']+=1
            self.delegate.calls+=1
            try:
                response=self.session.post(self.delegate.url,json=payload,
                    headers={'Authorization':'Bearer '+self.delegate.api_key,
                             'Content-Type':'application/json'},timeout=(5,20))
                if response.status_code!=200:
                    if response.status_code not in (408,429,500,502,503,504):
                        raise ValueError(f'caregiver HTTP {response.status_code}; request not committed')
                    raise RuntimeError(f'caregiver HTTP {response.status_code}')
                data=response.json()
                raw=(data.get('choices') or [{}])[0].get('message',{}).get('content')
                if not isinstance(raw,str) or not raw.strip():
                    raise RuntimeError('caregiver returned no usable text')
                if len(raw.encode('utf-8'))>8192:
                    raise ValueError('caregiver response exceeds bounded payload')
                usage=data.get('usage') or {}
                hit=usage.get('prompt_cache_hit_tokens')
                if hit is None:
                    hit=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0
                prompt=int(usage.get('prompt_tokens') or
                           (int(hit)+int(usage.get('prompt_cache_miss_tokens') or 0)))
                miss=usage.get('prompt_cache_miss_tokens')
                miss=int(miss) if miss is not None else max(0,prompt-int(hit))
                met=dict(prompt_tokens=prompt,cache_hit_tokens=int(hit),cache_miss_tokens=miss,
                         completion_tokens=int(usage.get('completion_tokens') or 0))
                if not usage:self.meter['unknown_usage_calls']+=1
                for k,v in met.items():self.meter[k]+=v
                self.meter['api_successes']+=1
                return dict(schema=1,raw=raw,reply=raw.strip(),usage=met,
                            request_id=str(data.get('id','')),finish_reason=(data.get('choices') or [{}])[0].get('finish_reason'))
            except ValueError:
                self.meter['api_failures']+=1
                raise
            except Exception as error:
                self.meter['api_failures']+=1
                last=type(error).__name__
                if attempt+1<self.delegate.retries:time.sleep(min(2**attempt,8))
            finally:
                self.meter['api_seconds']+=time.perf_counter()-started
        raise RuntimeError(f'caregiver request failed ({last}); world not advanced')

    def reply(self,transcript):
        started=time.perf_counter()
        if self.pending is not None:
            if self.pending['transcript_key']!=fingerprint(transcript):
                raise RuntimeError('previous caregiver event has not committed')
            return self.pending['answer']['reply']
        payload,receipt=self.prepare(transcript)
        key='context-v1:'+receipt['request_key']
        answer=self.response_cache.get(key)
        source='local'
        if answer is not None:
            self.meter['local_cache_hits']+=1
        else:
            answer=self._shared_get(key)
            source='shared'
            if answer is not None:self.meter['shared_cache_hits']+=1
            else:
                source='api'
                answer=self._request(payload)
        billed_usage=answer['usage'] if source=='api' else None
        answer=self._remember(key,answer)
        self.pending=dict(receipt,answer=answer)
        denom=self.meter['cache_hit_tokens']+self.meter['cache_miss_tokens']
        self.last_telemetry=dict(schema=1,source=source,turn=self.turn_count+1,
            rollover=receipt['rollover'],messages=len(payload['messages']),
            input_chars=sum(len(v['content']) for v in payload['messages']),
            request_key=receipt['request_key'],
            request_id=answer.get('request_id'),finish_reason=answer.get('finish_reason'),
            latency_seconds=time.perf_counter()-started,
            api_usage=billed_usage,
            cumulative=copy.deepcopy(self.meter),
            provider_hit_rate=self.meter['cache_hit_tokens']/denom if denom else None)
        return answer['reply']

    def commit_turn(self):
        if self.pending is None:raise RuntimeError('no caregiver observation to commit')
        p=self.pending
        self.context=p['base']+[p['current'],dict(role='assistant',content=p['answer']['raw'],
                                                delivered=p['answer']['reply'])]
        self.segment_count=p['count']+1
        self.rollovers+=int(p['rollover'])
        self.turn_count+=1
        self.pending=None

    def state_dict(self):
        return dict(super().state_dict(),context_harness=dict(schema=1,
            settings=dict(segment_turns=self.segment_turns,anchor_turns=self.anchor_turns,
                          max_context_chars=self.max_context_chars),
            context=copy.deepcopy(self.context),segment_count=self.segment_count,
            turn_count=self.turn_count,rollovers=self.rollovers,
            response_cache=copy.deepcopy(list(self.response_cache.items())),
            meter=copy.deepcopy(self.meter),last_telemetry=copy.deepcopy(self.last_telemetry)))

    def load_state_dict(self,state):
        super().load_state_dict(state)
        h=state.get('context_harness')
        if h is None:
            # Explicit migration, including restoration into an already used
            # object: never retain a future segment from a different branch.
            self.context=[]
            self.segment_count=self.turn_count=self.rollovers=0
            self.pending=None
            self.response_cache.clear()
            self.meter={k:0. if k=='api_seconds' else 0 for k in self.meter}
            self.last_telemetry={}
            return
        expected=dict(segment_turns=self.segment_turns,anchor_turns=self.anchor_turns,
                      max_context_chars=self.max_context_chars)
        if h.get('schema')!=1 or h['settings']!=expected:
            raise ValueError('incompatible caregiver context harness')
        if len(h['context'])>2*(self.segment_turns+self.anchor_turns) or len(h['response_cache'])>self.max_cache:
            raise ValueError('saved caregiver state exceeds fixed budget')
        self.context=copy.deepcopy(h['context'])
        self.segment_count,self.turn_count,self.rollovers=(int(h[k]) for k in
                                                        ('segment_count','turn_count','rollovers'))
        self.response_cache=OrderedDict(copy.deepcopy(h['response_cache']))
        self.meter=copy.deepcopy(h['meter'])
        self.last_telemetry=copy.deepcopy(h['last_telemetry'])
        self.pending=None


class ContextMotherWorld(MotherReplyWorld):
    def take(self,*args,**kwargs):
        response,heldout,info=super().take(*args,**kwargs)
        self.mother.commit_turn()
        return response,heldout,dict(info,mother_harness=copy.deepcopy(self.mother.last_telemetry))
