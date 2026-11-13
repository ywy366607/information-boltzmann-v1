import copy
from unittest.mock import patch

import pytest

from scripts.ib_local.mother_context_harness import ContextCachedMother,ContextMotherWorld
from test_mother_reply_world import Ambient


class Session:
    def __init__(self):self.payloads=[]
    def post(self,url,**kwargs):
        payload=copy.deepcopy(kwargs['json']);self.payloads.append(payload)
        n=len(self.payloads)
        class Response:
            status_code=200
            def json(self):
                return dict(choices=[dict(message=dict(content=f'  Reply {n}.\n'),finish_reason='stop')],
                            usage=dict(prompt_tokens=200,prompt_cache_hit_tokens=160,
                                       prompt_cache_miss_tokens=40,completion_tokens=8))
        return Response()


class Delegate:
    def __init__(self):
        self.cache,self.calls={},0
        self.base_system,self.model='fixed instructions','model'
        self.temperature,self.max_tokens,self.retries=.7,120,1
        self.api_key,self.url='test-only','http://test.invalid'
        self.thinking_field,self.thinking_off=True,{'thinking':{'type':'disabled'}}
        class Requests:Session=Session
        self.requests=Requests
    def get_prompt_parts(self,transcript):return self.base_system,'fixed final guidance'


def make(**kwargs):
    with patch('scratch.claude_mother_world_20261105.ToddlerMother',Delegate):
        return ContextCachedMother(**kwargs)


def test_adjacent_requests_reuse_full_input_and_raw_generated_prefix():
    m=make();w=ContextMotherWorld(Ambient(),m,history=1)
    w.take(b'hello',64,5);first=m.session.payloads[0]['messages']
    w.take(b'why',64,5);second=m.session.payloads[1]['messages']
    assert second[:len(first)]==first
    assert second[len(first)]==dict(role='assistant',content='  Reply 1.\n')
    assert second[0]['content']=='fixed instructions\n\nfixed final guidance'
    assert second[-1]==dict(role='user',content='why')
    assert len(w.transcript)==2 and len(m.context)==4
    assert m.last_telemetry['provider_hit_rate']==.8


def test_rollover_is_segmented_bounded_and_keeps_actual_recent_pairs():
    m=make(segment_turns=3,anchor_turns=1);w=ContextMotherWorld(Ambient(),m,history=1)
    for word in [b'a',b'b',b'c',b'd',b'e',b'f',b'g']:w.take(word,64,5)
    assert m.turn_count==7 and m.rollovers==2
    assert len(m.context)<=8
    assert m.session.payloads[3]['messages'][1]['content']=='c'
    assert m.session.payloads[3]['messages'][-1]['content']=='d'


def test_world_roundtrip_preserves_next_request_and_usage(tmp_path):
    m=make(shared_cache=tmp_path/'cache.sqlite');w=ContextMotherWorld(Ambient(),m,history=1)
    w.take(b'hello',64,5);state=copy.deepcopy(w.state_dict())
    b=make(shared_cache=tmp_path/'cache.sqlite');restored=ContextMotherWorld(Ambient(),b,history=1)
    restored.load_state_dict(state)
    next_turn=w.transcript+[('child','next')]
    assert m.prepare(next_turn)[0]==b.prepare(next_turn)[0]
    assert restored.state_dict()==state
    w.take(b'next',64,5);restored.take(b'next',64,5)
    assert m.context==b.context
    assert b.meter['shared_cache_hits']==1
    assert b.meter['api_calls']==1  # Restored accounting; cache read adds no bill.


def test_ambient_failure_does_not_commit_context_and_retry_reuses_real_reply():
    class Broken(Ambient):
        def take(self,*a):raise RuntimeError('ambient failure')
    m=make();w=ContextMotherWorld(Broken(),m)
    with pytest.raises(RuntimeError,match='ambient failure'):w.take(b'hello',64,5)
    assert m.context==[] and m.turn_count==0 and w.transcript==[]
    w.base=Ambient();w.take(b'hello',64,5)
    assert m.turn_count==1 and len(m.session.payloads)==1


def test_api_failure_never_becomes_silence_or_a_fake_observation():
    m=make();w=ContextMotherWorld(Ambient(),m)
    def fail(*a,**k):raise RuntimeError('unavailable')
    m.session.post=fail
    with pytest.raises(RuntimeError,match='world not advanced'):w.take(b'hello',64,5)
    assert w.base.clock==0 and w.transcript==[] and m.context==[]


def test_actual_shared_cache_key_includes_all_generation_settings(tmp_path):
    m=make(shared_cache=tmp_path/'cache.sqlite');a=ContextMotherWorld(Ambient(),m)
    a.take(b'hello',64,5)
    other=make(shared_cache=tmp_path/'cache.sqlite');other.delegate.temperature=.2
    b=ContextMotherWorld(Ambient(),other);b.take(b'hello',64,5)
    assert other.meter['api_calls']==1 and other.meter['shared_cache_hits']==0


def test_restoring_legacy_world_cannot_keep_future_context():
    m=make();w=ContextMotherWorld(Ambient(),m)
    old=m.state_dict();old.pop('context_harness')
    w.take(b'hello',64,5)
    m.load_state_dict(old)
    assert m.context==[] and m.turn_count==0 and m.pending is None
    assert m.meter['api_calls']==0


def test_context_budget_drops_complete_old_pairs_without_truncating_child():
    m=make(max_context_chars=2048)
    previous=[('child','a'*800),('mother','b'*800),('child','c'*800),('mother','d'*800)]
    payload,_=m.prepare(previous+[('child','literal new question')])
    assert sum(len(v['content']) for v in payload['messages'])<=2048
    assert payload['messages'][-1]['content']=='literal new question'
    assert payload['messages'][1]['content']=='c'*800


def test_transient_retry_uses_identical_request_and_commits_only_once(monkeypatch):
    m=make();m.delegate.retries=2
    actual=m.session.post;attempts=[]
    def request(url,**kwargs):
        attempts.append(copy.deepcopy(kwargs['json']))
        if len(attempts)==1:
            class Response:status_code=503
            return Response()
        return actual(url,**kwargs)
    m.session.post=request
    monkeypatch.setattr('scripts.ib_local.mother_context_harness.time.sleep',lambda _:None)
    w=ContextMotherWorld(Ambient(),m);w.take(b'hello',64,5)
    assert attempts[0]==attempts[1]
    assert m.turn_count==1 and w.base.clock==1
    assert m.meter['api_calls']==2 and m.meter['api_failures']==1


def test_unrelated_branch_requires_world_restore_before_request():
    m=make();w=ContextMotherWorld(Ambient(),m)
    w.take(b'hello',64,5)
    with pytest.raises(ValueError,match='does not match'):
        m.prepare([('child','other'),('mother','unrelated answer'),('child','next')])
    assert len(m.session.payloads)==1
