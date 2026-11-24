import numpy as np
from scripts.ib_local.natural_text_world import NaturalTextWorld
from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld


def make(policy):
    raw=b'x'*100+b' saturn has rings. '+b'y'*300
    return NaturalTextWorld(ReplyTopicWorld(np.frombuffer(raw,np.uint8),window=96,buckets=16,per_bucket=2,seed=11),{},navigation_policy=policy)


def test_whole_query_control_and_restored_miss():
    world=make('inquire_terms_v1'); quiet=make('inquire_read_control')
    for w,tag,start in ((world,b'HIT saturn',107),(quiet,b'NO HIT',0)):
        a,b,info=w.take(b'SATURN ???',80,16)
        assert tag in bytes(a[:64]) and (len(a),len(b),info['start'])==(80,16,start)
    restored=make('stable_context_v1'); restored.load_state_dict(world.state_dict())
    for w in (world,restored):
        a,_,info=w.take(b'unfindable',80,16)
        assert b'NO HIT' in bytes(a) and info['start']==139 and w.base.cursor==171
