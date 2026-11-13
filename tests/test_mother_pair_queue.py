import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ib_local import run_mother_pair_queue as queue


def setup_base(tmp_path, monkeypatch):
    base=tmp_path/'base'
    for mode in ('real','blocked'):
        source=base/f'pair_{mode}_navigating_1815/last.pt.gz'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'checkpoint')
        (source.parent/'config.json').write_text(json.dumps({'sources':{}}))
    validation=base/'validation/mother_validation.bin'
    validation.parent.mkdir(parents=True)
    validation.write_bytes(b'fixed')
    monkeypatch.setattr(queue,'BASE',base)
    return base


def test_queue_orders_two_arms_and_validates_after_both(tmp_path,monkeypatch):
    setup_base(tmp_path,monkeypatch)
    output=tmp_path/'queue'
    events=[]
    def fake_run(command, **kwargs):
        module=command[2]
        dest=Path(command[command.index('--output')+1])
        events.append(module)
        if module.endswith('run_mother_continuation'):
            dest.mkdir()
            (dest/'last.pt.gz').write_bytes(b'carried')
            (dest/'completed.json').write_text(json.dumps({'status':'complete','cycles':1,'final_cycle':1857}))
        else:
            dest.write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess,'run',fake_run)
    monkeypatch.setattr(queue.sys,'argv',['queue','--output',str(output),'--segments','1','--segment-cycles','1'])
    queue.main()
    state=json.loads((output/'status.json').read_text(encoding='utf-8'))
    assert state['status']=='complete'
    assert [r['mode'] for r in state['completed']]==['real','blocked']
    assert events==['scripts.ib_local.run_mother_continuation']*2+['scripts.ib_local.evaluate_mother_continuation']*2


def test_queue_does_not_run_control_if_real_fails(tmp_path,monkeypatch):
    setup_base(tmp_path,monkeypatch)
    output=tmp_path/'queue'
    called=[]
    def fail(command, **kwargs):
        called.append(command)
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(queue.subprocess,'run',fail)
    monkeypatch.setattr(queue.sys,'argv',['queue','--output',str(output),'--segments','1'])
    with pytest.raises(RuntimeError,match='exited 2'):
        queue.main()
    assert len(called)==1
    assert json.loads((output/'status.json').read_text(encoding='utf-8'))['status']=='failed'


def test_internal_control_keeps_existing_episodic_choice_enabled():
    assert queue.worker_modes('internal','observe') == [
        '--mode','real','--internal-rollout','observe','--allow-internal-mode-change']


def test_dsh_worker_sessions_are_stable_per_arm_and_forward_the_endpoint(tmp_path):
    a=queue.mother_context_args('dsh','http://127.0.0.1:8910',tmp_path,'real')
    b=queue.mother_context_args('dsh','http://127.0.0.1:8910',tmp_path,'observe')
    assert a[:2]==['--mother-context','dsh']
    assert a[-2:]==['--mother-harness-url','http://127.0.0.1:8910']
    assert a==queue.mother_context_args('dsh','http://127.0.0.1:8910',tmp_path,'real')
    assert a[a.index('--mother-session')+1]!=b[b.index('--mother-session')+1]
    assert queue.mother_context_args('segmented',None,tmp_path,'real')==['--mother-context','segmented']
    with pytest.raises(ValueError,match='requires'):
        queue.mother_context_args(None,'http://127.0.0.1:8910',tmp_path,'real')


def test_prune_only_superseded_owned_batches_not_sources_or_endpoints(tmp_path):
    output=tmp_path/'study'; output.mkdir()
    source=tmp_path/'source'/'last.pt.gz'; source.parent.mkdir(); source.write_bytes(b'original')
    prior=output/'real_01_01'/'last.pt.gz'; prior.parent.mkdir(); prior.write_bytes(b'old')
    status=dict(completed=[dict(output=str(prior.parent.resolve()),segment_end=True),{}])
    record=dict(cycles=8)
    queue.prune_batch_checkpoint(source,output,8,record,status)
    queue.prune_batch_checkpoint(prior,output,8,record,status)
    assert source.exists() and prior.exists()
    status['completed'][0]['segment_end']=False
    queue.prune_batch_checkpoint(prior,output,8,record,status)
    assert source.exists() and not prior.exists()
    assert len(status['retired_checkpoints'])==1
