import json

import pytest

from scripts.ib_local.event_journal import EventJournal, receipt


def make(path, cycle=10, sha='body', code='execution', config=None):
    return EventJournal(path, cycle=cycle, checkpoint_sha=sha, code_sha=code, config=config or {})


def test_durable_events_recover_without_advancing_complete_snapshot(tmp_path):
    a = make(tmp_path)
    row = dict(choice='reuse', utterance_hex='6162', metal_g=.001, updates=4)
    a.append(11, row, 'before', 'after')
    b = make(tmp_path)
    assert len(b.pending) == 1
    b.verify(b.pending[0], row, 'before', 'after')
    with pytest.raises(RuntimeError, match='diverged'):
        b.verify(b.pending[0], dict(row, utterance_hex='6163'), 'before', 'after')


def test_recovery_rejects_wrong_body_code_rng_and_changed_observation(tmp_path):
    a = make(tmp_path)
    a.append(11, dict(world={'start':10}), 'before', 'after')
    for kw in (dict(sha='other'), dict(code='other')):
        with pytest.raises(ValueError): make(tmp_path, **kw)
    with pytest.raises(RuntimeError):
        a.verify(make(tmp_path).pending[0], dict(world={'start':10}), 'wrong', 'after')


def test_crash_after_snapshot_before_journal_compaction_is_valid(tmp_path):
    a = make(tmp_path)
    a.append(11, dict(updates=4), 'before', 'after')
    config = {}
    a.bind(config, 11)
    b = make(tmp_path, cycle=11, sha='new-body', config=config)
    assert b.pending == []
    assert b.header['base_cycle'] == 11
    assert b.header['checkpoint_sha'] == 'new-body'


def test_only_torn_final_append_is_discarded(tmp_path):
    a = make(tmp_path)
    a.append(11, dict(updates=4), 'before', 'after')
    with a.path.open('ab') as stream: stream.write(b'{"cycle":12')
    b = make(tmp_path)
    assert len(b.pending) == 1
    data = b.path.read_text()
    lines = data.splitlines()
    item = json.loads(lines[-1]); item['row']['updates'] = 5
    b.path.write_text(lines[0]+'\n'+json.dumps(item)+'\n')
    with pytest.raises(ValueError, match='corrupt'): make(tmp_path)
