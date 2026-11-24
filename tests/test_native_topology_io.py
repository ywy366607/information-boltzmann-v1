"""Persistence interface checks only, no fabricated capability task."""
import json
from scripts.native_topology_io import iter_complete_jsonl,install_runtime_scan


def test_stream_retains_every_complete_record_and_only_removes_torn_tail(tmp_path):
    path=tmp_path/'metrics.jsonl'
    rows=[dict(event=i,values=[i,2*i]) for i in range(7)]
    complete=b''.join((json.dumps(r)+'\n').encode() for r in rows)
    path.write_bytes(complete+b'{"unfinished":')
    assert list(iter_complete_jsonl(path))==rows
    assert path.read_bytes()==complete


def test_original_pending_journal_reader_remains_in_use(tmp_path):
    from types import SimpleNamespace
    calls=[]
    def original(path):calls.append(str(path));return ['original']
    runtime=SimpleNamespace(complete_jsonl=original)
    assert install_runtime_scan(runtime) is original
    assert runtime.complete_jsonl(tmp_path/'events.autonomous.pending.jsonl')==['original']
    assert calls==[str(tmp_path/'events.autonomous.pending.jsonl')]
