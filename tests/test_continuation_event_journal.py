"""Runner transaction/STOP tests; fake engine is an interface fixture only."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local import run_mother_continuation as runner
CONTINUATION_PARSER = runner.continuation_parser


def install(tmp_path, monkeypatch, *, fail_at=None, stop_at=None, cycles=5):
    args = CONTINUATION_PARSER().parse_args(['--output',str(tmp_path),
        '--cycles',str(cycles),'--resume-output','--validate-every','0',
        '--checkpoint-every','3','--stop-file',str(tmp_path/'STOP')])
    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = torch.nn.Linear(2,2)
    model.encoder = SimpleNamespace(out_features=2)
    opt = torch.optim.AdamW(model.parameters())
    initial = copy.deepcopy(model.state_dict())
    saves = []
    def prepare(*a, **kw):
        model.load_state_dict(initial)
        opt.state.clear()
        torch.manual_seed(125); random.seed(125); np.random.seed(125)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(125)
        e = SimpleNamespace(model=model, optimizer=opt, device=torch.device('cpu'), count=0)
        def step(directions):
            e.count += 1
            if e.count == fail_at: raise RuntimeError('interrupted event')
            for p in model.parameters(): p.grad = torch.full_like(p, float(directions.mean()))
            opt.step()
            if e.count == stop_at: (tmp_path/'STOP').write_text('finish this event')
            return dict(choice='silence', utterance_hex='', external_bytes=960, self_bytes=0,
                response_nll=float(model.weight.detach().sum()), world={'source':'text'})
        e.step = step
        return SimpleNamespace(engine=e, config=dict(world_source='text'), cycle=0,
            life_card=None,life_branch=None,source='fixture',source_sha='initial')
    monkeypatch.setattr(runner,'prepare_continuation',prepare)
    monkeypatch.setattr(runner,'continuation_parser',lambda:SimpleNamespace(parse_args=lambda:args))
    monkeypatch.setattr(runner,'save',lambda e,out,cycle,config:saves.append(cycle) or f'checkpoint-{cycle}')
    return saves


def test_periodic_save_keeps_all_rows_and_commits_final_boundary(tmp_path, monkeypatch):
    saves = install(tmp_path,monkeypatch)
    runner.main()
    assert saves == [3,5]
    from scripts.ib_local.event_journal import complete_jsonl
    rows = complete_jsonl(tmp_path/'metrics.jsonl')
    assert [r['cycle'] for r in rows] == list(range(1,6))
    assert rows[0]['runtime']['durable_event_cycle'] == 1
    assert rows[0]['runtime']['complete_checkpoint_cycle'] == 0
    assert rows[-1]['runtime']['complete_checkpoint_cycle'] == 5


def test_stop_during_event_publishes_complete_state_before_exit(tmp_path,monkeypatch):
    saves = install(tmp_path,monkeypatch,stop_at=1)
    runner.main()
    assert saves == [1]


def test_interrupted_runner_recovers_receipt_before_new_events(tmp_path,monkeypatch):
    saves = install(tmp_path,monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError,match='interrupted event'): runner.main()
    assert saves == []
    saves = install(tmp_path,monkeypatch)
    runner.main()
    # Recovered event 1 is committed, then new events 2..5 continue.
    assert saves == [1,4,5]
    from scripts.ib_local.event_journal import complete_jsonl
    assert [r['cycle'] for r in complete_jsonl(tmp_path/'metrics.jsonl')] == list(range(1,6))


def test_new_output_journal_anchors_its_published_snapshot_not_ancestor(tmp_path,monkeypatch):
    import json
    install(tmp_path,monkeypatch,fail_at=1)
    original = runner.prepare_continuation
    def prepare(*args,**kwargs):
        loaded = original(*args,**kwargs)
        loaded.source_sha = 'ancestor'
        loaded.checkpoint_sha = 'published-new-output'
        return loaded
    monkeypatch.setattr(runner,'prepare_continuation',prepare)
    with pytest.raises(RuntimeError,match='interrupted event'): runner.main()
    header = json.loads((tmp_path/'events.pending.jsonl').read_text().splitlines()[0])
    assert header['checkpoint_sha'] == 'published-new-output'
