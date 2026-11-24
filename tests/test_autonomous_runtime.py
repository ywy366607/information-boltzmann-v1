"""Mixed-clock crash recovery and STOP; fixtures exercise the actual owner."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local import autonomous_runtime as runtime
from test_continuation_event_journal import install


def setup(tmp_path, monkeypatch, *, fail_at=None, stop_at=None):
    saves = install(tmp_path, monkeypatch, cycles=2)
    from scripts.ib_local import run_mother_continuation as runner
    args = runner.continuation_parser().parse_args()
    loaded = runner.prepare_continuation()
    e = loaded.engine
    e.autonomous_event_clock = e.joint_internal_ticks = 0
    e.autonomous_ready = False
    e.autonomous_choices = 0
    e.autonomous_last_admission = None
    e.joint_proposal_budget = 2
    loaded.checkpoint_sha = 'initial'
    def dispatch(engine):
        event = engine.autonomous_event_clock+1
        if event == fail_at:
            raise RuntimeError('interrupted internal event')
        if event % 3 == 0:
            row = engine.step(torch.randn(2, 32))
            row['autonomous_event'] = dict(kind='external', event=event, new_world_evidence=True)
        else:
            # No optimizer, world input, or fake text on internal/release events.
            torch.rand(1)
            engine.joint_internal_ticks += int(event % 3 == 1)
            row = dict(autonomous_event=dict(kind='internal' if event % 3 == 1 else 'release',
                event=event, new_world_evidence=False, updates=0, external_bytes=0), updates=0, external_bytes=0, self_bytes=0,
                internal_ticks=engine.joint_internal_ticks)
        engine.autonomous_event_clock = event
        if event == stop_at:
            args.stop_file.write_text('stop after quiet event')
        return row
    monkeypatch.setattr(runtime, 'dispatch', dispatch)
    from scripts.ib_local import mother_checkpoint
    def save(engine, out, cycle, config):
        saves.append(dict(cycle=cycle, event=engine.autonomous_event_clock,
                          config=copy.deepcopy(config)))
        return f'checkpoint-{engine.autonomous_event_clock}'
    monkeypatch.setattr(mother_checkpoint, 'save', save)
    return args, loaded, saves


def test_owner_stops_and_saves_during_no_input_without_forcing_external(tmp_path, monkeypatch):
    args, loaded, saves = setup(tmp_path, monkeypatch, stop_at=1)
    runtime.run(args, loaded, None, tmp_path)
    assert saves[-1]['event'] == 1 and saves[-1]['cycle'] == 0
    assert loaded.engine.count == 0
    assert loaded.engine.optimizer.state == {}
    status = json.loads((tmp_path/'completed.json').read_text())
    assert status['status'] == 'stopped' and status['cycles'] == 0


@pytest.mark.parametrize('fail_at', [2, 5])
def test_mixed_journal_replays_quiet_and_release_before_real_world(tmp_path, monkeypatch, fail_at):
    args, loaded, saves = setup(tmp_path, monkeypatch, fail_at=fail_at)
    with pytest.raises(RuntimeError, match='internal event'):
        runtime.run(args, loaded, None, tmp_path)
    assert saves == []
    args, loaded, saves = setup(tmp_path, monkeypatch)
    runtime.run(args, loaded, None, tmp_path)
    assert loaded.engine.count == 2
    assert saves[-1]['cycle'] == 2 and saves[-1]['event'] == 6
    events = [json.loads(line) for line in (tmp_path/'autonomous_events.jsonl').read_text().splitlines()]
    assert [r['event'] for r in events] == list(range(1, 7))
    metrics = [json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert [r['cycle'] for r in metrics] == [1, 2]
    assert sum(r['updates'] for r in events) == 2


def test_stop_before_first_event_has_defined_status_and_no_world(tmp_path, monkeypatch):
    args, loaded, saves = setup(tmp_path, monkeypatch)
    args.stop_file.touch()
    runtime.run(args, loaded, None, tmp_path)
    assert saves == [] and loaded.engine.count == 0
    assert json.loads((tmp_path/'completed.json').read_text())['cycles'] == 0
