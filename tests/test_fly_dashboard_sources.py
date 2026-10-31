"""The dashboard must retain source counts and current segment freshness."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'fly_dashboard', Path(__file__).resolve().parents[1] / 'scripts/ib/dashboard/server.py')
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def test_score_sources_and_chunks_remain_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'RUNS', tmp_path)
    run = tmp_path / 'child'
    run.mkdir()
    row = {'trained': 24, 'scores': [2., 4., 99.], 'unigram': [6., 8., 1.],
           'sources': ['mother_reply', 'mother_reply', 'self_hearing']}
    (run / 'scores_c6.jsonl').write_text(json.dumps(row) + '\n{"partial":', encoding='utf-8')
    r = dashboard.windows('child')[0]
    assert r['chunk'] == 6 and r['n_tokens'] == 3
    assert r['by_source']['mother_reply'] == {'n': 2, 'nll': 3., 'unigram': 7.}
    assert r['by_source']['self_hearing']['nll'] == 99.
    assert r['by_source']['mother_call']['n'] == 0
    assert r['by_source']['mother_call']['nll'] is None


def test_active_segment_overrides_previous_completed_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'RUNS', tmp_path)
    run = tmp_path / 'child'
    run.mkdir()
    (run / 'progress.json').write_text(json.dumps(
        {'status': 'completed', 'bptt_train_tokens': 16, 'target_bptt_tokens': 16,
         'dopamine_phasic_sd': .3}), encoding='utf-8')
    (run / 'chunk_plan.json').write_text(json.dumps(
        {'status': 'running', 'start_tokens': 16, 'target_tokens': 32}), encoding='utf-8')
    r = dashboard.parse_status('child')['runtime']
    assert r['status'] == 'running' and r['target'] == 32
    assert r['trained'] == 16  # Log clock is not invented from the new target.
    assert r['neural']['dopamine_phasic_sd'] == pytest.approx(.3)
