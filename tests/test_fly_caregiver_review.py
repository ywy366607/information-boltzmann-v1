"""A sleeping or absent caregiver must not authorize another training segment."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'grow_mengli', Path(__file__).resolve().parents[1] / 'scripts/ib/fly_interface/grow_mengli.py')
grow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grow)


def prepare(tmp_path):
    checkpoint = tmp_path / 'last.pt'
    checkpoint.write_bytes(b'saved individual')
    now = datetime.now(timezone.utc)
    stat = checkpoint.stat()
    review = {'reviewer': 'Ling', 'agent_thread': 'this-thread', 'decision': 'Continue shared reading',
              'next_chunk': {'status': 'reviewed', 'start_tokens': 100, 'target_tokens': 200,
                             'checkpoint_mtime_ns': stat.st_mtime_ns, 'checkpoint_size': stat.st_size,
                             'expires_at': (now + timedelta(minutes=15)).isoformat()}}
    path = tmp_path / 'caregiver_presence.json'
    path.write_text(json.dumps(review), encoding='utf-8')
    return checkpoint, path, review, now


def test_agent_review_is_required_and_consumed_once(tmp_path):
    checkpoint, path, review, now = prepare(tmp_path)
    grow.claim_caregiver_review(tmp_path, checkpoint, 100, 200, now)
    assert json.loads(path.read_text())['next_chunk']['status'] == 'consumed'
    with pytest.raises(RuntimeError, match='does not match'):
        grow.claim_caregiver_review(tmp_path, checkpoint, 100, 200, now)
    path.unlink()
    with pytest.raises(RuntimeError, match='required'):
        grow.claim_caregiver_review(tmp_path, checkpoint, 100, 200, now)


def test_stale_review_cannot_launch(tmp_path):
    checkpoint, _, _, now = prepare(tmp_path)
    with pytest.raises(RuntimeError, match='expired'):
        grow.claim_caregiver_review(tmp_path, checkpoint, 100, 200, now + timedelta(minutes=16))


def test_checkpoint_metadata_follows_verified_archive(tmp_path):
    cp = tmp_path / 'last.pt'
    cp.write_bytes(b'complete state')
    point = tmp_path / 'recovery'
    point.mkdir()
    stat = cp.stat()
    (point / 'manifest.json').write_text(json.dumps({
        'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
        'trained': 200, 'checkpoint_sha256': 'verified-digest'}), encoding='utf-8')
    grow.publish_checkpoint_metadata(tmp_path, cp, point)
    metadata = json.loads((tmp_path / 'checkpoint_metadata.json').read_text())
    assert metadata['trained'] == 200 and metadata['mtime_ns'] == stat.st_mtime_ns
    cp.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='does not match'):
        grow.publish_checkpoint_metadata(tmp_path, cp, point)


@pytest.mark.parametrize('change', ['checkpoint', 'target', 'start'])
def test_review_is_bound_to_saved_state_and_budget(tmp_path, change):
    checkpoint, _, _, now = prepare(tmp_path)
    if change == 'checkpoint':
        checkpoint.write_bytes(b'a different saved individual')
    with pytest.raises(RuntimeError, match='does not match'):
        grow.claim_caregiver_review(tmp_path, checkpoint, 101 if change == 'start' else 100,
                                   201 if change == 'target' else 200, now)
