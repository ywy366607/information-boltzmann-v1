"""Full byte recovery, provenance and overwrite protection; no learning claim."""
import json
from pathlib import Path

import pytest

from scripts.ib.fly_interface.mengli_state_archive import save_point, restore_point


def test_recovery_preserves_full_bytes_context_and_requires_fresh_review(tmp_path):
    root = tmp_path / 'source'; root.mkdir()
    (root / 'information_boltzmann').mkdir()
    (root / 'information_boltzmann/model.py').write_text('original code', encoding='utf8')
    run = tmp_path / 'run'; run.mkdir()
    storage = tmp_path / 'weights'; storage.mkdir()
    checkpoint = storage / 'last.pt'
    original = b'complete optimizer brain world and pending credit' * 100
    checkpoint.write_bytes(original)
    (run / 'config.json').write_text('{"clock":16}', encoding='utf8')
    (run / 'mother.jsonl').write_text('actual experience\n', encoding='utf8')
    (run / 'caregiver_presence.json').write_text('{"next_chunk":{"status":"consumed"}}', encoding='utf8')
    point = save_point(run, checkpoint, 100, 'before_change', root)
    assert save_point(run, checkpoint, 100, 'before_change', root) == point
    checkpoint.write_bytes(b'later possibly incorrect state')
    branch, branch_weights = tmp_path / 'branch', tmp_path / 'branch_weights'
    restore_point(point, branch, branch_weights)
    assert (branch_weights / 'last.pt').read_bytes() == original
    assert checkpoint.read_bytes() == b'later possibly incorrect state'
    assert (branch / 'mother.jsonl').read_text() == 'actual experience\n'
    assert not (branch / 'caregiver_presence.json').exists()
    assert (branch / 'STOP').exists()
    assert json.loads((branch / 'progress.json').read_text())['needs_caregiver_review']
    assert (point / 'runtime.zip').exists()
    with pytest.raises(ValueError, match='empty branch'):
        restore_point(point, branch, branch_weights)


def test_corrupted_recovery_is_never_published_as_last(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    cp = tmp_path / 'last.pt'; cp.write_bytes(b'the complete saved individual')
    point = save_point(run, cp, 10, 'candidate', tmp_path)
    manifest_path = point / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['checkpoint_sha256'] = 'invalid'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='checksum'):
        restore_point(point, tmp_path / 'branch', tmp_path / 'branch_weights')
    assert not (tmp_path / 'branch_weights/last.pt').exists()
