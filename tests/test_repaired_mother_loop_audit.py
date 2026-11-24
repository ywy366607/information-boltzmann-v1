import json

import pytest

from scripts.audit_repaired_mother_loop import ambient_score_key, analyze, read_committed


def test_ignore_incomplete_concurrent_log_tail(tmp_path):
    log = tmp_path / 'metrics.jsonl'
    log.write_bytes(b'{"cycle":3885}\n{"cycle":3886')
    assert read_committed(log) == [{'cycle': 3885}]
    with log.open('ab') as stream:
        stream.write(b'}\n')
    assert read_committed(log) == [{'cycle': 3885}, {'cycle': 3886}]


def test_match_actual_ambient_suffix_instead_of_raw_mother_length():
    row = lambda length: {'world': {'start': 123, 'anchor': 100, 'mother_bytes': length},
                          'external_bytes': 960}
    assert ambient_score_key(row(78)) == ambient_score_key(row(79))
    assert ambient_score_key(row(79)) != ambient_score_key(row(80))


def test_reject_gaps_between_saved_continuation_segments(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.audit_repaired_mother_loop.ROOT', tmp_path)
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    (first / 'metrics.jsonl').write_text(json.dumps({'cycle': 3885}) + '\n')
    (second / 'metrics.jsonl').write_text(json.dumps({'cycle': 3887}) + '\n')
    with pytest.raises(ValueError, match='overlap or have a gap'):
        analyze((first, second))
