"""Durable complete-individual saves and explicit restart journal boundaries.

A progress row describes the live process; a save receipt describes a successfully
replaced checkpoint. They are deliberately separate. Restarting retains old logs
and tags new rows with an attempt identity rather than counting a replayed suffix
as new, independently stored experience.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Mapping
import uuid

import torch

POSITION_KEYS = ('step', 'cursor', 'val_cursor', 'next_eval',
                 'events', 'optimizer_updates', 'pending_gradient_events')


class CheckpointWriteError(OSError):
    """A checkpoint or its receipt was not successfully committed."""


def checkpoint_position(payload: Mapping) -> dict:
    learner = payload['learner']
    return {**{key: int(payload[key]) for key in ('step', 'cursor', 'val_cursor', 'next_eval')},
            'fresh_training_tokens': int(payload['cursor']) - 1,
            'events': int(learner['events']),
            'optimizer_updates': int(learner['optimizer_updates']),
            'pending_gradient_events': int(learner['pending'])}


def _replace_with_retry(temporary: Path, destination: Path, *, attempts: int,
                        delay: float) -> None:
    if attempts < 1 or delay < 0:
        raise ValueError('At least one replacement attempt and nonnegative delay required')
    for attempt in range(attempts):
        try:
            os.replace(temporary, destination)
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def save_complete_checkpoint(path: Path, payload: Mapping, *, attempt_id: str,
                             replace_attempts: int = 50,
                             retry_delay: float = .05) -> dict:
    """Return a receipt only after checkpoint and JSON receipt replacements succeed.

The receipt identifier is embedded in the actual serialized payload, allowing a
later CPU load to verify provenance. The sidecar adds the final byte count and
mtime; no additional multi-gigabyte hash/read pass is needed at every save. A
failed temporary file is retained for recovery, and the previous destination is
never removed first.
    """
    path = Path(path)
    checkpoint_id = uuid.uuid4().hex
    position = checkpoint_position(payload)
    receipt = {'protocol': 'complete_individual_checkpoint_v1',
               'checkpoint_id': checkpoint_id, 'attempt_id': attempt_id,
               'path': str(path.resolve()), 'position': position,
               'serialized_at_unix': time.time()}
    serialized = dict(payload)
    serialized['checkpoint_receipt'] = dict(receipt)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{checkpoint_id}.tmp')
    stage = 'serialization'
    try:
        with temporary.open('wb') as handle:
            torch.save(serialized, handle)
            handle.flush()
            os.fsync(handle.fileno())
        stage = 'checkpoint replacement'
        _replace_with_retry(temporary, path, attempts=replace_attempts, delay=retry_delay)
        stat = path.stat()
        receipt.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                       committed_at_unix=time.time())
        receipt_path = path.with_name(f'{path.name}.receipt.json')
        receipt_temporary = receipt_path.with_name(f'{receipt_path.name}.{checkpoint_id}.tmp')
        stage = 'receipt serialization'
        with receipt_temporary.open('w', encoding='utf-8') as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        stage = 'receipt replacement'
        _replace_with_retry(receipt_temporary, receipt_path,
                            attempts=replace_attempts, delay=retry_delay)
        return receipt
    except (OSError, RuntimeError) as error:
        raise CheckpointWriteError(f'{stage} failed for {path}; no successful save receipt: {error}') from error


def require_saved_position(receipt: Mapping | None, position: Mapping) -> None:
    """A terminal success state requires a receipt for this exact live position."""
    if receipt is None:
        raise CheckpointWriteError('Terminal success requires a committed complete checkpoint')
    mismatches = [key for key in POSITION_KEYS
                  if receipt.get('position', {}).get(key) != position.get(key)]
    if mismatches:
        raise CheckpointWriteError(f'Last saved checkpoint trails live state: {mismatches}')


def _json_rows(path: Path):
    if not path.exists():
        return
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                yield number, row


def inspect_resume_gap(output: Path, position: Mapping) -> dict:
    """Report retained rows beyond the loaded physical individual; never delete logs.

Legacy rows without attempt IDs cannot be uniquely credited to a restart. This
reports that ambiguity explicitly; new rows carry exact source ranges and IDs.
    """
    result = {'protocol': 'retained_restart_boundary_v1', 'checkpoint_position': dict(position),
              'files': {}, 'logs_preserved': True}
    for name in ('metrics.jsonl', 'lifelong_evaluation.jsonl'):
        ahead, positions, attempts, legacy, repeated, seen = [], [], set(), 0, 0, set()
        replayed_intervals, source_intervals = 0, set()
        for line, row in _json_rows(Path(output) / name):
            fresh = row.get('fresh_training_tokens', row.get('bptt_train_tokens'))
            if fresh is None:
                continue
            key = (row.get('attempt_id'), row.get('step'), fresh,
                   row.get('fresh_validation_cursor'))
            repeated += int(key in seen)
            seen.add(key)
            if all(row.get(key) is not None for key in ('source_sha256', 'source_start', 'source_end')):
                interval = (row['source_sha256'], row['source_start'], row['source_end'])
                replayed_intervals += int(interval in source_intervals)
                source_intervals.add(interval)
            if row.get('attempt_id'):
                attempts.add(row['attempt_id'])
            else:
                legacy += 1
            if (int(fresh) > position['fresh_training_tokens'] or
                    (row.get('events') is not None and int(row['events']) > position['events']) or
                    (row.get('fresh_validation_cursor') is not None and
                     int(row['fresh_validation_cursor']) > position['val_cursor'])):
                ahead.append(line)
                positions.append(int(fresh))
        result['files'][name] = {'rows_ahead_of_checkpoint': len(ahead),
                                'first_ahead_line': min(ahead, default=None),
                                'last_ahead_line': max(ahead, default=None),
                                'fresh_range_ahead': [min(positions), max(positions)] if positions else None,
                                'legacy_rows_without_attempt': legacy,
                                'repeated_record_positions': repeated,
                                'replayed_source_interval_rows': replayed_intervals,
                                'prior_attempt_ids': sorted(attempts)}
    progress_path = Path(output) / 'progress.json'
    try:
        old_progress = json.loads(progress_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        old_progress = {}
    result['previous_progress'] = {key: old_progress.get(key) for key in
                                   ('status', 'step', 'fresh_training_tokens', 'events', 'optimizer_updates')}
    result['uncheckpointed_fresh_suffix'] = max(
        0, int(old_progress.get('fresh_training_tokens', 0)) - position['fresh_training_tokens'])
    result['uncheckpointed_event_suffix'] = max(
        0, int(old_progress.get('events', 0)) - position['events'])
    result['uncheckpointed_optimizer_update_suffix'] = max(
        0, int(old_progress.get('optimizer_updates', 0)) - position['optimizer_updates'])
    return result


def append_attempt_journal(output: Path, *, attempt_id: str, position: Mapping,
                           source_checkpoint: Path | None,
                           source_receipt: Mapping | None,
                           source_gap: Mapping | None = None) -> dict:
    result = {'protocol': 'continuous_individual_attempt_v1', 'attempt_id': attempt_id,
              'started_at_unix': time.time(), 'restored_position': dict(position),
              'source_checkpoint': None if source_checkpoint is None else str(Path(source_checkpoint).resolve()),
              'parent_checkpoint_id': None if source_receipt is None else source_receipt.get('checkpoint_id'),
              'parent_attempt_id': None if source_receipt is None else source_receipt.get('attempt_id'),
              'recovery': inspect_resume_gap(output, position),
              'source_recovery': None if source_gap is None else dict(source_gap),
              'freshness_contract': 'new rows identify source intervals; replayed source intervals are not independent fresh experience'}
    with (Path(output) / 'resume_journal.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(result, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return result


def committed_log_rows(output: Path, name: str, receipt: Mapping) -> dict:
    """Select rows on the actually saved restart branch, deduplicating source ranges.

Only the committed prefix of each ancestor attempt belongs to the restored
individual. Rows in a failed attempt's later suffix remain in the raw file but
are excluded. Legacy rows may be positional evidence without verified attempt
provenance; that limitation is returned explicitly rather than hidden.
    """
    output = Path(output)
    journals = {row['attempt_id']: row for _, row in _json_rows(output / 'resume_journal.jsonl')
                if row.get('attempt_id')}
    allowed, legacy_position = {}, None
    current, position = receipt['attempt_id'], receipt['position']
    visited = set()
    while current is not None:
        if current in visited:
            raise ValueError('Restart attempt ancestry contains a cycle')
        visited.add(current)
        allowed[current] = position
        journal = journals.get(current)
        if journal is None:
            break
        parent = journal.get('parent_attempt_id')
        position = journal['restored_position']
        if parent is None:
            if journal.get('source_checkpoint') is not None:
                legacy_position = position
            break
        current = parent
    selected, excluded, duplicate_intervals = {}, 0, 0
    for line, row in _json_rows(output / name):
        attempt = row.get('attempt_id')
        limit = legacy_position if attempt is None else allowed.get(attempt)
        fresh = row.get('fresh_training_tokens', row.get('bptt_train_tokens'))
        if limit is None or fresh is None or int(fresh) > limit['fresh_training_tokens']:
            excluded += 1
            continue
        if ((row.get('events') is not None and int(row['events']) > limit['events']) or
                (row.get('fresh_validation_cursor') is not None and
                 int(row['fresh_validation_cursor']) > limit['val_cursor'])):
            excluded += 1
            continue
        if all(row.get(key) is not None for key in ('source_sha256', 'source_start', 'source_end')):
            key = ('source', row['source_sha256'], row['source_start'], row['source_end'])
        else:
            key = ('legacy_position', row.get('step'), fresh, row.get('fresh_validation_cursor'))
        duplicate_intervals += int(key in selected)
        selected[key] = (line, row)
    rows = [row for _, row in sorted(selected.values())]
    legacy_selected = sum(row.get('attempt_id') is None for row in rows)
    return {'protocol': 'committed_restart_branch_rows_v1', 'checkpoint_id': receipt['checkpoint_id'],
            'rows': rows, 'selected_rows': len(rows), 'excluded_raw_rows': excluded,
            'deduplicated_source_rows': duplicate_intervals,
            'unattributed_legacy_rows': legacy_selected,
            'legacy_provenance_verified': legacy_selected == 0,
            'allowed_attempt_ids': list(allowed)}
