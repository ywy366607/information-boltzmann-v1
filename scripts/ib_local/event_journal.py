"""Durable natural-text event receipts between complete organism snapshots.

Recovery reexecutes recorded events from their complete boundary, checks actual
actions/observations and all RNG clocks, and refuses a changed execution graph.
This is a persistence protocol, never an extra learning/replay objective.
"""
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rng_digest():
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy(),
              [v.numpy() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    return hashlib.sha256(pickle.dumps(states, protocol=4)).hexdigest()


def execution_digest(root):
    root = Path(root)
    paths = sorted([*(root/'scripts/ib_local').glob('*.py'),
                    *(root/'information_boltzmann').glob('*.py')])
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def receipt(row):
    keys = ('choice', 'utterance_hex', 'self_bytes', 'external_bytes', 'world',
            'response_nll', 'heldout_pre_nll', 'heldout_post_nll', 'metal_g',
            'updates', 'short_loss', 'long_loss', 'outcome_error', 'actor_norm',
            'since_birth_updates', 'since_birth_external_bytes', 'since_birth_self_bytes')
    result = {k: row[k] for k in keys if k in row}
    joint = row.get('joint_language_search', {})
    result['language_search'] = {k: joint[k] for k in
        ('selected', 'distribution', 'candidates', 'relation_identity', 'relation_version') if k in joint}
    if 'autonomous_event' in row:
        result['autonomous_event'] = row['autonomous_event']
        result['internal_ticks'] = row.get('internal_ticks')
    if 'adaptive_dynamics' in row:
        result['adaptive_dynamics'] = row['adaptive_dynamics']
    return result


def complete_jsonl(path):
    """Remove only an unacknowledged torn final write; preserve complete rows."""
    path = Path(path)
    if not path.exists(): return []
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
        with path.open('r+b') as stream:
            stream.truncate(sum(map(len, lines)))
            stream.flush(); os.fsync(stream.fileno())
    return [json.loads(line) for line in lines]


class EventJournal:
    def __init__(self, directory, *, cycle, checkpoint_sha, code_sha, config,
                 filename='events.pending.jsonl', config_key='event_journal'):
        self.path = Path(directory)/filename
        self.config_key = config_key
        self.pending = []
        self.code_sha = code_sha
        self.header = dict(schema=1, base_cycle=cycle, checkpoint_sha=checkpoint_sha, code_sha=code_sha)
        self.head = digest(self.header)
        if self.path.exists():
            records = complete_jsonl(self.path)
            if not records:
                raise ValueError('event journal has no complete anchor')
            header, *events = records
            if header.get('schema') != 1 or header['base_cycle'] > cycle:
                raise ValueError('event journal boundary does not match checkpoint')
            head = digest(header)
            at_checkpoint = head if header['base_cycle'] == cycle else None
            for index, item in enumerate(events, header['base_cycle']+1):
                payload = {k:v for k,v in item.items() if k != 'sha'}
                if item['cycle'] != index or item['previous'] != head or item['sha'] != digest(payload):
                    raise ValueError('event journal chain is corrupt')
                head = item['sha']
                if index == cycle: at_checkpoint = head
                if index > cycle: self.pending.append(item)
            if header['base_cycle'] == cycle:
                if header['checkpoint_sha'] != checkpoint_sha:
                    raise ValueError('event journal anchor checkpoint differs')
            elif config.get(self.config_key, {}).get('head') != at_checkpoint:
                raise ValueError('complete checkpoint is not bound to this event journal')
            if self.pending:
                if header['code_sha'] != code_sha:
                    raise ValueError('pending events require their original execution code')
                self.header, self.head = header, head
                return
        self.reset(cycle, checkpoint_sha)

    def reset(self, cycle, checkpoint_sha):
        self.header = dict(schema=1, base_cycle=cycle, checkpoint_sha=checkpoint_sha,
                           code_sha=self.code_sha)
        self.head = digest(self.header)
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            stream.write((json.dumps(self.header)+'\n').encode())
            stream.flush(); os.fsync(stream.fileno())
        from scripts.ib_local.checkpoint_commit import publish_checkpoint
        publish_checkpoint(temporary, self.path)
        self.pending = []

    def append(self, cycle, row, before, after):
        item = dict(cycle=cycle, row=row, receipt=receipt(row), before_rng=before,
                    after_rng=after, previous=self.head)
        item['sha'] = digest(item)
        with self.path.open('ab') as stream:
            stream.write((json.dumps(item, ensure_ascii=True, allow_nan=False)+'\n').encode())
            stream.flush(); os.fsync(stream.fileno())
        self.head = item['sha']

    def verify(self, item, row, before, after):
        if before != item['before_rng'] or after != item['after_rng'] or receipt(row) != item['receipt']:
            raise RuntimeError(f"event recovery diverged at cycle {item['cycle']}; no new event allowed")

    def bind(self, config, cycle):
        config[self.config_key] = dict(schema=1, cycle=cycle, head=self.head, code_sha=self.code_sha)
