"""Bounded, prequential use of Lean-checked relation data.

This is a research-side adapter: it never changes model parameters or runs
candidate-provided source code. Checking examples is not a proof of causality.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from scripts.ib_local.executable_causal_relation import execute, parse_candidate
from scripts.ib_local.probe_executable_causal_relation import lean_source

ROOT = Path(__file__).resolve().parents[2]
LEAN_PROJECT = ROOT / 'lean' / 'CGBAxioms'


@dataclass(frozen=True)
class CheckedRelation:
    code: tuple
    inputs: int
    digest: str


def admit(raw, input_count, examples, *, timeout=30):
    """Check bounded data and specific executions with a fixed Lean template."""
    code = parse_candidate(raw, input_count)
    if type(examples) is not list or not 1 <= len(examples) <= 16:
        raise ValueError('one to sixteen check inputs required')
    rows = []
    for xs in examples:
        if type(xs) is not list or len(xs) != input_count:
            raise ValueError('wrong input shape')
        execute(code, xs)
        rows.append({'x': xs})
    # The existing generator uses precisely two inputs. Refuse other shapes
    # instead of silently checking a different program.
    if input_count != 2:
        raise ValueError('Lean template currently supports exactly two inputs')
    source = lean_source(code, rows)
    with tempfile.TemporaryDirectory(prefix='cgb_relation_') as tmp:
        file = Path(tmp) / 'Candidate.lean'
        file.write_text(source, encoding='utf-8')
        result = subprocess.run(['lake', 'env', 'lean', str(file)], cwd=LEAN_PROJECT,
                                capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise ValueError('Lean rejected bounded execution: ' + result.stdout + result.stderr)
    payload = json.dumps({'inputs': input_count, 'program': raw}, sort_keys=True,
                         separators=(',', ':')).encode()
    return CheckedRelation(code, input_count, hashlib.sha256(payload).hexdigest())


class RelationMemory:
    """Fixed capacity; score only after the matching prediction was issued."""
    def __init__(self, capacity=8):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('invalid capacity')
        self.capacity = capacity
        self.entries = {}
        self.pending = None

    def insert(self, relation):
        if not isinstance(relation, CheckedRelation):
            raise ValueError('relation must first pass Lean admission')
        if relation.digest not in self.entries and len(self.entries) >= self.capacity:
            raise ValueError('capacity reached; no automatic replacement policy')
        self.entries.setdefault(relation.digest, {'relation': relation, 'count': 0,
                                                  'absolute_error_sum': 0})

    def predict(self, digest, inputs):
        if self.pending is not None:
            raise ValueError('previous outcome still pending')
        entry = self.entries[digest]
        output = execute(entry['relation'].code, inputs)
        self.pending = (digest, tuple(inputs), output)
        return output

    def settle(self, observed):
        if self.pending is None or type(observed) is not int:
            raise ValueError('a pending prediction and integer outcome are required')
        digest, _, prediction = self.pending
        error = abs(prediction - observed)
        entry = self.entries[digest]
        entry['count'] += 1
        entry['absolute_error_sum'] += error
        self.pending = None
        return error


def revise_constant_relation(state, *, step_micro=20000, prechecked=None):
    """Propose one bounded constant edit using settled local loss feedback.

    This is a deliberately restricted update, not model-written Lean. The
    caller must install the checked result at the next observation boundary.
    """
    if type(step_micro) is not int or not 1 <= step_micro <= 100000:
        raise ValueError('bounded positive edit step required')
    code = state['code']
    if len(code) != 1 or code[0]['op'] != 'const':
        raise ValueError('constant relation required')
    feedback = state['feedback_sum']
    if not state['count'] or not math.isfinite(feedback):
        raise ValueError('actual settled finite feedback required')
    old = code[0]['arg']
    direction = (feedback > 0) - (feedback < 0)
    new = min(1000000, max(-1000000, old + direction * step_micro))
    if prechecked is None:
        candidate = admit([{'op':'const','arg':new}], 2, [[0,0],[1000000,128]])
    else:
        candidate = prechecked.get(new)
        if (not isinstance(candidate, CheckedRelation) or candidate.inputs != 2
                or len(candidate.code) != 1 or candidate.code[0].op != 'const'
                or candidate.code[0].arg != new):
            raise ValueError('matching prechecked program required')
    return candidate, {'from_micro':old, 'to_micro':new, 'feedback_sum':feedback,
                       'observations':state['count'], 'old_digest':state['digest'],
                       'new_digest':candidate.digest}
