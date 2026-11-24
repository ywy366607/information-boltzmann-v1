"""Read-only, settlement-time measurement of a fixed finite-U revision.

This observer captures the actual pair of states returned by production revise.
Offline scoring changes queries only: it never invokes revise or retention.
"""
from contextlib import contextmanager
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.ib_local.causal_relation_program import Port
from scripts.ib_local.finite_use_value import score_scope


def fact_digest(fact):
    digest = hashlib.sha256()
    digest.update(f'{fact.kind.value}:{fact.observed}:{fact.encoded}'.encode())
    for key in ('source', 'action', 'outcome', 'mother'):
        tensor = getattr(fact, key)
        if tensor is not None:
            value = tensor.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_fact(fact):
    return replace(fact, **{key: getattr(fact, key).detach().cpu().clone()
                           for key in ('source', 'action', 'outcome', 'mother')
                           if torch.is_tensor(getattr(fact, key))})


def cpu_state(state):
    # score_state has already removed organization and linked variable uses.
    assert state.organization is None and not state.variable_uses
    fields = {key: value.detach().cpu().clone()
              for key, value in vars(state).items() if torch.is_tensor(value)}
    return replace(state, **fields,
                   constraints=tuple(cpu_fact(f) for f in state.constraints),
                   extra=tuple(cpu_fact(f) for f in state.extra))


class SettlementScorePort(nn.Module):
    """Exact dependencies of canonical MOTHER score_scope, at settlement."""
    def __init__(self, model):
        super().__init__()
        assert model.joint_relation_update_revision >= 4
        for name in ('encoder', 'long_predictor', 'mother_from_boundary', 'relation_basis'):
            setattr(self, name, copy.deepcopy(getattr(model, name)).cpu())
        self.joint_relation_update_revision = model.joint_relation_update_revision
        self.identity_preview_revision = model.identity_preview_revision
        self.joint_relation_update_schema = model.joint_relation_update_schema
        width = model.memory.source.shape[-1]
        empty = model.memory.source.new_empty((0, width)).cpu()
        self.memory = SimpleNamespace(source=empty, action=empty,
                                      mother_next=empty, internal_next=empty)


def known_problem(archive, identity):
    row = archive.db.execute(
        'SELECT signature,payload,version,anchor_episode FROM unknown_bindings WHERE id=?',
        (identity,)).fetchone()
    assert row is not None
    return dict(local_identity=identity, signature=row[0], payload=json.loads(row[1]),
                version=row[2], anchor_episode=row[3])


def common_fact_digests(snapshot, ids):
    """Read immutable pre-fork MOTHER facts without opening a live archive."""
    from scripts.ib_local.causal_relation_program import Evidence
    db = sqlite3.connect(Path(snapshot).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        schema = json.loads(db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0])
        offsets, offset = {}, 0
        for key in sorted(schema):
            spec = schema[key]
            size = int(np.prod(spec['shape'])) * np.dtype(spec['dtype']).itemsize
            offsets[key] = offset
            offset += size
        result = {}
        for identity in ids:
            if not identity.startswith('episode:'):
                continue
            row = db.execute('SELECT payload FROM episodes WHERE id=?',
                             (int(identity.split(':')[1]),)).fetchone()
            if row is None:
                continue
            def tensor(key):
                spec = schema[key]
                array = np.frombuffer(row[0], dtype=spec['dtype'],
                    count=int(np.prod(spec['shape'])), offset=offsets[key]).copy()
                return torch.from_numpy(array).reshape(1, -1)
            fact = Evidence(tensor('source'), tensor('action'), tensor('mother_next'),
                            Port.MOTHER, True, fact_id=identity)
            result[identity] = fact_digest(fact)
        return result
    finally:
        db.close()


@contextmanager
def capture_second_settlement(path, common_snapshot):
    """Observe the second real finite-U settlement, before its deferred step."""
    from scripts.ib_local import finite_use_value as finite
    original_observed, original_revise = finite.observed, finite.revise
    count, active, transitions = 0, False, []

    def revise(*args, **kwargs):
        updated = original_revise(*args, **kwargs)
        if active:
            transitions.append((args[0], updated, args[1]))
        return updated

    def observed(engine, receipt, slot, merged):
        nonlocal count, active
        count += 1
        active = count == 2
        try:
            comparisons, report = original_observed(engine, receipt, slot, merged)
        finally:
            active = False
        if count != 2:
            return comparisons, report
        assert report['committed'] and len(transitions) == 1
        before, after, incoming = transitions[0]
        assert before.evidence_bound or before.binding is not None
        assert before.organization is None and not before.variable_uses
        facts = receipt['finite']['facts']
        versions = tuple(p._version for p in engine.model.parameters())
        rng = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        writes = engine.model._causal_archive.db.total_changes
        with torch.no_grad():
            port = SettlementScorePort(engine.model)
            old, new = cpu_state(before), cpu_state(after)
            queries = tuple(cpu_fact(f) for f in facts)
            e = SimpleNamespace(model=port, relation_ridge=engine.relation_ridge)
            a, b = score_scope(e, old, queries), score_scope(e, new, queries)
            expected = a.new_tensor([c['before'] for c in comparisons])
            expected_after = a.new_tensor([c['after'] for c in comparisons])
            torch.testing.assert_close(a, expected, rtol=2e-5, atol=3e-6)
            torch.testing.assert_close(b, expected_after, rtol=2e-5, atol=3e-6)
            origin = common_fact_digests(common_snapshot, [f.fact_id for f in queries])
            capsule = dict(model=port, before=old, after=new, queries=queries,
                incoming=cpu_fact(incoming), relation_ridge=engine.relation_ridge,
                problem=known_problem(engine.model._causal_archive, before.binding.identity),
                prediction_event=receipt['prediction_event'],
                settlement_event=int(engine.autonomous_event_clock),
                settlement_update=int(engine.model.memory.clock),
                actual_U=sum(c['contribution'] for c in comparisons)/len(comparisons),
                diagonal_cpu=float((b-a).mean()),
                diagonal_max_error=float(torch.maximum((a-expected).abs(), (b-expected_after).abs()).max()),
                common_query_digests=origin,
                capture='actual production revise output; before deferred value optimizer step')
            torch.save(capsule, path)
        assert versions == tuple(p._version for p in engine.model.parameters())
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
        assert writes == engine.model._causal_archive.db.total_changes
        return comparisons, report

    finite.observed, finite.revise = observed, revise
    try:
        yield
        assert count == 2 and Path(path).exists()
    finally:
        finite.observed, finite.revise = original_observed, original_revise


@torch.no_grad()
def cross_score(capsule, donor):
    """Keep recipient modules and both revised states fixed; change Q only."""
    facts = donor['queries']
    reasons = []
    for fact in facts:
        if fact.kind != Port.MOTHER or not fact.observed or fact.encoded:
            reasons.append(f'{fact.fact_id}: incompatible historical MOTHER query')
        if capsule is not donor and donor['common_query_digests'].get(fact.fact_id) != fact_digest(fact):
            reasons.append(f'{fact.fact_id}: raw query not bound to common pre-fork evidence')
    # Both columns ask historical MOTHER predictions with explicit source/action.
    # An inverse unknown's local ID is never used to identify another problem.
    if reasons:
        return dict(applicable=False, reasons=reasons, value=None)
    engine = SimpleNamespace(model=capsule['model'], relation_ridge=capsule['relation_ridge'])
    before = score_scope(engine, capsule['before'], facts)
    after = score_scope(engine, capsule['after'], facts)
    return dict(applicable=True, value=float((after-before).mean()),
                per_query=[dict(query=f.fact_id, before=float(a), after=float(b),
                                contribution=float(b-a)) for f, a, b in zip(facts, before, after)],
                meaning='same historical MOTHER prediction questions; recipient actual R/Rprime and settlement modules fixed')


def four_cells(silence, generate, first_difference, gamma):
    capsules = dict(silence=silence, generate=generate)
    cells = {row: {column: cross_score(c, q) for column, q in capsules.items()}
             for row, c in capsules.items()}
    for name, c in capsules.items():
        value = cells[name][name]['value']
        assert value is not None and abs(value-c['actual_U']) < 1e-5
    deltas = {q: (None if any(cells[r][q]['value'] is None for r in capsules) else
                 first_difference+gamma*(cells['generate'][q]['value']-cells['silence'][q]['value']))
              for q in capsules}
    values = list(deltas.values())
    kind = ('queries_inapplicable' if any(v is None for v in values) else
            'both_prefer_generate' if all(v > 0 for v in values) else
            'both_prefer_silence' if all(v < 0 for v in values) else 'question_dependent_ordering')
    return dict(cells=cells, two_step_generate_minus_silence=deltas,
                classification=kind,
                same_problem_signature=silence['problem']['signature']==generate['problem']['signature'],
                problems={k:c['problem'] for k,c in capsules.items()},
                capture={k:{field:c[field] for field in ('prediction_event','settlement_event',
                    'settlement_update','actual_U','diagonal_cpu','diagonal_max_error')} for k,c in capsules.items()})
