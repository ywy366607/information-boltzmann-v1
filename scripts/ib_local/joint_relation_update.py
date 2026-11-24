"""Bounded shared-variable dependencies and legal-overlap reconciliation."""
from __future__ import annotations

from dataclasses import replace
import torch


def configure_joint_relation_update(engine, *, revision=2):
    if revision not in (1,2,3,4):
        raise ValueError('unknown joint relation implementation revision')
    if not hasattr(engine.model, 'relation_basis'):
        raise ValueError('joint update requires the shared mechanism basis')
    archive = getattr(engine.model, '_causal_archive', None)
    if archive is None or not hasattr(engine, 'shared_unknown_schema'):
        raise ValueError('joint update requires persistent shared questions')
    archive.db.executescript('''
        CREATE TABLE IF NOT EXISTS relation_variables (
            relation_id INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
            unknown_id TEXT NOT NULL REFERENCES unknown_bindings(id) ON DELETE CASCADE,
            port TEXT NOT NULL, generation INTEGER NOT NULL, seen_version INTEGER NOT NULL,
            touched INTEGER NOT NULL, PRIMARY KEY(relation_id,unknown_id));
        CREATE INDEX IF NOT EXISTS variable_dependents ON relation_variables(unknown_id,relation_id);
    ''')
    engine.joint_relation_update_schema = engine.model.joint_relation_update_schema = 1
    engine.joint_relation_update_revision = engine.model.joint_relation_update_revision = revision
    if revision>=3:
        from scripts.ib_local.relation_evidence import configure_support_ids
        configure_support_ids(engine)


def configure_reference_write_cap(engine, config, requested=None):
    """Keep the private write quota fixed across continuation boundaries."""
    saved = config.get('relation_reference_write_cap')
    if saved is not None and requested is not None and saved != requested:
        raise ValueError('cannot change the saved relation reference write cap')
    cap = saved if saved is not None else requested
    if cap is None:
        return
    if (getattr(engine.model, 'joint_relation_update_schema', None) != 1 or
            cap not in (6, 7, 8) or cap > engine.model.causal_relations.evidence_budget):
        raise ValueError('reference write cap requires joint relations and a supported quota')
    engine.model.relation_reference_write_cap = config['relation_reference_write_cap'] = cap


def register_use(engine, state, question):
    """Only a real selected binding creates an edge, never an imagined branch."""
    archive = engine.model._causal_archive
    archive.sync()
    db = archive.db
    budget = getattr(engine.model, 'relation_reference_write_cap',
                     engine.model.causal_relations.evidence_budget)
    with db:
        db.execute('DELETE FROM relation_variables WHERE relation_id=? AND generation<>?',
                   (state.identity, state.generation))
        values = (state.identity, question['identity'], question['payload']['role'],
                  state.generation, question['version'], int(engine.model.memory.clock))
        if getattr(engine.model, 'reference_use_enabled', False):
            from scripts.ib_local.reference_use import credit, rank
            db.execute('''INSERT INTO relation_variables VALUES (?,?,?,?,?,?)
                ON CONFLICT(relation_id,unknown_id) DO UPDATE SET
                seen_version=excluded.seen_version,touched=excluded.touched''', values)
            rows = db.execute('SELECT unknown_id FROM relation_variables WHERE relation_id=? '
                              'ORDER BY touched DESC,unknown_id DESC', (state.identity,)).fetchall()
            ordered = rank(rows, credit(engine.model, (state.identity,)), required=question['identity'])
            db.executemany('DELETE FROM relation_variables WHERE relation_id=? AND unknown_id=?',
                           ((state.identity, uid) for uid, in ordered[budget:]))
            archive._relation_vars_cache.clear()
            return
        db.execute('INSERT OR REPLACE INTO relation_variables VALUES (?,?,?,?,?,?)', values)
        db.execute('DELETE FROM relation_variables WHERE relation_id=? AND unknown_id NOT IN '
                   '(SELECT unknown_id FROM relation_variables WHERE relation_id=? '
                   'ORDER BY touched DESC,unknown_id DESC LIMIT ?)',
                   (state.identity, state.identity, budget))
    archive._relation_vars_cache.clear()


def seed_actual_uses(engine, rows):
    """Migrate edges only from recorded real question/program participation."""
    db=engine.model._causal_archive.db
    inserted=0
    with db:
        for row in rows:
            q=row.get('shared_unknown',{})
            if not q.get('revision',{}).get('committed'):
                continue
            rid=q['relation_identity'];identity=q['identity']
            generation=row['causal_program']['roles']['generation']
            relation=db.execute('SELECT generation FROM relations WHERE id=?',(rid,)).fetchone()
            unknown=db.execute('SELECT version FROM unknown_bindings WHERE id=?',(identity,)).fetchone()
            if relation is None or unknown is None or relation[0]!=generation:
                continue
            db.execute('INSERT OR REPLACE INTO relation_variables VALUES (?,?,?,?,?,?)',
                (rid,identity,q['role'],generation,q['version'],row['cycle']))
            inserted+=1
        budget=engine.model.causal_relations.evidence_budget
        for (rid,) in db.execute('SELECT DISTINCT relation_id FROM relation_variables').fetchall():
            db.execute('DELETE FROM relation_variables WHERE relation_id=? AND unknown_id NOT IN '
                '(SELECT unknown_id FROM relation_variables WHERE relation_id=? '
                'ORDER BY touched DESC,unknown_id DESC LIMIT ?)',(rid,rid,budget))
    return dict(recorded_real_uses=inserted,edges=db.execute(
        'SELECT COUNT(*) FROM relation_variables').fetchone()[0],new_parameters=0)


def share_variables(model, left_identity, right_identity):
    """A functionally accepted real sharing plan shares variable USES, not IDs.

    The two scopes remain distinct. Each retains the same fixed edge quota;
    merely recalling similar vectors can never create this connection.
    """
    archive=model._causal_archive
    archive.sync()
    db=archive.db
    budget=getattr(model, 'relation_reference_write_cap', model.causal_relations.evidence_budget)
    uses=db.execute('SELECT unknown_id,port,seen_version,touched FROM relation_variables '
                    'WHERE relation_id IN (?,?) ORDER BY touched DESC,unknown_id',
                    (left_identity,right_identity)).fetchall()
    unique={}
    for row in uses:
        if row[0] not in unique:unique[row[0]]=row
    use_credit = getattr(model, 'reference_use_enabled', False)
    if use_credit:
        from scripts.ib_local.reference_use import credit, rank
        selected = rank(unique.values(), credit(model, (left_identity, right_identity)))[:budget]
    else:
        selected = list(unique.values())[:budget]
    with db:
        for rid in (left_identity,right_identity):
            generation=db.execute('SELECT generation FROM relations WHERE id=?',(rid,)).fetchone()[0]
            if use_credit:
                kept = {(uid, port) for uid, port, _, _ in selected}
                old = db.execute('SELECT unknown_id,port,generation FROM relation_variables '
                                 'WHERE relation_id=?', (rid,)).fetchall()
                db.executemany('DELETE FROM relation_variables WHERE relation_id=? AND unknown_id=?',
                    ((rid, uid) for uid, port, gen in old if gen != generation or (uid, port) not in kept))
                db.executemany('''INSERT INTO relation_variables VALUES (?,?,?,?,?,?)
                    ON CONFLICT(relation_id,unknown_id) DO UPDATE SET
                    seen_version=excluded.seen_version,touched=excluded.touched''',
                    ((rid, uid, port, generation, version, touched)
                     for uid, port, version, touched in selected))
                continue
            db.execute('DELETE FROM relation_variables WHERE relation_id=?',(rid,))
            db.executemany('INSERT INTO relation_variables VALUES (?,?,?,?,?,?)',
                ((rid,identity,port,generation,version,touched) for identity,port,version,touched in selected))
    archive._relation_vars_cache.clear()


def relation_variables(model, state):
    """Read current versions on demand; a pending state keeps its own snapshot."""
    if getattr(model,'joint_relation_update_schema',None)!=1:
        return state
    rows = reference_rows(model, state)
    return resolve_references(model, state, rows) if rows else state


def reference_rows(model, state):
    archive = model._causal_archive
    rv_cache = getattr(archive, '_relation_vars_cache', None)
    key = (state.identity, state.generation, len(state.valid))
    if rv_cache is not None and key in rv_cache:
        return rv_cache[key]
    rows = archive.db.execute('''SELECT v.unknown_id,v.port,u.version,v.seen_version
        FROM relation_variables v JOIN unknown_bindings u ON u.id=v.unknown_id
        WHERE v.relation_id=? AND v.generation=? ORDER BY v.touched DESC,v.unknown_id
        LIMIT ?''', key).fetchall()
    if rv_cache is not None:
        if len(rv_cache) >= 512:
            rv_cache.pop(next(iter(rv_cache)))
        rv_cache[key] = rows
    return rows


def resolve_references(model, state, rows, *, supports=None):
    from scripts.ib_local.shared_unknown_bindings import _supports, retain_constraints
    from scripts.ib_local.causal_relation_program import Variable, Port
    if not rows:
        return replace(state, constraints=(), variable_uses=(), evidence_bound=False)
    if supports is None:
        supports = {identity: _supports(model._causal_archive, identity)
                    for identity, _, _, _ in rows}
    facts = tuple(e for identity, _, _, _ in rows for e in supports[identity])
    unique = {(e.fact_id,e.kind):e for e in facts}
    kept = retain_constraints(model, tuple(unique.values()), (), len(state.valid),
                              use=getattr(state,'retention_use',None))
    return replace(state, constraints=kept,evidence_bound=getattr(model,'joint_relation_update_revision',1)>=3,
        variable_uses=tuple((Variable(identity,Port(port)),version,seen)
                            for identity,port,version,seen in rows))


def overlap_consensus(distributions, weights):
    """Exact weighted quadratic solve on ONE identically bound output port.

    Normalized weights do not count correlated programs as independent data.
    Predictive disagreement remains part of reported variance.
    """
    means = torch.stack([d['mean'] for d in distributions])
    variances = torch.stack([d['scale2'] for d in distributions])
    if not bool(torch.isfinite(means).all() and torch.isfinite(variances).all() and (variances>0).all()):
        raise ValueError('finite overlapping predictions required')
    if (weights.shape != (len(distributions),) or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any()) or not bool(weights.sum() > 0)):
        raise ValueError('nonnegative active program weights required')
    w = (weights/weights.sum())[:,None,None]
    precision = w/variances.clamp_min(torch.finfo(variances.dtype).tiny)
    center = (precision*means).sum(0)/precision.sum(0)
    initial = (w*means).sum(0)
    objective = lambda z: (precision*(means-z).square()).sum(0).mean(-1)
    variance = (w*(variances+(means-center).square())).sum(0)
    # Keep the heaviest active predictive tail instead of becoming certain
    # merely because another path has more support.
    active_df=[torch.as_tensor(d['df'],device=means.device,dtype=means.dtype)
               for i,d in enumerate(distributions) if bool(weights[i]>0)]
    return dict(mean=center,scale2=variance,df=torch.stack(active_df).min(),
                objective_before=objective(initial),objective_after=objective(center),
                disagreement=(w*(means-center).square()).sum(0),iterations=1)
