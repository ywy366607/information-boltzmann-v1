"""Later predictive use of an individual shared-question reference."""
from dataclasses import replace
import math

import torch


def configure(engine, config):
    model = engine.model
    if (getattr(model, 'joint_relation_update_revision', 0) < 3 or
            getattr(engine, 'causal_use_arm', None) != 'C2'):
        raise ValueError('reference use requires canonical relations and finite C2 settlement')
    db = model._causal_archive.db
    db.executescript('''
        CREATE TABLE IF NOT EXISTS reference_use (
            relation_id INTEGER NOT NULL, unknown_id TEXT NOT NULL,
            first REAL NOT NULL, second REAL NOT NULL, count INTEGER NOT NULL,
            event TEXT NOT NULL, PRIMARY KEY(relation_id,unknown_id),
            FOREIGN KEY(relation_id,unknown_id) REFERENCES relation_variables
                (relation_id,unknown_id) ON DELETE CASCADE);
    ''')
    model.reference_use_enabled = True
    config['reference_use'] = 1


def credit(model, ids):
    db = model._causal_archive.db
    b1, b2 = model.memory.read_credit_betas
    result = {}
    for rid in ids:
        for uid, first, second, count in db.execute('''SELECT u.unknown_id,u.first,u.second,u.count
                FROM reference_use u JOIN relation_variables v USING(relation_id,unknown_id)
                JOIN relations r ON r.id=v.relation_id AND r.generation=v.generation
                WHERE u.relation_id=?''', (rid,)):
            bias = math.tanh((first/(1-b1**count)) /
                (math.sqrt(second/(1-b2**count))+model.memory.read_credit_eps))
            result[uid] = max(result.get(uid, 0.), bias)
    return result


def rank(rows, benefits, required=None):
    # Python's stable sort preserves the original SQL order within equal credit.
    return sorted(rows, key=lambda row: (row[0] == required, benefits.get(row[0], 0.)), reverse=True)


def capture(model, state):
    from scripts.ib_local.causal_relation_program import Port
    from scripts.ib_local.joint_relation_update import reference_rows, resolve_references
    from scripts.ib_local.relation_evidence import mechanism_evidence, evidence_key
    from scripts.ib_local.shared_unknown_bindings import _supports

    rows = reference_rows(model, state)
    if not rows:
        return None
    supports = {row[0]: _supports(model._causal_archive, row[0]) for row in rows}

    def ids(candidate):
        return tuple(evidence_key(f) for f in mechanism_evidence(model, candidate, Port.MOTHER))

    variants = {}
    full = ids(state)
    for row in rows:
        cut = resolve_references(model, state, tuple(other for other in rows if other != row),
                                 supports=supports)
        key = row[0], row[1], row[2]
        variants[key] = None if ids(cut) == full else cut
    return state, variants


@torch.no_grad()
def begin(engine, captured, source, action):
    if captured is None:
        return None
    from scripts.ib_local.causal_relation_program import mechanism

    state, variants = captured
    def freeze(candidate):
        d = mechanism(engine.model, candidate, source, action, ridge=engine.relation_ridge)
        return {k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in d.items() if k in ('mean', 'scale2', 'df')}

    encoder = engine.model.encoder
    return dict(scope=(state.identity, state.generation), full=freeze(state),
                cuts={k: None if cut is None else freeze(cut) for k, cut in variants.items()},
                encoder=encoder.weight.detach().clone(),
                bias=None if encoder.bias is None else encoder.bias.detach().clone())


@torch.no_grad()
def settle(frozen, target, event):
    if frozen is None or target is None:
        return None
    from torch.nn.functional import linear
    from scripts.ib_local.causal_relation_program import log_score

    y = linear(target.reshape(1, -1), frozen['encoder'], frozen['bias'])
    full = float(log_score(frozen['full'], y).mean())
    return dict(scope=frozen['scope'], event=event, score=full, references=[
        dict(unknown=uid, port=port, version=version, changed=cut is not None,
             benefit=0. if cut is None else full-float(log_score(cut, y).mean()))
        for (uid, port, version), cut in frozen['cuts'].items()])


def commit(model, report):
    if report is None:
        return
    db = model._causal_archive.db
    rid, gen = report['scope']
    b1, b2 = model.memory.read_credit_betas
    with db:
        for item in report['references']:
            uid, port, benefit = item['unknown'], item['port'], item['benefit']
            # A scope that split or lost the route cannot collect this payment.
            if db.execute('''SELECT 1 FROM relation_variables v JOIN relations r ON r.id=v.relation_id
                    WHERE v.relation_id=? AND v.generation=? AND r.generation=v.generation
                    AND v.unknown_id=? AND v.port=?''', (rid, gen, uid, port)).fetchone() is None:
                continue
            if not math.isfinite(benefit):
                raise FloatingPointError('nonfinite reference contribution')
            if not item['changed']:
                continue
            db.execute('''INSERT INTO reference_use VALUES (?,?,?,?,1,?)
                ON CONFLICT(relation_id,unknown_id) DO UPDATE SET
                first=?*first+excluded.first, second=?*second+excluded.second,
                count=count+1,event=excluded.event WHERE event<>excluded.event''',
                (rid, uid, (1-b1)*benefit, (1-b2)*benefit**2, report['event'], b1, b2))
