from dataclasses import replace
from types import SimpleNamespace
import sqlite3

import pytest
import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, log_score
from scripts.ib_local.joint_relation_update import (
    register_use, relation_variables, share_variables,
)
from scripts.ib_local import reference_use, shared_unknown_bindings


def test_reference_cut_uses_the_same_finite_inputs(monkeypatch, tmp_path):
    from test_joint_relation_update import body
    engine, archive = body(tmp_path, revision=4)
    model = engine.model
    try:
        v = model.memory.source[0:1].detach().clone()
        facts = {f'episode:{i}': Evidence(v+i, v-i, v+i, Port.MOTHER, observed=True,
                                         fact_id=f'episode:{i}') for i in (1, 2, 3)}
        fixtures = {'a': ('episode:1', 'episode:2'), 'b': ('episode:1', 'episode:3'),
                    'c': ('episode:1',)}
        monkeypatch.setattr(shared_unknown_bindings, '_supports',
                            lambda _, uid: tuple(facts[eid] for eid in fixtures[uid]))
        # Use the production merge/dedup, with deterministic finite ranking.
        monkeypatch.setattr(shared_unknown_bindings, 'retain_constraints',
                            lambda _, facts, extra, cap, use: facts[:cap])
        archive.db.execute("INSERT INTO relations VALUES (1,1,0,X'',0)")
        archive.db.executemany('INSERT INTO unknown_bindings(id,signature,version,payload,touched) '
                               "VALUES (?, ?, 1, '{}', 0)", ((k,k) for k in fixtures))
        archive.db.executemany('INSERT INTO relation_variables VALUES (1,?,\'condition\',1,0,?)',
                               [('a',3),('b',2),('c',1)])
        state = model.causal_relations.state(0)
        state = replace(state, identity=1, generation=1, valid=state.valid[:3])
        full, cuts = reference_use.capture(model, relation_variables(model, state))
        ids = lambda st: [f.fact_id for f in st.constraints]
        assert ids(full) == ['episode:1', 'episode:2', 'episode:3']
        assert cuts[('c','condition',1)] is None
        assert ids(cuts[('b','condition',1)]) == ['episode:1', 'episode:2']
        # The budget can substitute another eligible fact when a route drops.
        fixtures['a'] = ('episode:1', 'episode:2', 'episode:3')
        fixtures['b'] = ('episode:1',)
        fixtures['c'] = ('episode:4',)
        facts['episode:4'] = replace(facts['episode:3'], fact_id='episode:4')
        captured = reference_use.capture(model, relation_variables(model, state))
        assert 'episode:4' in ids(captured[1][('a','condition',1)])
        frozen = reference_use.begin(engine, captured, v, v)
        y = v @ frozen['encoder'].T
        if frozen['bias'] is not None:
            y = y+frozen['bias']
        before = log_score(frozen['full'],y).mean().item()
        with torch.no_grad():
            model.encoder.weight.add_(3.)
        report = reference_use.settle(frozen, v, 'episode:5')
        assert report['score'] == pytest.approx(before)
        cuts = {row['unknown']: row for row in report['references']}
        assert cuts['a']['changed'] and cuts['a']['benefit'] != 0
        assert cuts['b']['benefit'] == cuts['c']['benefit'] == 0
    finally:
        archive.close()


def test_only_later_live_references_consume_credit(tmp_path):
    path = tmp_path/'credit.sqlite'
    db = sqlite3.connect(path)
    db.executescript('''PRAGMA foreign_keys=ON;
        CREATE TABLE relations(id INTEGER PRIMARY KEY, generation INTEGER);
        CREATE TABLE relation_variables(relation_id INTEGER, unknown_id TEXT, port TEXT,
          generation INTEGER, seen_version INTEGER, touched INTEGER,
          PRIMARY KEY(relation_id,unknown_id));
        INSERT INTO relations VALUES (1,1),(2,1);
    ''')
    model = SimpleNamespace(_causal_archive=SimpleNamespace(db=db, sync=lambda: None),
                            joint_relation_update_revision=4, causal_relations=SimpleNamespace(evidence_budget=2),
                            memory=SimpleNamespace(clock=4,read_credit_betas=(.9,.99),read_credit_eps=1e-8))
    e = SimpleNamespace(model=model,causal_use_arm='C2')
    config = {}
    reference_use.configure(e, config)
    db.executemany('INSERT INTO relation_variables VALUES (1,?,\'condition\',1,0,?)', [('old',1),('recent',2)])
    report = dict(scope=(1,1),event='episode:5',references=[
        dict(unknown='old',port='condition',benefit=.2,changed=True),
        dict(unknown='recent',port='condition',benefit=0.,changed=False)])
    assert reference_use.credit(model,(1,)) == {}
    reference_use.commit(model, report)
    reference_use.commit(model, report)
    assert db.execute('SELECT unknown_id,count FROM reference_use').fetchall() == [('old',1)]
    db.close()
    db = model._causal_archive.db = sqlite3.connect(path)
    try:
        db.execute('PRAGMA foreign_keys=ON')
        e.model.reference_use_enabled = False
        assert config['reference_use'] == 1
        reference_use.configure(e, config)
        register_use(e, SimpleNamespace(identity=1,generation=1),
                     dict(identity='new', version=0, payload={'role':'condition'}))
        assert db.execute('SELECT unknown_id FROM relation_variables WHERE relation_id=1').fetchall() == [('new',),('old',)]
        share_variables(model, 1, 2)
        assert reference_use.credit(model,(1,))['old'] > 0
        assert reference_use.credit(model,(2,)) == {}
        db.execute('UPDATE relations SET generation=2 WHERE id=1')
        register_use(e, SimpleNamespace(identity=1,generation=2),
                     dict(identity='old', version=0, payload={'role':'condition'}))
        reference_use.commit(model, report)
        assert reference_use.credit(model,(1,)) == {}
        assert db.execute('SELECT * FROM reference_use').fetchall() == []
    finally:
        db.close()
