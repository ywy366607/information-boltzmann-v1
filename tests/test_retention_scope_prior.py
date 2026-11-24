"""Identity-preserving retention priors in actual typed-unknown interfaces."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from scripts.ib_local import relation_retention_use as use
from test_persistent_rhythm import body
from scripts.ib_local.autonomous_cycle import dispatch


def test_prior_is_read_only_identity_bound_and_local_observations_take_over():
    model = SimpleNamespace(retention_scope_prior_mode='relation')
    parent = use.UseSnapshot('relation:2:1', mode='apply', revision=7,
        moments=(('episode:1', .1, .001, 1), ('episode:2', -.1, .001, 1)))
    local = use.UseSnapshot('unknown:3', mode='apply')
    cold = use.with_prior(model, local, parent, ['episode:1'])
    assert cold.moments == () and cold.revision == 0
    assert cold.prior_moments == parent.moments[:1]
    assert parent.revision == 7 and local.prior_moments == ()
    like = torch.ones(2)
    assert use.biases(cold, ['episode:1', 'episode:2'], like)[0] > 0
    assert use.biases(cold, ['episode:1', 'episode:2'], like)[1] == 0
    arrived = use.advance(cold, ['episode:1'], {'episode:1': -1.}, 'episode:4')
    assert arrived.moments[0][3] == 1  # one local observation, not parent count + one
    assert use.biases(arrived, ['episode:1'], like)[0] < 0
    assert use.advance(arrived, ['episode:1'], {'episode:1': -1.}, 'episode:4') == arrived
    assert use.filtered(arrived, []).prior_moments == ()
    off = use.with_prior(SimpleNamespace(retention_scope_prior_mode='off'), arrived, parent, ['episode:1'])
    assert off.prior_moments == () and off.moments == arrived.moments


def test_real_unknown_preview_commit_and_lagged_credit_share_the_prior(tmp_path, monkeypatch):
    from scripts.ib_local import shared_unknown_bindings as unknowns
    from scripts.ib_local.relation_evidence import evidence_key
    torch.set_num_threads(1)
    torch.manual_seed(201120)
    e, archive = body(tmp_path, root_mode='observe', scope_prior='relation')
    calls = []
    original = unknowns._with_constraints
    def verify(a, state, question):
        before = use._load(e.model, question['identity'])
        got = original(a, state, question)
        if got.organization is not None:
            parent = got.organization.retention_use
            wanted = {evidence_key(f) for f in got.constraints}
            assert got.retention_use.prior_scope == parent.scope
            assert got.retention_use.prior_moments == tuple(r for r in parent.moments if r[0] in wanted)
            assert got.retention_use.moments == before.moments
            calls.append(got.retention_use.prior_scope)
        return got
    monkeypatch.setattr(unknowns, '_with_constraints', verify)
    try:
        rows = []
        for _ in range(128):
            r = dispatch(e)
            if r['autonomous_event']['kind'] == 'external':
                rows.append(r)
            if len(rows) == 3:
                break
        assert len(rows) == 3 and calls
        for row in rows:
            assert row['causal_use']['finite_U']['verified']
            fresh = row['causal_use']['finite_U']['fresh_use']
            assert fresh['prior_scope'].startswith('relation:')
            assert row['policy_replay_max_error'] < 1e-3
        assert archive.db.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        archive.close()
