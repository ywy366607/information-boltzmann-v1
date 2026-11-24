"""Full production-interface observation parity and finite-U score binding."""
import copy

import torch

from scripts.ib_local.causal_use_value import configure
from scripts.ib_local.causal_use_transport import finish


def same(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def test_complete_learning_loop_transport_preserves_trajectory(tmp_path):
    from test_joint_relation_update import body
    engines = []
    try:
        for name, enabled in [('plain', False), ('observed', True)]:
            torch.manual_seed(21)
            e, archive = body(tmp_path/name, revision=4)
            configure(e, 'C')
            e.model.identity_preview_revision = 1
            e.causal_use_transport_audit = enabled
            engines.append((e, archive))
        trajectories = []
        for e, archive in engines:
            torch.manual_seed(301)
            rows = []
            for event in range(3):
                e.autonomous_event_clock = event
                row = e.step(torch.randn(e.model.encoder.out_features, 4))
                finish(e, row)
                rows.append(row)
                if e.causal_use_transport_audit:
                    r = row['use_transport']
                    if not r['committed']:
                        continue
                    assert r['committed_count'] <= r['budget']
                    assert r['decomposition_residual'] < 1e-12
                    for q in r['queries']:
                        assert all(q['query'] not in ids for ids in q['supports'])
                    assert not hasattr(e, '_use_transport_pending')
            trajectories.append((rows, copy.deepcopy(e.model.state_dict()), e.recurrent.clone()))
            if e.causal_use_transport_audit:
                assert any(r['use_transport']['committed'] for r in rows)
        for a, b in zip(trajectories[0][0], trajectories[1][0]):
            for key in ('choice', 'utterance_hex', 'metal_g', 'response_nll', 'choice_probability'):
                assert a[key] == b[key]
        same(trajectories[0][1], trajectories[1][1])
        torch.testing.assert_close(trajectories[0][2], trajectories[1][2], rtol=0, atol=0)
    finally:
        for _, archive in engines:
            archive.close()
