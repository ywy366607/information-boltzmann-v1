"""Bind first-step interventions to the actual complete learner, not a toy task."""
import pytest
import torch

from test_persistent_rhythm import body
from scripts.ib_local.action_value_pair_audit import first_action
from scripts.ib_local.autonomous_cycle import dispatch


@pytest.mark.parametrize('option', [0, 1])
def test_complete_body_first_intervention_and_normal_next_event(tmp_path, option):
    torch.set_num_threads(1)
    torch.manual_seed(201120)
    e, archive = body(tmp_path, root_mode='observe', scope_prior='off')
    updates = []
    step = e.optimizer.step
    def counted(*args, **kwargs):
        result = step(*args, **kwargs)
        updates.append(1)
        return result
    e.optimizer.step = counted
    captured, rows = [], []
    term = e._choice_policy_term
    try:
        with first_action(e, option, captured.append):
            for _ in range(256):
                row = dispatch(e)
                if row['autonomous_event']['kind'] == 'external':
                    rows.append(row)
                if len(rows) == 2:
                    break
        assert len(rows) == 2 and len(captured) == 1 and len(updates) > 0
        assert rows[0]['choice'] == ('silence', 'generate', 'reuse')[option]
        assert bool(rows[0]['utterance_hex']) == bool(option)
        assert captured[0]['behavior']['categorical_score_gradient'] == 0
        assert all(r['causal_use']['finite_U']['verified'] for r in rows)
        assert max(r['policy_replay_max_error'] for r in rows) < 1e-3
        assert '_utter' not in vars(e) and e._choice_policy_term == term
    finally:
        del e.optimizer.step
        archive.close()


def test_first_intervention_keeps_unavailable_reuse_unavailable(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(201120)
    engine, archive = body(tmp_path, root_mode='observe', scope_prior='off')
    try:
        with pytest.raises(ValueError, match='positive actual probability'):
            with first_action(engine, 2, lambda receipt: None):
                for _ in range(256):
                    dispatch(engine)
        assert '_utter' not in vars(engine)
    finally:
        archive.close()
