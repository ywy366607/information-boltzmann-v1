"""Contracts for the production reward path and full finite-U integration."""
import torch

from scripts.ib_local.causal_use_value import configure
from scripts.ib_local.causal_value_learning import prediction
from test_joint_relation_update import body


def test_shared_gain_is_the_actual_plain_read_and_learns_at_zero_confidence(tmp_path):
    e, archive = body(tmp_path, revision=4)
    try:
        configure(e, 'C2')
        z = torch.randn(3, e.model.encoder.out_features)
        d = torch.randn_like(z)
        opt = torch.arange(3)
        raw = prediction(e.model, z, d, opt)
        x = torch.cat((z, d, torch.eye(3)), -1)
        expected = e.causal_use_value_evidence.bases()+e.model.kanli.outcome(x)[:, :4]
        torch.testing.assert_close(raw, expected)
        assert e.causal_use_value_evidence.confidence().count_nonzero() == 0
        grad = torch.autograd.grad((raw[:, 3]-1).square().mean(),
                                   e.model.kanli.outcome[-1].weight)[0]
        assert grad[3].norm() > 0
        assert grad[:3].count_nonzero() == 0
    finally:
        archive.close()


def test_full_C2_step_trains_same_gain_and_replays_compatible_actual_labels(tmp_path):
    e, archive = body(tmp_path, revision=4)
    try:
        configure(e, 'C2')
        e.autonomous_event_clock = 0
        rows = [e.step(torch.randn(e.model.encoder.out_features, 4)) for _ in range(3)]
        for row in rows:
            contract = row['causal_gain_learning']
            assert abs(contract['current_target']-10*row['metal_g']) < 1e-6
            assert row['episode_slot'] not in contract['replay_slots']
            assert contract['replay_samples'] <= e.replay_batch
            assert row['causal_use']['finite_U'].get('verified', True)
        assert rows[-1]['causal_gain_learning']['replay_samples'] > 0
    finally:
        archive.close()
