"""Native execution/measurement contracts; these are interface checks."""
import json

import pytest
import torch

from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local import latent_action_planner as planner
from scripts.ib_local import native_amortized_decision as amort
from scripts.native_action_contrast_geometry import observe
from test_relation_retention_use import configured


@pytest.mark.parametrize('speech_only', [False, True])
def test_four_corner_observer_at_native_teacher_preserves_budget_and_learning(tmp_path, speech_only):
    torch.set_num_threads(1)
    engine, archive = configured(tmp_path)
    original = planner.teach
    try:
        engine.gain_units_schema = 1
        engine.model._expression_speech_only = speech_only
        ev.configure(engine, schema=5, word_execution='native_anchor')
        engine.native_word_runtime = 'batch'
        engine.model._preview_batch_enabled = True
        amort.configure(engine, dict(schema=1, seed=201120, planner='action_batch',
                                     correction='decision_budget'))
        with observe(engine, max_cases=2, option_calibration=True,
                     capsule_dir=tmp_path/'capsules') as records:
            row = engine.step(torch.randn(8, 4))
        assert planner.teach is original
        assert records
        receipt = row['causal_use']['native_amortization']['action_planner']
        assert 0 < receipt['teacher_count'] <= 2
        assert receipt['native_labels'] > 0
        assert receipt['new_world_label']
        for record in records:
            assert record['teacher_anchor_max_error'] < 1e-4
            assert record['score_reconstruction_max_error'] < 1e-4
            assert record['prior_reconstruction_max_error'] < 1e-6
            assert record['bank_prediction_reconstruction_max_error'] < 1e-4
            assert record['extra_world_calls'] == record['extra_teacher_calls'] == 0
            calibration = record['option_calibration']
            assert calibration is not None
            if calibration['bank_rows']:
                assert calibration['equation_error'] < 1e-8
                assert calibration['orthogonality_error'] < 1e-8
            for pair in record['pairs']:
                assert pair['reconstruction_error'] < 1e-4
            for policy in record['policies'].values():
                assert all(abs(sum(p)-1) < 1e-6 for p in policy['actor_probability'])
        json.dumps(records, allow_nan=False)
    finally:
        archive.close()


def test_observer_scope_restores_native_function_after_exception():
    original = planner.teach
    with pytest.raises(RuntimeError, match='scope test'):
        with observe(object()):
            raise RuntimeError('scope test')
    assert planner.teach is original
