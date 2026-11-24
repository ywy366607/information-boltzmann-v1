"""Settlement observation on the existing complete learner; no capability claim."""
from dataclasses import replace

import torch

from test_persistent_rhythm import body
from scripts.ib_local.action_value_pair_audit import first_action
from scripts.ib_local.action_value_cross_score import (
    capture_second_settlement, cross_score, four_cells,
)
from scripts.ib_local.autonomous_cycle import dispatch


def test_actual_settlement_capsule_reconstructs_diagonal_and_survives_later_update(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(201120)
    engine, archive = body(tmp_path, root_mode='observe', scope_prior='off')
    origin = tmp_path/'origin'
    origin.mkdir()
    snapshot = archive.checkpoint(origin)['snapshot']
    path = tmp_path/'second_settlement.pt'
    rows = []
    try:
        with capture_second_settlement(path, snapshot), first_action(engine, 0, lambda r: None):
            for _ in range(256):
                row = dispatch(engine)
                if row['autonomous_event']['kind'] == 'external':
                    rows.append(row)
                if len(rows) == 2:
                    break
        capsule = torch.load(path, map_location='cpu', weights_only=False)
        assert len(rows) == 2
        assert abs(capsule['actual_U']-rows[1]['metal_g']) < 1e-8
        assert capsule['diagonal_max_error'] < 1e-5
        original = capsule['model'].encoder.weight.detach().clone()
        with torch.no_grad():
            engine.model.encoder.weight.add_(1)
        assert torch.equal(original,capsule['model'].encoder.weight)
        diagonal = cross_score(capsule,capsule)
        assert diagonal['applicable']
        assert abs(diagonal['value']-capsule['actual_U']) < 1e-5
        table = four_cells(capsule,capsule,0.,.8)
        assert table['two_step_generate_minus_silence'] == dict(silence=0.,generate=0.)
        foreign = dict(capsule, queries=(replace(capsule['queries'][0],fact_id='episode:999999999'),))
        assert not cross_score(capsule,foreign)['applicable']
        assert not cross_score(capsule,dict(capsule,queries=(replace(capsule['queries'][0],encoded=True),)))['applicable']
    finally:
        archive.close()
