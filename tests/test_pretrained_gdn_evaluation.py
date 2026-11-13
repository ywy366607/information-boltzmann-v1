"""Accounting/restore checks for the cross-window evaluation protocol."""
import json

import pytest
import torch

from scripts.ib_local.analyze_pretrained_gdn_cross_window import paired, journal
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def test_paired_accounting_separates_fresh_and_revisit_targets():
    full, held = [], []
    for i in range(32):
        for arm, fresh, revisit in [(full, 1., 2.), (held, 1.25, 1.5)]:
            arm.append(dict(step=i+1, phase='A', nll_sum=64*fresh+64*revisit,
                target_tokens=128, traffic=dict(fresh=dict(nll_sum=64*fresh, tokens=64),
                                              revisit=dict(nll_sum=64*revisit, tokens=64))))
    assert paired(full, held)['full_minus_held'] == pytest.approx(.125)
    assert paired(full, held, traffic='fresh')['full_minus_held'] == pytest.approx(-.25)
    assert paired(full, held, traffic='revisit')['full_minus_held'] == pytest.approx(.5)


def test_resumed_journal_retains_latest_actual_record_per_position(tmp_path):
    path = tmp_path/'steps.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in [dict(step=1, ce=2),
        dict(step=2, ce=5), dict(step=2, ce=3)]), encoding='utf-8')
    assert [r['ce'] for r in journal(path)] == [2, 3]


def test_complete_restore_check_detects_changed_optimizer_moment():
    first = dict(native=torch.ones(2), optimizer=dict(moment=torch.ones(2)))
    second = dict(native=torch.ones(2), optimizer=dict(moment=torch.tensor([1., 1.001])))
    with pytest.raises(AssertionError, match='optimizer/moment'):
        assert_same(first, second)
