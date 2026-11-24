"""Finite-menu admission for the production two-native-teacher budget."""
from types import SimpleNamespace

import torch

from scripts.ib_local.competing_action_teachers import retain, select


def test_root_coverage_survives_high_novelty_and_order_changes():
    roots = {'a':[0], 'b':[1], 'c':[2]}
    rows = [dict(key=str(i), priority=float(i), root_key='a') for i in range(12)]
    rows += [dict(key='b', priority=.1, root_key='b'), dict(key='c', priority=.2, root_key='c')]
    def run(order):
        queue = {}
        for row in order:
            retain(queue, row, 8)
        assert len(queue)==8 and {r['root_key'] for r in queue.values()}==set(roots)
        menu = sorted(queue.values(), key=lambda r:(-r['priority'],r['key']))
        e = SimpleNamespace(_competing_action_roots=roots)
        got, trace = select(e, menu, torch.tensor([.3,.9,.8]))
        assert [r['root_key'] for r in got]==['b','c']
        assert trace['covered_roots']==3 and trace['top_margin']>0
        return [r['key'] for r in got]
    assert run(rows)==run(list(reversed(rows)))
