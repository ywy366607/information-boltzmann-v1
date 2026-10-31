import json

import numpy as np
import pandas as pd
import torch

from information_boltzmann.core import fly_bptt_learning as fl
from information_boltzmann.core.fly_circuit_observer import CircuitObserver, cell_groups
from tests.test_fly_bptt_learning import make_model, physical


def annotations(tmp_path):
    path = tmp_path / 'ann.feather'
    pd.DataFrame({'bodyId': np.arange(6), 'class': ['Kenyon_Cell', 'Kenyon_Cell', 'DAN', 'MBON', None, None],
                  'superclass': ['cb_intrinsic'] * 4 + ['cb_motor', 'cb_motor'],
                  'type': ['KCg', 'KCg', 'PAM01', 'MBON11', None, 'APL']}).to_feather(path)
    return path


def test_groups_split_dopamine_and_name_apl(tmp_path):
    coarse, fine = cell_groups(np.arange(6), annotations(tmp_path))
    assert coarse == ['KC', 'KC', 'DAN_PAM', 'MBON', 'cb_motor', 'APL']
    assert fine[4] == 'untyped:cb_motor' and fine[0] == 'KCg'


def test_observer_counts_live_ticks_only_and_never_changes_the_dynamics(tmp_path):
    torch.manual_seed(3)
    model = make_model(tmp_path)
    observer = CircuitObserver(model, np.arange(6), annotations(tmp_path), tmp_path / 'obs')
    state = physical(model)
    state.ring = (torch.tensor([[1., 1., 0., 2., 0., 0.]]),) + tuple(torch.zeros_like(state.h) for _ in range(3))
    observer(model, state)                       # inactive: ignored
    observer.active = True
    for _ in range(4):
        observer(model, state)
    read = torch.zeros(1, model.embedding.embedding_dim)
    observer.end_word(model, token=3, who='mother', source='mother_reply', read=read)
    row = json.loads((tmp_path / 'obs' / 'circuits.jsonl').read_text().splitlines()[0])
    assert row['ticks'] == 4 and row['rates']['KC'] == 1.0 and row['rates']['MBON'] == 1.0
    assert row['rates']['DAN_PAM'] == 0.0 and row['surprisal'] > 0
    observer.flush()
    rates = np.load(tmp_path / 'obs' / 'fine_0000.npz')['rates']
    assert rates.shape == (1, len(observer.fine_names))
    # as a tick observer it must leave the physical state bit-identical
    fl.TICK_OBSERVERS.append(observer)
    try:
        before = state.h.clone()
        observer(model, state)
        assert torch.equal(before, state.h)
    finally:
        fl.TICK_OBSERVERS.remove(observer)
