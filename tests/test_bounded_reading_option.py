from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ib_local.bounded_reading_option import stop_reason, finish
from scripts.ib_local.navigation_gain_units import td_gain_target
from scripts.ib_local.world_learning_cycle import observed_pieces, StreamPiece


def test_actual_question_identity_version_stops_extended_action(monkeypatch):
    from scripts.ib_local import bounded_reading_option as option
    from scripts.ib_local import shared_unknown_bindings as shared
    row = dict(identity='unknown:real', version=7)
    monkeypatch.setattr(shared, '_row', lambda archive, identity: row if identity == row['identity'] else None)
    engine = SimpleNamespace(bounded_reading_option_schema=1,
        _program_question_roles=dict(unknown_identity='unknown:real'),
        model=SimpleNamespace(_causal_archive=object()),
        reading_option_state=dict(surprise_ema=1.6),
        reading_option_spec=dict(max_blocks=4, surprise_excess=.5),
        world=SimpleNamespace(base=SimpleNamespace(data=bytes(10000), cursor=960, window=4096)))
    session = option.begin(engine)
    assert session['unknown_id'] == 'unknown:real'
    assert session['unknown_version'] == 7
    info = dict(start=0, source='text')
    assert option.completed_block(engine, session, 1.5, info)
    row['version'] = 8
    assert not option.completed_block(engine, session, 1.5, info)
    assert session['stop'] == 'target_changed'


def test_actual_stream_partition_and_incremental_extension():
    blocks = [np.arange(960, dtype=np.int64).astype(np.uint8) for _ in range(4)]
    combined, pieces = observed_pieces(blocks[0][:480], blocks[0][480:], dict(source='text'))
    seen = []
    # Matches the real mutable piece list, including extension after update #2.
    for index, piece in enumerate(pieces):
        seen.extend(combined[piece.start:piece.end].tolist())
        if index % 2 == 1 and index < 7:
            offset = len(combined)
            data, extra = observed_pieces(blocks[0][:480], blocks[0][480:], dict(source='text'))
            combined = np.concatenate((combined, data))
            pieces.extend(StreamPiece(p.source, offset+p.start, offset+p.end,
                None if p.probe_start is None else offset+p.probe_start,
                None if p.probe_end is None else offset+p.probe_end) for p in extra)
    assert seen == np.concatenate(blocks).tolist()
    assert len(pieces) == 8
    assert [(p.start, p.end) for p in pieces] == [(i*480, (i+1)*480) for i in range(8)]
    assert all(p.probe_end == p.end+480 for p in pieces[::2])


def decision(**overrides):
    values = dict(blocks=1, max_blocks=4, nll=1.5, baseline=1.6, excess=.5,
                  target_valid=True, source_remaining=9600)
    values.update(overrides)
    return stop_reason(**values)


def test_observed_surprise_target_change_eof_and_cap_stop():
    assert decision() is None
    assert decision(nll=2.2) == 'surprise'
    assert decision(nll=float('nan')) == 'surprise'
    assert decision(target_valid=False) == 'target_changed'
    assert decision(source_remaining=959) == 'source_boundary'
    assert decision(blocks=4) == 'budget'
    assert decision(blocks=1, max_blocks=1) == 'budget'


@pytest.mark.parametrize('duration', [1, 2, 3, 4, 8])
def test_credit_time_matches_real_completed_blocks(duration):
    engine = SimpleNamespace(gamma=.8, gain_units_schema=1,
        prev={'duration_blocks': duration}, joint_proposal_budget=4,
        reading_option_spec={'max_blocks': 8},
        reading_option_state=dict(options=0, blocks=0, surprise_ema=1.6, terminations={}))
    session = dict(blocks=duration, block_nll=[1.5]*duration, stop='budget')
    engine._reading_option_session = session
    r = finish(engine, session)
    u, future = .13, .07
    # Critic units SCALE=100, raw-gain target units stay explicit.
    target = td_gain_target(engine, u*r['terminal_gain_discount'], 100*future)
    assert target == pytest.approx(.8**(duration-1)*u + .8**duration*future)
    assert r['language_updates'] == 2*duration
    assert r['native_plannings'] == r['action_settlements'] == 1
    assert engine.reading_option_state['blocks'] == duration
    assert td_gain_target(engine, .2, 1., False) == .2
    with pytest.raises(RuntimeError, match='already settled'):
        finish(engine, session)


def test_legacy_credit_is_one_block():
    engine = SimpleNamespace(gamma=.8, gain_units_schema=1, prev={'g': .2})
    assert td_gain_target(engine, .2, 2.) == pytest.approx(.216)
