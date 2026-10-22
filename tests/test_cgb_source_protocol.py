"""Numerical source/energy accounting, not an artificial capability study."""
import copy

import numpy as np
import pytest

from information_boltzmann.cgb.energy_observation import material_energy_components
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.mobile_exchange import MobileExchange
from scripts.ib.generative.compare_support_transport import same
from scripts.ib.generative.native_learning_stream import advance_byte, new_session
from scripts.ib.generative.source_protocol import SourceProtocol
from test_cgb_growing_material import growing


def ports():
    m = MobileExchange([[1., 1., 1.], [1.8, 1., 1.]], [3, 3], [3, 3], [4, 4], [4, 4],
                      pre=[0], post=[1], port_nodes=[0, 1]*4, seed=31)
    m.field_scheme = 'symmetric_ou'
    s = new_session(m, b'interface state')
    s.update(fixed_frequency_frozen=True, phase='fresh_E', next_phase_byte=0)
    return m, s


def test_text_protocol_repeats_original_physical_learning_and_scores():
    m, s = ports()
    other = MobileExchange.from_state_dict(m.state_dict())
    session = copy.deepcopy(s)
    protocol = SourceProtocol('text_drive')
    for target in (65, 66, 67):
        row = protocol.advance(m, s, target)
        original = advance_byte(other, session, target, 'fresh_E')
        assert all(row[key] == value for key, value in original.items())
    same(m.state_dict(), other.state_dict())
    assert s == session
    assert protocol.text_exposures == protocol.intervals == 3


@pytest.mark.parametrize('mode', ['text_drive', 'held_source', 'source_disconnected'])
def test_refinement_advances_same_external_interval_and_rejects_fractional_clock(mode):
    m, s = ports()
    other = MobileExchange.from_state_dict(m.state_dict())
    protocol = SourceProtocol(mode, [3.]*8 if mode == 'held_source' else None)
    before, session = m.state_dict(), copy.deepcopy(s)
    with pytest.raises(ValueError, match='divide'):
        protocol.advance(m, s, 255 if mode == 'text_drive' else None, dt_ms=3.)
    same(m.state_dict(), before)
    assert s == session and protocol.intervals == 0
    protocol.advance(m, s, 255 if mode == 'text_drive' else None, dt_ms=1.)
    for _ in range(4):
        other.step(1., None if mode == 'source_disconnected' else np.full(8, 3.))
    same(m.state_dict(), other.state_dict())
    assert m.t == 4. and protocol.intervals == 1


@pytest.mark.parametrize('mode', ['held_source', 'source_disconnected'])
def test_control_advances_physical_time_without_fictitious_text(mode, tmp_path):
    m, s = ports()
    protocol = SourceProtocol(mode, [3.]*8 if mode == 'held_source' else None)
    scored = s['scored_bytes'];cursor = s['next_phase_byte']
    work, source_charge = m.work_in, m.source_charge
    for _ in range(4):
        row = protocol.advance(m, s)
        assert row['scores'] is None and row['target'] is None
    assert m.t == 16. and s['scored_bytes'] == scored and s['next_phase_byte'] == cursor
    assert protocol.text_exposures == 0 and protocol.intervals == 4
    if mode == 'source_disconnected':
        assert m.work_in == work and m.source_charge == source_charge
    else:
        assert m.work_in != work
    path = tmp_path/'control.npz'
    save_exchange(path, m, dict(session=s, source_protocol=protocol.state_dict()))
    restored, meta = load_exchange(path)
    other_protocol = SourceProtocol(**meta['source_protocol'])
    other_session = meta['session']
    for _ in range(4):
        assert protocol.advance(m, s) == other_protocol.advance(restored, other_session)
    same(m.state_dict(), restored.state_dict())
    assert s == other_session and protocol.state_dict() == other_protocol.state_dict()
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-10


def test_control_refuses_labels_and_text_refuses_missing_target_before_mutating():
    m, s = ports();initial = m.state_dict()
    with pytest.raises(ValueError, match='unexposed'):
        SourceProtocol('source_disconnected').advance(m, s, 65)
    with pytest.raises(ValueError, match='byte'):
        SourceProtocol('text_drive').advance(m, s)
    same(m.state_dict(), initial)


def test_energy_components_close_and_do_not_change_the_physical_future():
    m = growing()
    other = type(m).from_state_dict(m.state_dict())
    for _ in range(8):
        breakdown = material_energy_components(m)
        assert abs(breakdown['total']-m.energy()) < 1e-12
        assert abs(breakdown['terms']['domains']-m.domain_formation_energy*m.n) < 1e-13
        m.step(2., [2.]);other.step(2., [2.])
    same(m.state_dict(), other.state_dict())
    m.energy = lambda: other.energy()+1
    with pytest.raises(ValueError, match='does not cover'):
        material_energy_components(m)
