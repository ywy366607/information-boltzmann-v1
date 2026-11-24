import math
from pathlib import Path

from scripts.audit_critic_repair_history import audit


def test_rescore_old_targets_and_replay_saved_error(monkeypatch):
    # Same observed forecasts: old target g+.8*V/10, new target g+.8*V/100.
    old = {'cycle': 3884, 'value_now': 10., 'td_previous': {
        'g': .1, 'y': .9, 'imagined': 12.}, 'critic_rms': 20.}
    error2 = (12.-100*.18)**2
    saved = math.sqrt(.98*old['critic_rms']**2+.02*error2)
    new = {'cycle': 3885, 'value_now': 10., 'td_previous': {
        'g': .1, 'y': .18, 'imagined': 12.}, 'critic_rms': saved}
    monkeypatch.setattr('scripts.audit_critic_repair_history.read_committed',
                        lambda p: [old] if p.parent.name == 'old' else [new])
    out = audit((Path('new'),), previous_segments=(Path('old'),))
    assert math.isclose(out['recomputed_history']['bias_corrected_rms'], 6.)
    assert out['recomputed_history']['target_replay_max_abs_error'] < 1e-12
    assert out['replay_max_abs_squared_error'] < 1e-12


def test_replay_tracks_explicit_migration_boundary(monkeypatch):
    td = {'g': .1, 'y': .18, 'imagined': 12.}
    old_variance = .98*100 + .02*36
    new_variance = 4.
    rows = [
        {'cycle': 3885, 'value_now': 10., 'td_previous': td,
         'critic_rms': math.sqrt(old_variance)},
        {'cycle': 3886, 'value_now': 10., 'td_previous': td,
         'critic_rms': math.sqrt(.98*new_variance+.02*36)},
    ]
    monkeypatch.setattr('scripts.audit_critic_repair_history.read_committed', lambda p: rows)
    migration = {'cycle': 3885, 'old_critic_error': old_variance,
                 'new_critic_error': new_variance}
    out = audit((Path('events'),), migration=migration)
    assert out['replay_max_abs_squared_error'] < 1e-12
