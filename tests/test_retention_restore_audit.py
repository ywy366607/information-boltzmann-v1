"""Archive compatibility preserves every existing retention and prior value."""
from copy import deepcopy

import pytest

from scripts.audit_native_relation_use_restore import assert_restored_payload


def payload():
    return dict(scope='unknown:1727', revision=1, mode='apply',
                moments=[['episode:1382', -.0026, .000007, 1]],
                last_event='episode:2483', betas=[.9, .999], eps=1e-8)


def test_legacy_archive_accepts_only_new_empty_defaults():
    old = payload()
    restored = dict(old, prior_scope='', prior_revision=0, prior_moments=[])
    assert_restored_payload(old, restored)
    assert 'prior_scope' not in old


@pytest.mark.parametrize('key,value', [
    ('revision', 2), ('moments', []), ('mode', 'observe'),
    ('last_event', 'changed'), ('betas', [.8, .999]), ('eps', 1e-6),
    ('prior_scope', 'relation:1:1'), ('prior_revision', 1),
    ('prior_moments', [['episode:1382', .1, .01, 1]]),
])
def test_legacy_comparison_rejects_changed_values(key, value):
    old = payload()
    restored = dict(old, prior_scope='', prior_revision=0, prior_moments=[])
    restored[key] = value
    with pytest.raises(AssertionError):
        assert_restored_payload(old, restored)


def test_new_archive_preserves_nonempty_prior_exactly():
    stored = dict(payload(), prior_scope='relation:241:5', prior_revision=3,
                  prior_moments=[['episode:1616', .013, .00002, 3]])
    assert_restored_payload(stored, deepcopy(stored))
    changed = deepcopy(stored)
    changed['prior_moments'][0][3] += 1
    with pytest.raises(AssertionError):
        assert_restored_payload(stored, changed)
