from types import SimpleNamespace

import pytest

from scripts.ib_local.run_unreinforced_navigation_v2 import unreinforced_utter


def test_own_sample_restores_random_mode():
    engine = SimpleNamespace(mode='random')
    assert unreinforced_utter(engine, lambda e: (e.mode, b'hello')) == ('active', b'hello')
    assert engine.mode == 'random'


def test_mode_restored_after_failure():
    engine = SimpleNamespace(mode='random')
    def fail(e):
        raise RuntimeError('test')
    with pytest.raises(RuntimeError):
        unreinforced_utter(engine, fail)
    assert engine.mode == 'random'
