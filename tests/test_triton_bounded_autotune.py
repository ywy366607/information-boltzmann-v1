from types import SimpleNamespace

import pytest

from scripts.ib_local.triton_bounded_autotune import patch_autotuner


def config(warps=4, stages=1, **kwargs):
    return SimpleNamespace(num_warps=warps, num_stages=stages, kwargs=kwargs)


def test_policy_only_chooses_from_original_pruned_set_and_keeps_objects():
    seen = []
    excluded, first, chosen = config(1), config(8), config(4, BLOCK=64)
    class Tuner:
        base_fn = staticmethod(lambda: None)
        def prune_configs(self, kwargs):
            seen.append(kwargs)
            return [first, chosen]  # Backend has rejected excluded already.
    rows = []
    patch_autotuner(Tuner, rows.append)
    result = Tuner().prune_configs({'N': 128})
    assert len(result) == 1 and result[0] is chosen
    assert result[0] is not excluded
    assert seen == [{'N': 128}]
    assert rows[0]['original_candidates'] == 2


def test_policy_is_deterministic_idempotent_and_does_not_rewrite_configs():
    a, b = config(4, 2, BLOCK=32), config(4, 1, BLOCK=64)
    class Tuner:
        base_fn = staticmethod(lambda: None)
        def prune_configs(self, kwargs):
            return [a, b]
    rows = []
    patch_autotuner(Tuner, rows.append)
    patched = Tuner.prune_configs
    patch_autotuner(Tuner, rows.append)
    assert Tuner.prune_configs is patched
    assert Tuner().prune_configs({})[0] is b
    assert b.kwargs == {'BLOCK': 64} and a.num_stages == 2
    assert len(rows) == 1


def test_no_fallback_if_original_pruning_cannot_admit_a_kernel():
    class Tuner:
        def prune_configs(self, kwargs):
            return []
    patch_autotuner(Tuner)
    with pytest.raises(RuntimeError, match='no legal candidate'):
        Tuner().prune_configs({})


def test_original_single_config_run_skips_benchmark_and_restores_config_list():
    a, b = config(8), config(4)
    class Tuner:
        base_fn = staticmethod(lambda: None)
        arg_names = ['x']
        def __init__(self):
            self.configs = [a, b]
        def prune_configs(self, kwargs):
            assert self.nargs == {'x': 123}
            return self.configs
        def run(self, *args, **kwargs):
            assert len(self.configs) == 1, 'would enter the benchmark path'
            assert self.configs[0] is b
            if kwargs.get('fail'):
                raise RuntimeError('original resource check')
            return args
    patch_autotuner(Tuner)
    t = Tuner()
    held = t.configs
    assert t.run(123) == (123,)
    assert t.configs is held
    with pytest.raises(RuntimeError, match='original resource check'):
        t.run(123, fail=True)
    assert t.configs is held
