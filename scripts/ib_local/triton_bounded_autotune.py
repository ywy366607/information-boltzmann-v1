"""Opt-in, process-local bound on Triton autotune candidate enumeration.

Select one configuration from the backend's existing, officially pruned list.
No kernel implementation, shape, arithmetic branch, driver resource check or
learner setting is replaced. This is a resource policy, not a fastest-config
claim. Apply consistently to all experimental arms before loading any kernel.
"""
from __future__ import annotations


def configuration_key(config):
    """Favor fewer warps/stages; use deterministic kwargs only to break ties."""
    return (config.num_warps, config.num_stages,
            tuple(sorted((str(k), repr(v)) for k, v in config.kwargs.items())))


def patch_autotuner(autotuner_class, on_selection=None):
    original = autotuner_class.prune_configs
    if getattr(original, '_bounded_candidates', False):
        return

    def prune(self, kwargs):
        candidates = original(self, kwargs)
        if not candidates:
            raise RuntimeError('original Triton pruning produced no legal candidate')
        chosen = min(candidates, key=configuration_key)
        if on_selection is not None:
            on_selection(dict(kernel=self.base_fn.__name__,
                original_candidates=len(candidates), num_warps=chosen.num_warps,
                num_stages=chosen.num_stages, kwargs=chosen.kwargs))
        return [chosen]

    prune._bounded_candidates = True
    autotuner_class.prune_configs = prune
    if not hasattr(autotuner_class, 'run'):
        return
    original_run = autotuner_class.run

    def run(self, *args, **kwargs):
        if len(self.configs) <= 1:
            return original_run(self, *args, **kwargs)
        # Returning one item from prune_configs alone STILL benchmarks it,
        # allocating Triton's 256 MiB cache-flush buffer and timing events.
        # Enter the original supported one-config execution path instead.
        self.nargs = dict(zip(self.arg_names, args))
        held_configs = self.configs
        self.configs = prune(self, kwargs)
        try:
            return original_run(self, *args, **kwargs)
        finally:
            self.configs = held_configs

    autotuner_class.run = run


def enable_bounded_autotune(on_selection=None):
    from triton.runtime.autotuner import Autotuner
    patch_autotuner(Autotuner, on_selection)
    return dict(policy='one_original_pruned_config_min_warps_stages_no_benchmark',
                kernel_math_changed=False, original_resource_checks=True,
                library_files_modified=False)
