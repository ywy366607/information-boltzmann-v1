"""Layer-streamed evaluation of the same passive physical measurement."""
import numpy as np


def layer_streamed(original, s, p, theta, context, frequency_scale, impedance, previous=None):
    values, physical = [], []
    for i in range(len(s)):
        row, z = original(s[i:i+1], p[i:i+1], theta[i:i+1], context[i:i+1],
            frequency_scale[i:i+1], impedance[i:i+1],
            None if previous is None else previous[i:i+1])
        values.append(row)
        physical.append(z)
    return ({k:np.concatenate([row[k] for row in values],axis=0) for k in values[0]},
            np.concatenate(physical,axis=0))


def install(*, release_workspace=False):
    from scripts import native_topology_observer as module
    original = module.matrix_observation
    original_sample = module.NativeTopologyObserver.sample
    module.matrix_observation = lambda *args, **kwargs: layer_streamed(original,*args,**kwargs)
    if release_workspace:
        import gc
        import torch
        def sample(self, kind, *args, **kwargs):
            if kind == 'update_before':
                gc.collect()
                torch.cuda.empty_cache()
            return original_sample(self, kind, *args, **kwargs)
        module.NativeTopologyObserver.sample = sample
    def undo():
        module.matrix_observation = original
        module.NativeTopologyObserver.sample = original_sample
    return undo
