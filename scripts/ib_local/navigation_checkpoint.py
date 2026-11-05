"""Preserve carried action/wood state and isolate optimizer restore buffers."""
import copy

from scripts.ib_local import run_navigation_residual_wood as legacy

CARRY = ('_wood_source', '_wood_action', '_wood_predictions', '_forecast',
         '_batch_baseline', '_wood_replay', '_pending_record',
         'draft_diagnostics', '_draft_actions')


def snapshot(engine):
    state = legacy.snapshot(engine)
    state['engine'].update(copy.deepcopy({k:getattr(engine,k) for k in CARRY if hasattr(engine,k)}))
    return state


def restore(engine, state):
    # load_state_dict may reuse moment tensors supplied by the caller. Each
    # branch must receive its own copy, preserving the snapshot for reuse.
    # A legacy snapshot must not accidentally keep a candidate's coordinate
    # transform when reusing the same engine for an observational branch.
    for name in ('value_normalization', 'value_scale', 'reward_memory_kind', 'reward_memory_migration'):
        if name not in state['engine'] and hasattr(engine, name):
            delattr(engine, name)
    legacy.restore(engine, {**state, 'optimizer':copy.deepcopy(state['optimizer'])})


def save(engine, output, cycle, config):
    # Reuse the streaming array writer; change only which carried attributes
    # it includes. This process runs one learner and never yields in this block.
    derived = legacy._DERIVED
    try:
        legacy._DERIVED = derived.difference(CARRY)
        legacy.save_checkpoint(engine, output, cycle, config)
    finally:
        legacy._DERIVED = derived
