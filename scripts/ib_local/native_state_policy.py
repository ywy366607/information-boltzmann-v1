"""Explicit per-layer S boundary policy, shared by training and inference.

This is an experimental state organization, not a necessary CGB mechanism.
Only selected associative matrices are cleared; short-convolution history,
other layers, pending byte, lexical history and experience are preserved.
"""


def configure_native_policy(engine, reset_layers=(), boundary_bytes=128):
    layers = tuple(int(i) for i in reset_layers)
    count = len(engine.model.base.layers)
    if len(set(layers)) != len(layers) or any(i < 0 or i >= count for i in layers):
        raise ValueError('unique valid zero-based reset layer indices required')
    if boundary_bytes != 128:
        raise ValueError('this paired protocol uses the existing128-byte window')
    policy = dict(schema=1, reset_layers=list(layers), boundary_bytes=boundary_bytes,
                  scope='associative_S_only_keep_convolution')
    engine.native_state_policy = policy
    return policy


def apply_native_boundary(engine, state, window=128):
    policy = getattr(engine, 'native_state_policy', None)
    if policy is None:
        return state
    if policy.get('schema') != 1 or policy.get('scope') != 'associative_S_only_keep_convolution':
        raise ValueError('unknown saved native state policy')
    if window != policy['boundary_bytes']:
        raise ValueError('training/inference native boundary periods differ')
    indices = policy['reset_layers']
    if not indices:
        return state
    layers = engine.model.base.layers
    if len(state) != len(layers):
        raise ValueError('native state/layer count mismatch')
    result = state.clone()
    for index in indices:
        layer = layers[index]
        if layer.recurrence not in ('legacy', 'decay_then_delta', 'feedback_delta'):
            raise ValueError('exchange-state reset needs its own declared semantics')
        if layer.conv_kernel or layer.share_size:
            result[index, :, :layer.matrix_size] = 0
        else:
            result[index] = 0
    return result
