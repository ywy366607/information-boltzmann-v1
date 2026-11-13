import numpy as np

from scripts.ib_local.water_source_audit import WaterSourceAudit, stable_norm


def test_tiny_nonzero_source_is_not_reported_as_exact_zero():
    x = np.array([3e-210, 4e-210], dtype=np.float64)
    assert np.linalg.norm(x) == 0.
    np.testing.assert_allclose(stable_norm(x) / 1e-210, 5., rtol=1e-14)


def test_first_window_direction_is_held_input_finite_difference():
    specs = {layer: dict(heads=1, key_dim=2, value_dim=2, width=2) for layer in (1, 12)}
    audit = WaterSourceAudit(specs, rank=2, block_size=16)
    states = {layer: np.array([[[.2, .3], [.4, .1]]], np.float32) for layer in specs}
    initial = {k: v.copy() for k, v in states.items()}
    packets = {}
    for layer in specs:
        packets[layer] = dict(branch='chunk', x=np.ones((128, 2), np.float32),
            k=np.tile(np.array([[[.8, .6]]], np.float32), (128, 1, 1)),
            v=np.ones((128, 1, 2), np.float32) * .1,
            g=np.ones((128, 1), np.float32) * -.01,
            beta=np.ones((128, 1), np.float32) * .2,
            a_slope=np.ones((128, 1), np.float32) * .5,
            Aexp=np.ones(1, np.float32))
    for window in range(2):
        audit.open_window(window * 128, 0, states)
        for layer in specs:
            audit.add_chunk(layer, 0, packets[layer])
        audit.advance_window()
        # Continue physical state using the same reference transition.
        for layer in specs:
            for tick in range(128):
                k = packets[layer]['k'][tick, 0].astype(np.float64)
                k /= np.sqrt(k @ k + 1e-6)
                s = states[layer][0].astype(np.float64)
                states[layer] = (np.exp(-.01) * s + .2 * np.outer(k, .1 - k @ (np.exp(-.01) * s)))[None]
    def execute(eps):
        s = initial[1][0].astype(np.float64)
        for tick in range(256):
            k = packets[1]['k'][0, 0].astype(np.float64)
            k /= np.sqrt(k @ k + 1e-6)
            alpha = np.exp(-.01 * np.exp(eps if tick < 128 else 0))
            s = alpha * s + .2 * np.outer(k, .1 - k @ (alpha * s))
        return s
    eps = 1e-4
    numerical = (execute(eps) - execute(-eps)) / (2 * eps)
    sl = audit.layout.slices[audit.layout.head_fields[1, 0]]
    analytic = audit.exact[0][sl].reshape(2, 2)
    np.testing.assert_allclose(analytic, numerical, atol=2e-7, rtol=2e-6)
    assert audit.snapshots[-1]['strict_old_all_sources']
    assert audit.snapshots[-1]['youngest_source_age'] == 129
    assert len(audit.exact) == 2
    assert all(d.shape == (audit.layout.size,) for d in audit.exact)
