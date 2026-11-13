"""CPU numerical/interface tests, not training or capability experiments."""
import json
import pickle

import numpy as np
import pytest

import scripts.ib_local.native_s_eligibility as native
from scripts.ib_local.bounded_causal_tag_reference import ErrorTerm, NumericalBudget
from scripts.ib_local.bounded_causal_tag_blocked import ArraySource, capture_credit_blocked
from scripts.ib_local.native_s_eligibility import NativeSEligibility


SPECS = {layer: dict(heads=1, key_dim=2, value_dim=2, width=2) for layer in (1, 12)}


def make_producer(**kwargs):
    return NativeSEligibility(SPECS, block_size=3, **kwargs)


def initial_s(seed=1):
    rng = np.random.default_rng(seed)
    return {layer: rng.normal(size=(1, 2, 2)).astype(np.float32) for layer in SPECS}


def packet(seed=7, length=128, branch='recurrent'):
    rng = np.random.default_rng(seed)
    return dict(x=rng.normal(size=(length, 2)).astype(np.float32),
                k=rng.normal(size=(length, 1, 2)).astype(np.float32),
                v=rng.normal(size=(length, 1, 2)).astype(np.float32),
                g=-rng.uniform(.001, .08, size=(length, 1)).astype(np.float32),
                beta=rng.uniform(.1, .9, size=(length, 1)).astype(np.float32),
                a_slope=rng.uniform(.2, .8, size=(length, 1)).astype(np.float32),
                Aexp=np.array([.7], dtype=np.float32), branch=branch)


def portion(p, lo, hi):
    return {name: (value[lo:hi].copy() if name not in ('Aexp', 'branch') else value)
            for name, value in p.items()}


def fill(producer, start=0, version=0, states=None):
    producer.open_window(start, version, initial_s() if states is None else states)
    for layer in SPECS:
        p = packet(layer)
        producer.add_chunk(layer, 0, portion(p, 0, 64))
        p['branch'] = 'chunk'
        producer.add_chunk(layer, 64, portion(p, 64, 128))


def reconstructed(producer):
    result = {l: s.astype(np.float64) for l, s in producer.pending['initial_s'].items()}
    for layer, chunks in producer.pending['chunks'].items():
        for p in chunks:
            for t in range(len(p['x'])):
                k = p['k'][t, 0].astype(np.float64)
                k /= (np.linalg.norm(k) + 1e-6 if p['branch'] == 'recurrent'
                      else np.sqrt(k @ k + 1e-6))
                s = np.exp(float(p['g'][t, 0])) * result[layer][0]
                v, beta = p['v'][t, 0].astype(np.float64), float(p['beta'][t, 0])
                result[layer][0] = s + beta * np.outer(k, v - k @ s)
    return result


def assert_code_equal(a, b):
    for name in ('U', 'V', 'fields', 'bills'):
        np.testing.assert_array_equal(getattr(a.code, name), getattr(b.code, name))
    assert a.clock == b.clock


def test_selected_layout_fixed_owner_order_and_current_budget():
    p = make_producer()
    assert p.layout.size == 8
    assert len(p.layout.fields) == 2
    assert all('.s.' in name for name in p.layout.fields)
    assert p.parameter_shapes == {
        'layer1.a_proj.weight': (1, 2), 'layer1.b_proj.weight': (1, 2),
        'layer1.dt_bias': (1,), 'layer1.A_log': (1,),
        'layer12.a_proj.weight': (1, 2), 'layer12.b_proj.weight': (1, 2),
        'layer12.dt_bias': (1,), 'layer12.A_log': (1,)}
    assert list(p.parameter_shapes) == list(p.parameter_slices)
    assert p.code.U.dtype == p.code.V.dtype == np.float32
    specs = {l: dict(heads=4, key_dim=256, value_dim=256, width=1024) for l in SPECS}
    current = NativeSEligibility(specs)
    assert current.code.U.shape == (2, 2 * 4 * 256**2)
    assert current.code.V.shape == (2, 16400)
    assert len(current.layout.fields) == 8
    assert current.nbytes == current.code.nbytes
    assert current.nbytes + current.estimated_temporary_bytes < 64 * 2**20
    with pytest.raises(ValueError, match='exactly 1 and 12'):
        NativeSEligibility({0: SPECS[1], 1: SPECS[12]})


def test_capture_is_old_until_advance_and_raw_credit_is_original_shapes():
    p = make_producer()
    fill(p, start=450757, version=524)
    before = p.code.copy()
    teach = initial_s(2)
    assert p.is_complete()
    assert all(not g.any() for g in p.raw_credit(teach).values())
    for name in ('U', 'V', 'fields', 'bills'):
        np.testing.assert_array_equal(getattr(p.code, name), getattr(before, name))
    end = reconstructed(p)
    p.set_end_s({l: s[None] for l, s in end.items()})
    result = p.advance_window()
    assert result['processed'] == 128 and result['inserted_sources'] == 512
    assert p.pending is None and not p.is_complete()
    assert p.clock == dict(seen=450885, parameter_version=524, windows=1, inserted_sources=512)
    assert max(d['max_abs'] for d in result['reconstruction_delta'].values()) < 1e-7
    assert result['captured_bytes'] > result['bytes'] == p.code.nbytes
    assert result['estimated_peak_bytes'] < p.max_bytes and result['elapsed'] >= 0
    json.dumps(result)  # adapter logging does not need a NumPy JSON encoder
    assert np.linalg.norm(p.code.U) > 0 and np.linalg.norm(p.code.V) > 0
    fill(p, start=450885, version=525, states=end)
    saved = p.code.copy()
    gradients = p.raw_credit(teach)
    flat_teach = np.concatenate([teach[l][h].ravel() for l, h in p.layout.head_fields])
    expected = (saved.U.astype(np.float64) @ flat_teach) @ saved.V.astype(np.float64)
    for name, sl in p.parameter_slices.items():
        assert gradients[name].shape == p.parameter_shapes[name]
        assert gradients[name].dtype == np.float32
        np.testing.assert_allclose(gradients[name].ravel(), expected[sl], rtol=1e-6, atol=1e-7)
    for name in ('U', 'V', 'fields', 'bills'):
        np.testing.assert_array_equal(getattr(p.code, name), getattr(saved, name))


@pytest.mark.parametrize('branch', ['recurrent', 'chunk'])
def test_two_sparse_gate_sources_match_direct_parameter_finite_difference(monkeypatch, branch):
    p = make_producer()
    states = initial_s(4)
    data = packet(branch=branch)
    # Tiny keys distinguish norm+eps from sqrt(norm squared+eps).
    data['k'][0, 0] = [2e-4, -3e-4]
    x = data['x'][0].astype(np.float64)
    a, b, dt, a_log = np.array([.2, -.1]), np.array([-.3, .4]), .15, -.2
    z = a @ x + dt
    data['g'][0, 0] = -np.exp(a_log) * np.logaddexp(0., z)
    data['beta'][0, 0] = 1 / (1 + np.exp(-(b @ x)))
    data['a_slope'][0, 0] = 1 / (1 + np.exp(-z))
    data['Aexp'][0] = np.exp(a_log)
    p.open_window(0, 0, states)
    for layer in SPECS:
        p.add_chunk(layer, 0, data)
    events = []
    calls = 0
    def register(code, source, owner, layout, block_size):
        nonlocal calls
        calls += 1
        assert source.token[-1] in ('R', 'B')
        assert source.matrix.size == 4
        if source.token[1:4] == (1, 0, 0):
            # Provider remains stable and yields no coordinates outside this head.
            first = np.concatenate([source(lo, min(lo + block_size, layout.size))
                                    for lo in range(0, layout.size, block_size)])
            second = np.concatenate([source(lo, min(lo + block_size, layout.size))
                                     for lo in range(0, layout.size, block_size)])
            np.testing.assert_array_equal(first, second)
            assert not first[4:].any()
            events.append((first.copy(), owner.copy()))
    monkeypatch.setattr(native, 'insert_source_inplace', register)
    p.advance_window()
    assert calls == 512 and len(events) == 2
    derivative = sum(effect[:, None] * owner for effect, owner in events)
    raw_k = data['k'][0, 0].astype(np.float64)
    k = raw_k / (np.linalg.norm(raw_k) + 1e-6 if branch == 'recurrent'
                 else np.sqrt(raw_k @ raw_k + 1e-6))
    s, v = states[1][0].astype(np.float64), data['v'][0, 0].astype(np.float64)
    theta = np.concatenate([a, b, [dt, a_log]])
    def forward(params):
        z = params[:2] @ x + params[4]
        alpha = np.exp(-np.exp(params[5]) * np.logaddexp(0., z))
        beta = 1 / (1 + np.exp(-(params[2:4] @ x)))
        decayed = alpha * s
        return (decayed + beta * np.outer(k, v - k @ decayed)).ravel()
    numerical = np.empty((4, 6))
    eps = 1e-5
    for j in range(6):
        d = np.eye(6)[j] * eps
        numerical[:, j] = (forward(theta + d) - forward(theta - d)) / (2 * eps)
    np.testing.assert_allclose(derivative[:4, :6], numerical, atol=2e-7, rtol=2e-6)
    assert not derivative[:, 6:].any()
    assert not events[0][1][p.parameter_slices['layer1.b_proj.weight']].any()
    assert not events[1][1][p.parameter_slices['layer1.a_proj.weight']].any()


@pytest.mark.parametrize('alpha,beta,k', [(.3, .6, [1., 0.]), (1., 0., [0., 0.]),
                                         (.8, 1., [2., 0.])])
def test_source_free_transport_self_only_radii_and_untouched_identity(alpha, beta, k):
    p = make_producer()
    rng = np.random.default_rng(11)
    p.code.U[:] = rng.normal(size=p.code.U.shape)
    p.code.V[:] = rng.normal(size=p.code.V.shape)
    p.code.bills[:] = [2., 0., 0.]
    p.code.fields[:] = [3., 5.]
    before = p.code.copy()
    name = p.layout.head_fields[1, 0]
    sl = p.layout.slices[name]
    k = np.array(k)
    old = before.U[:, sl].astype(np.float64).reshape(2, 2, 2)
    expected = np.stack([alpha * (u - beta * np.outer(k, k @ u)) for u in old])
    p._transport_head(name, k, alpha, beta)
    np.testing.assert_array_equal(p.code.U[:, sl], expected.astype(np.float32).reshape(2, -1))
    np.testing.assert_array_equal(p.code.U[:, 4:], before.U[:, 4:])
    np.testing.assert_array_equal(p.code.V, before.V)
    gain = abs(alpha) * max(1., abs(1. - beta * (k @ k)))
    assert p.code.bills[0] == pytest.approx(2 * max(1., gain))
    assert p.code.fields[0] >= 3 * gain
    assert p.code.fields[0] - 3 * gain < 1e-5
    assert p.code.fields[1] == 5.


def test_raw_credit_ignores_strict_margin_and_does_not_rescale_teaching():
    p = make_producer()
    p.code.U[:] = .25
    p.code.V[:] = 2.
    p.code.bills[:] = 1e10
    p.code.fields[:] = 1e10
    teaching = {l: np.ones((1, 1, 2, 2), dtype=np.float32) * 10 for l in SPECS}
    readout = ArraySource(np.full(8, 10.))
    strict = capture_credit_blocked(p.code, readout, p.layout,
                                   NumericalBudget(contraction=ErrorTerm(0.)), block_size=3)
    assert not strict['margin_pass'].any()
    assert not strict['conditional_credit'].any()
    raw = p.raw_credit(teaching)
    np.testing.assert_array_equal(np.concatenate([g.ravel() for g in raw.values()]), strict['estimate'])
    assert all(np.all(g == 80.) for g in raw.values())
    assert p.last_credit['norm'] > 0


def test_state_disk_restore_incomplete_capture_and_next_advance_equal(tmp_path):
    p = make_producer()
    fill(p, start=900, version=4)
    p.advance_window()
    p.open_window(1028, 5, initial_s(5))
    p.add_chunk(1, 0, packet(1, 64))
    p.add_chunk(12, 0, packet(12, 64))
    state = p.state_dict()
    path = tmp_path / 'eligibility.pkl'
    with path.open('wb') as f:
        pickle.dump(state, f)
    with path.open('rb') as f:
        restored = pickle.load(f)
    restored['specs'] = {str(l): s for l, s in restored['specs'].items()}
    for name in ('initial_s', 'counts', 'chunks'):
        restored['pending'][name] = {str(l): v for l, v in restored['pending'][name].items()}
    q = make_producer()
    q.load_state_dict(restored)
    assert_code_equal(p, q)
    assert not q.is_complete() and q.pending['counts'] == {1: 64, 12: 64}
    # Both export and import own their arrays.
    state['code']['U'].fill(42)
    restored['pending']['initial_s']['1'].fill(42)
    assert not np.all(p.code.U == 42)
    assert not np.all(q.pending['initial_s'][1] == 42)
    for producer in (p, q):
        for layer in SPECS:
            producer.add_chunk(layer, 64, packet(20 + layer, 64, 'chunk'))
        producer.set_end_s(reconstructed(producer))
    teach = initial_s(6)
    for name, value in p.raw_credit(teach).items():
        np.testing.assert_array_equal(value, q.raw_credit(teach)[name])
    a, b = p.advance_window(), q.advance_window()
    assert_code_equal(p, q)
    assert a['reconstruction_delta'] == b['reconstruction_delta']
    fill(p, 1156, 6)
    fill(q, 1156, 6)
    p.advance_window()
    q.advance_window()
    assert_code_equal(p, q)


def test_counts_versions_incomplete_and_migration_never_backfill():
    p = make_producer()
    with pytest.raises(ValueError, match='incomplete'):
        p.advance_window()
    p.open_window(450757, 524, initial_s())
    assert not p.code.U.any() and p.clock['windows'] == 0
    p.add_chunk(1, 0, packet(length=64))
    with pytest.raises(ValueError, match='already open'):
        p.open_window(450757, 524, initial_s())
    with pytest.raises(ValueError, match='incomplete'):
        p.advance_window()
    assert p.pending['counts'] == {1: 64, 12: 0}
    assert not p.code.U.any()
    with pytest.raises(ValueError, match='offset'):
        p.add_chunk(1, 0, packet(length=64))
    with pytest.raises(ValueError, match='exceeds 128'):
        p.add_chunk(1, 64, packet(length=65))
    p.add_chunk(1, 64, packet(length=64))
    assert not p.is_complete()
    p.add_chunk(12, 0, packet())
    p.advance_window()
    with pytest.raises(ValueError, match='no backfill'):
        p.open_window(0, 525, initial_s())
    with pytest.raises(ValueError, match='version'):
        p.open_window(450885, 523, initial_s())
    p.open_window(450885, 524, initial_s())  # observe may retain parameter version


@pytest.mark.parametrize('field', ['x', 'k', 'v', 'g', 'beta', 'a_slope', 'Aexp'])
def test_packet_shape_validation_and_shared_head_count(field):
    p = make_producer()
    p.open_window(0, 0, initial_s())
    data = packet()
    data[field] = data[field][..., None]
    with pytest.raises(ValueError, match='expected CPU numpy array|packet length'):
        p.add_chunk(1, 0, data)
    assert p.pending['counts'][1] == 0


def test_bounded_storage_copies_and_budget_guard():
    p = make_producer()
    states, data = initial_s(), packet(length=1)
    p.open_window(0, 0, states)
    original = states[1].copy()
    states[1].fill(30)
    np.testing.assert_array_equal(p.pending['initial_s'][1], original)
    for layer in SPECS:
        for offset in range(128):
            p.add_chunk(layer, offset, data)
    data['x'].fill(90)
    assert not np.all(p.pending['chunks'][1][0]['x'] == 90)
    assert len(p.pending['chunks'][1]) == 128
    payload = sum(sum(value.nbytes for value in chunk.values() if isinstance(value, np.ndarray))
                  for chunks in p.pending['chunks'].values() for chunk in chunks)
    assert p.nbytes == p.code.nbytes + payload + 32
    assert p.nbytes + p.estimated_temporary_bytes < p.max_bytes
    p.advance_window()
    baseline = p.nbytes
    for version in range(1, 3):
        fill(p, version * 128, version)
        assert len(p.pending['chunks'][1]) == 2
        p.advance_window()
        assert p.nbytes == baseline
    with pytest.raises(MemoryError, match='not RSS'):
        NativeSEligibility(SPECS, max_bytes=10)
    limited = make_producer()
    limited.max_bytes = limited.nbytes + limited.estimated_temporary_bytes + 32
    limited.open_window(0, 0, initial_s())
    with pytest.raises(MemoryError):
        limited.add_chunk(1, 0, packet())
    assert limited.pending['counts'] == {1: 0, 12: 0}


def test_multihead_sources_touch_only_their_own_s_and_gate_owner_rows(monkeypatch):
    specs = {l: dict(heads=2, key_dim=2, value_dim=3, width=2) for l in SPECS}
    p = NativeSEligibility(specs, block_size=5)
    states = {l: np.ones((1, 2, 2, 3), dtype=np.float32) for l in SPECS}
    p.open_window(0, 0, states)
    data = dict(x=np.ones((128, 2), dtype=np.float32),
                k=np.ones((128, 2, 2), dtype=np.float32),
                v=np.ones((128, 2, 3), dtype=np.float32),
                g=np.full((128, 2), -.02, dtype=np.float32),
                beta=np.full((128, 2), .3, dtype=np.float32),
                a_slope=np.full((128, 2), .5, dtype=np.float32),
                Aexp=np.ones(2, dtype=np.float32), branch='chunk')
    for layer in specs:
        p.add_chunk(layer, 0, data)
    calls = 0
    def register(code, source, owner, layout, block_size):
        nonlocal calls
        calls += 1
        _, layer, tick, head, kind = source.token
        expected = np.zeros_like(owner, dtype=bool)
        prefix = f'layer{layer}.'
        if kind == 'R':
            sl = p.parameter_slices[prefix + 'a_proj.weight']
            expected[sl.start + head * 2:sl.start + (head + 1) * 2] = True
            for suffix in ('dt_bias', 'A_log'):
                expected[p.parameter_slices[prefix + suffix].start + head] = True
        else:
            sl = p.parameter_slices[prefix + 'b_proj.weight']
            expected[sl.start + head * 2:sl.start + (head + 1) * 2] = True
        np.testing.assert_array_equal(owner != 0, expected)
        assert source.field_slice == layout.slices[layout.head_fields[layer, head]]
        assert source.matrix.shape == (6,)
        assert source(0, layout.size).shape == (24,)
    monkeypatch.setattr(native, 'insert_source_inplace', register)
    result = p.advance_window()
    assert calls == result['inserted_sources'] == 1024


def test_state_integrity_rejects_count_version_or_owner_map_corruption():
    p = make_producer()
    fill(p)
    p.advance_window()
    fill(p, 128, 1)
    for damage in ('count', 'version', 'map', 'clock'):
        state = p.state_dict()
        if damage == 'count':
            state['pending']['counts'][1] = 127
        elif damage == 'version':
            state['pending']['parameter_version'] = -1
        elif damage == 'map':
            state['parameter_slices']['layer1.dt_bias'] = slice(0, 1)
        else:
            state['clock']['inserted_sources'] = 0
        q = make_producer()
        with pytest.raises(ValueError):
            q.load_state_dict(state)
        assert q.pending is None and not q.code.U.any()
