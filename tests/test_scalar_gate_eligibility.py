"""Conditional local derivatives and bounded source accounting, not training."""
import copy

import numpy as np
import pytest

from scripts.ib_local.scalar_gate_eligibility import EPS, ScalarGateEligibility


SPECS = {layer: dict(heads=2, key_dim=2, value_dim=3, width=4) for layer in (1,12)}


def packet(seed, branch='recurrent'):
    rng = np.random.default_rng(seed)
    return dict(x=rng.normal(size=(128,4)).astype(np.float32),
                k=rng.normal(size=(128,2,2)).astype(np.float32),
                v=rng.normal(size=(128,2,3)).astype(np.float32),
                g=-rng.uniform(.001,.009,size=(128,2)).astype(np.float32),
                beta=rng.uniform(.1,.9,size=(128,2)).astype(np.float32),
                a_slope=np.full((128,2),.5,np.float32),
                Aexp=np.ones(2,np.float32), branch=branch, offset=0)


def fixture(branch='recurrent', windows=3):
    rng = np.random.default_rng(42)
    initial = {l: rng.normal(size=(2,2,3)) for l in SPECS}
    packets = [{l: [packet(20*w+l,branch)] for l in SPECS} for w in range(windows)]
    return initial, packets


def physical(initial, chunks, selected=None, theta=0.):
    """Independent primal reference with direct selected A_log perturbation."""
    state = {l: v.copy().astype(np.float64) for l,v in initial.items()}
    for layer in SPECS:
        for p in chunks[layer]:
            for tick in range(len(p['x'])):
                for h in range(SPECS[layer]['heads']):
                    raw = p['k'][tick,h].astype(np.float64)
                    k = raw / (np.linalg.norm(raw)+EPS if p['branch']=='recurrent'
                               else np.sqrt(raw@raw+EPS))
                    g = float(p['g'][tick,h]) * (np.exp(theta) if (layer,h)==selected else 1.)
                    alpha, beta = np.exp(g), float(p['beta'][tick,h])
                    s = state[layer][h]
                    state[layer][h] = alpha*s + beta*np.outer(
                        k, p['v'][tick,h].astype(np.float64) - k@(alpha*s))
    return state


@pytest.mark.parametrize('branch',['recurrent','chunk'])
@pytest.mark.parametrize('windows',[2,3])
@pytest.mark.parametrize('selected',[(1,0),(12,1)])
def test_strict_old_direct_scalar_matches_finite_difference(branch, windows, selected):
    initial, packets = fixture(branch,windows)
    producer = ScalarGateEligibility(SPECS)
    nominal = initial
    for i,chunks in enumerate(packets):
        producer.advance(nominal,chunks,128*i)
        nominal = physical(nominal,chunks)
    eps = 1e-5
    variations = []
    for theta in (-eps,eps):
        state = initial
        for i,chunks in enumerate(packets):
            state = physical(state,chunks,selected if i < windows-1 else None,theta)
        variations.append(state)
    numerical = (variations[1][selected[0]] - variations[0][selected[0]])/(2*eps)
    expected = producer.older[selected[0]]
    np.testing.assert_allclose(numerical[selected[1]],expected[selected[1]],rtol=2e-7,atol=2e-9)
    other = 1-selected[1]
    np.testing.assert_array_equal(numerical[other],np.zeros_like(numerical[other]))


def test_source_age_order_and_credit_only_older():
    initial, packets = fixture()
    p = ScalarGateEligibility(SPECS)
    teaching = {l: np.ones((1,2,2,3),np.float32) for l in SPECS}
    assert all(not a.any() for a in p.credit(teaching).values())
    p.advance(initial,packets[0],4096)
    assert all(not a.any() for a in p.credit(teaching).values())
    assert any(a.any() for a in p.previous.values())
    p.advance(physical(initial,packets[0]),packets[1],4224)
    for layer in SPECS:
        np.testing.assert_allclose(p.credit(teaching)[f'layer{layer}.A_log'],
                                   p.older[layer].sum(axis=(1,2)),rtol=1e-7)
        assert not p.current[layer].any()


def test_split_chunks_equal_whole_and_inputs_untouched():
    initial, packets = fixture(windows=1)
    chunks = packets[0]
    original = copy.deepcopy(chunks)
    split = {}
    for layer, entries in chunks.items():
        p = entries[0]
        split[layer] = []
        for lo,hi in ((0,39),(39,128)):
            q = {k: v[lo:hi].copy() if isinstance(v,np.ndarray) and k!='Aexp'
                 else copy.deepcopy(v) for k,v in p.items()}
            q['offset'] = lo
            split[layer].append(q)
    a,b = ScalarGateEligibility(SPECS),ScalarGateEligibility(SPECS)
    a.advance(initial,chunks,0)
    b.advance(initial,split,0)
    for layer in SPECS:
        np.testing.assert_array_equal(a.previous[layer],b.previous[layer])
        for key,value in chunks[layer][0].items():
            if isinstance(value,np.ndarray):
                np.testing.assert_array_equal(value,original[layer][0][key])


def test_restore_continuation_identical_and_state_copies():
    initial, packets = fixture()
    a,b = ScalarGateEligibility(SPECS),ScalarGateEligibility(SPECS)
    a.advance(initial,packets[0],0)
    state = a.state_dict()
    b.load_state_dict(state)
    state['previous'][1].fill(99.)
    nominal = physical(initial,packets[0])
    for i in (1,2):
        a.advance(nominal,packets[i],128*i)
        b.advance(nominal,packets[i],128*i)
        nominal = physical(nominal,packets[i])
    for name in ('older','previous','current'):
        for layer in SPECS:
            np.testing.assert_array_equal(getattr(a,name)[layer],getattr(b,name)[layer])
    assert a.seen == b.seen == 384 and a.windows == b.windows == 3


def test_real_shape_resident_budget_constant():
    specs = {l: dict(heads=4,key_dim=256,value_dim=256,width=1024) for l in (1,12)}
    p = ScalarGateEligibility(specs)
    assert p.nbytes == 3*8*256*256*8 == 12*2**20


@pytest.mark.parametrize('kind',['gap','incomplete','nan','branch'])
def test_reject_bad_windows_before_mutation(kind):
    initial, packets = fixture(windows=1)
    p = ScalarGateEligibility(SPECS)
    chunks = packets[0]
    if kind=='gap':
        p.seen = 128
    elif kind=='incomplete':
        chunks[1] = []
    elif kind=='nan':
        chunks[1][0]['g'][0,0] = np.nan
    else:
        chunks[1][0]['branch'] = 'unknown'
    with pytest.raises(ValueError):
        p.advance(initial,chunks,0)
    assert all(not v.any() for c in (p.older,p.previous,p.current) for v in c.values())


def test_load_rejects_nonfinite_unsettled_and_fictional_history():
    p = ScalarGateEligibility(SPECS)
    for field,value in (('older',np.nan),('current',1.),('older',1.)):
        state = p.state_dict()
        state[field][1][0,0,0] = value
        with pytest.raises(ValueError):
            p.load_state_dict(state)
