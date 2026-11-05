"""Check frozen table reads against the existing boundary read operation."""
import numpy as np
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_live_witness import frozen_table_logp
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.language_navigation_kanli_v4s import RepresentativeWitnessEngine
from types import SimpleNamespace


def test_frozen_read_matches_boundary_peek_and_does_not_write():
    table = RetrievedCreditTable(orders=(0,1,2), heads=1, slots_log2=4)
    table.run_credit(np.frombuffer(b'abracadabra and abracadabra', np.uint8))
    before = table.state_dict()
    ids = np.frombuffer(b'another natural string', np.uint8)
    logp = frozen_table_logp(table, ids)
    history = table.history
    table.history = np.uint64(0)
    for i,b in enumerate(ids):
        np.testing.assert_allclose(np.exp(logp[i]), peek(table, b), rtol=2e-6, atol=1e-7)
        table.history = np.uint64(((int(table.history) << 8) | int(b)) & ((1<<64)-1))
    table.history = history
    after = table.state_dict()
    assert before['history'] == after['history'] and before['events'] == after['events']
    assert all(np.array_equal(before['arrays'][k],after['arrays'][k]) for k in before['arrays'])


def test_admission_is_independent_of_surprise_and_capacity_stays_bounded():
    def engine():
        return SimpleNamespace(w_rng=np.random.default_rng(11),device='cpu',capacity=3,
            w_ids=[],w_tgt=[],w_logp=[],w_hear=[],w_admitted=0)
    low,high=engine(),engine()
    response=np.arange(200,dtype=np.uint8)
    probs=np.full((200,256),1/256,np.float32)
    for _ in range(20):
        assert RepresentativeWitnessEngine._admit(low,response,probs,np.zeros(200),0)
        assert RepresentativeWitnessEngine._admit(high,response,probs,np.full(200,10.),0)
    assert low.w_admitted==high.w_admitted==20
    assert len(low.w_ids)==len(high.w_ids)==3
    for a,b in zip(low.w_ids,high.w_ids):
        np.testing.assert_array_equal(a.numpy(),b.numpy())
    assert low.w_hear==[0.]*3 and high.w_hear==[10.]*3


def test_frozen_continuation_score_sees_table_learning_without_learning_targets():
    table = RetrievedCreditTable(orders=(0,1,2), heads=1, slots_log2=8)
    response = np.frombuffer(b'ab'*32, np.uint8)
    continuation = np.frombuffer(b'ab'*16, np.uint8)
    prefix = np.concatenate((np.array([ord('b')], np.uint8), response, continuation[:-1]))
    def score():
        logp = frozen_table_logp(table, prefix)[len(response):]
        return float(-logp[np.arange(len(continuation)), continuation].mean())
    before = score()
    assert before-score() == 0.
    table.run_credit(np.concatenate((np.array([ord('b')], np.uint8), response)))
    events = table.events
    after = score()
    assert after < before
    assert after == score() and table.events == events
