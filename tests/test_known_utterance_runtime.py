"""Execution equivalence, not a synthetic memory capability task."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.latent_navigation import LatentNavigationBody
from scripts.ib_local.variable_expression import configure_variable_expression, film
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.navigation_retrieval import ExactUtteranceTable
from scripts.ib_local.language_navigation_kanli_v4e import EqualEngine


def fixture():
    torch.manual_seed(91)
    base = GatedDeltaNetLM(256, 32, 2, 1, 'decay_then_delta', 64, 4)
    model = LatentNavigationBody(TableGDN(base, True, inject_layer=0), latent_dim=8,
                                capacity=8, horizon=2, max_chunk=256)
    model.retrieval = ExactUtteranceTable(8)
    params = list(model.parameters())
    e = SimpleNamespace(model=model, optimizer=torch.optim.AdamW(params), params=params,
        groups=(params,), prev=None, device=torch.device('cpu'), pending=97,
        recurrent=model.initial_state(1), _draft_actions=[])
    configure_variable_expression(e)
    e.table = RetrievedCreditTable(orders=(0, 1), heads=1, slots_log2=2)
    return e


def table_state(table):
    return [table.rows.copy(), table.count.copy(), table.tag.copy(), table.err.copy(), table.history]


@pytest.mark.parametrize('length', [0, 17, 128])
def test_known_film_preserves_table_rng_endpoint_and_recurrent(length):
    words = bytes((np.arange(length) % 256).astype(np.uint8))
    ref, opt = fixture(), fixture()
    rng = torch.get_rng_state().clone()
    expected = film(ref, words, ref.table)
    opt._utterance_payload_cache = {}
    decoded = []
    handle = opt.model.base.decoder.register_forward_hook(lambda *a:decoded.append(True))
    actual = film(opt, words, opt.table)
    handle.remove()
    if length >= 32: assert decoded == []
    assert torch.equal(rng, torch.get_rng_state())
    assert expected[0] == actual[0] and expected[1] == actual[1]
    assert np.array_equal(np.asarray(expected[2]), np.asarray(actual[2]))
    for a, b in zip(table_state(ref.table), table_state(opt.table)):
        assert np.array_equal(a, b)
    torch.testing.assert_close(ref._draft_actions[-1], opt._draft_actions[-1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ref.decision.state, opt.decision.state, atol=1e-6, rtol=1e-5)


def test_delivery_reuses_executed_state_but_commits_actual_table_once():
    ref, opt = fixture(), fixture()
    words = b'An already executed utterance.'
    opt._utterance_payload_cache = {}
    # Draft changes a branch, not the live table.
    branch = RetrievedCreditTable(orders=(0, 1), heads=1, slots_log2=2)
    film(opt, words, branch)
    ref.decision.start(ref.recurrent)
    a, b = dict(probs=[], choice=2), dict(probs=[], choice=2)
    assert EqualEngine._speak_bytes(ref, words, a, False) == words
    assert EqualEngine._speak_bytes(opt, words, b, False) == words
    assert np.array_equal(a['probs'], b['probs'])
    assert a['ending_reason'] == b['ending_reason']
    for x, y in zip(table_state(ref.table), table_state(opt.table)):
        assert np.array_equal(x, y)
    torch.testing.assert_close(ref.decision.state, opt.decision.state, atol=1e-6, rtol=1e-5)


def test_save_digest_is_reused_only_when_supplied(monkeypatch):
    from scripts.ib_local import life_card
    hashes, cards = [], []
    monkeypatch.setattr(life_card, 'sha256_file', lambda path: hashes.append(path) or 'fallback')
    monkeypatch.setattr(life_card, 'from_state_dict', lambda *a, **kw: kw)
    monkeypatch.setattr(life_card, 'write_card', lambda card, *a, **kw: cards.append(card))
    kwargs = dict(checkpoint='checkpoint', config={}, cycle=4, row={}, output='display', branch_id='branch')
    engine = SimpleNamespace(world=SimpleNamespace(), model=SimpleNamespace(state_dict=lambda:{}))
    life_card.export_live(engine, **kwargs, committed_sha='just-committed')
    assert hashes == [] and cards[-1]['source_sha'] == 'just-committed'
    life_card.export_live(engine, **kwargs)
    assert hashes == ['checkpoint'] and cards[-1]['source_sha'] == 'fallback'


def test_handover_reuses_live_packed_budget_and_preserves_head_gradient():
    from scripts.ib_local.variable_expression import _handover_remaining, handover_logits
    model = SimpleNamespace(handover=torch.nn.Linear(9, 1))
    feature = torch.randn(1, 8)
    for remaining in (1., .75, .125):
        packed = torch.tensor([[[remaining]]])
        decision = SimpleNamespace(capture=True, _input_host=object(), budget=packed)
        actual_remaining = _handover_remaining(decision, remaining)
        assert actual_remaining.data_ptr() == packed.data_ptr()
        expected = handover_logits(model, feature, remaining)
        actual = handover_logits(model, feature, actual_remaining)
        assert torch.equal(expected, actual)
        left = torch.autograd.grad(expected.sum(), tuple(model.handover.parameters()))
        right = torch.autograd.grad(actual.sum(), tuple(model.handover.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(left, right))
    assert _handover_remaining(SimpleNamespace(capture=False), .25) == .25
