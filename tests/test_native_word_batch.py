"""Numerical equivalence of private original-GDN word trajectory batching."""
import copy

import numpy as np
import pytest
import torch

from scripts.ib_local.known_utterance_runtime import known_film
from scripts.ib_local.native_word_batch import endpoints, request
from scripts.ib_local.preview_runtime import preview_scope
from scripts.ib_local.table_branch import table_branch
from test_known_utterance_runtime import fixture, table_state


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('factorized', [False, True])
def test_original_endpoints_match_padding_questions_tables_and_state(batch_size, factorized):
    e = fixture()
    e.native_factorized_endpoints = factorized
    words = [b'', b'Natural language.', b'A short observation of the world.', bytes(range(64))]
    e.model.question_to_hidden = torch.nn.Linear(8, 64, bias=False)
    for value in [torch.randn(8), torch.randn(8)]:
        e.model._question_context = value
        e.recurrent.normal_(std=.03)
        old_state = e.decision.state.clone()
        old_table = table_state(e.table)
        rng = torch.get_rng_state().clone()
        expected = {}
        for w in words:
            with table_branch(e.table) as branch:
                known_film(e, w, branch)
            expected[w] = e._draft_actions.pop().clone()
        e.decision.state.copy_(old_state)
        receipt = copy.deepcopy(e._last_expression_receipt)
        with preview_scope(e.model):
            actual = endpoints(e, words, batch_size=batch_size)
        for w in words:
            torch.testing.assert_close(actual[w][0], expected[w], atol=3e-6, rtol=2e-5)
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(e.decision.state, old_state)
        assert e._last_expression_receipt.keys() == receipt.keys()
        assert np.array_equal(e._last_expression_receipt['final_probability'], receipt['final_probability'])
        for a, b in zip(old_table, table_state(e.table)):
            assert np.array_equal(a, b)
        assert request.get() is None
        assert len(e._draft_actions) == 0


def test_private_batch_uses_original_feature_boundary_and_removes_hooks(monkeypatch):
    e = fixture()
    e.model.experience_layer = 0
    calls = []
    handle = e.model.base.layers[1].register_forward_hook(lambda *a:calls.append(True))
    old = len(e.model.base.layers[0]._forward_pre_hooks)
    with preview_scope(e.model):
        result = endpoints(e, [b'An observed event.', b'Another observation.'])
    assert len(result) == 2 and calls == []
    assert len(e.model.base.layers[0]._forward_pre_hooks) == old
    handle.remove()
    assert request.get() is None


def test_preview_scoped_multi_read_batches_only_missing_keys():
    from scripts.ib_local.preview_runtime import read_many
    model = object(); calls = []
    def compute(keys):
        calls.append(list(keys)); return [k*2 for k in keys]
    with torch.no_grad(), preview_scope(model):
        assert read_many(model, 'x', [1, 2, 1], compute) == [2, 4, 2]
        assert read_many(model, 'x', [2, 3], compute) == [4, 6]
    assert calls == [[1, 2], [3]]


def test_different_questions_share_physical_prefix_and_keep_their_own_film(monkeypatch):
    from scripts.ib_local import native_word_batch as batch
    e = fixture()
    e.model.experience_layer = 1
    e.model.question_to_hidden = torch.nn.Linear(8, 64, bias=False)
    e.model.expression_budget_to_hidden.weight.data.normal_(std=.03)
    e.recurrent.normal_(std=.03)
    words = [b'', b'A known payload.', b'A longer historical experience.']
    contexts = [torch.randn(8), torch.randn(8)]
    expected = []
    for q in contexts:
        e.model._question_context = q
        expected.append(batch._endpoints(e, words))
    prefix_calls = []
    run = batch._endpoints

    def counted(*args, **kwargs):
        prefix_calls.append(kwargs.get('raw'))
        return run(*args, **kwargs)

    monkeypatch.setattr(batch, '_endpoints', counted)
    e.native_factorized_endpoints = True
    with torch.no_grad(), preview_scope(e.model):
        for q, reference in zip(contexts, expected):
            e.model._question_context = q
            actual = endpoints(e, words)
            for word in words:
                torch.testing.assert_close(actual[word], reference[word], atol=3e-6, rtol=2e-5)
    assert prefix_calls == [True]
    assert not torch.equal(expected[0][words[1]], expected[1][words[1]])
    assert request.get() is None
