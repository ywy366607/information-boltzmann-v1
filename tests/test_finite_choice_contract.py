"""Numeric/interface binding to the production selectors and actual finite U."""
import pytest
import torch

from scripts.ib_local.finite_choice_contract import choose, trace_choices, tree_expectation
from test_relation_retention_use import configured


def test_trace_preserves_original_draw_rng_and_scoped_conditioning():
    p = torch.tensor([.2, .3, .5])
    state = torch.get_rng_state()
    expected = int(torch.multinomial(p, 1))
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    with trace_choices() as trace:
        assert choose(p, stage='question') == expected
    assert torch.equal(torch.get_rng_state(), after)
    torch.set_rng_state(state)
    with trace_choices({'question': (expected+1) % 3}) as fixed:
        assert choose(p, stage='question') == (expected+1) % 3
    assert torch.equal(torch.get_rng_state(), after)
    assert fixed.events[0]['sampled'] == expected
    assert trace.events[0]['selected'] == expected
    with trace_choices({'question': 3}):
        with pytest.raises(ValueError):
            choose(p, stage='question')


def test_complete_body_selectors_use_actual_probabilities_and_tree(tmp_path):
    e, archive = configured(tmp_path)
    try:
        e.step(torch.randn(8, 4))
        with trace_choices() as trace:
            row = e.step(torch.randn(8, 4))
        events = {x['stage']: x for x in trace.events}
        question, proposal = events['question'], events['proposal']
        torch.testing.assert_close(question['probability'],
            question['probability'].new_tensor(row['generative_relation']['query_probabilities']))
        torch.testing.assert_close(proposal['probability'],
            proposal['probability'].new_tensor(e._joint_selection['distribution']))
        assert row['causal_use']['finite_U']['verified']
        assert row['generative_relation']['binding_replay_error'] < 1e-5
        assert row['joint_language_search']['replay_max_error'] < 1e-3
        # Bind the contraction to this body's actual actor and actual U values.
        pi = torch.tensor(row['choice_probability'], dtype=torch.float64)
        q = torch.tensor(row['causal_use']['epistemic_value']['local_expected'], dtype=torch.float64)
        result = tree_expectation(torch.ones(1), [torch.ones(1)], [[pi]], [[q]])
        torch.testing.assert_close(result['expected'], (pi*q).sum())
        torch.testing.assert_close(result['total_mass'], torch.ones((), dtype=torch.float64))
    finally:
        archive.close()
