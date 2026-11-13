"""Answer-mask and prompt-boundary checks, not synthetic capability studies."""
import pytest
import torch

from scripts.ib_local.pretrained_gdn_learning import language_loss
from scripts.ib_local.probe_pretrained_gdn_sft_credit import windows


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_answer_supervision_has_zero_gradient_on_prompt_and_padding():
    logits = torch.randn(1, 5, 7, requires_grad=True)
    target = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor([[False, False, True, True, False]])
    _, loss = language_loss(logits, target, mask)
    loss.backward()
    assert logits.grad[0, :2].count_nonzero() == 0
    assert logits.grad[0, 4].count_nonzero() == 0
    assert logits.grad[0, 2:4].norm() > 0


def test_answer_mask_matches_all_answer_tokens_and_ignores_eos_padding():
    tokenizer = CharacterTokenizer()
    ids, mask, accounting = windows(tokenizer, 'A real passage.', 'Which passage?', 'A real passage')
    assert len(mask) % 128 == 0
    assert len(ids) == len(mask)+1
    assert int(mask.sum()) == accounting['answer_tokens']
    assert ''.join(chr(c) for c in ids[1:][mask].tolist()) == ' A real passage'
    assert not bool(mask[accounting['real_tokens']-1:].any())


def test_empty_supervision_is_rejected():
    with pytest.raises(ValueError, match='nonempty'):
        language_loss(torch.randn(1, 3, 5), torch.zeros(1, 3, dtype=torch.long),
                      torch.zeros(1, 3, dtype=torch.bool))


def test_branch_restore_keeps_source_cpu_optimizer_moments_immutable(monkeypatch):
    import types
    import scripts.ib_local.pretrained_gdn_learning as module
    from scripts.ib_local.pretrained_gdn_experience import copy_tree
    learner = module.PretrainedExperienceLearner.__new__(module.PretrainedExperienceLearner)
    learner.model = torch.nn.Linear(2, 1)
    learner.native, learner.masters, learner.memories = [], [], {}
    learner.null_parameters = []
    learner.layers, learner.capacity = (1,), 2
    learner.directions = torch.zeros(2)
    # Match the learner's native, recall and null parameter-group schema.
    learner.optimizer = torch.optim.Adam([
        dict(params=learner.model.parameters()), dict(params=[]), dict(params=[])])
    learner.model(torch.ones(1, 2)).sum().backward()
    learner.optimizer.step()
    state = dict(schema=1, layers=(1,), capacity=2, native={}, masters=[], memories={},
        optimizer=copy_tree(learner.optimizer.state_dict()), directions=torch.zeros(2),
        cache=dict(states=[], seen_tokens=0), updates=1, tokens=128,
        rng=torch.get_rng_state(), cuda_rng=torch.zeros(1, dtype=torch.uint8))
    expected = copy_tree(state['optimizer'])
    monkeypatch.setattr(module, 'new_recurrent_cache', lambda n: types.SimpleNamespace(states=[]))
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state: None)
    learner.load_state_dict(state)
    learner.optimizer.step()
    for key, moments in expected['state'].items():
        for name, value in moments.items():
            torch.testing.assert_close(state['optimizer']['state'][key][name], value, atol=0, rtol=0)


def test_restored_capsule_slots_do_not_modify_source_checkpoint():
    from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall
    memory = LayerExperienceRecall(6, capacity=2, latent_dim=3, topk=2)
    state = memory.get_extra_state()
    memory.set_extra_state(state)
    memory.capsules[0] = object()
    assert state['capsules'][0] is None


def test_state_cut_preserves_other_route_and_source_cache(monkeypatch):
    import copy
    import types
    import scripts.ib_local.probe_pretrained_gdn_state_route as module
    cache = types.SimpleNamespace(states=[dict(recurrent_state=torch.ones(2, 3),
        conv_state=(torch.full((2, 4), 2.),), untouched=7)], _seen_tokens=512)
    monkeypatch.setattr(module, 'clone_cache', copy.deepcopy)
    state_cut = module.cut_cache(cache, 'S')
    assert state_cut.states[0]['recurrent_state'].count_nonzero() == 0
    torch.testing.assert_close(state_cut.states[0]['conv_state'][0], cache.states[0]['conv_state'][0])
    conv_cut = module.cut_cache(cache, 'conv')
    torch.testing.assert_close(conv_cut.states[0]['recurrent_state'], cache.states[0]['recurrent_state'])
    assert conv_cut.states[0]['conv_state'][0].count_nonzero() == 0
    both = module.cut_cache(cache, 'all')
    assert both.states[0]['recurrent_state'].count_nonzero() == 0
    assert both.states[0]['conv_state'][0].count_nonzero() == 0
    assert cache.states[0]['recurrent_state'].min() == 1
    assert cache.states[0]['conv_state'][0].min() == 2
    assert state_cut._seen_tokens == 512


def test_historical_gate_intervention_excludes_current_credit_window():
    from scripts.ib_local.probe_pretrained_gdn_historical_forgetting import historical_interval
    start,end=historical_interval(350)
    assert start==128
    assert end==221
    assert 350-end>128
    with pytest.raises(ValueError):
        historical_interval(257)


def test_temporary_decay_change_restores_parameters_even_on_failure():
    import types
    from scripts.ib_local.probe_pretrained_gdn_historical_forgetting import shift_decay
    parameter=torch.nn.Parameter(torch.tensor([-.3,.7]))
    model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[
        types.SimpleNamespace(attn=types.SimpleNamespace(A_log=parameter))]))
    original=parameter.detach().clone()
    with pytest.raises(RuntimeError):
        with shift_decay(model,.25):
            torch.testing.assert_close(parameter,original+.25)
            raise RuntimeError('interrupted observation')
    torch.testing.assert_close(parameter,original,atol=0,rtol=0)
