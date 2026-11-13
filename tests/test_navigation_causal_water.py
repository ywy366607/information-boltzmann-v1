"""Same-body storage, read-credit derivatives and continuation contracts.

Numerical/interface tests only; these are not language-capability experiments.
"""
import copy

import pytest
import torch
from torch import nn

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.language_navigation_kanli_v4 import attach
from scripts.ib_local.language_navigation_kanli_v4 import KanLiEngine
from scripts.ib_local.language_navigation_kanli_v4l import LiveSigregBody
from scripts.ib_local.language_navigation_v2 import DialogueEngine
from scripts.ib_local.latent_navigation_v3 import ExperienceMemory
from scripts.ib_local.navigation_read_credit import configure_read_credit


def body():
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32.,
                          decay_init='log_timescale')
    return attach(LiveSigregBody(TableGDN(base, True), latent_dim=8, capacity=8,
                                short=1, long=4, topk=2, history_size=4, max_chunk=4))


def populated(long=2):
    memory = ExperienceMemory(2, capacity=2, short=1, long=long, topk=2,
                              mode='fifo', history_size=2)
    memory.read_credit_enabled = True
    memory.clock.fill_(10000)
    encoder = nn.Identity()
    x, y = torch.eye(2)
    memory.commit_episode(x, x, x, x, encoder, progress=.2, source_time=0)
    memory.commit_episode(y, y, y, y, encoder, progress=.6, source_time=1)
    memory.begin_read_credit()
    return memory, encoder


def test_exact_words_are_separate_and_legacy_body_load_is_lossless():
    model = body()
    m = model.memory
    m.clock.fill_(10)
    x = torch.arange(16, dtype=torch.float32)
    slot, _ = m.commit_experience(x, x, x, x, model.encoder, b'past words', .2, 0)
    assert m.words.data_ptr() == model.retrieval.words.data_ptr()
    assert not any('words' in k or 'word_count' in k for k in m.state_dict())
    assert bytes(model.retrieval.words[slot, :10].tolist()) == b'past words'
    state = copy.deepcopy(model.state_dict())
    # Reconstruct the exact legacy serialization: byte payload under memory,
    # no read-credit state. Every learned tensor is retained unchanged.
    legacy = copy.deepcopy(state)
    for name in ('words', 'word_count'):
        legacy['memory.' + name] = legacy.pop('retrieval.' + name)
    for name in list(legacy):
        if name.startswith('memory.read_credit_'):
            del legacy[name]
    for name in list(legacy['memory._extra_state']):
        if name.startswith('read_credit_'):
            del legacy['memory._extra_state'][name]
    restored = body()
    restored.load_state_dict(legacy, strict=True)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            if k.startswith('memory.read_credit_'):
                assert not restored.state_dict()[k].any()
                continue
            torch.testing.assert_close(restored.state_dict()[k], v, rtol=0, atol=0)
    # deepcopy must bind the copied memory to its own retrieval, never the live body.
    clone = copy.deepcopy(restored)
    assert clone.memory.words.data_ptr() == clone.retrieval.words.data_ptr()
    assert clone.memory.words.data_ptr() != restored.retrieval.words.data_ptr()


def test_language_error_changes_next_prediction_and_choice_through_old_records():
    from types import SimpleNamespace
    torch.manual_seed(31)
    model = body().double()
    ids = torch.tensor([[31, 32, 33, 34]])
    targets = torch.tensor([[32, 33, 34, 35]])
    logp = torch.full((1, 4, 256), -5.545177, dtype=torch.float64)
    state = model.initial_state(1)
    with torch.no_grad():
        _, _, _, features, _ = model.predict_observed(ids, state, logp)
        source = features[0].clone()
        action = torch.eye(16, dtype=torch.float64)[0]
        model.memory.clock.fill_(10000)
        for i, sign in enumerate((1., -1.)):
            model.memory.commit_experience(source, sign*action, source, source,
                model.encoder, bytes([65+i]), .2, i)
        engine = SimpleNamespace(model=model)
        before_logits = model.predict(ids, state, logp)[0].clone()
        before_choice = KanLiEngine.plan(engine, source[None])[0].clone()
    assert int(model.memory.occupied.sum()) == 2
    parameters = [p.detach().clone() for p in model.parameters()]
    loss, _, _, _ = model.objective(ids, targets, state, logp, torch.randn(8, 4, dtype=torch.float64))
    loss.backward()
    info = model.memory.settle_read_credit()
    assert info['records'] == 2 and model.memory.read_credit_mean.abs().sum() > 0
    with torch.no_grad():
        after_logits = model.predict(ids, state, logp)[0]
        after_choice = KanLiEngine.plan(engine, source[None])[0]
    assert not torch.equal(after_logits, before_logits)
    assert not torch.equal(after_choice, before_choice)
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), parameters))


def test_present_error_credits_old_records_and_changes_next_read_not_old_gain():
    m, enc = populated()
    query = torch.tensor([[1., 1.]])
    gains, content = m.progress.clone(), m.action.clone()
    m.begin_read_credit()
    read = m.read(query, enc)
    target = torch.tensor([[1., 0.]])
    loss = (read.value-target).square().sum()
    loss.backward()
    packet = m._read_participation[0]
    # Exact central difference of the very additive logits probed by this read.
    eps = 1e-3
    for j in range(2):
        b = torch.zeros_like(packet.bias)
        b[0, j] = eps
        values = m.action[packet.slots]
        plus = ((b.softmax(-1)[..., None]*values).sum(-2)-target).square().sum()
        minus = (((-b).softmax(-1)[..., None]*values).sum(-2)-target).square().sum()
        assert float(packet.bias.grad[0, j]) == pytest.approx(float((plus-minus)/(2*eps)), abs=5e-5)
    info = m.settle_read_credit()
    assert info['records'] == 2 and info['participations'] == 2
    assert m.read_credit_mean[0] > 0 and m.read_credit_mean[1] < 0
    with torch.no_grad():
        after = m.read(query, enc)
    assert (after.value-target).square().sum() < loss.detach()
    assert torch.equal(m.progress, gains) and torch.equal(m.action, content)
    assert not m._read_participation
    assert m.settle_read_credit()['participations'] == 0


def test_overwritten_record_cannot_receive_an_old_receipt():
    m, enc = populated()
    m.begin_read_credit()
    read = m.read(torch.tensor([[1., 1.]]), enc)
    (read.value-torch.tensor([[1., 0.]])).square().sum().backward()
    x = torch.tensor([1., 0.])
    m.commit_episode(x, x, x, x, enc, progress=.9, source_time=4)
    info = m.settle_read_credit()
    assert info['stale'] == 1 and info['records'] == 1
    assert m.read_credit_count[0] == 0 and m.read_credit_mean[0] == 0


def test_merge_invalidates_receipt_without_rewriting_observed_gain_as_credit():
    m, enc = populated()
    m.mode = 'merge'
    read = m.read(torch.tensor([[1., 1.]]), enc)
    (read.value-torch.tensor([[1., 0.]])).square().sum().backward()
    x = torch.tensor([1., 0.])
    slot, merged = m.commit_episode(x, x, x, x, enc, progress=.8, source_time=4)
    assert slot == 0 and merged
    gain = m.progress.clone()
    info = m.settle_read_credit()
    assert info['stale'] == 1 and m.read_credit_count[0] == 0
    assert torch.equal(m.progress, gain)


def test_no_grad_and_disabled_reads_do_not_create_eligibility():
    m, enc = populated()
    with torch.no_grad():
        m.read(torch.tensor([[1., 1.]]), enc)
    assert not m._read_participation
    m.read_credit_enabled = False
    m.read(torch.tensor([[1., 1.]]), enc)
    assert not m._read_participation


def test_read_credit_workspace_is_bounded_and_closed_after_settlement():
    m, enc = populated()
    for _ in range(6):
        m.read(torch.tensor([[1., 1.]]), enc)
    with pytest.raises(RuntimeError, match='bounded read-credit batch'):
        m.read(torch.tensor([[1., 1.]]), enc)
    assert m._read_credit_entries == 12
    m.settle_read_credit()
    for _ in range(20):
        m.read(torch.tensor([[1., 1.]]), enc)
    assert not m._read_participation and m._read_credit_entries == 0


def test_nonfinite_credit_is_rejected_before_any_persistent_update():
    m, enc = populated()
    read = m.read(torch.tensor([[1., 1.]]), enc)
    read.value[:, 0].sum().backward()
    m._read_participation[0].bias.grad.fill_(float('nan'))
    before = m.read_credit_mean.clone()
    with pytest.raises(FloatingPointError):
        m.settle_read_credit()
    assert torch.equal(m.read_credit_mean, before)
    assert not m.read_credit_count.any()


def test_read_preference_matches_adam_on_the_actual_attention_gradient():
    m, enc = populated()
    q = nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([q], lr=3e-4)
    configure_read_credit(m, 'adam', optimizer.param_groups[0])
    query = torch.tensor([[1., 1.]])
    for target in (torch.tensor([[1., 0.]]), torch.tensor([[.3, .7]]), torch.tensor([[1., 0.]])):
        m.begin_read_credit()
        read = m.read(query, enc)
        (read.value-target).square().sum().backward()
        m.settle_read_credit()
        optimizer.zero_grad(set_to_none=True)
        weights = q.tanh().softmax(-1)
        (weights[None]-target).square().sum().backward()
        optimizer.step()
        torch.testing.assert_close(m.read_credit_mean, q, rtol=3e-5, atol=1e-9)
    assert m.read_credit_steps.tolist() == [3, 3]
    restored, _ = populated()
    restored.load_state_dict(copy.deepcopy(m.state_dict()))
    assert restored.read_credit_rule == 'adam'
    for instance in (m, restored):
        instance.begin_read_credit()
        read = instance.read(query, enc)
        read.value[:, 0].sum().backward()
        instance.settle_read_credit()
    for name in ('read_credit_mean', 'read_credit_first','read_credit_second','read_credit_steps'):
        assert torch.equal(getattr(m, name), getattr(restored, name))


def test_replacement_resets_read_preference_and_adam_history_together():
    m, enc = populated()
    configure_read_credit(m, 'adam', torch.optim.Adam([nn.Parameter(torch.zeros(1))]).param_groups[0])
    m.read(torch.tensor([[1., 1.]]), enc).value[:, 0].sum().backward()
    m.settle_read_credit()
    assert m.read_credit_steps[0] == 1
    x = torch.tensor([1., 0.])
    slot, merged = m.commit_episode(x, x, x, x, enc, source_time=4)
    assert slot == 0 and not merged
    for name in ('read_credit_mean','read_credit_first','read_credit_second','read_credit_steps'):
        assert getattr(m, name)[slot] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA interface check')
@pytest.mark.parametrize('rule', ['mean','adam'])
def test_cuda_repeated_slot_reduction_resumes_bitwise(rule):
    memory, encoder = populated(long=128)
    memory.cuda()
    configure_read_credit(memory, rule, torch.optim.Adam([nn.Parameter(torch.zeros(1))]).param_groups[0])
    saved = copy.deepcopy(memory.state_dict())
    query = torch.tensor([[1., 1.]], device='cuda').expand(128, -1)
    target = torch.linspace(-1., 1., 256, device='cuda').reshape(128, 2)
    results = []
    for _ in range(3):
        memory.load_state_dict(copy.deepcopy(saved))
        memory.begin_read_credit()
        read = memory.read(query, encoder)
        (read.value-target).square().mean().backward()
        memory.settle_read_credit()
        results.append(memory.read_credit_mean.clone())
    assert torch.equal(results[0], results[1]) and torch.equal(results[1], results[2])


@pytest.mark.parametrize('rule', ['mean','adam'])
def test_full_body_advance_settles_credit_and_restore_repeats_read_exactly(rule):
    torch.manual_seed(17)
    model = body()
    ids = torch.tensor([[31, 32, 33, 34]])
    targets = torch.tensor([[32, 33, 34, 35]])
    logp = torch.full((1, 4, 256), -5.545177)
    state = model.initial_state(1)
    with torch.no_grad():
        _, _, _, h, _ = model.predict_observed(ids, state, logp)
        model.memory.clock.fill_(10)
        for i in range(2):
            x = h[i]
            model.memory.commit_experience(x, h[i+1], x, x, model.encoder,
                                           bytes([65+i]), .2, i)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    engine = DialogueEngine(model, optimizer, state, None, None, 31, None,
                            utterance_bytes=1, response_bytes=2, holdout_bytes=2)
    configure_read_credit(model.memory, rule, optimizer.param_groups[0])
    loss, next_state, h, diagnostics = model.objective(
        ids, targets, state, logp, torch.randn(8, 4))
    before = model.memory.progress.clone()
    engine._advance(loss, h, diagnostics, next_state)
    assert diagnostics['read_credit']['records'] > 0
    assert diagnostics['read_credit']['participations'] > 0
    assert torch.equal(model.memory.progress, before)
    assert not model.memory._read_participation
    restored = body()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    with torch.no_grad():
        first = model.predict(ids, engine.recurrent, logp)[0]
        second = restored.predict(ids, engine.recurrent, logp)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in restored.parameters())
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_engine = DialogueEngine(restored, restored_optimizer, engine.recurrent.clone(),
        None, None, 31, None, utterance_bytes=1, response_bytes=2, holdout_bytes=2)
    directions = torch.randn(8, 4)
    for learner in (engine, restored_engine):
        loss, next_state, features, diag = learner.model.objective(
            ids, targets, learner.recurrent, logp, directions)
        learner._advance(loss, features, diag, next_state)
    for k, value in model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, restored.state_dict()[k], rtol=0, atol=0)
    for original, resumed in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for k, value in original.items():
            torch.testing.assert_close(value, resumed[k], rtol=0, atol=0)
