"""Finite historical participation, provenance and derivative protocol tests."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ib_local.delayed_prediction_credit import (
    PendingAnswer, PredictionParticipation, capture_reads, capture_views, fork_view,
    anchor_participation, unique_tensor_bytes,
    release_view,
    align_numeric_feedback,
)
from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience


def populated():
    torch.manual_seed(73)
    wrapper = LayerExperienceRecall(6, capacity=2, latent_dim=3, topk=2)
    wrapper.memory.clock.fill_(256)
    for i in range(2):
        x = torch.randn(6)
        wrapper.memory.commit_episode(x, x+1, x+2, x+3, wrapper.encoder, source_time=64+i)
        wrapper.capsules[i] = LocalExperience(torch.randn(1, 65, 6), None,
                                             int(wrapper.memory.read_credit_version[i]))
    return wrapper


def test_read_view_keeps_past_participants_after_live_slot_replacement():
    wrapper = populated()
    learner = SimpleNamespace(memories={1: wrapper})
    views = capture_views(learner)
    original_capsule = wrapper.capsules[0]
    wrapper.memory.source[0].fill_(9)
    wrapper.memory.read_credit_version[0].add_(1)
    wrapper.capsules[0] = LocalExperience(torch.full((1, 65, 6), 99.), None,
                                          int(wrapper.memory.read_credit_version[0]))
    # Simulate the no-grad decode cache's instance replacement.
    wrapper.retrieve = lambda *args: (_ for _ in ()).throw(AssertionError('live/future view used'))
    shadow = fork_view(wrapper, views[1])
    assert shadow.capsules[0] is original_capsule
    assert not torch.equal(shadow.memory.source, wrapper.memory.source)
    assert shadow.encoder is wrapper.encoder and shadow.inlet is wrapper.inlet
    native = nn.Linear(6, 6)
    output = shadow.retrieve(torch.randn(1, 6), lambda c: native(c.inputs))
    native_grad, encoder_grad = torch.autograd.grad(output.square().sum(),
                                                   (native.weight, wrapper.encoder.weight))
    assert native_grad.norm() > 0 and encoder_grad.norm() > 0
    assert not wrapper.memory._read_credit_active
    assert shadow.memory.read_credit_version[0] != wrapper.memory.read_credit_version[0]


def record():
    return PredictionParticipation(torch.tensor([[2]]), {'states': [], 'seen_tokens': 99},
                                   {}, 0, 7, 100, logits=torch.randn(11))


def test_delayed_teaching_requires_finished_emission_and_later_external_feedback():
    pending = PendingAnswer(capacity=1)
    pending.add(record(), 3)
    learner = SimpleNamespace(updates=7)
    with pytest.raises(RuntimeError, match='feedback must arrive'):
        pending.validate_feedback(learner, 101)
    pending.finish_emission(101)
    with pytest.raises(RuntimeError, match='feedback must arrive'):
        pending.validate_feedback(learner, 101)
    pending.validate_feedback(learner, 102)
    learner.updates += 1
    with pytest.raises(RuntimeError, match='parameters changed'):
        pending.validate_feedback(learner, 102)
    learner.updates = 7
    pending.consumed = True
    with pytest.raises(RuntimeError, match='already settled'):
        pending.validate_feedback(learner, 102)


def test_one_bounded_marker_per_actual_emission_and_recovery_preserves_identity():
    pending = PendingAnswer(capacity=2)
    first = record()
    pending.add(first, 4)
    with pytest.raises(RuntimeError, match='one prediction'):
        pending.add(first, 5)
    pending.add(record(), 6)
    with pytest.raises(RuntimeError, match='full'):
        pending.add(record(), 7)
    pending.finish_emission(102)
    restored = copy.deepcopy(pending)
    assert [r.emitted for r in restored.records] == [4, 6]
    assert restored.records[0].parameter_version == 7
    assert restored.answer_end_seen == 102
    assert restored.records[0].inputs.grad_fn is None
    assert restored.records[0].logits.grad_fn is None
    assert restored.nbytes == pending.nbytes
    assert set(vars(first)) == {'inputs', 'boundary', 'views', 'clock_offset',
                               'parameter_version', 'prediction_seen', 'logits', 'emitted', 'reads'}


def test_shared_native_capsules_are_counted_once_in_pending_storage():
    wrapper = populated()
    views = capture_views(SimpleNamespace(memories={1: wrapper}))
    assert unique_tensor_bytes([views, views]) == unique_tensor_bytes(views)
    assert unique_tensor_bytes(views) > 0


def test_actual_speech_uses_one_boundary_and_shares_unchanged_read_views():
    from scripts.ib_local.irreversible_credit_stream import PretrainedStreamBackend
    wrapper = populated()

    class Learner:
        model = nn.Linear(6, 6)
        memories = {1: wrapper}
        updates = 7
        cache = SimpleNamespace(states=[{'state': torch.ones(4)}], _seen_tokens=100)

        def predict(self, x, clock_offset, **options):
            cache = SimpleNamespace(states=self.cache.states,
                                    _seen_tokens=self.cache._seen_tokens+x.numel())
            return torch.zeros(1, x.numel(), 11), cache

    backend = PretrainedStreamBackend(Learner())
    backend.set_prediction_capture(True)
    backend.advance([2, 3], clock_offset=0)
    first = backend.last_prediction_participation
    assert first.boundary['seen_tokens'] == 100
    backend.mark_prediction_used()
    backend.advance([4], clock_offset=2)
    second = backend.last_prediction_participation
    assert second.boundary is None
    assert first.views is second.views
    wrapper.memory.clock.add_(128)
    backend.advance([5], clock_offset=0)
    third = backend.last_prediction_participation
    assert third.boundary is None
    assert third.views is not second.views
    assert first.prediction_seen == 102 and third.prediction_seen == 104
    backend.set_prediction_capture(False)
    assert backend.last_prediction_participation is None
    backend.close()


def test_held_participation_keeps_real_query_and_values_with_local_derivatives():
    wrapper = populated()
    native = nn.Linear(6, 6)
    query = torch.randn(1, 6)
    with torch.no_grad():
        expected = wrapper.retrieve(query, lambda c: native(c.inputs))
        wrapper.current = dict(source=query[None], query_positions=[0])
        held = capture_reads(SimpleNamespace(memories={1:wrapper}))[1]
    view = capture_views(SimpleNamespace(memories={1:wrapper}))[1]
    shadow = fork_view(wrapper, view)
    anchor_participation(shadow, held)
    # A perturbed reconstruction must keep the actual participating route.
    reconstructed = (query+10).requires_grad_()
    actual = shadow.retrieve(reconstructed, lambda c: native(c.inputs)+3)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert reconstructed.grad.norm()>0 and native.weight.grad.norm()>0
    assert wrapper.encoder.weight.grad.norm()>0
    import weakref
    reference=weakref.ref(shadow)
    release_view(shadow)
    del shadow, actual
    # Refcount release must not wait for a future cyclic-GC pass. Keeping one
    # old view per emitted token exhausts RAM before a case can be checkpointed.
    assert reference() is None


def test_numeric_feedback_supervises_actual_answer_content_not_prefix_or_stop():
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):
            return [ord(c) for c in text]
        def decode(self,tokens):
            return ''.join(chr(t) for t in tokens)
    pending=PendingAnswer(capacity=16)
    for token in ': 6566011.':
        pending.add(record(),ord(token))
    positions,labels=align_numeric_feedback(pending,Tokenizer(),'9556663')
    assert positions==list(range(2,9))
    assert ''.join(chr(label) for label in labels)=='9556663'
    assert [r.emitted for r in pending.records]==list(map(ord,': 6566011.'))
    assert not pending.consumed


def test_numeric_feedback_rejects_absent_participation_and_does_not_invent_positions():
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):
            return list(map(ord,text))
        def decode(self,tokens):
            return ''.join(map(chr,tokens))
    pending=PendingAnswer(capacity=16)
    for token in 'unknown':
        pending.add(record(),ord(token))
    with pytest.raises(ValueError,match='no actually emitted'):
        align_numeric_feedback(pending,Tokenizer(),'1234567')
    with pytest.raises(ValueError,match='ASCII'):
        align_numeric_feedback(pending,Tokenizer(),'abc')
