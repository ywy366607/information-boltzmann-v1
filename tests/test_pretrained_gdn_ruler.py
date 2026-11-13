"""Benchmark adapters: no gold leaks into prompt or stopping controls."""
import pytest
import torch
from types import SimpleNamespace

from scripts.ib_local.evaluate_pretrained_gdn_ruler import (
    answer_diagnostics, encode_case, frozen_decode_replays,
)
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.train_pretrained_gdn_ruler import supervised_sections


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode('utf-8'))


def test_provenance_hash_never_reads_an_entire_checkpoint(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
    data = b'checkpoint blocks'*(2**17)
    path = tmp_path/'checkpoint.bin'
    path.write_bytes(data)
    def forbidden(*args, **kwargs):
        raise AssertionError('whole-file checkpoint allocation is forbidden')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    assert sha256(path) == hashlib.sha256(data).hexdigest()


def test_actual_feedback_names_its_question_instead_of_an_ambiguous_answer():
    from scripts.ib_local.continue_irreversible_feedback import correction_language
    text=correction_language(dict(answer_prefix=' The special number for key-A is',outputs=['1234567']))
    assert text=='\nCorrection: The special number for key-A is 1234567.\n\n'


def test_gold_cannot_change_ruler_prompt_or_answer_prefix():
    row = dict(input='context\nQuestion:',answer_prefix=' Answer:',outputs=['1234567'])
    prefix, gold = encode_case(Tokenizer(),row)
    other, other_gold = encode_case(Tokenizer(),dict(row,outputs=['7654321']))
    assert prefix == other == list(b'context\nQuestion: Answer:')
    assert gold != other_gold
    assert gold == list(b' 1234567')


def test_missing_answer_prefix_is_not_guessed_from_gold():
    prefix, gold = encode_case(Tokenizer(),dict(input='prompt',outputs=['X','Y']))
    assert prefix == list(b'prompt')
    assert gold == list(b' X, Y')


def test_sentence_initial_dummy_prefix_is_not_inserted_into_ruler_answer():
    class DummyPrefixTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            return [999]+super().encode(text,add_special_tokens)
    tok = DummyPrefixTokenizer()
    row = dict(input='prompt',outputs=['123'])
    prefix,gold = encode_case(tok,row)
    assert prefix+gold == tok.encode('prompt 123')
    assert gold == list(b' 123')
    assert gold != tok.encode(' 123')


def test_retokenized_prompt_boundary_is_rejected():
    class MergingTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text,add_special_tokens)
            return [998]+ids if ' ' in text else ids
    with pytest.raises(ValueError,match='boundary'):
        encode_case(MergingTokenizer(),dict(input='prompt',outputs=['123']))


def test_termination_diagnostic_does_not_hide_behind_low_answer_loss():
    logits = torch.tensor([[[0.,10.,0.],[0.,10.,0.],[0.,10.,0.]]])
    result = answer_diagnostics(logits,[1,1],2)
    assert result['answer_nll'] < .001
    assert result['eos_nll'] > 9
    assert result['answer_with_eos_nll'] > 3
    assert result['eos_rank'] == 2


def test_numeric_diagnostics_separate_substring_from_complete_number():
    result = numeric_scores('127610277777 text',['1276102'])
    assert result['official_contains'] and result['first_number_prefix']
    assert not result['first_number_exact'] and not result['standalone_number']


def test_decode_replay_reuse_is_bounded_temporary_and_exact():
    capsule = object()
    calls = []
    class Wrapper:
        def retrieve(self,query,replay):
            return query+replay(capsule)
    wrapper = Wrapper()
    learner = SimpleNamespace(memories={1:wrapper})
    def replay(c):
        calls.append(c)
        return torch.tensor([2.])
    with pytest.raises(RuntimeError,match='frozen'):
        with frozen_decode_replays(learner):
            pass
    with torch.no_grad(), frozen_decode_replays(learner):
        a = wrapper.retrieve(torch.tensor([1.]),replay)
        b = wrapper.retrieve(torch.tensor([3.]),replay)
    assert len(calls) == 1
    assert a.item() == 3 and b.item() == 5
    assert 'retrieve' not in wrapper.__dict__
    wrapper.retrieve(torch.tensor([1.]),replay)
    assert len(calls) == 2


@pytest.mark.parametrize('reset_native,expected_start', [(True, 0), (False, 73)])
def test_ruler_carried_mode_preserves_incoming_cache_and_legacy_reset_is_explicit(
        monkeypatch, reset_native, expected_start):
    from scripts.ib_local import evaluate_pretrained_gdn_ruler as module
    class Learner:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.cache = SimpleNamespace(_seen_tokens=73)
            self.memories = {}
            self.first_seen = None
        def observe(self, x, return_logits=False):
            self.first_seen = self.cache._seen_tokens
            self.cache._seen_tokens += x.shape[1]
            logits = torch.zeros(1, x.shape[1], 256)
            logits[..., 49] = 5
            return {}, logits
        def predict(self, x, cache):
            logits = torch.zeros(1, x.shape[1], 256)
            logits[..., 49] = 5
            return logits, SimpleNamespace(_seen_tokens=cache._seen_tokens+x.shape[1])
    class Tok(Tokenizer):
        eos_token_id = 0
        def decode(self, ids, skip_special_tokens=True):
            return bytes(ids).decode()
    monkeypatch.setattr(module, 'new_recurrent_cache', lambda:SimpleNamespace(_seen_tokens=0))
    learner = Learner()
    result = module.run_case(learner, Tok(), dict(index=0, input='a'*200, outputs=['1']),
                             4, reset_native=reset_native)
    assert learner.first_seen == expected_start
    assert learner.cache._seen_tokens == expected_start+200
    assert result['prediction'] == '1111'


def test_real_interference_preserves_fact_and_question_without_using_gold():
    from scripts.ib_local.audit_pretrained_continued_answers import insert_interference
    row = dict(input='A stored fact.\nWhat is the fact?', outputs=['123'])
    modified = insert_interference(row, 'Actual external text.')
    assert modified['input'] == 'A stored fact.\nActual external text.\n\nWhat is the fact?'
    other = insert_interference(dict(row, outputs=['987']), 'Actual external text.')
    assert modified['input'] == other['input']
    assert modified['outputs'] == ['123']


@pytest.mark.parametrize('length',[362,383,384,385,1024,1920])
def test_answer_learning_has_exact_128_window_without_padding_or_dropped_input(length):
    prefix = list(range(length))
    gold = [5001,5002,5003]
    eos = 5004
    chunks = list(supervised_sections(prefix,gold,eos))
    full = prefix+gold+[eos]
    assert [i for x,_,_ in chunks for i in x] == full[:-1]
    assert [i for _,y,_ in chunks for i in y] == full[1:]
    active = [(x,y,m) for x,y,m in chunks if any(m)]
    assert len(active) == 1 and len(active[0][0]) == 128
    assert [t for _,y,m in chunks for t,yes in zip(y,m) if yes] == gold+[eos]


def test_long_answer_refused_instead_of_silently_expanding_credit_window():
    with pytest.raises(ValueError,match='short answers'):
        list(supervised_sections(list(range(512)),[1]*128,0))
