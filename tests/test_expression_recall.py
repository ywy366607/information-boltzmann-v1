from types import SimpleNamespace

import torch

from scripts.ib_local.expression_recall import Recall, byte_log_probability, retrieve


def test_recalled_bytes_are_optional_and_can_join_inside_a_phrase():
    logits = torch.zeros(256)
    recalled = (Recall(4, 9, b'abc hello def', 1., 0.),)
    original = logits.log_softmax(-1)
    result = byte_log_probability(logits, b'a new hell', recalled)
    assert result[ord('o')] > original[ord('o')]
    assert result.isfinite().all()
    torch.testing.assert_close(result.exp().sum(), torch.tensor(1.))
    torch.testing.assert_close(byte_log_probability(logits, b'qz', recalled), original)


def test_feeling_changes_ranking_without_touching_the_payload():
    words = torch.tensor([[97, 98], [99, 100]], dtype=torch.uint8)
    m = SimpleNamespace(occupied=torch.ones(2, dtype=torch.bool), source=torch.tensor([[1.,0.],[.8,.6]]),
                        feeling=torch.tensor([[2.,2.,2.],[0.,0.,0.]]), read_credit_version=torch.tensor([7,9]),
                        topk=1, temperature=.2)
    model = SimpleNamespace(memory=m, encoder=torch.nn.Identity(),
                            retrieval=SimpleNamespace(words=words,word_count=torch.tensor([2,2])))
    h, feeling = torch.tensor([1.,0.]), torch.zeros(3)
    assert retrieve(model, h, feeling, felt=False)[0] == Recall(0,7,b'ab',1.,2.)
    assert retrieve(model, h, feeling, felt=True)[0] == Recall(1,9,b'cd',1.,0.)
