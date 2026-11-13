"""Scoring and numerical-probe integrity, not synthetic capability evidence."""
import pytest
import torch

from scripts.ib_local.evaluate_pretrained_gdn_tasks import (
    answer_metrics, chunk_slices, normalize_answer, trajectory_comparison,
)


def test_answer_metrics_use_token_multiplicity_and_human_alternatives():
    assert normalize_answer(' The CAT, a dog! ') == 'cat dog'
    assert answer_metrics('a cat', ['dog', 'the cat']) == dict(em=1., f1=1.)
    assert answer_metrics('cat cat', ['cat'])['f1'] == pytest.approx(2/3)
    assert answer_metrics('', ['cat']) == dict(em=0., f1=0.)


@pytest.mark.parametrize('length', [128,129,255,256,257,2049])
def test_real_prompt_partition_does_not_pad_drop_or_repeat_tokens(length):
    sections=list(chunk_slices(length))
    recovered=[i for section in sections for i in range(length)[section]]
    assert recovered == list(range(length))
    assert all(128 <= s.stop-s.start <= 255 for s in sections)


def test_too_short_prompt_is_rejected_instead_of_filling_it():
    with pytest.raises(ValueError, match='actual input'):
        list(chunk_slices(127))


def test_state_recovery_does_not_hide_loss_of_between_content_difference():
    a=[dict(age_tokens=t,states={'1':torch.tensor([0.])},probability=None) for t in (0,128)]
    b=[dict(age_tokens=t,states={'1':torch.tensor([v])},probability=None)
       for t,v in [(0,2.),(128,.2)]]
    p=[dict(age_tokens=t,states={'1':torch.tensor([v])},probability=None)
       for t,v in [(0,1.),(128,.1)]]
    row=trajectory_comparison(a,p,b)[1]
    assert row['perturbation_remaining_fraction'] == pytest.approx(.1)
    assert row['content_retained_fraction'] == pytest.approx(.1)
