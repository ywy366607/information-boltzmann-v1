import copy
from types import SimpleNamespace

import torch

from scripts.ib_local.language_navigation_kanli_v4t import ProbabilityActionEvidence
from scripts.ib_local.navigation_gain_clipping import ClippedSharedActionEvidence, prepare_gain_clipping


def initialized():
    e = ProbabilityActionEvidence(.95)
    e.seen.fill_(3)
    engine = SimpleNamespace(model=SimpleNamespace(wood_evidence=e))
    prepare_gain_clipping(engine)
    e.gain_innovation_scale.fill_(1.)
    return e


def test_gain_clip_preserves_real_stats_and_other_channels():
    clipped = initialized()
    original = ProbabilityActionEvidence(.95)
    original.load_state_dict({k:v for k,v in clipped.state_dict().items() if k != 'gain_innovation_scale'})
    base = torch.zeros(4); target = torch.tensor([4., -2., 1., 100.])
    clipped.observe(1, target, base, base)
    original.observe(1, target, base, base)
    torch.testing.assert_close(clipped.bases()[:, :3], original.bases()[:, :3])
    torch.testing.assert_close(clipped.unconditional, original.unconditional)
    torch.testing.assert_close(clipped.base_error, original.base_error)
    assert clipped.bases()[1, 3] == torch.tensor(.15)
    assert clipped.gain_innovation_scale < 2.


def test_sustained_gain_change_can_increase_scale_and_baseline():
    e = initialized()
    for _ in range(60):
        base = e.bases()[1].clone()
        target = torch.tensor([0., 0., 0., 10.])
        e.observe(1, target, base, base)
    assert e.bases()[1, 3] > 5.
    assert e.gain_innovation_scale > 1.


def test_scale_survives_state_dict_roundtrip():
    e = initialized(); e.gain_innovation_scale.fill_(2.75)
    other = initialized(); other.load_state_dict(copy.deepcopy(e.state_dict()))
    assert isinstance(other, ClippedSharedActionEvidence)
    assert other.gain_innovation_scale.item() == 2.75
