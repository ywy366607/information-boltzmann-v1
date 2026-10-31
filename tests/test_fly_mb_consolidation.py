import torch
from torch import nn

from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_mb_consolidation import (
    MB_CONSOLIDATION_PARAMETER_NAMES,
    dpm_forgetting_factor,
    step_mb_consolidation,
)


class TinyMB(nn.Module):
    def __init__(self):
        super().__init__()
        self.mb_consolidation_enabled = True
        self.register_buffer("dpm_index", torch.tensor([0], dtype=torch.long), persistent=False)
        self.register_buffer("apl_index", torch.tensor([1], dtype=torch.long), persistent=False)
        self.register_buffer("comp_kc", torch.tensor([2, 3], dtype=torch.long), persistent=False)
        self.mb_dpm_logit_decay = nn.Parameter(torch.tensor(4.0))
        self.mb_apl_logit_decay = nn.Parameter(torch.tensor(2.0))
        self.mb_dpm_consolidation = nn.Parameter(torch.tensor(0.0))
        self.mb_apl_feedback = nn.Parameter(torch.tensor(0.0))


def state():
    h = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
    ring = (torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros_like(h),
            torch.zeros_like(h), torch.zeros_like(h))
    return FlyPhysicalState(h, ring, torch.zeros_like(h), torch.zeros_like(h),
                            torch.zeros_like(h), torch.ones_like(h), torch.zeros_like(h),
                            torch.zeros_like(h))


def test_dpm_and_apl_traces_are_native_and_bounded():
    model = TinyMB()
    thresholds, dpm, apl = step_mb_consolidation(model, state(), torch.ones(1, 4))
    assert dpm.shape == (1, 1) and apl.shape == (1, 1)
    assert dpm[0, 0] > 0 and 0 < apl[0, 0] < 1
    assert torch.equal(thresholds, torch.ones(1, 4))
    model.mb_apl_feedback.data.fill_(0.5)
    thresholds, _, _ = step_mb_consolidation(model, state(), torch.ones(1, 4))
    assert torch.all(thresholds[0, [2, 3]] < 1.0)


def test_mb_parameters_receive_gradients():
    model = TinyMB()
    thresholds, dpm, apl = step_mb_consolidation(model, state(), torch.ones(1, 4))
    (thresholds.square().sum() + dpm.sum() + apl.sum() + dpm_forgetting_factor(model, dpm).sum()).backward()
    assert model.mb_dpm_logit_decay.grad is not None
    assert model.mb_apl_logit_decay.grad is not None
    assert model.mb_dpm_consolidation.grad is not None
    assert model.mb_apl_feedback.grad is not None
    assert set(MB_CONSOLIDATION_PARAMETER_NAMES) == {name for name, _ in model.named_parameters()}


def test_mb_disabled_is_exact_identity():
    model = TinyMB()
    model.mb_consolidation_enabled = False
    threshold = torch.randn(1, 4)
    out, dpm, apl = step_mb_consolidation(model, state(), threshold)
    assert torch.equal(out, threshold)
    assert dpm.numel() == 0 and apl.numel() == 0
