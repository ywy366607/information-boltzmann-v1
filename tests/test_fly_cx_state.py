import torch
from torch import nn

from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_cx_state import CX_PARAMETER_NAMES, step_cx_state


class TinyCX(nn.Module):
    def __init__(self):
        super().__init__()
        self.cx_enabled = True
        self.register_buffer("cx_index", torch.tensor([1, 3], dtype=torch.long), persistent=False)
        self.cx_logit_decay = nn.Parameter(torch.tensor([3.0, 3.0]))
        self.cx_feedback = nn.Parameter(torch.tensor([0.0, 0.0]))


def make_state():
    h = torch.zeros(1, 4)
    ring = (torch.tensor([[0.0, 1.0, 0.0, 0.0]]), torch.zeros_like(h),
            torch.zeros_like(h), torch.zeros_like(h))
    return FlyPhysicalState(h, ring, torch.zeros_like(h), torch.zeros_like(h),
                            torch.zeros_like(h), torch.ones_like(h), torch.zeros_like(h),
                            torch.zeros_like(h))


def test_cx_trace_persists_and_feedback_uses_native_threshold():
    model = TinyCX()
    thresholds, trace = step_cx_state(model, make_state(), torch.ones(1, 4))
    assert trace.shape == (1, 2)
    assert trace[0, 0] > 0
    assert torch.equal(thresholds, torch.ones(1, 4))
    model.cx_feedback.data.fill_(0.4)
    thresholds, trace = step_cx_state(model, make_state(), torch.ones(1, 4))
    assert thresholds[0, 1] < 1.0
    assert thresholds[0, 3] == 1.0


def test_cx_decay_and_feedback_receive_gradients():
    model = TinyCX()
    thresholds, trace = step_cx_state(model, make_state(), torch.ones(1, 4))
    (thresholds.square().sum() + trace.sum()).backward()
    assert model.cx_logit_decay.grad is not None
    assert model.cx_feedback.grad is not None
    assert set(CX_PARAMETER_NAMES) == {name for name, _ in model.named_parameters()}


def test_cx_disabled_is_exact_identity():
    model = TinyCX()
    model.cx_enabled = False
    threshold = torch.randn(1, 4)
    out, trace = step_cx_state(model, make_state(), threshold)
    assert torch.equal(out, threshold)
    assert trace.numel() == 0
