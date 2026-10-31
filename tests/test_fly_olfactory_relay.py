import torch
from torch import nn

from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_olfactory_relay import AL_PARAMETER_NAMES, step_olfactory_relay


class TinyAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.olfactory_relay_enabled = True
        self.register_buffer('al_local_index', torch.tensor([0, 1], dtype=torch.long), persistent=False)
        self.register_buffer('al_pn_index', torch.tensor([2, 3], dtype=torch.long), persistent=False)
        self.register_buffer('al_contact_local_slot', torch.tensor([0, 1], dtype=torch.long), persistent=False)
        self.register_buffer('al_contact_pn_slot', torch.tensor([0, 1], dtype=torch.long), persistent=False)
        self.register_buffer('al_contact_weight', torch.ones(2), persistent=False)
        self.al_local_logit_decay = nn.Parameter(torch.tensor(3.0))
        self.al_pn_feedback = nn.Parameter(torch.zeros(2))


def make_state():
    h = torch.zeros(1, 4)
    ring = (torch.tensor([[1.0, 0.5, 0.0, 0.0]]), torch.zeros_like(h),
            torch.zeros_like(h), torch.zeros_like(h))
    return FlyPhysicalState(h, ring, torch.zeros_like(h), torch.zeros_like(h),
                            torch.zeros_like(h), torch.ones_like(h), torch.zeros_like(h),
                            torch.zeros_like(h))


def test_local_trace_reaches_only_pn_thresholds():
    model = TinyAL()
    model.al_pn_feedback.data.fill_(0.5)
    thresholds, trace, drive = step_olfactory_relay(model, make_state(), torch.ones(1, 4))
    assert trace.shape == (1, 2) and drive.shape == (1, 2)
    assert torch.all(thresholds[0, 2:] > 1.0)
    assert torch.equal(thresholds[0, :2], torch.ones(2))


def test_relay_parameters_receive_gradients():
    model = TinyAL()
    thresholds, trace, drive = step_olfactory_relay(model, make_state(), torch.ones(1, 4))
    (thresholds.square().sum() + trace.sum() + drive.sum()).backward()
    assert model.al_local_logit_decay.grad is not None
    assert model.al_pn_feedback.grad is not None
    assert set(AL_PARAMETER_NAMES) == {name for name, _ in model.named_parameters()}


def test_relay_disabled_is_identity():
    model = TinyAL()
    model.olfactory_relay_enabled = False
    threshold = torch.randn(1, 4)
    out, trace, drive = step_olfactory_relay(model, make_state(), threshold)
    assert torch.equal(out, threshold)
    assert trace.numel() == 0 and drive.numel() == 0
