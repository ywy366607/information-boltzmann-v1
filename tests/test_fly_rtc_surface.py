"""Exact read interface and autonomous rollout; no capability claims."""
import copy
from types import SimpleNamespace

import torch
from torch import nn

from information_boltzmann.core.fly_rtc_surface import MotorSurfaceForecaster
from information_boltzmann.core.fly_rtc_student import TickResponseStudent, PhysicalObservationCodec
from information_boltzmann.core.fly_streaming_observer import DelayRegionGraph


def prototype():
    graph = DelayRegionGraph([0, 0, 1, 1], [1], torch.zeros(4, 2, 2), torch.zeros(4, 2, 2))
    codec = PhysicalObservationCodec(graph, 2, latent_dim=3, sample_per_region=2)
    latent = TickResponseStudent(codec, 2, horizon=3)
    return MotorSurfaceForecaster(latent, 2)


def test_zero_tick_exact_read_and_current_weight_revision():
    student = prototype()
    history = torch.zeros(4, 1, 2, 3)
    motor, mean = torch.tensor([[.03, -.08]]), torch.tensor([[.01, -.02]])
    _, drafts, means = student(history, motor, mean)
    model = SimpleNamespace(read_centering=True, output_read=nn.Linear(2, 2, bias=False))
    torch.testing.assert_close(student.decode(model, drafts[0], means[0]),
                               model.output_read(motor - mean), rtol=0, atol=0)
    original = drafts.detach().clone()
    model.output_read.weight.data.mul_(2)
    torch.testing.assert_close(student.decode(model, drafts[0], means[0]),
                               model.output_read(motor - mean), rtol=0, atol=0)
    torch.testing.assert_close(drafts, original)
    model.read_centering = False
    torch.testing.assert_close(student.decode(model, drafts[0], means[0]),
                               model.output_read(motor), rtol=0, atol=0)


def test_motor_feedback_is_autonomous_and_control_zeros_all_ticks():
    student = prototype()
    with torch.no_grad():
        student.motor_delta.weight.zero_()
        student.motor_delta.weight[:, -2:] = .1 * torch.eye(2)
    control = copy.deepcopy(student)
    control.use_motor_state = False
    history = torch.zeros(4, 1, 2, 3)
    motor = torch.tensor([[1., 2.]])
    _, drafts, means = student(history, motor, torch.zeros_like(motor))
    _, held, _ = control(history, motor, torch.zeros_like(motor))
    for tick in range(4):
        torch.testing.assert_close(drafts[tick], motor * 1.1**tick)
        torch.testing.assert_close(held[tick], motor)
    torch.testing.assert_close(means[1], .01 * drafts[1])
    drafts[-1].sum().backward()
    assert student.motor_delta.weight.grad is not None
    assert torch.isfinite(student.motor_delta.weight.grad).all()
