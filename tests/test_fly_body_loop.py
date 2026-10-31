import torch
from torch import nn
from types import SimpleNamespace

from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_body_loop import (
    BODY_LOOP_PARAMETER_NAMES, step_body_loop, quarantine_body_proxy_credit,
)


class TinyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.body_loop_enabled = True
        self.body_proxy_compatibility = True  # reproduce historical equations only
        self.register_buffer('body_descending_index', torch.tensor([0, 1], dtype=torch.long), persistent=False)
        self.register_buffer('body_ascending_index', torch.tensor([2, 3], dtype=torch.long), persistent=False)
        self.body_logit_decay = nn.Parameter(torch.tensor(3.0))
        self.body_feedback = nn.Parameter(torch.zeros(2))


def make_state():
    h = torch.zeros(1, 4)
    ring = (torch.tensor([[1.0, 0.5, 0.0, 0.0]]), torch.zeros_like(h),
            torch.zeros_like(h), torch.zeros_like(h))
    return FlyPhysicalState(h, ring, torch.zeros_like(h), torch.zeros_like(h),
                            torch.zeros_like(h), torch.ones_like(h), torch.zeros_like(h),
                            torch.zeros_like(h))


def test_body_loop_writes_only_to_ascending_cells_and_persists_trace():
    model = TinyBody()
    model.body_feedback.data.fill_(0.5)
    source, action, feedback = step_body_loop(model, make_state(), None)
    assert action.shape == feedback.shape == (1, 1)
    assert action.item() > 0
    assert torch.equal(source[0, :2], torch.zeros(2))
    assert torch.all(source[0, 2:] > 0)


def test_body_loop_parameters_receive_gradients():
    model = TinyBody()
    model.body_feedback.data.fill_(0.2)
    source, action, feedback = step_body_loop(model, make_state(), None)
    (source.square().sum() + action.sum() + feedback.sum()).backward()
    assert model.body_logit_decay.grad is not None
    assert model.body_feedback.grad is not None
    assert set(BODY_LOOP_PARAMETER_NAMES) == {name for name, _ in model.named_parameters()}


def test_body_loop_disabled_is_exact_identity():
    model = TinyBody()
    model.body_loop_enabled = False
    source = torch.randn(1, 4)
    out, action, feedback = step_body_loop(model, make_state(), source)
    assert torch.equal(out, source)
    assert action.numel() == 0 and feedback.numel() == 0


def test_native_execution_never_injects_population_mean_proxy():
    model = TinyBody()
    del model.body_proxy_compatibility
    model.body_feedback.data.fill_(100.)
    source = torch.randn(1, 4)
    state = make_state()
    out, action, feedback = step_body_loop(model, state, source)
    assert out is source
    assert action is state.body_action_trace
    assert feedback is state.body_feedback_trace


def test_proxy_derivatives_are_archived_without_losing_real_pending_credit():
    model = TinyBody()
    model.body_proxy_compatibility = False
    real = torch.tensor([.2])
    old = torch.tensor([.7])
    credit = {'clock_ticks': 19, 'E': {'body_feedback': old, 'comp_eta': real},
              'response_score': {'body_muscle_logit_decay': old, 'ai_w': real},
              'deferred_response_scores': {'pending': {'body_feedback': old, 'comp_eta': real}}}
    learner = SimpleNamespace(model=model, consequence_state={'native_credit': credit,
                                                              'source': 'mother_reply'})
    assert quarantine_body_proxy_credit(learner)
    assert credit['clock_ticks'] == 19
    assert credit['E'] == {'comp_eta': real}
    assert credit['response_score'] == {'ai_w': real}
    assert credit['deferred_response_scores']['pending'] == {'comp_eta': real}
    archive = learner.consequence_state['quarantined_body_proxy_credit'][0]
    assert torch.equal(archive['traces']['E']['body_feedback'], old)
    assert learner.consequence_state['source'] == 'mother_reply'
    assert not quarantine_body_proxy_credit(learner)  # idempotent, no history spam
