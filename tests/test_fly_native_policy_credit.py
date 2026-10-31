"""Analytic score derivatives and continuation contracts, not capability studies."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from information_boltzmann.core.fly_native_policy_credit import (
    native_parameter_names, native_policy_credit, prepare_native_policy_backward,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp_eta = nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        self.comp_weight = nn.Parameter(torch.tensor([.3, -.2], dtype=torch.float64))
        self.comp_logit_trace = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.log_tau_m = nn.Parameter(torch.tensor([.4], dtype=torch.float64))
        self.mbon_kappa = nn.Parameter(torch.tensor([.1], dtype=torch.float64))
        self.dopamine_logit_fast = nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        self.log_kc_release = nn.Parameter(torch.tensor([-.4], dtype=torch.float64))
        self.intrinsic_kappa = nn.Parameter(torch.tensor([.9], dtype=torch.float64))
        self.edge_weight_e = nn.Parameter(torch.tensor([.7], dtype=torch.float64))
        self.voice_log_gain = nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        self.ai_w = nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        self.read_norm = nn.Identity()
        self.decoder = nn.Linear(2, 3, dtype=torch.float64)
        with torch.no_grad():
            self.decoder.weight.copy_(torch.tensor([[1., -.4], [-.5, .2], [.3, .8]]))
            self.decoder.bias.zero_()


def learner(model=None):
    return SimpleNamespace(model=model or Model(), consequence_state={'keep': 'untouched'})


def trajectory(m, n):
    logits = m.comp_eta + m.comp_weight[0] + m.ai_w + m.intrinsic_kappa[0]
    h = torch.sigmoid(logits).expand(n)
    r = torch.stack((m.comp_weight[1] + m.log_tau_m[0], m.comp_eta)).expand(n, 2)
    return h, r


def timing(start=0, length=2, first=0, tail=1):
    return {'kind': 'timing', 'start': start, 'length': length,
            'first': first, 'tail_ticks': tail}


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_host_score_archive_preserves_delayed_values_and_exact_policy_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    base = Model().to(device)
    objects = [learner(copy.deepcopy(base)), learner(copy.deepcopy(base))]
    results = []
    for obj, host in zip(objects, (False, True)):
        obj.native_credit_cpu = host
        m = obj.model
        h, r = trajectory(m, 2)
        start = [{'kind':'clock','ticks':2}, timing(first=1,tail=0),
                 {'kind':'content','tick':1,'word':1,'temperature':1.},
                 {'kind':'response_begin'}]
        native_policy_credit(obj,start,h,r,[2],include_policy_heads=True)
        trace = obj.consequence_state['native_credit']
        for archive in (trace['E'],trace['response_score']):
            assert all(v.device.type == ('cpu' if host else device) for v in archive.values())
        snapshot = {k:{n:v.detach().cpu().clone() for n,v in trace[k].items()}
                    for k in ('E','response_score')}
        h,r = trajectory(m,1)
        end = [{'kind':'clock','ticks':1},timing(length=1,first=None,tail=0),
               {'kind':'response_end','advantage':1.7},{'kind':'trial_end'}]
        j,diag = native_policy_credit(obj,end,h,r,[1],include_policy_heads=True)
        names = list(dict(m.named_parameters()))
        gradients = torch.autograd.grad(j,tuple(m.parameters()),allow_unused=True)
        results.append((snapshot,j.detach().cpu(),
                        {n:g.cpu() for n,g in zip(names,gradients) if g is not None}))
        assert diag['native_eligibility_gpu_bytes'] == 0
    for kind in ('E','response_score'):
        assert results[0][0][kind].keys() == results[1][0][kind].keys()
        for name,value in results[0][0][kind].items():
            torch.testing.assert_close(value,results[1][0][kind][name],rtol=0,atol=0)
    torch.testing.assert_close(results[0][1],results[1][1],rtol=0,atol=0)
    assert results[0][2].keys()==results[1][2].keys()
    for name,gradient in results[0][2].items():
        torch.testing.assert_close(gradient,results[1][2][name],rtol=0,atol=0)


def test_native_whitelist_includes_synapses_and_excludes_ports_heads_homeostat():
    names = native_parameter_names(Model())
    assert {'comp_weight', 'comp_eta', 'comp_logit_trace', 'log_tau_m',
            'mbon_kappa', 'dopamine_logit_fast', 'log_kc_release'} <= set(names)
    assert not {'decoder.weight', 'decoder.bias', 'ai_w', 'intrinsic_kappa',
                'edge_weight_e', 'voice_log_gain'} & set(names)


def test_large_host_trace_validation_bounds_temporaries_and_checks_last_coordinate(monkeypatch):
    from information_boltzmann.core.fly_native_policy_credit import _finite
    values = torch.zeros((1 << 20) + 7)
    real = torch.isfinite
    sizes = []

    def inspect(block):
        sizes.append(block.numel())
        return real(block)

    monkeypatch.setattr(torch, 'isfinite', inspect)
    _finite(values, 'large archive')
    assert sizes == [1 << 20, 7]
    values[-1] = float('inf')
    with pytest.raises(ValueError, match='large archive'):
        _finite(values, 'large archive')


def test_native_retained_graph_disables_buffer_donation_without_disabling_compilation():
    import torch._functorch.config as config
    m = Model()
    m._compiled_coefficients, m._compiled_finish = object(), object()
    with config.patch(donated_buffer=True):
        prepare_native_policy_backward(m)
        assert config.donated_buffer is False
        assert not hasattr(m, '_compiled_coefficients') and not hasattr(m, '_compiled_finish')
        replacement = object()
        m._compiled_finish = replacement
        prepare_native_policy_backward(m)
        assert m._compiled_finish is replacement  # No repeated compile invalidation.


def test_native_motor_score_reaches_brain_without_changing_policy_or_ordinary_gradients():
    from information_boltzmann.core.fly_active_inference import ai_step
    m = nn.Module()
    m.ai_mode, m.ai_brain_gradient, m.ai_precision = 'motor', False, False
    m.read_indices = torch.tensor([0, 1])
    m.ai_lambda = torch.tensor(.7)
    m.ai_mu = None
    m.ai_w = nn.Parameter(torch.tensor([[.2, -.1], [.8, .3]]))
    m.ai_b = nn.Parameter(torch.tensor([.1, -.2]))
    m.comp_weight = nn.Parameter(torch.tensor([.4, .2]))
    m.comp_logit_trace = nn.Parameter(torch.tensor(2.))
    m.read_norm, m.decoder = nn.Identity(), nn.Linear(2, 3)
    s = SimpleNamespace(ring=(m.comp_weight[None],))
    regular_g, native_g = torch.zeros(1, 2), torch.zeros(1, 2)
    regular, native = [], []
    rng = torch.get_rng_state().clone()
    for commit in (False, True):
        hz, regular_g, _ = ai_step(m, s, regular_g, taped=commit)
        nhz, native_g, _ = ai_step(m, s, native_g, taped=commit, brain_gradient=True)
        torch.testing.assert_close(hz, nhz, rtol=0, atol=0)
        torch.testing.assert_close(regular_g, native_g, rtol=0, atol=0)
        regular.append(hz); native.append(nhz)
    assert torch.equal(rng, torch.get_rng_state())  # Shadow scoring consumes no extra randomness.
    ordinary = torch.autograd.grad(torch.cat(regular).sum(), m.comp_weight,
                                   allow_unused=True, retain_graph=True)[0]
    assert ordinary is None
    l = SimpleNamespace(model=m, consequence_state={})
    events = [{'kind': 'clock', 'ticks': 2}, timing(first=1, tail=0),
              {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 1.}]
    j, diag = native_policy_credit(l, events, torch.cat(native), m.comp_weight[None].expand(2, 2), [2])
    j.backward()
    assert m.comp_weight.grad.abs().sum() > 0
    assert m.ai_w.grad is None and m.ai_b.grad is None
    assert diag['native_timing_connected'] == 1


def test_same_window_settlement_matches_analytic_native_gradient_only():
    l = learner()
    h, r = trajectory(l.model, 2)
    events = [{'kind': 'clock', 'ticks': 2}, timing(),
              {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 3.}]
    j, diag = native_policy_credit(l, events, h, r, [2])
    expected = 3 * .5 * (1 - float(h[0].detach()))
    j.backward()
    assert float(l.model.comp_eta.grad) == pytest.approx(expected)
    assert l.model.comp_weight.grad.tolist() == pytest.approx([expected, 0.])
    assert l.model.intrinsic_kappa.grad is None
    assert l.model.ai_w.grad is None
    assert l.model.decoder.weight.grad is None
    assert diag['native_response_settlements'] == 1
    assert diag['native_score_connected_events'] == 1
    assert l.consequence_state['keep'] == 'untouched'
    assert not l.consequence_state['native_credit'].get('response_score')


@pytest.mark.parametrize('first', [None, 0, 1, 2])
def test_first_commit_survival_is_exact_for_the_observed_prefix(first):
    l = learner()
    h, r = trajectory(l.model, 3)
    events = [{'kind': 'clock', 'ticks': 3}, timing(length=3, first=first, tail=0),
              {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 1.}]
    j, _ = native_policy_credit(l, events, h, r, [3])
    p = float(h[0].detach())
    expected = -3 * p if first is None else (1 - p) - first * p
    j.backward()
    assert float(l.model.comp_eta.grad) == pytest.approx(expected)


def test_content_matches_sampling_temperature_and_post_temperature_bias():
    l = learner()
    h, r = trajectory(l.model, 2)
    bias = torch.tensor([.6, -.4, .2], requires_grad=True)
    logits = l.model.decoder(r[0:1]).float() / .7 + bias.detach()
    score = torch.log_softmax(logits, -1)[0, 2]
    expected = torch.autograd.grad(score, l.model.comp_weight, retain_graph=True)[0] * .5 * 2.
    events = [{'kind': 'clock', 'ticks': 2},
              {'kind': 'content', 'tick': 0, 'word': 2, 'temperature': .7,
               'logit_bias': bias, 'tail_ticks': 1},
              {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 2.}]
    j, _ = native_policy_credit(l, events, h, r, [2])
    j.backward()
    assert torch.allclose(l.model.comp_weight.grad, expected)
    assert bias.grad is None
    assert l.model.decoder.weight.grad is None
    assert l.model.comp_weight.grad[1] != 0


def test_pending_response_survives_checkpoint_and_is_not_discounted_as_physiology():
    l = learner()
    h, r = trajectory(l.model, 2)
    events = [{'kind': 'clock', 'ticks': 2}, timing(tail=0), {'kind': 'response_begin'}]
    first_j, _ = native_policy_credit(l, events, h, r, [2])
    assert float(first_j) == 0
    restored = learner(copy.deepcopy(l.model))
    restored.consequence_state = copy.deepcopy(l.consequence_state)
    h2, r2 = trajectory(restored.model, 3)
    j, diag = native_policy_credit(restored, [
        {'kind': 'clock', 'ticks': 3}, {'kind': 'response_end', 'advantage': 2.}], h2, r2, [3])
    j.backward()
    expected = 2 * (1 - float(h[0].detach()))
    assert float(restored.model.comp_eta.grad) == pytest.approx(expected)
    e = restored.consequence_state['native_credit']['E']['comp_eta']
    assert float(e) == pytest.approx((1 - float(h[0].detach())) * .5 ** 3)
    assert diag['native_total_clock_ticks'] == 5
    assert not diag['native_response_pending']
    h3, r3 = trajectory(restored.model, 1)
    j3, diag3 = native_policy_credit(restored, [
        {'kind': 'clock', 'ticks': 1}, {'kind': 'response_end', 'advantage': 2.}], h3, r3, [1])
    assert float(j3) == 0
    assert diag3['native_unmatched_response_ends'] == 1


def test_later_same_window_action_is_excluded_from_earlier_response():
    l = learner()
    h, r = trajectory(l.model, 4)
    events = [{'kind': 'clock', 'ticks': 2}, timing(first=0, tail=0),
              {'kind': 'response_begin'}, {'kind': 'clock', 'ticks': 2},
              timing(start=2, first=None, tail=0),
              {'kind': 'response_end', 'advantage': 1.}]
    j, _ = native_policy_credit(l, events, h, r, [2, 2])
    j.backward()
    assert float(l.model.comp_eta.grad) == pytest.approx(1 - float(h[0].detach()))


def test_graph_is_retained_for_the_original_prediction_loss():
    l = learner()
    h, r = trajectory(l.model, 1)
    j, _ = native_policy_credit(l, [
        {'kind': 'clock', 'ticks': 1}, timing(length=1, tail=0),
        {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 1.}], h, r, [1])
    (j + r.square().sum()).backward()
    assert torch.isfinite(l.model.comp_weight.grad).all()
    assert l.model.comp_weight.grad[1] != 0


def test_disconnected_native_graph_is_reported_and_prior_state_is_not_cleared():
    l = learner()
    h, r = trajectory(l.model, 1)
    j, diag = native_policy_credit(l, [
        {'kind': 'clock', 'ticks': 1}, timing(length=1, tail=0),
        {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 1.}],
        h.detach(), r.detach(), [1])
    assert float(j) == 0
    assert diag['native_score_no_graph_events'] == 1
    assert l.consequence_state['keep'] == 'untouched'


@pytest.mark.parametrize('bad', ['hazard', 'advantage', 'bias', 'clock', 'first'])
def test_invalid_or_nonfinite_events_fail_without_partly_mutating_state(bad):
    l = learner()
    h, r = trajectory(l.model, 2)
    events = [{'kind': 'clock', 'ticks': 2}, timing(),
              {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 1.}]
    if bad == 'hazard':
        h = h * float('nan')
    elif bad == 'advantage':
        events[-1]['advantage'] = float('nan')
    elif bad == 'bias':
        events[1] = {'kind': 'content', 'tick': 0, 'word': 1, 'temperature': 1.,
                     'logit_bias': torch.tensor([0., float('inf'), 0.])}
    elif bad == 'clock':
        events[0]['ticks'] = 1
    elif bad == 'first':
        events[1]['first'] = 2
    with pytest.raises(ValueError):
        native_policy_credit(l, events, h, r, [2])
    assert l.consequence_state == {'keep': 'untouched'}


def test_nonfinite_native_derivative_is_rejected():
    l = learner()
    # Forward finite, derivative undefined: reject the score before saving it.
    native_zero = l.model.comp_eta - l.model.comp_eta.detach()
    h = torch.sigmoid(torch.sqrt(native_zero)).expand(1)
    r = torch.zeros(1, 2)
    with pytest.raises(ValueError, match='score derivative'):
        native_policy_credit(l, [{'kind': 'clock', 'ticks': 1},
                                 timing(length=1, tail=0)], h, r, [1])
    assert l.consequence_state == {'keep': 'untouched'}


def test_many_scores_use_at_most_three_reverse_passes():
    l = learner()
    h, r = trajectory(l.model, 20)
    events = []
    for tick in range(20):
        events.extend([{'kind': 'clock', 'ticks': 1},
                       timing(start=tick, length=1, tail=0)])
    events.extend([{'kind': 'response_begin'},
                   {'kind': 'response_end', 'advantage': 2.}])
    j, diag = native_policy_credit(l, events, h, r, [1] * 20)
    j.backward()
    expected = 2 * (1 - float(h[0].detach())) * sum(.5 ** k for k in range(20))
    assert float(l.model.comp_eta.grad) == pytest.approx(expected)
    assert diag['native_score_events'] == 20
    assert diag['native_score_connected_events'] == 20
    assert diag['native_reverse_passes'] <= 3


def test_splitting_event_stream_preserves_credit_and_pending_state():
    whole = learner()
    split = learner(copy.deepcopy(whole.model))
    first = [{'kind': 'clock', 'ticks': 1}, timing(length=1, tail=0),
             {'kind': 'response_begin'}, {'kind': 'clock', 'ticks': 1},
             timing(start=1, length=1, first=None, tail=0),
             {'kind': 'response_end', 'advantage': 2.},
             {'kind': 'response_begin'}]
    second = [{'kind': 'clock', 'ticks': 2},
              {'kind': 'response_end', 'advantage': -1.}]
    h, r = trajectory(whole.model, 4)
    jw, _ = native_policy_credit(whole, first + second, h, r, [2, 2])
    hs, rs = trajectory(split.model, 2)
    j1, d1 = native_policy_credit(split, first, hs, rs, [2])
    assert d1['native_response_pending']
    hs2, rs2 = trajectory(split.model, 2)
    j2, d2 = native_policy_credit(split, second, hs2, rs2, [2])
    jw.backward()
    (j1 + j2).backward()
    # Split windows differentiate partial linear sums separately; their final
    # floating-point association differs, although the analytic score is equal.
    assert torch.allclose(whole.model.comp_eta.grad, split.model.comp_eta.grad,
                          rtol=0., atol=2e-15)
    ew = whole.consequence_state['native_credit']['E']
    es = split.consequence_state['native_credit']['E']
    assert ew.keys() == es.keys()
    assert all(torch.allclose(ew[k], es[k], rtol=0., atol=2e-15) for k in ew)
    assert not d2['native_response_pending']


class SideEffect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, model):
        ctx.model = model
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        from information_boltzmann.core import triton_synapse
        m = ctx.model
        assert triton_synapse.DIRECT_EDGE_GRAD
        if torch.is_tensor(getattr(m, 'homeo_spikes', None)):
            m.homeo_spikes.add_(9.)
        else:
            m.homeo_spikes = torch.tensor([9.])
        m.homeo_ticks = (getattr(m, 'homeo_ticks', 0) or 0) + 11
        m.step_grad_clips = torch.tensor(7.)
        # Mimic the fused kernel's direct leaf gradient side effect.
        if m.edge_weight_e.grad is None:
            m.edge_weight_e.grad = torch.ones_like(m.edge_weight_e)
        else:
            m.edge_weight_e.grad.add_(1.)
        return gradient, None


@pytest.mark.parametrize('initial', ['absent', 'none', 'existing'])
def test_auxiliary_recompute_preserves_homeostat_counts_and_edge_gradients(initial):
    from information_boltzmann.core import triton_synapse
    l = learner()
    m = l.model
    original_spikes = torch.tensor([2.])
    original_gradient = torch.full_like(m.edge_weight_e, 5.)
    m.edge_weight_e.grad = original_gradient
    if initial == 'none':
        m.homeo_spikes = m.homeo_ticks = m.step_grad_clips = None
    elif initial == 'existing':
        m.homeo_spikes, m.homeo_ticks, m.step_grad_clips = original_spikes, 4, torch.tensor(3.)
    h = torch.sigmoid(SideEffect.apply(m.comp_eta, m)).expand(1)
    r = torch.zeros(1, 2)
    j, diag = native_policy_credit(l, [
        {'kind': 'clock', 'ticks': 1}, timing(length=1, tail=0),
        {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 1.}], h, r, [1])
    assert not triton_synapse.DIRECT_EDGE_GRAD
    assert m.edge_weight_e.grad is original_gradient
    assert float(m.edge_weight_e.grad[0]) == 5.
    if initial == 'absent':
        assert not any(hasattr(m, key) for key in ('homeo_spikes', 'homeo_ticks', 'step_grad_clips'))
    elif initial == 'none':
        assert m.homeo_spikes is m.homeo_ticks is m.step_grad_clips is None
    else:
        assert m.homeo_spikes is original_spikes
        assert torch.equal(m.homeo_spikes, torch.tensor([2.]))
        assert m.homeo_ticks == 4
        assert float(m.step_grad_clips) == 3.
    assert diag['native_auxiliary_homeostasis_preserved']
    j.backward()
    assert m.comp_eta.grad is not None


def test_score_must_follow_its_real_clock():
    l = learner()
    h, r = trajectory(l.model, 1)
    with pytest.raises(ValueError, match='precedes'):
        native_policy_credit(l, [timing(length=1, tail=0), {'kind': 'clock', 'ticks': 1}], h, r, [1])
    assert l.consequence_state == {'keep': 'untouched'}
