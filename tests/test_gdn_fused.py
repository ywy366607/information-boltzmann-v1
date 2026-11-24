"""The fused Triton recurrence must reproduce the reference GatedDeltaNetLayer exactly
(up to float32 summation order): outputs, next state, diagnostics and every gradient."""

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ib_local"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CASES = [
    # recurrence, conv, budget, eta, gain, decay_init, feedback_init
    ("decay_then_delta", 0, 0.0, None, None, "constant", "constant"),
    ("decay_then_delta", 4, 32.0, None, None, "constant", "constant"),
    ("decay_then_delta", 4, 0.6, None, None, "constant", "constant"),  # cap active
    ("feedback_delta", 4, 32.0, 0.1, 2.0, "constant", "constant"),
    ("feedback_delta", 4, 0.0, 0.8, 4.0, "constant", "constant"),      # latch region
    ("feedback_delta", 0, 0.6, 0.8, 4.0, "constant", "constant"),      # latch + active cap
    ("decay_then_delta", 4, 32.0, None, None, "log_timescale", "constant"),
    ("feedback_delta", 4, 32.0, None, None, "log_timescale", "threshold_spread"),
]


def run(layer, x, state):
    out, nxt, diag = layer(x, state)
    loss = (out * torch.linspace(-1, 1, out.shape[-1], device=out.device)).square().mean() \
        + nxt.square().sum() * 1e-2
    loss.backward()
    grads = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
    return out, nxt, diag, grads, x.grad.clone(), state.grad.clone()


@pytest.mark.parametrize("recurrence,conv,budget,eta,gain,decay_init,feedback_init", CASES)
def test_fused_matches_reference(recurrence, conv, budget, eta, gain, decay_init, feedback_init):
    import gated_deltanet
    import gdn_fused
    torch.manual_seed(0)
    layer = gated_deltanet.GatedDeltaNetLayer(64, 2, recurrence, mlp_hidden=96,
                                              conv_kernel=conv, content_budget=budget,
                                              decay_init=decay_init,
                                              feedback_init=feedback_init).cuda().double()
    if eta is not None:
        with torch.no_grad():
            layer.feedback_logit.fill_(math.log(eta / (1 - eta)))
            layer.feedback_gain_raw.fill_(math.log(math.expm1(gain)))
    layer = layer.float()
    other = copy.deepcopy(layer)
    B, T = 2, 24
    x0 = torch.randn(B, T, 64, device="cuda")
    s0 = torch.randn(B, layer.state_size if conv else layer.matrix_size, device="cuda") * 0.3
    if not conv:
        s0 = s0.view(B, 2, 32, 32)
    ref = run(layer, x0.clone().requires_grad_(), s0.clone().requires_grad_())
    gdn_fused.enable()
    try:
        fus = run(other, x0.clone().requires_grad_(), s0.clone().requires_grad_())
    finally:
        gdn_fused.disable()
    for a, b, name in ((ref[0], fus[0], "output"), (ref[1], fus[1], "state"),
                       (ref[4], fus[4], "dx"), (ref[5], fus[5], "dstate")):
        torch.testing.assert_close(b, a, rtol=2e-4, atol=2e-5, msg=name)
    assert ref[2].keys() == fus[2].keys()
    for key in ref[2]:
        torch.testing.assert_close(fus[2][key], ref[2][key], rtol=2e-4, atol=2e-6, msg=key)
    assert ref[3].keys() == fus[3].keys()
    for key in ref[3]:
        torch.testing.assert_close(fus[3][key], ref[3][key], rtol=5e-4, atol=5e-5, msg=key)


@pytest.mark.parametrize('steps', (1, 16, 17, 128))
def test_state_checkpoints_match_full_scan_and_inference(steps):
    from gdn_fused import fused_delta

    torch.manual_seed(17)
    shape = (1, 8, steps, 32)
    values = [torch.randn(shape, device='cuda') * .15 for _ in range(3)]
    values += [torch.rand(shape[:-1], device='cuda') for _ in range(2)]
    values += [torch.rand(8, device='cuda') for _ in range(2)]
    values.append(torch.randn((1, 8, 32, 32), device='cuda') * .15)

    def run(block):
        inputs = [v.detach().requires_grad_() for v in values]
        retained = []
        def pack(t):
            if t.ndim == 5:
                retained.append(t.shape[2])
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = fused_delta(*inputs, 0.6, True, block=block)
        grad = torch.autograd.grad(out[0].square().sum()+out[1].square().sum(), inputs)
        return out, grad, retained

    full, d_full, old = run(1)
    compact, d_compact, new = run(16)
    with torch.no_grad():
        inference = fused_delta(*values, 0.6, True)
    assert old == [steps+1]
    assert new == [math.ceil(steps/16)+1 if steps > 16 else steps+1]
    for expected, actual in zip((*full, *full, *d_full), (*inference, *compact, *d_compact)):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_birth_initialization():
    """Log-timescale heads span short to long memory; feedback heads straddle the latch
    threshold; shared weights equal the plain control's; old configs keep old behaviour."""
    import gated_deltanet
    torch.manual_seed(11)
    control = gated_deltanet.GatedDeltaNetLM(256, 128, 4, 4, "decay_then_delta", 256, 4, 32.0,
                                             decay_init="log_timescale")
    torch.manual_seed(11)
    spread = gated_deltanet.GatedDeltaNetLM(256, 128, 4, 4, "feedback_delta", 256, 4, 32.0,
                                            decay_init="log_timescale",
                                            feedback_init="threshold_spread")
    shared = dict(control.named_parameters())
    for name, p in spread.named_parameters():
        if "feedback" not in name:
            assert torch.equal(p, shared[name]), name
    alphas = torch.cat([layer._birth_retention() for layer in control.layers])
    assert alphas.min() < 0.7 and alphas.max() > 0.995
    for layer in spread.layers:
        alpha = layer._birth_retention()
        eta = layer.feedback_logit.sigmoid()
        gain = torch.nn.functional.softplus(layer.feedback_gain_raw)
        ratio = 0.5 * eta * gain / (1 - alpha * 0.5)
        assert ratio[0] < 0.01 and ratio[1] < 1 and ratio[-1] > 1 and ratio.max() <= 3.01
    old = gated_deltanet.GatedDeltaNetLM(256, 128, 4, 4, "feedback_delta", 256, 4, 32.0)
    assert "decay_init" in old.config_dict() and old.layers[0].alpha_proj.bias is not None
    assert torch.allclose(old.layers[0].feedback_logit.sigmoid(), torch.full((4,), 0.1))
