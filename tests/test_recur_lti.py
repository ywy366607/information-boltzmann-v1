"""R0 shared-Φ recurrence: LTI ρ<1, no loop identity, full-depth BPTT."""
from __future__ import annotations

import inspect
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.native_mot import LTIInjection, NativeMoTLayer, NativeMoTStack


def test_lti_rho_lt_1_any_loga():
    inj = LTIInjection(16)
    with torch.no_grad():
        inj.log_A.fill_(12.0)
        inj.log_dt.fill_(-8.0)
    A = inj.get_A()
    assert A.min() > 0.0
    assert A.max() < 1.0
    assert float(inj.rho()) < 1.0
    with torch.no_grad():
        inj.log_A.uniform_(-5.0, 5.0)
        inj.log_dt.fill_(3.0)
    assert float(inj.rho()) < 1.0


def test_share_layers_one_module_four_loops():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=4, n_heads=4,
        share_layers=True, n_loops=4,
    )
    assert len(stack.layers) == 1
    assert stack.share_layers is True
    assert stack.lti_x is not None and stack.lti_h is not None
    img = torch.rand(2, 3, 8, 8)
    emb = torch.randn(2, 5, 32)
    X, H, tok, traces = stack.forward_native(img, emb, torch.ones(2, 5))
    assert X.shape == (2, 64, 32)
    assert H.shape == emb.shape
    assert len(traces) == 4
    assert len(stack._last_step_H) == 4
    rho = stack.injection_rho()
    assert rho["rho_x"] < 1.0 and rho["rho_h"] < 1.0


def test_loop_grads_flow_no_detach():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=4, n_heads=4,
        share_layers=True, n_loops=4,
    )
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 4, 32)
    _, H, _, _ = stack.forward_native(img, emb, torch.ones(1, 4))
    steps = stack._last_step_H
    assert all(h.requires_grad for h in steps)
    loss = steps[0].sum() + steps[-1].sum() + H.sum()
    loss.backward()
    w = stack.layers[0].mot.Wq_v.weight
    assert w.grad is not None and float(w.grad.abs().sum()) > 0
    assert stack.lti_x.log_A.grad is not None
    assert float(stack.lti_x.log_A.grad.abs().sum()) > 0


def test_lti_assemble_adds_delta_not_full_state():
    inj = LTIInjection(4)
    with torch.no_grad():
        inj.B.zero_()
    h = torch.randn(2, 3, 4)
    e = torch.randn(2, 3, 4)
    delta = torch.randn(2, 3, 4)
    x_phi = h + delta
    out = inj.assemble(h, e, x_phi - h)
    naive = inj(h, e) + x_phi
    assert torch.allclose(out, inj.get_A() * h + delta)
    assert not torch.allclose(out, naive, atol=1e-4)


def test_shared_loop_calls_phi_on_unmixed_field():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=1, n_heads=4,
        share_layers=True, n_loops=1,
    )
    seen = {}

    def _hook(mod, args):
        seen["X"] = args[0].detach().clone()

    handle = stack.layers[0].register_forward_pre_hook(_hook)
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 3, 32)
    with torch.no_grad():
        X0 = stack.encode_X(img)
        stack.forward_native(img, emb, torch.ones(1, 3), n_loops=1)
    handle.remove()
    assert "X" in seen
    assert torch.allclose(seen["X"], X0, atol=1e-5)


def test_phi_has_no_loop_index():
    sig = inspect.signature(NativeMoTLayer.forward)
    assert "loop_t" not in sig.parameters
    assert "loop_idx" not in sig.parameters
    lti_sig = inspect.signature(LTIInjection.forward)
    assert list(lti_sig.parameters) == ["self", "h", "e"]


def test_unshared_forward_matches_default():
    kw = dict(d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=2, n_heads=4)
    torch.manual_seed(0)
    a = NativeMoTStack(**kw, share_layers=False)
    torch.manual_seed(0)
    b = NativeMoTStack(**kw)
    assert a.lti_x is None and b.lti_x is None
    assert len(a.layers) == 2 and len(b.layers) == 2
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 3, 32)
    mask = torch.ones(1, 3)
    a.eval()
    b.eval()
    with torch.no_grad():
        Xa, Ha, ta, tra = a.forward_native(img, emb, mask)
        Xb, Hb, tb, trb = b.forward_native(img, emb, mask)
    assert torch.allclose(Xa, Xb)
    assert torch.allclose(Ha, Hb)
    assert torch.allclose(ta, tb)
    assert len(tra) == len(trb) == 2


def test_infer_n_loops_override():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=2, n_heads=4,
        share_layers=True, n_loops=2,
    )
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 3, 32)
    _, _, _, tr2 = stack.forward_native(img, emb, torch.ones(1, 3), n_loops=2)
    _, _, _, tr4 = stack.forward_native(img, emb, torch.ones(1, 3), n_loops=4)
    assert len(tr2) == 2 and len(tr4) == 4


def test_dualstream_deep_supervise_ce():
    from scripts.run_v0_surprise_eval import DualStreamVQAModel

    model = DualStreamVQAModel(
        d_model=32, n_slices=8, n_layers=4, res=8, n_heads=4,
        share_layers=True, n_loops=4, surprise_mode="baseline",
    )
    imgs = torch.rand(2, 3, 8, 8)
    prompts = ["What color is the small square ?", "What color is the small square ?"]
    out = model(imgs, prompts)
    assert len(out["logits_k"]) == 4
    assert out["logits"].shape == out["logits_k"][-1].shape
    targets = torch.zeros(2, dtype=torch.long)
    loss = model.task_loss(out, targets)
    loss.backward()
    assert model.mot_stack.layers[0].mot.Wq_v.weight.grad is not None
    assert model.mot_stack.lti_x.log_A.grad is not None


def test_unshared_dualstream_no_deep_sup():
    from scripts.run_v0_surprise_eval import DualStreamVQAModel

    model = DualStreamVQAModel(
        d_model=32, n_slices=8, n_layers=2, res=8, n_heads=4,
        surprise_mode="baseline",
    )
    assert model.share_layers is False
    assert model.deep_supervise is False
    imgs = torch.rand(1, 3, 8, 8)
    out = model(imgs, ["What color is the small square ?"])
    assert out["logits_k"] == []
    assert model.mot_stack.lti_x is None
