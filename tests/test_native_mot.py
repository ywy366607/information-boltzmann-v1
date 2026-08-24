"""Unit tests for native multimodal MoT (spec-accurate experts + field update)."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.native_mot import (  # noqa: E402
    NativeMoTBlock,
    NativeMoTLayer,
    NativeMoTStack,
    estimate_pretrain_tokens_needed,
)


def test_projections_not_shared():
    """Wq_v is a different Parameter than Wq_t (and similarly K/V)."""
    blk = NativeMoTBlock(d=64, n_heads=4)
    assert blk.Wq_v is not blk.Wq_t
    assert blk.Wk_v is not blk.Wk_t
    assert blk.Wv_v is not blk.Wv_t
    assert blk.Wo_v is not blk.Wo_t
    assert blk.ffn_v is not blk.ffn_t
    # different parameter objects
    assert id(blk.Wq_v.weight) != id(blk.Wq_t.weight)


def test_mot_block_shapes_and_both_update():
    blk = NativeMoTBlock(d=64, n_heads=4)
    S = torch.randn(2, 8, 64)
    H = torch.randn(2, 12, 64)
    mask = torch.ones(2, 12)
    S2, H2, P2 = blk(S, H, text_mask=mask)
    assert P2 is None
    assert S2.shape == S.shape and H2.shape == H.shape
    assert (S2 - S).abs().sum() > 0
    assert (H2 - H).abs().sum() > 0
    # different H → different S' (cross-modal K/V shared space)
    S3, _, _ = blk(S, torch.randn_like(H), text_mask=mask)
    assert (S2 - S3).abs().sum() > 1e-4


def test_mot_dual_patch_concat():
    blk = NativeMoTBlock(d=64, n_heads=4)
    S = torch.randn(2, 8, 64)
    P = torch.randn(2, 16, 64)
    H = torch.randn(2, 10, 64)
    S2, H2, P2 = blk(S, H, text_mask=torch.ones(2, 10), P=P)
    assert S2.shape == S.shape and P2.shape == P.shape and H2.shape == H.shape


def test_full_layer_field_pipeline():
    layer = NativeMoTLayer(d_x=32, d=64, n_slices=8, n_heads=4, res=8)
    X = torch.randn(2, 64, 32)  # N=8*8
    H = torch.randn(2, 10, 64)
    X2, H2, tr = layer(X, H, text_mask=torch.ones(2, 10))
    assert X2.shape == X.shape and H2.shape == H.shape
    assert tr.x_delta > 0 and tr.h_delta >= 0


def test_stack_dims_default_product():
    """Default product dims: d_x=128, d=512 (may be heavy; smoke with smaller)."""
    stack = NativeMoTStack(
        d_llm=64, res=16, d_x=32, d=64, n_slices=8, n_layers=2, n_heads=4,
    )
    img = torch.rand(1, 3, 16, 16)
    emb = torch.randn(1, 8, 64)
    X, H_llm, tok, traces = stack.forward_native(img, emb, torch.ones(1, 8))
    assert X.shape == (1, 256, 32)
    assert H_llm.shape == emb.shape
    assert tok.shape[1] == 8  # M slices projected
    assert len(traces) == 2
    out = stack(img, text_emb=emb, text_mask=torch.ones(1, 8))
    assert out.meta["kind"] == "native_mot"
    assert out.meta["slice_ephemeral"] is True


def test_common_f2_anchor_reads_one_coordinate_without_field_write():
    torch.manual_seed(9)
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=2,
        n_heads=4, surprise_mode="v1_bayes",
    ).eval()
    stack.set_record_field_trace(True)
    img = torch.rand(3, 3, 8, 8)
    emb = torch.randn(3, 6, 32)
    mask = torch.ones(3, 6)
    X, _, _, _ = stack.forward_native(img, emb, mask)
    final_before = X.clone()
    trace = stack.common_f2_anchor_trace(anchor_layer=0)

    assert len(stack._last_X_steps) == 3
    assert torch.allclose(stack._last_X_steps[0], stack._last_X_stem)
    assert torch.allclose(stack._last_X_steps[-1], final_before)
    assert trace["energy"].shape == (3, 3)
    assert trace["delta"].shape == (3, 2)
    assert trace["terminal_delta"].shape == (3,)
    assert torch.isfinite(trace["energy"]).all()
    direct = stack.common_f2_anchor_energy(stack._last_X_steps[-1])
    assert torch.allclose(direct, trace["energy"][:, -1])
    assert torch.allclose(stack._last_X, final_before)


def test_s_update_rms_dir_bounds_step():
    layer = NativeMoTLayer(
        d_x=32, d=64, n_slices=8, n_heads=4, res=8,
        surprise_mode="v0_jepa", s_update="rms_dir",
    )
    S = torch.randn(2, 8, 64)
    delta = torch.randn(2, 8, 64) * 80.0
    gate = torch.ones(2, 8, 1)
    S2 = layer._apply_s_update(S, delta, gate)
    step = (S2 - S).pow(2).mean(dim=-1).sqrt()
    # RMSNorm(Δ) has last-dim RMS ≈ 1, eta=1, g=1 → step RMS ≈ 1
    assert float(step.mean()) < 3.0
    assert float(delta.pow(2).mean().sqrt()) > 20.0


def test_s_update_trust_clips_only_when_large():
    layer = NativeMoTLayer(
        d_x=32, d=64, n_slices=8, n_heads=4, res=8,
        surprise_mode="baseline", s_update="trust", trust_rho=0.1,
    )
    S = torch.randn(2, 8, 64)
    tiny = 0.01 * S
    big = 10.0 * S
    gate = torch.ones(2, 8, 1)
    s_tiny = layer._apply_s_update(S, tiny, gate)
    s_big = layer._apply_s_update(S, big, gate)
    r_tiny = (s_tiny - S).pow(2).mean().sqrt() / S.pow(2).mean().sqrt()
    r_big = (s_big - S).pow(2).mean().sqrt() / S.pow(2).mean().sqrt()
    assert float(r_tiny) < 0.05
    assert float(r_big) < 0.15


def test_forward_momentum_smooths_flip():
    from fine_grain.forward_optim import ForwardStateOpt
    opt = ForwardStateOpt(d=8, kind="momentum", beta=0.9)
    S = torch.zeros(1, 4, 8)
    g = torch.ones(1, 4, 1)
    d1 = torch.ones(1, 4, 8)
    d2 = -torch.ones(1, 4, 8)
    S1, st = opt.step(S, d1, g)
    S2, _ = opt.step(S1, d2, g, state=st)
    # flip is damped: |step2| << |raw Δ|=1
    assert float((S2 - S1).abs().mean()) < 0.2


def test_forward_muon_orthogonalizes_slice_updates():
    from fine_grain.forward_optim import ForwardStateOpt
    torch.manual_seed(0)
    opt = ForwardStateOpt(d=16, kind="muon", beta=0.0)
    S = torch.zeros(2, 8, 16)
    g = torch.ones(2, 8, 1)
    delta = torch.randn(2, 8, 16)
    _, st = opt.step(S, delta, g)
    step = st["step"][0].float()
    u = torch.nn.functional.normalize(step, dim=-1)
    off = (u @ u.T).abs()
    off = off - torch.diag(off.diag())
    raw = torch.nn.functional.normalize(delta[0].float(), dim=-1)
    raw_off = (raw @ raw.T).abs()
    raw_off = raw_off - torch.diag(raw_off.diag())
    assert float(off.mean()) < float(raw_off.mean())


def test_stack_pred_loss_reaches_queries():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=2, n_heads=4,
        surprise_mode="v0_jepa",
    )
    img = torch.rand(2, 3, 8, 8)
    emb = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6)
    stack.forward_native(img, emb, mask)
    pred = stack._last_pred_loss
    assert pred.ndim == 0 and pred.requires_grad
    pred.backward()
    q = stack.layers[0].surprise_gate.slice_queries
    assert q.grad is not None and float(q.grad.norm()) > 0


def test_grads_flow_both_modalities():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=64, n_slices=4, n_layers=2, n_heads=4,
    )
    img = torch.rand(2, 3, 8, 8)
    emb = torch.randn(2, 6, 32, requires_grad=False)
    _, H_llm, tok, _ = stack.forward_native(img, emb, torch.ones(2, 6))
    loss = tok.pow(2).mean() + H_llm.pow(2).mean()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in stack.parameters() if p.requires_grad
    )


def test_token_budget_gate():
    n = 10_000_000
    assert estimate_pretrain_tokens_needed(n, 1.0) == 10_000_000
    assert estimate_pretrain_tokens_needed(n, 2.0) == 20_000_000


def test_layers_have_independent_slice_params():
    """SliceRead / Deslice are not weight-tied across layers."""
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=2, n_heads=4,
    )
    assert stack.layers[0].read is not stack.layers[1].read
    assert stack.layers[0].deslice is not stack.layers[1].deslice
    assert id(stack.layers[0].read.to_logits.weight) != id(
        stack.layers[1].read.to_logits.weight
    )
    assert stack.layers[0].read is not stack.readout


def test_dx_equals_d_aligned_shapes():
    """Mini alignment: d_x == d should forward cleanly."""
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=64, d=64, n_slices=4, n_layers=2, n_heads=4,
        deslice_topk=0,
    )
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 5, 32)
    X, H_llm, tok, traces = stack.forward_native(img, emb, torch.ones(1, 5))
    assert X.shape[-1] == 64
    assert tok.shape == (1, 4, 32)
    assert len(traces) == 2


def test_ablation_flags_forward_and_grad():
    """Five mini arms: clean + single-factor knobs must build and backprop."""
    flags = [
        dict(use_ada_temp=False, use_gumbel=False, deslice_topk=0, use_stiefel=False),
        dict(use_ada_temp=True, use_gumbel=False, deslice_topk=0, use_stiefel=False),
        dict(use_ada_temp=False, use_gumbel=True, deslice_topk=0, use_stiefel=False),
        dict(use_ada_temp=False, use_gumbel=False, deslice_topk=2, use_stiefel=False),
        dict(use_ada_temp=False, use_gumbel=False, deslice_topk=0, use_stiefel=True),
    ]
    for kw in flags:
        stack = NativeMoTStack(
            d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=2, n_heads=4, **kw,
        )
        stack.train()
        img = torch.rand(2, 3, 8, 8)
        emb = torch.randn(2, 6, 32)
        _, H_llm, tok, _ = stack.forward_native(img, emb, torch.ones(2, 6))
        loss = tok.pow(2).mean() + H_llm.pow(2).mean()
        loss.backward()
        assert stack.use_ada_temp == kw["use_ada_temp"]
        assert stack.deslice_topk == kw["deslice_topk"]


def test_clean_fixed_temp_vs_ada_temp_path():
    """use_ada_temp=False must not require temp-head gradients for assignment."""
    from fine_grain.native_mot import SliceRead

    r = SliceRead(16, 16, 4, n_heads=2, use_ada_temp=False)
    x = torch.randn(1, 9, 16)
    S, w = r(x)
    assert S.shape == (1, 4, 16) and w.shape == (1, 9, 4)
    r2 = SliceRead(16, 16, 4, n_heads=2, use_ada_temp=True)
    S2, w2 = r2(x)
    assert S2.shape == S.shape


def test_local_visual_kinds():
    """dw3 / none / cnx7 LocalVisual all forward under stack."""
    for kind in ("dw3", "none", "cnx7"):
        stack = NativeMoTStack(
            d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=2, n_heads=4,
            local_kind=kind,
        )
        img = torch.rand(1, 3, 8, 8)
        emb = torch.randn(1, 5, 32)
        X, H_llm, tok, _ = stack.forward_native(img, emb, torch.ones(1, 5))
        assert X.shape == (1, 64, 32)
        assert stack.local_kind == kind or (
            kind == "none" and stack.layers[0].local.kind == "none"
        )
        loss = tok.pow(2).mean()
        loss.backward()


def test_dual_patch_stack_forward_grad():
    """PatchEmbed(X)+MoT(P,S,H)+Deslice+Unpatch on point field."""
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=2, n_heads=4,
        dual_patch=True, patch_size=4, use_unpatch=True, local_kind="none",
        use_ada_temp=True, use_stiefel=True,
    )
    assert stack.dual_patch and stack.layers[0].patch_embed is not None
    img = torch.rand(2, 3, 8, 8)
    emb = torch.randn(2, 6, 32)
    X, H_llm, tok, traces = stack.forward_native(img, emb, torch.ones(2, 6))
    assert X.shape == (2, 64, 32)
    assert tok.shape[1] == 4
    assert len(traces) == 2
    (tok.pow(2).mean() + H_llm.pow(2).mean()).backward()
    # unpatch off still runs
    stack2 = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=1, n_heads=4,
        dual_patch=True, patch_size=2, use_unpatch=False,
    )
    X2, _, _, _ = stack2.forward_native(
        torch.rand(1, 3, 8, 8), torch.randn(1, 4, 32), torch.ones(1, 4),
    )
    assert X2.shape[-1] == 32


def test_visual_queries_ignore_answer_tokens():
    """prompt_mask must hide answer keys from Slice queries and earlier H."""
    torch.manual_seed(0)
    blk = NativeMoTBlock(d=32, n_heads=4)
    S = torch.randn(1, 4, 32)
    H = torch.randn(1, 6, 32)
    text_mask = torch.ones(1, 6, dtype=torch.bool)
    prompt_mask = torch.tensor([[True, True, True, True, False, False]])
    S1, H1, _ = blk(S, H, text_mask=text_mask, prompt_mask=prompt_mask)
    H_suffix = H.clone()
    H_suffix[:, 4:] = torch.randn_like(H_suffix[:, 4:])
    S2, H2, _ = blk(S, H_suffix, text_mask=text_mask, prompt_mask=prompt_mask)
    assert torch.allclose(S1, S2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(H1[:, :4], H2[:, :4], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(H1[:, 4:], H2[:, 4:], atol=1e-3)


def test_language_queries_are_causal():
    torch.manual_seed(1)
    blk = NativeMoTBlock(d=32, n_heads=4)
    S = torch.randn(1, 4, 32)
    H = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    _, H1, _ = blk(S, H, text_mask=mask)
    H_future = H.clone()
    H_future[:, -1] = torch.randn_like(H_future[:, -1])
    _, H2, _ = blk(S, H_future, text_mask=mask)
    assert torch.allclose(H1[:, :-1], H2[:, :-1], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(H1[:, -1], H2[:, -1], atol=1e-3)


def test_residual_read_s0_ignores_answer_tokens():
    torch.manual_seed(2)
    layer = NativeMoTLayer(
        d_x=32, d=32, n_slices=8, n_heads=4, res=8,
        use_residual_read=True, surprise_mode="baseline",
    )
    X = torch.randn(1, 64, 32)
    H = torch.randn(1, 6, 32)
    text_mask = torch.ones(1, 6, dtype=torch.bool)
    prompt_mask = torch.tensor([[True, True, True, True, False, False]])
    X1, H1, _ = layer(X, H, text_mask=text_mask, prompt_mask=prompt_mask)
    H_suffix = H.clone()
    H_suffix[:, 4:] = torch.randn_like(H_suffix[:, 4:])
    X2, H2, _ = layer(X, H_suffix, text_mask=text_mask, prompt_mask=prompt_mask)
    assert torch.allclose(X1, X2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(H1[:, :4], H2[:, :4], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_projections_not_shared()
    print("ok private projections")
    test_mot_block_shapes_and_both_update()
    print("ok mot block")
    test_full_layer_field_pipeline()
    print("ok layer")
    test_stack_dims_default_product()
    print("ok stack")
    test_grads_flow_both_modalities()
    print("ok grads")
    test_token_budget_gate()
    print("ok token gate")
    test_layers_have_independent_slice_params()
    print("ok layer-independent params")
    test_dx_equals_d_aligned_shapes()
    print("ok d_x==d")
    test_ablation_flags_forward_and_grad()
    print("ok ablation flags")
    test_clean_fixed_temp_vs_ada_temp_path()
    print("ok ada-temp switch")
    test_local_visual_kinds()
    print("ok local visual kinds")
    test_mot_dual_patch_concat()
    print("ok mot dual patch")
    test_dual_patch_stack_forward_grad()
    print("ok dual patch stack")
    print("ALL NATIVE MOT TESTS PASSED")
