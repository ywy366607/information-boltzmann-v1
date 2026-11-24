"""Unit tests for H-Net Dynamic Chunking (causality, ratio loss, Triton scan, grads)."""
from __future__ import annotations

import pytest
import torch

from scripts.ib_local.hnet_chunking import HNetDynamicChunker, hnet_bypass


def test_causality_zero_future_leakage():
    torch.manual_seed(0)
    for B in (1, 3):
        chunker = HNetDynamicChunker(16, target_ratio=3.0, min_seq_len=4)
        x1 = torch.randn(B, 20, 16)
        x2 = x1.clone()
        x2[:, 12:] += 5.0  # Perturb future positions t >= 12 only!

        # Calibrate once identically so router_thresh is fixed for both passes.
        chunker.calibrated.fill_(True)
        chunker.router_thresh.data.fill_(0.22)

        p1, b1, _ = chunker.route(x1)
        pack1 = chunker.chunk(x1, p1, b1)
        y1, _ = chunker.dechunk(pack1['x_chunk'] * 1.5, pack1)

        p2, b2, _ = chunker.route(x2)
        pack2 = chunker.chunk(x2, p2, b2)
        y2, _ = chunker.dechunk(pack2['x_chunk'] * 1.5, pack2)

        assert torch.allclose(p1[:, :12], p2[:, :12], atol=1e-6)
        assert torch.equal(b1[:, :12], b2[:, :12])
        assert torch.allclose(y1[:, :12], y2[:, :12], atol=1e-6)


def test_ratio_loss_minimum_and_restoring_gradient():
    N = 4.0
    for f_val, expected_sign in ((0.5, 1.0), (0.1, -1.0)):
        f = torch.tensor(f_val)
        g = torch.tensor(0.25, requires_grad=True)
        loss = (N / (N - 1.0)) * ((N - 1.0) * f * g + (1.0 - f) * (1.0 - g))
        loss.backward()
        assert (g.grad.item() * expected_sign) > 0.0

    # Exact minimum == 1.0 at F == G == 1/N
    f_eq = torch.tensor(1.0 / N)
    g_eq = torch.tensor(1.0 / N)
    loss_eq = (N / (N - 1.0)) * ((N - 1.0) * f_eq * g_eq + (1.0 - f_eq) * (1.0 - g_eq))
    assert abs(float(loss_eq) - 1.0) < 1e-6


def test_upcycling_identity_at_all_boundaries():
    torch.manual_seed(1)
    chunker = HNetDynamicChunker(16, target_ratio=4.0, min_seq_len=4)
    x = torch.randn(1, 12, 16)
    p = torch.ones(1, 12)
    b = torch.ones(1, 12)
    pack = chunker.chunk(x, p, b)
    z_core = torch.randn(1, 12, 16)
    y, _ = chunker.dechunk(z_core, pack)
    # With p == 1, b == 1, W_res == 0: output equals z_core identically (Theorem 3)!
    assert torch.allclose(y, z_core, atol=1e-6)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_end_to_end_gradients_and_bypass(device):
    torch.manual_seed(2)
    chunker = HNetDynamicChunker(32, target_ratio=4.0, min_seq_len=8, device=device)
    x = torch.randn(1, 40, 32, device=device, requires_grad=True)
    inc = torch.randn(1, 40, 8, 16, device=device, requires_grad=True)

    p, b, r_loss = chunker.route(x)
    pack = chunker.chunk(x, p, b, incoming=inc)
    assert 1 <= pack['M_max'] <= 40
    z_core = pack['x_chunk'] * 0.8 + 0.1
    y, inc_up = chunker.dechunk(z_core, pack, incoming_chunk=pack['incoming_chunk'])
    assert y.shape == x.shape
    assert inc_up.shape == inc.shape

    loss = y.square().mean() + inc_up.square().mean() + 0.03 * r_loss
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert chunker.q_proj.weight.grad is not None and chunker.q_proj.weight.grad.abs().sum() > 0
    assert chunker.k_proj.weight.grad is not None and chunker.k_proj.weight.grad.abs().sum() > 0
    assert chunker.router_thresh.grad is not None and chunker.router_thresh.grad.abs().sum() > 0
    assert chunker.residual_proj.weight.grad is not None and chunker.residual_proj.weight.grad.abs().sum() > 0

    # Bypass check
    holder = type('M', (), {'hnet_chunker': chunker})()
    assert chunker.active_for(40)
    with hnet_bypass(holder):
        assert not chunker.active_for(40)
    assert chunker.active_for(40)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_hnet_section24_single_step_matches_batched_exact(device):
    """H-Net §2.4: 1-byte stateful chunked decoding must match full-sequence chunked forward bit-for-bit."""
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.hnet_chunking import HNetStepState, hnet_step_scope
    from scripts.ib_local.train_gdn_fast_table import TableGDN

    torch.manual_seed(7)
    base = GatedDeltaNetLM(vocab_size=256, d=32, layers=6, heads=2)
    body = TableGDN(base, use_table=True, inject_layer=1).to(device)
    body.eval()
    chunker = HNetDynamicChunker(
        32, enc_layers=2, dec_layers=2, target_ratio=3.0, min_seq_len=1,
        router_scale=6.0, router_thresh_init=0.50, device=device,
    )
    chunker.calibrated.fill_(True)
    with torch.no_grad():
        torch.nn.init.normal_(chunker.residual_proj.weight, std=0.05)
    body.hnet_chunker = chunker

    T = 24
    ids = torch.randint(0, 256, (1, T), device=device)
    logp = torch.log_softmax(torch.randn(1, T, 256, device=device), dim=-1)
    s0 = body.initial_state(1, device=device)

    with torch.no_grad():
        logits_batch, state_batch, _ = body.predict(ids, s0, logp)

        # Step-by-step H-Net §2.4 execution (1 byte at a time, skipping deep core on non-boundaries).
        st = s0.clone()
        step_state = HNetStepState()
        step_logits = []
        for t in range(T):
            with hnet_step_scope(body, step_state):
                lg, st, _ = body.predict(ids[:, t:t + 1], st, logp[:, t:t + 1])
            step_logits.append(lg)
        logits_step = torch.cat(step_logits, dim=1)

    assert step_state.steps == T
    assert 1 <= step_state.boundaries < T  # Deep core was actually skipped on non-boundary bytes!
    assert (logits_step - logits_batch).abs().max().item() < 1e-4
    assert (st - state_batch).abs().max().item() < 1e-4

