"""Unit tests for GatedDeltaNetLM capacity-matched baseline."""
import math
import pytest
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.cbim_torus3d import CBIMTorus3D


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_exchange_allocation_still_erases_content_contrast(device):
    """Actual V1.7 kernel: a fixed allocation is not a retained association.

    Force posterior=prior by its analytical address, without changing the
    production kernel. This audits the Lean obstruction, not language ability.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scripts.ib_local.gated_deltanet import exchange_association_content

    torch.manual_seed(31)
    share = torch.softmax(torch.randn(2, 32, dtype=torch.float64, device=device), -1)
    conditional = torch.randn(2, 32, 8, dtype=torch.float64, device=device)
    conditional = conditional / conditional.abs().sum(-1, keepdim=True).clamp_min(1)
    value = torch.randn(2, 8, dtype=torch.float64, device=device)
    value = value / value.abs().sum(-1, keepdim=True).clamp_min(1)
    gain = torch.full((2, 1), 12., dtype=torch.float64, device=device)
    rate = torch.full((2, 1), .7, dtype=torch.float64, device=device)
    evidence = -.5 * (conditional-value.unsqueeze(-2)).square().sum(-1)
    address = share.log()-gain*share-evidence
    content = share.unsqueeze(-1)*conditional
    new_content, new_share = exchange_association_content(
        content, share, address, value, gain, rate, 1.)
    torch.testing.assert_close(new_share, share, atol=1e-14, rtol=1e-12)
    actual = new_content/new_share.unsqueeze(-1)
    expected = (1-rate).unsqueeze(-1)*conditional + rate.unsqueeze(-1)*value.unsqueeze(-2)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(actual[:, 0]-actual[:, 1],
                               (1-rate)*(conditional[:, 0]-conditional[:, 1]),
                               atol=1e-14, rtol=1e-12)
    assert (actual-conditional).abs().max().item() > .1


def recurrent_reference(conv_kernel=0, content_budget=0.0):
    return GatedDeltaNetLM(vocab_size=256, d=32, layers=2, heads=2,
                          recurrence="decay_then_delta", mlp_hidden=64,
                          conv_kernel=conv_kernel, content_budget=content_budget)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_contact_exchange_joint_reference_retention_and_tiny_shares(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from scripts.ib_local.gated_deltanet import contact_exchange_content
    torch.manual_seed(94)
    c = torch.randn(2, 7, 5, dtype=torch.float64, device=device)
    c = c / c.abs().sum(-1, keepdim=True).clamp_min(1)
    a = torch.softmax(torch.randn(2, 7, dtype=c.dtype, device=device), -1)
    a[0, 0] = 0  # Retain conditional content even at zero allocation.
    a[1, 0] = 1e-300
    a = a/a.sum(-1, keepdim=True)
    k = torch.randn_like(a)
    v = torch.randn(2, 5, dtype=c.dtype, device=device)
    gain, rate = torch.full_like(a[:, :1], 3), torch.full_like(a[:, :1], .7)
    y = v/v.abs().sum(-1, keepdim=True).clamp_min(1)
    r = torch.softmax(gain*a+k-.5*(c-y[:, None]).square().sum(-1), -1)
    nc, na = contact_exchange_content(c, a, k, v, gain, rate, 1.)
    torch.testing.assert_close(nc, c+rate[..., None]*r[..., None]*(y[:, None]-c))
    torch.testing.assert_close(na, (1-rate)*a+rate*r)
    torch.testing.assert_close((rate*r).sum(-1), rate.squeeze(-1))
    assert torch.isfinite(nc).all() and nc.abs().sum(-1).max() <= 1+1e-12
    assert (nc.abs()*na[..., None]).sum((-1, -2)).max() <= 1+1e-12
    assert (na[:, 0] > 0).all()
    # Exact actual-read decomposition: allocation motion is not omitted.
    q = torch.randn_like(a)
    difference = torch.einsum('bi,bij->bj', q*na, nc)-torch.einsum('bi,bij->bj', q*a, c)
    expected = (torch.einsum('bi,bij->bj', q*(na-a), c)
                + torch.einsum('bi,bij->bj', q*na*rate*r, y[:, None]-c))
    torch.testing.assert_close(difference, expected)
    held, shares = contact_exchange_content(c, a, k, v, gain, torch.zeros_like(rate), 1.)
    torch.testing.assert_close(held, c, rtol=0, atol=0)
    torch.testing.assert_close(shares, a, rtol=0, atol=0)
    # Force r=a at strictly positive shares. Each row now decays by its LOCAL
    # participation instead of the old global 1-gamma factor.
    a = torch.full_like(a, 1/7)
    address = a.log()-gain*a+.5*(c-y[:, None]).square().sum(-1)
    nc, na = contact_exchange_content(c, a, address, v, gain, rate, 1.)
    torch.testing.assert_close(na, a)
    torch.testing.assert_close(nc[:, 0]-nc[:, 1], (1-rate/7)*(c[:, 0]-c[:, 1]))


@pytest.mark.parametrize('availability', [False, True])
def test_contact_exchange_derivatives_and_body_weighted_read(availability):
    from scripts.ib_local.gated_deltanet import contact_exchange_content, GatedDeltaNetLayer
    torch.manual_seed(95)
    args = [torch.randn(1, 3, 2, dtype=torch.float64)*.1,
            torch.softmax(torch.randn(1, 3, dtype=torch.float64), -1),
            torch.randn(1, 3, dtype=torch.float64),
            torch.randn(1, 2, dtype=torch.float64)*.1,
            torch.tensor([[2.]], dtype=torch.float64), torch.tensor([[.4]], dtype=torch.float64)]
    assert torch.autograd.gradcheck(lambda *x: contact_exchange_content(*x, 1., availability=availability),
                                    tuple(x.requires_grad_() for x in args))
    layer = GatedDeltaNetLayer(8, 2, recurrence='contact_exchange', content_budget=1.,
                              contact_availability=availability).double()
    x = torch.randn(1, 1, 8, dtype=torch.float64)
    c = torch.randn(1, 2, 4, 4, dtype=x.dtype)*.05
    a = torch.softmax(torch.randn(1, 2, 4, dtype=x.dtype), -1)
    packed = torch.cat((c.flatten(1), a.flatten(1)), -1)
    xn = layer.norm(x)
    q = torch.nn.functional.normalize(layer.q_proj(xn).reshape(1, 2, 4), dim=-1)
    k, v = [proj(xn).reshape(1, 2, 4) for proj in (layer.k_proj, layer.v_proj)]
    gain = 8*torch.sigmoid(layer.alpha_proj(xn)).reshape(1, 2, 1)
    rate = torch.sigmoid(layer.beta_proj(xn)).reshape(1, 2, 1)
    nc, na = contact_exchange_content(c, a, k, v, gain, rate, 1., availability=availability)
    read = ((nc*na[..., None]).transpose(-1, -2) @ q[..., None]).squeeze(-1).reshape(1, 1, 8)
    expected = x+layer.o_proj(read*torch.nn.functional.silu(layer.g_proj(xn)))
    out, state, diag = layer(x, packed)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(state, torch.cat((nc.flatten(1), na.flatten(1)), -1))
    torch.testing.assert_close(diag['content_mass'], (nc.abs()*na[..., None]).sum((-1,-2)).mean())


@pytest.mark.parametrize("rule,expected", [("legacy", 0.0), ("decay_then_delta", 0.25)])
def test_decay_order_has_explicit_checkpoint_semantics(rule, expected):
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer

    layer = GatedDeltaNetLayer(2, 1, recurrence=rule).double()
    with torch.no_grad():
        layer.k_proj.weight.copy_(torch.eye(2))
        layer.v_proj.weight.zero_()
        for gate in (layer.alpha_proj, layer.beta_proj):
            gate.weight.zero_()
            gate.bias.zero_()  # alpha=beta=1/2
    state = torch.eye(2, dtype=torch.float64)[None, None]
    _, after, _ = layer(torch.tensor([[[1.0, 0.0]]], dtype=torch.float64), state)
    target = torch.diag(torch.tensor([expected, 0.5], dtype=torch.float64))[None, None]
    torch.testing.assert_close(after, target)
    torch.testing.assert_close(state, torch.eye(2, dtype=torch.float64)[None, None])


@pytest.mark.parametrize("conv_kernel", [0, 1, 4])
@pytest.mark.parametrize("content_budget", [0.0, 0.2])
def test_continuous_logits_match_arbitrary_chunking_and_are_causal(conv_kernel, content_budget):
    torch.manual_seed(83)
    model = recurrent_reference(conv_kernel, content_budget).double()
    ids = torch.randint(0, 256, (2, 9))
    initial = model.initial_state(2).normal_()
    saved = initial.clone()
    with torch.no_grad():
        whole, end, _ = model.predict_logits(ids, initial)
        first, state, _ = model.predict_logits(ids[:, :4], initial)
        second, chunk_end, _ = model.predict_logits(ids[:, 4:], state)
        torch.testing.assert_close(torch.cat((first, second), 1), whole)
        torch.testing.assert_close(chunk_end, end)
        parts, state = [], initial
        for t in range(ids.shape[1]):
            one, state, _ = model.predict_logits(ids[:, t:t+1], state)
            parts.append(one)
        torch.testing.assert_close(torch.cat(parts, 1), whole)
        torch.testing.assert_close(state, end)
        assert state.numel() == initial.numel()
        assert state.untyped_storage().nbytes() == state.numel() * state.element_size()
        changed = ids.clone()
        changed[:, 4:] = (changed[:, 4:] + 17) % 256
        other, _, _ = model.predict_logits(changed, initial)
        torch.testing.assert_close(other[:, :4], whole[:, :4])
        # Target labels are read only by the loss, never the state transition.
        _, end_a, _ = model(ids, ids, initial)
        _, end_b, _ = model(ids, changed, initial)
        torch.testing.assert_close(end_a, end_b)
    torch.testing.assert_close(initial, saved)


@pytest.mark.parametrize("conv_kernel", [0, 4])
@pytest.mark.parametrize("content_budget", [0.0, 0.2])
def test_all_slow_parameters_and_incoming_state_receive_credit(conv_kernel, content_budget):
    torch.manual_seed(84)
    model = recurrent_reference(conv_kernel, content_budget)
    state = model.initial_state(2).normal_().requires_grad_()
    ids, targets = torch.randint(0, 256, (2, 2, 8))
    loss, _, _ = model(ids, targets, state)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert state.grad is not None and state.grad.abs().sum() > 0


@pytest.mark.parametrize("conv_kernel", [0, 4])
@pytest.mark.parametrize("content_budget", [0.0, 0.2])
def test_window_boundary_keeps_values_and_restores_complete_learner(conv_kernel, content_budget):
    import io

    torch.manual_seed(85)
    model = recurrent_reference(conv_kernel, content_budget)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ids, targets = torch.randint(0, 256, (2, 1, 6))
    loss, state, _ = model(ids, targets)
    loss.backward()
    optimizer.step()
    carried = state.detach().clone()
    assert carried.grad_fn is None
    torch.testing.assert_close(carried, state)
    payload = io.BytesIO()
    torch.save(dict(config=model.config_dict(), model=model.state_dict(),
                    state=carried, optimizer=optimizer.state_dict()), payload)
    payload.seek(0)
    checkpoint = torch.load(payload, weights_only=True)
    restored = GatedDeltaNetLM(**checkpoint["config"])
    restored.load_state_dict(checkpoint["model"])
    resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    outcomes = []
    for body, opt, incoming in ((model, optimizer, carried),
                                (restored, resumed_optimizer, checkpoint["state"])):
        opt.zero_grad(set_to_none=True)
        loss, following, _ = body(ids, targets, incoming)
        loss.backward()
        opt.step()
        assert following.numel() == carried.numel()
        outcomes.append((loss.detach(), following.detach()))
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right)
    for left, right in zip(*outcomes):
        torch.testing.assert_close(left, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("conv_kernel", [0, 4])
@pytest.mark.parametrize("content_budget", [0.0, 0.2])
def test_recurrent_reference_captured_learning_step(conv_kernel, content_budget):
    from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer

    torch.manual_seed(86)
    model = recurrent_reference(conv_kernel, content_budget).cuda()
    trainer = TruncatedInternalTimeGraphTrainer(model, tokens=8, chunk_tokens=8)
    ids, targets = torch.randint(0, 256, (2, 1, 8), device="cuda")
    loss, state, _ = trainer.step(ids, targets)
    assert torch.isfinite(loss)
    assert torch.isfinite(state).all()
    assert state.shape == model.state_shape(1)


def test_short_conv_uses_raw_history_in_causal_order():
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLayer

    layer = GatedDeltaNetLayer(2, 1, conv_kernel=4).double()
    with torch.no_grad():
        layer.qkv_conv.weight.copy_(torch.tensor([1., 2., 3., 4.]).expand(6, 1, 4))
    history = torch.tensor([-3., -2., -1.], dtype=torch.float64).expand(1, 6, 3)
    projected = torch.tensor([0., 1.], dtype=torch.float64).expand(1, 6, 2)
    mixed, cache = layer._mix_qkv(projected, history)
    expected = torch.nn.functional.silu(torch.tensor([-10., 0.], dtype=torch.float64))
    torch.testing.assert_close(mixed, expected.expand_as(mixed))
    torch.testing.assert_close(cache, torch.tensor([-1., 0., 1.], dtype=torch.float64).expand_as(cache))


def test_short_conv_chunk_gradient_matches_whole_and_cache_is_used():
    torch.manual_seed(88)
    model = recurrent_reference(4).double()
    ids = torch.randint(0, 256, (1, 7))
    initial = model.initial_state(1).normal_().requires_grad_()
    whole, end, _ = model.predict_logits(ids, initial)
    params = (initial, *model.parameters())
    full_grad = torch.autograd.grad(whole.square().sum() + end.square().sum(), params)
    first, middle, _ = model.predict_logits(ids[:, :2], initial)
    last, chunk_end, _ = model.predict_logits(ids[:, 2:], middle)
    chunk_grad = torch.autograd.grad(
        torch.cat((first, last), 1).square().sum() + chunk_end.square().sum(), params)
    for actual, expected in zip(chunk_grad, full_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    with torch.no_grad():
        erased = middle.detach().clone()
        erased[..., model.layers[0].matrix_size:] = 0
        wrong, _, _ = model.predict_logits(ids[:, 2:], erased)
    assert (last - wrong).abs().max() > 1e-8
    with pytest.raises(ValueError, match="expected recurrent state"):
        model.predict_logits(ids, torch.zeros(2, 1, 2, 16, 16))


def test_parameter_count_matching():
    """Verify that GatedDeltaNetLM matches CBIM Torus3D non-embedding parameters within 0.1%."""
    cbim = CBIMTorus3D(shape=(8, 8, 4), velocities=8, content_dim=16)
    gdn = GatedDeltaNetLM(vocab_size=50257, d=128, layers=4, heads=4)

    cbim_non_emb = sum(
        p.numel() for n, p in cbim.named_parameters()
        if "embedding" not in n and "decoder" not in n
    )
    gdn_non_emb = sum(
        p.numel() for n, p in gdn.named_parameters()
        if "embedding" not in n and "decoder" not in n
    )

    print(f"CBIM Torus3D non-embedding: {cbim_non_emb:,}")
    print(f"GatedDeltaNetLM non-embedding: {gdn_non_emb:,}")
    diff = abs(cbim_non_emb - gdn_non_emb)
    ratio = diff / cbim_non_emb
    print(f"Difference: {diff} ({ratio:.4%})")

    assert cbim_non_emb == 332724
    assert gdn_non_emb == 332448
    assert ratio < 0.001  # Within 0.1%!


def test_forward_backward_cpu():
    """Verify CPU forward and backward execution, loss scale, and state shape."""
    model = GatedDeltaNetLM(vocab_size=1000, d=64, layers=2, heads=2)
    ids = torch.randint(0, 1000, (2, 8))
    targets = torch.randint(0, 1000, (2, 8))

    loss, next_state, diags = model(ids, targets)
    assert loss is not None
    assert math.isfinite(loss.item())
    assert next_state.shape == (2, 2, 2, 32, 32)
    assert "alpha_mean" in diags
    assert "beta_mean" in diags
    assert "state_norm" in diags

    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU test")
def test_cuda_graph_trainer_compatibility():
    """Verify TruncatedInternalTimeGraphTrainer capture and replay on GPU."""
    from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer

    model = GatedDeltaNetLM(vocab_size=50257, d=128, layers=4, heads=4).cuda()
    trainer = TruncatedInternalTimeGraphTrainer(model, tokens=128, chunk_tokens=128)

    ids = torch.randint(0, 50257, (1, 128), device="cuda")
    targets = torch.randint(0, 50257, (1, 128), device="cuda")

    loss, state, diags = trainer.step(ids, targets)
    assert math.isfinite(float(loss))
    assert math.isfinite(float(trainer.grad_norm))
    assert 10.0 < float(loss) < 12.0  # Initial loss around ln(50257) = 10.82


def test_initial_loss_near_entropy():
    """Verify initial loss matches theoretical entropy ln(50257) = 10.82."""
    model = GatedDeltaNetLM(vocab_size=50257, d=128, layers=4, heads=4)
    ids = torch.randint(0, 50257, (1, 32))
    targets = torch.randint(0, 50257, (1, 32))

    with torch.no_grad():
        loss, _, _ = model(ids, targets)
    expected = math.log(50257)
    assert abs(loss.item() - expected) < 0.5


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_available_contact_budget_permutation_and_corner_release(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from scripts.ib_local.gated_deltanet import contact_exchange_content as step
    torch.manual_seed(96)
    c = torch.randn(3, 7, 5, device=device, dtype=torch.float64)
    c = c/c.abs().sum(-1, keepdim=True).clamp_min(1)
    a, k = torch.randn(3, 7, device=device, dtype=c.dtype).softmax(-1), torch.randn(3, 7, device=device, dtype=c.dtype)
    v = torch.randn(3, 5, device=device, dtype=c.dtype)*.03
    gain, rate = a.new_full((3, 1), 5.), a.new_full((3, 1), .3)
    logits = gain*a+k-.5*(c-v[:, None]).square().sum(-1)
    weight = (logits-logits.max(-1, keepdim=True).values).exp()*(1-a)
    p = weight/weight.sum(-1, keepdim=True)
    nc, na = step(c, a, k, v, gain, rate, 1., availability=True)
    torch.testing.assert_close(na, (1-rate)*a+rate*p)
    torch.testing.assert_close(nc, (1-rate[..., None]*p[..., None])*c+rate[..., None]*p[..., None]*v[:, None])
    assert (na >= 0).all() and (nc.abs()*na[..., None]).sum((-1,-2)).max() <= 1+1e-12
    perm = torch.randperm(7, device=device)
    pc, pa = step(c[:, perm], a[:, perm], k[:, perm], v, gain, rate, 1., availability=True)
    torch.testing.assert_close(pc, nc[:, perm]); torch.testing.assert_close(pa, na[:, perm])
    # Full occupation blocks its own admission despite extremely strong score;
    # allocation is transferred, not discarded. Closed contact content holds.
    a = a.new_tensor([1., 0.]).requires_grad_()
    c = c.new_tensor([[.2, -.1], [-.1, .2]]).requires_grad_()
    nc, na = step(c, a, torch.zeros_like(a), c.new_tensor([.1, .1]),
                  a.new_tensor([10000.]), a.new_tensor([.3]), 1., availability=True)
    torch.testing.assert_close(na, a.new_tensor([.7, .3]))
    torch.testing.assert_close(nc[0], c[0], rtol=0, atol=0)
    (nc.sum()+na.square().sum()).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(c.grad).all()


def test_available_contact_joint_latch_survives_and_releases():
    """Conditional local stability of the ACTUAL (C,a) transition, not a task."""
    from scripts.ib_local.gated_deltanet import contact_exchange_content as step
    value = torch.tensor([.2, -.1], dtype=torch.float64)
    gamma, gain = value.new_tensor([.3]), value.new_tensor([4*math.log(3)])
    neutral = torch.zeros(2, dtype=value.dtype)
    def update(z, address=neutral):
        a = torch.stack((z[0], 1-z[0]))
        c, a = step(z[1:].reshape(2, 2), a, address, value, gain, gamma, 1., availability=True)
        return torch.cat((a[:1], c.flatten()))
    for p in (.25, .75):
        fixed = torch.cat((value.new_tensor([p]), value.repeat(2)))
        torch.testing.assert_close(update(fixed), fixed)
        radius = torch.linalg.eigvals(torch.autograd.functional.jacobian(update, fixed)).abs().max()
        assert radius < .95
        z = fixed+value.new_tensor([.001, .0002, -.0001, -.0001, .0002])
        for _ in range(250): z = update(z)
        torch.testing.assert_close(z, fixed, rtol=0, atol=1e-9)
        evidence = value.new_tensor([(1 if p < .5 else -1)*(float(gain)+math.log(3)), 0])
        for _ in range(20): z = update(z, evidence)
        assert (float(z[0])-.5)*(p-.5) < 0
        for _ in range(300): z = update(z)
        expected = fixed.clone(); expected[0] = 1-p
        torch.testing.assert_close(z, expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_contact_exchange_does_not_inherit_delta_read_invariants(device):
    """A scope counterexample, not a capability task or a desired behavior.

    GenerationColumnMemory's exact-prediction/no-write and orthogonal-read
    laws concern tied delta writes. They cannot be marked satisfied merely
    because each conditional contact follows an innovation update.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from scripts.ib_local.gated_deltanet import contact_exchange_content as step
    C = torch.tensor([[.4, 0.], [0., .2]], dtype=torch.float64, device=device)
    a = C.new_tensor([.5, .5])
    M = a[:, None]*C
    q = C.new_tensor([1., 0.])
    gamma, gain = C.new_tensor([.5]), C.new_tensor([0.])
    value = M.T @ q  # Actual read is already exactly correct.
    # Even using q itself as the address does not recover tied delta no-write.
    tied_c, tied_a = step(C, a, q, value, gain, gamma, 1., availability=True)
    assert ((tied_c.T @ (tied_a*q))-value).abs().max() > 1e-3
    # Equalize evidence to isolate content writing from allocation changes.
    address = .5*(C-value).square().sum(-1)
    nc, na = step(C, a, address, value, gain, gamma, 1., availability=True)
    torch.testing.assert_close(na, a)
    new_read = nc.T @ (na*q)
    torch.testing.assert_close(new_read, C.new_tensor([.175, 0.]))
    assert (new_read-value).abs().max() > .02
    delta = M + gamma*torch.outer(q, value-M.T @ q)
    torch.testing.assert_close(delta, M, rtol=0, atol=0)
    # A write key orthogonal to q leaves the delta read unchanged (alpha=1).
    # The current contact update has no corresponding key-overlap identity.
    key, value = C.new_tensor([0., 1.]), C.new_tensor([.1, -.1])
    delta = M + gamma*torch.outer(key, value-M.T @ key)
    torch.testing.assert_close(delta.T @ q, M.T @ q, rtol=0, atol=0)
    nc, na = step(C, a, key, value, gain, gamma, 1., availability=True)
    assert ((nc.T @ (na*q))-(M.T @ q)).abs().max() > 1e-3


def test_actual_content_admission_budgets_fixed_points_and_derivative():
    from scripts.ib_local.gated_deltanet import admit_association_content as admit

    torch.manual_seed(91)
    # Rectangular, signed, batched: no 2x2-only or nonnegative-content assumption.
    u = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    bounded = admit(u, 1.0)
    row, col = bounded.abs().sum(-1), bounded.abs().sum(-2)
    assert row.max() <= 1 + 1e-14 and col.max() <= 1 + 1e-14
    torch.testing.assert_close(admit(bounded, 1.0), bounded)
    # Independently compare every entry with the exact-real scalar formula.
    oracle = torch.empty_like(u)
    for b in range(2):
        for h in range(3):
            for i in range(5):
                for j in range(7):
                    denominator = max(1.0, float(u[b, h, i].detach().abs().sum()),
                                      float(u[b, h, :, j].detach().abs().sum()))
                    oracle[b, h, i, j] = u[b, h, i, j].detach() / denominator
    torch.testing.assert_close(bounded, oracle)
    assert torch.autograd.gradcheck(lambda x: admit(x, 1.0), (u,))
    # Zero signed net flow must not conceal large absolute occupancy.
    cancel = torch.tensor([[10., -10.], [-10., 10.]], dtype=torch.float64)
    torch.testing.assert_close(admit(cancel, 1.0).abs().sum(-1), torch.ones(2, dtype=u.dtype))
    for invalid in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            admit(u, invalid)


def test_content_competition_changes_real_read_and_includes_retention_cost():
    from scripts.ib_local.gated_deltanet import admit_association_content as admit

    # An excessive new association displaces actual old content at a shared
    # endpoint. The orthogonal-read identity of unconstrained delta no longer holds.
    old = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)
    proposal = old.clone()
    proposal[0, 1] = 2.0
    after = admit(proposal, 1.0)
    query = torch.tensor([1., 0.], dtype=old.dtype)
    assert (after.T @ query)[0] < (old.T @ query)[0]
    torch.testing.assert_close(after.T @ query,
                               proposal.T @ query + (after - proposal).T @ query)
    # A filled entry can reverse sign under conflicting content; capacity does
    # not create an absorbing zero update. This is NOT a stable-latch test.
    full = torch.eye(2, dtype=old.dtype)
    reverse = full.clone()
    reverse[0, 0] -= 2.0
    assert admit(reverse, 1.0)[0, 0] == -1


def test_budgeted_body_uses_content_without_extra_state_or_parameters():
    torch.manual_seed(92)
    baseline, bounded = recurrent_reference(4), recurrent_reference(4, 0.02)
    bounded.load_state_dict(baseline.state_dict())
    assert bounded.state_shape(1) == baseline.state_shape(1)
    assert sum(p.numel() for p in bounded.parameters()) == sum(p.numel() for p in baseline.parameters())
    ids = torch.randint(0, 256, (1, 9))
    with torch.no_grad():
        original, _, _ = baseline.predict_logits(ids)
        changed, state, _ = bounded.predict_logits(ids)
        matrices = state[..., :bounded.layers[0].matrix_size].reshape(2, 1, 2, 16, 16)
        assert matrices.abs().sum(-1).max() <= 0.02 + 1e-7
        assert matrices.abs().sum(-2).max() <= 0.02 + 1e-7
        assert (original - changed).abs().max() > 1e-7
    assert bounded.config_dict()['content_budget'] == 0.02


@pytest.mark.parametrize('conv_kernel', [0, 4])
@pytest.mark.parametrize('recurrence,availability', [('evidence_exchange', False),
                         ('contact_exchange', False), ('contact_exchange', True)])
def test_joint_exchange_body_causal_learning_budgets_and_restore(conv_kernel, recurrence, availability):
    torch.manual_seed(93)
    model = GatedDeltaNetLM(vocab_size=256, d=32, layers=2, heads=2,
                           recurrence=recurrence, mlp_hidden=64,
                           conv_kernel=conv_kernel, content_budget=1.0,
                           contact_availability=availability).double()
    ids = torch.randint(0, 256, (2, 9))
    target = torch.randint(0, 256, (2, 9))
    initial = model.initial_state(2)
    logits, following, _ = model.predict_logits(ids, initial)
    first, middle, _ = model.predict_logits(ids[:, :4], initial)
    last, end, _ = model.predict_logits(ids[:, 4:], middle)
    torch.testing.assert_close(torch.cat((first, last), 1), logits)
    torch.testing.assert_close(end, following)
    changed = ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 17) % 256
    other, _, _ = model.predict_logits(changed, initial)
    torch.testing.assert_close(other[:, :4], logits[:, :4])
    _, same_state, _ = model(ids, changed, initial)
    torch.testing.assert_close(same_state, following)
    layer = model.layers[0]
    matrix = following[..., :layer.matrix_size].reshape(2, 2, 2, 16, 16)
    share = following[..., layer.matrix_size:layer.matrix_size+layer.share_size].reshape(2, 2, 2, 16)
    if recurrence == 'contact_exchange':
        assert matrix.abs().sum(-1).max() <= 1 + 1e-12
        matrix = matrix*share.unsqueeze(-1)
    torch.testing.assert_close(share.sum(-1), torch.ones_like(share.sum(-1)))
    assert (share >= 0).all()
    assert (matrix.abs().sum(-1) <= share + 1e-12).all()
    assert matrix.abs().sum(-2).max() <= 1 + 1e-12
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), target.flatten())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    optimizer.step()
    import copy
    restored = GatedDeltaNetLM(**model.config_dict()).double()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored_opt = torch.optim.Adam(restored.parameters(), lr=0.001)
    restored_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    outcomes = []
    for body, opt in ((model, optimizer), (restored, restored_opt)):
        opt.zero_grad(set_to_none=True)
        loss, state, _ = body(ids, target, following.detach().clone())
        loss.backward()
        opt.step()
        outcomes.append((loss.detach(), state.detach()))
    for a, b in zip(outcomes[0], outcomes[1]):
        torch.testing.assert_close(a, b)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    baseline = recurrent_reference(conv_kernel)
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in model.parameters())
    assert following.numel()-baseline.initial_state(2).numel() == 2*2*2*16
    assert following.untyped_storage().nbytes() == following.numel()*following.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_joint_exchange_cuda_graph_learning():
    from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer
    model = GatedDeltaNetLM(vocab_size=256, d=32, layers=2, heads=2,
                           recurrence='evidence_exchange', mlp_hidden=64,
                           conv_kernel=4, content_budget=1.0).cuda()
    trainer = TruncatedInternalTimeGraphTrainer(model, tokens=8, chunk_tokens=8)
    ids, targets = torch.randint(0, 256, (2, 1, 8), device='cuda')
    for _ in range(2):
        loss, state, _ = trainer.step(ids, targets)
        assert torch.isfinite(loss) and torch.isfinite(state).all()


def test_continuous_byte_protocol_scores_jumps():
    import numpy as np
    from scripts.ib_local.train_gated_deltanet import byte_protocol
    raw = np.arange(101)
    stream, phases = byte_protocol(raw, 10, 10)
    assert stream.shape == (101,)
    assert stream[79:82].tolist() == [79, 60, 61]
    assert stream[89:92].tolist() == [69, 80, 81]
    assert [p['stop_update'] for p in phases] == [7, 8, 9, 10]
    assert sum(p['stop_update']-p['start_update'] for p in phases) == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('recurrence,availability', [('decay_then_delta', False),
                         ('feedback_delta', False), ('evidence_exchange', False), ('contact_exchange', False),
                         ('contact_exchange', True)])
def test_graph_accumulation_matches_eager_update(recurrence, availability):
    import copy
    from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer
    torch.manual_seed(11)
    model = GatedDeltaNetLM(vocab_size=256, d=32, layers=2, heads=2,
                           recurrence=recurrence, mlp_hidden=64, conv_kernel=4,
                           contact_availability=availability,
                           content_budget=1. if recurrence in ('evidence_exchange', 'contact_exchange') else 0.).cuda()
    reference = copy.deepcopy(model)
    runner = TruncatedInternalTimeGraphTrainer(model, tokens=16, chunk_tokens=8)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, foreach=True)
    state = reference.initial_state(1)
    for _ in range(2):
        ids, targets = torch.randint(0, 256, (2, 1, 16), device='cuda')
        actual_loss, actual_state, _ = runner.step(ids, targets)
        optimizer.zero_grad(set_to_none=True)
        expected_loss = 0.
        for start in (0, 8):
            loss, state, _ = reference(ids[:, start:start+8], targets[:, start:start+8], state)
            (loss/2).backward()
            expected_loss += loss.detach()/2
            state = state.detach()
        for lexical in (True, False):
            group = [p for n,p in reference.named_parameters()
                     if (n == 'source.embedding.weight') == lexical]
            torch.nn.utils.clip_grad_norm_(group, 1., foreach=True)
        optimizer.step()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_state, state, rtol=2e-5, atol=2e-6)
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_feedback_actual_kernel_potential_and_zero_endpoint(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from scripts.ib_local.gated_deltanet import feedback_delta_step
    torch.manual_seed(731)
    S = torch.randn(2, 6, 4, dtype=torch.float64, device=device, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn(2, 6, 1, dtype=S.dtype, device=device), dim=-2)
    v = torch.randn(2, 4, 1, dtype=S.dtype, device=device, requires_grad=True)
    alpha, beta, eta, gain = .7, .4, .6, 2.
    following, _ = feedback_delta_step(S, k, v, alpha, beta, eta, gain)
    # Per-event potential holds the language-generated cue fixed, while outer
    # language gradients must propagate through BOTH uses of v.
    r = S.transpose(-1, -2) @ k
    drive = gain*r+v
    phi = drive.abs()-torch.log1p(drive.abs())
    energy = ((1-alpha)/(2*beta)*S.square().sum()+alpha/2*r.square().sum()
              -(1-eta)*(v*r).sum()-eta/gain*phi.sum())
    grad, = torch.autograd.grad(energy, S, retain_graph=True)
    torch.testing.assert_close(following, S-beta*grad, atol=1e-12, rtol=1e-12)
    base, _ = feedback_delta_step(S, k, v, alpha, beta, 0., gain)
    expected = alpha*S + beta*k @ (v-(alpha*S).transpose(-1, -2)@k).transpose(-1,-2)
    torch.testing.assert_close(base, expected)
    assert torch.autograd.gradcheck(lambda m, x: feedback_delta_step(m,k,x,alpha,beta,eta,gain)[0], (S,v))


@pytest.mark.parametrize('conv_kernel,budget', [(0,0.), (4,16.)])
def test_feedback_body_causal_full_learning_capacity_and_restore(conv_kernel, budget):
    import copy
    kwargs = dict(vocab_size=256,d=32,layers=2,heads=2,mlp_hidden=64,
                  conv_kernel=conv_kernel,content_budget=budget)
    torch.manual_seed(11)
    baseline = GatedDeltaNetLM(recurrence='decay_then_delta', **kwargs).double()
    torch.manual_seed(11)
    model = GatedDeltaNetLM(recurrence='feedback_delta', **kwargs).double()
    for name, p in baseline.named_parameters():
        torch.testing.assert_close(p, dict(model.named_parameters())[name], rtol=0,atol=0)
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in baseline.parameters()) == 8
    ids, targets = torch.randint(0,256,(2,2,12))
    state = model.initial_state(2)
    assert state.shape == baseline.initial_state(2).shape
    logits, following, _ = model.predict_logits(ids,state)
    first, middle, _ = model.predict_logits(ids[:,:5],state)
    last, end, _ = model.predict_logits(ids[:,5:],middle)
    torch.testing.assert_close(torch.cat((first,last),1),logits)
    torch.testing.assert_close(end,following)
    changed=ids.clone();changed[:,5:] = (changed[:,5:]+7)%256
    other, _, _=model.predict_logits(changed,state)
    torch.testing.assert_close(other[:,:5],logits[:,:5])
    _, no_target_write, _=model(ids, changed,state)
    torch.testing.assert_close(no_target_write,following)
    if budget:
        matrix=following[...,:model.layers[0].matrix_size].reshape(2,2,2,16,16)
        assert matrix.abs().sum(-1).max() <= budget+1e-10
        assert matrix.abs().sum(-2).max() <= budget+1e-10
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,256),targets.flatten())
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    opt.step()
    restored=GatedDeltaNetLM(**model.config_dict()).double()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored_opt=torch.optim.AdamW(restored.parameters(),lr=3e-4)
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    outcomes=[]
    for body,optim in ((model,opt),(restored,restored_opt)):
        optim.zero_grad(set_to_none=True)
        loss,next_state,_=body(ids,targets,following.detach().clone())
        loss.backward();optim.step()
        outcomes.append((loss.detach(),next_state.detach()))
    for a,b in zip(outcomes[0],outcomes[1]):torch.testing.assert_close(a,b)
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b)


def test_feedback_runtime_kernel_keeps_rest_releases_and_accounts_capacity():
    from scripts.ib_local.gated_deltanet import feedback_delta_step, admit_association_content
    key=torch.tensor([[1.],[0.]],dtype=torch.float64)
    fixed=torch.tensor([[5/12], [0.]],dtype=torch.float64)
    value=torch.zeros(1,1,dtype=torch.float64)
    following,_=feedback_delta_step(fixed,key,value,.5,.5,1.,4.)
    torch.testing.assert_close(following,fixed)
    torch.testing.assert_close(admit_association_content(following,1.),fixed)
    released,_=feedback_delta_step(fixed,key,torch.full_like(value,-8.),.5,.5,1.,4.)
    for _ in range(40):released,_=feedback_delta_step(released,key,value,.5,.5,1.,4.)
    torch.testing.assert_close(released,-fixed,atol=1e-10,rtol=0)
    # A tighter capacity changes the certified fixed point; do not claim that
    # a separate capacity proof preserves the original latch automatically.
    assert not torch.allclose(admit_association_content(fixed,.1),fixed)


def test_feedback_pair_orchestrates_same_budget_and_existing_evaluators(tmp_path, monkeypatch):
    import argparse,json
    from scripts.ib_local import train_gated_deltanet as train
    from scripts.ib_local import evaluate_gated_deltanet as evaluate
    calls=[]
    monkeypatch.setattr(evaluate,'aligned_revisit_blocks',lambda config,rows: {'checked':True})
    def execute(command, **kwargs):
        calls.append(command)
        destination=Path(command[command.index('--output')+1])
        if '--continuous-bytes' in command:
            destination.mkdir()
            config=dict(stream_sha256='same',validation_sha256='same',state_bytes=32,
                        steps=5000,tokens=512,chunk_tokens=32)
            (destination/'summary.json').write_text(json.dumps({'config':config}))
            (destination/'metrics.jsonl').write_text('')
        else:destination.write_text('{}')
    from pathlib import Path
    monkeypatch.setattr(train.subprocess,'run',execute)
    args=argparse.Namespace(output=tmp_path/'pair',data=tmp_path,steps=5000,tokens=512,
                           chunk_tokens=32,d=128,layers=4,heads=4,mlp_hidden=256,conv_kernel=4,
                           association_budget=32.,validate_every=500,validation_tokens=32768,
                           seed=11,decay_init='constant',feedback_init='constant',
                           fused_recurrence=False)
    train.run_feedback_pair(args)
    assert len(calls)==6
    assert json.loads((args.output/'progress.json').read_text())['status']=='complete'
    assert all('--association-budget' in c for c in calls[:2])
    assert all(c[c.index('--steps')+1]=='5000' for c in calls[:2])
    context=[c for c in calls if '--prefixes' in c]
    assert len(context)==2
    assert all(c[c.index('--exclude')+1]=='32800' for c in context)
