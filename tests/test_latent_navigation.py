"""Implementation identities and causal interfaces, not capability experiments."""
import copy

import pytest
import torch
from torch import nn

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.latent_navigation import (
    LatentGraphTrainer, LatentNavigationBody, LatentTransitionTable, sigreg,
)


def make_body(device='cpu'):
    torch.manual_seed(12)
    body = TableGDN(GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta',
                                   32, 4, 32., decay_init='log_timescale'), True).to(device)
    return LatentNavigationBody(body, 8, 16, 3, 4, max_chunk=16)


def inputs(body):
    device = next(body.parameters()).device
    ids = torch.tensor([[104, 101, 108, 108, 111, 32, 119, 111]], device=device)
    targets = torch.tensor([[101, 108, 108, 111, 32, 119, 111, 114]], device=device)
    logp = torch.randn(1, 8, 256, device=device).log_softmax(-1)
    return ids, targets, body.initial_state(1, device), logp


def test_zero_gate_recovers_existing_body_and_constructor_preserves_rng():
    model = make_body()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    clone = LatentNavigationBody(model.body, 8, 16, 3, 4)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []):
        assert torch.equal(a, b)
    ids, _, state, logp = inputs(model)
    # Nonempty memory: equality must follow from the zero gate, not empty data.
    model.memory.commit(torch.randn(8, 16))
    a, sa, _ = model.body.predict(ids, state, logp)
    b, sb, _ = model.predict(ids, state, logp)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)


def test_immutable_empty_read_and_state_address_distinction():
    memory = LatentTransitionTable(3, 4, 2, 1)
    encoder = nn.Identity()
    query = torch.eye(3)[:2]
    before = copy.deepcopy(memory.state_dict())
    empty = memory.read(query, encoder)
    assert torch.equal(empty.value, torch.zeros_like(query))
    for name, value in memory.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[name])
    memory.append_pairs(query, torch.tensor([[4., 0, 0], [0, 7., 0]]))
    read = memory.read(query, encoder)
    assert read.slots[0, 0] != read.slots[1, 0]
    assert not torch.equal(read.value[0], read.value[1])
    # Current state, not a byte suffix or hand-labelled direction, is the query.


def test_one_hot_write_preserves_other_slots_and_exact_successor():
    memory = LatentTransitionTable(3, 4, 2, 1)
    memory.append_pairs(torch.eye(3), torch.eye(3)*2)
    before = memory.successor.clone()
    memory.append_pairs(torch.tensor([[0., 0., 1.]]), torch.tensor([[7., 8., 9.]]))
    torch.testing.assert_close(memory.successor[:3], before[:3], atol=0, rtol=0)
    torch.testing.assert_close(memory.successor[3], torch.tensor([7., 8., 9.]))
    # beta=1 one-hot tied write contracts error to zero at the chosen slot.
    query = torch.tensor([.1, .2, .3, .4])
    expected = query[3] * (memory.successor[3] - before[3])
    torch.testing.assert_close(query @ memory.successor - query @ before, expected)


def test_horizon_alignment_across_chunks_and_bounded_fifo():
    memory = LatentTransitionTable(2, 7, 3, 2)
    features = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    memory.commit(features[:2])
    assert not memory.occupied.any()
    src, dst = memory.pairs(features[2:6])
    torch.testing.assert_close(src, features[:3])
    torch.testing.assert_close(dst, features[3:6])
    memory.commit(features[2:6])
    size = memory.nbytes
    memory.commit(features[6:])
    assert memory.nbytes == size and memory.pending_count == 3
    indices = (memory.cursor + torch.arange(7)) % 7
    torch.testing.assert_close(memory.source[indices], features[10:17])
    torch.testing.assert_close(memory.successor[indices], features[13:20])
    torch.testing.assert_close(memory.pending, features[-3:])
    assert all(not x.requires_grad for x in memory.buffers())


def test_reencoding_memory_uses_current_projector_not_stale_latents():
    memory = LatentTransitionTable(3, 4, 2, 1)
    encoder = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(3))
    memory.append_pairs(torch.eye(3), torch.eye(3)*2)
    old = memory.read(torch.eye(3), encoder).value
    with torch.no_grad():
        encoder.weight.mul_(3)
    new = memory.read(torch.eye(3), encoder).value
    torch.testing.assert_close(new, old*3)


def test_read_gradient_is_local_credit_not_weight_sum():
    memory = LatentTransitionTable(2, 4, 1, 2)
    memory.append_pairs(torch.eye(2), torch.tensor([[1., 0], [0, 3.]]))
    q = torch.tensor([[.8, .6]], requires_grad=True)
    read = memory.read(q, nn.Identity())
    derivative = torch.autograd.grad(read.value[0, 0], q)[0]
    eps = 1e-3
    shift = torch.tensor([[eps, 0.]])
    finite = (memory.read(q.detach()+shift, nn.Identity()).value[0, 0] -
              memory.read(q.detach()-shift, nn.Identity()).value[0, 0]) / (2*eps)
    assert derivative[0, 0].item() == pytest.approx(finite.item(), abs=2e-4)
    # Identical values give zero query credit despite positive participation.
    memory.successor[:2].fill_(1.)
    read = memory.read(q, nn.Identity())
    derivative = torch.autograd.grad(read.value.sum(), q)[0]
    assert read.weights.sum() == 1
    torch.testing.assert_close(derivative, torch.zeros_like(derivative), atol=1e-6, rtol=0)


def test_future_suffix_cannot_change_earlier_logits_or_retrieval():
    model = make_body()
    model.memory.commit(torch.randn(8, 16))
    with torch.no_grad():
        model.read_gain.fill_(.3)
    ids, _, state, logp = inputs(model)
    other = ids.clone(); other[:, 4:] = 200
    a, _, _, _, ar = model.predict_observed(ids, state, logp)
    b, _, _, _, br = model.predict_observed(other, state, logp)
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)
    torch.testing.assert_close(ar.value[:4], br.value[:4], atol=0, rtol=0)


def test_observed_target_branch_and_memory_reach_language_gradients():
    model = make_body()
    model.memory.commit(torch.randn(8, 16))
    with torch.no_grad():
        model.read_gain.fill_(.3)
    ids, targets, state, logp = inputs(model)
    loss, ns, features, diag = model.objective(ids, targets, state, logp, torch.randn(8, 12))
    features.retain_grad()
    loss.backward()
    # Final h has no future target; its nonzero gradient in the JEPA target
    # branch can be checked independently of the language and SIGReg terms.
    model.zero_grad(set_to_none=True)
    loss, _, h, _ = model.objective(ids, targets, state, logp, torch.randn(8, 12))
    src, dst = model.memory.pairs(h)
    z_target = model.encoder(dst)
    z_target.retain_grad()
    pred = model.predictor(torch.cat((model.encoder(src), model.memory.read(src, model.encoder).value), -1))
    (pred-z_target).square().mean().backward()
    assert z_target.grad.abs().sum() > 0
    for module in (model.encoder, model.predictor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    a, sa, _ = model.predict(ids, state, logp)
    with torch.no_grad():
        model.memory.successor.mul_(-1)
    b, sb, _ = model.predict(ids, state, logp)
    assert (a-b).abs().max() > 1e-5
    assert (sa-sb).abs().max() > 1e-7


def test_pending_previous_chunk_cannot_read_future_table_for_training():
    model = make_body()
    model.memory.commit(torch.randn(8, 16))
    ids, targets, state, logp = inputs(model)
    directions = torch.randn(8, 12)
    first = model.objective(ids, targets, state, logp, directions)
    # Pending history matters for future pair commits, but must not be used as
    # a source with a table that has already observed its own future.
    with torch.no_grad():
        model.memory.pending.add_(100.)
    second = model.objective(ids, targets, state, logp, directions)
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    torch.testing.assert_close(first[3]['latent'], second[3]['latent'], atol=0, rtol=0)


def test_cuda_graph_matches_eager_over_continuing_writes_and_updates():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from scripts.ib_local import gdn_fused
    gdn_fused.enable()
    torch.set_num_threads(1)
    a, b = make_body('cuda'), make_body('cuda')
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    for model in (a, b):
        model.memory.commit(torch.randn(8, 16, device='cuda', generator=torch.Generator(device='cuda').manual_seed(5)))
        with torch.no_grad():
            model.read_gain.fill_(.1)
    oa = torch.optim.AdamW(a.parameters(), lr=1e-3, weight_decay=0)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-3, weight_decay=0)
    graph = LatentGraphTrainer(a, oa, tokens=8, projections=12, capture=True)
    eager = LatentGraphTrainer(b, ob, tokens=8, projections=12, capture=False)
    ids, targets, state, logp = inputs(a)
    graph.state.copy_(state); eager.state.copy_(state)
    for seed in (3, 4, 5):
        directions = torch.randn(8, 12, device='cuda', generator=torch.Generator(device='cuda').manual_seed(seed))
        la, sa, ha, _ = graph.step(ids, targets, logp, directions)
        lb, sb, hb, _ = eager.step(ids, targets, logp, directions)
        torch.testing.assert_close(la, lb, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(sa, sb, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(ha, hb, atol=2e-6, rtol=2e-5)
        for x, y in zip(a.parameters(), b.parameters()):
            if x.grad is not None or y.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-5)
        graph.finish(); eager.finish()
        oa.step(); ob.step()
        for x, y in zip(a.state_dict().values(), b.state_dict().values()):
            if isinstance(x, torch.Tensor):
                torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)


def test_sigreg_and_centered_branch_rank_boundary():
    torch.manual_seed(14)
    directions = torch.randn(8, 32)
    normal = torch.randn(2048, 8, requires_grad=True)
    collapsed = torch.zeros_like(normal)
    assert sigreg(normal, directions) < sigreg(collapsed, directions)
    sigreg(normal, directions).backward()
    assert normal.grad.abs().sum() > 0
    branches = torch.randn(4, 64, dtype=torch.float64)
    residual = branches - branches.mean(0)
    assert torch.linalg.matrix_rank(residual) <= 3
    common = torch.randn(64, dtype=torch.float64)
    shifted = branches + common
    torch.testing.assert_close(shifted-shifted.mean(0), residual)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_complete_continuation_restore_and_bounded_training(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    torch.set_num_threads(1)
    model = make_body(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    ids, targets, state, logp = inputs(model)
    directions = torch.randn(8, 12, device=device)

    def advance(m, opt, s):
        opt.zero_grad(set_to_none=True)
        loss, ns, h, _ = m.objective(ids, targets, s, logp, directions)
        loss.backward()
        m.commit_observed(h)  # after backward, before changing the feature generator
        opt.step()
        return loss.detach(), ns.detach()

    _, state = advance(model, optimizer, state)
    clone = make_body(device)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    second = torch.optim.AdamW(clone.parameters(), lr=1e-3, weight_decay=0)
    second.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    a, sa = advance(model, optimizer, state)
    b, sb = advance(clone, second, state.clone())
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, clone.state_dict()[key], atol=0, rtol=0)
        else:
            assert value == clone.state_dict()[key]
    with pytest.raises(ValueError):
        model.predict(ids.repeat(1, 3), state, logp.repeat(1, 3, 1))
