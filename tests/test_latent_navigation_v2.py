"""Finite-memory and causal-gradient checks; not language capability tests."""
import copy

import pytest
import torch
from torch import nn

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.latent_navigation_v2 import (
    LatentNavigationV2, TransitionMemory, TransitionRead, V2GraphTrainer, weighted_sigreg,
)


def body(device='cpu', mode='merge'):
    torch.manual_seed(8)
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32.,
                          decay_init='log_timescale').to(device)
    return LatentNavigationV2(TableGDN(base, True).to(device), 8, 8, 2, 4, 2,
                              max_chunk=4, threshold=.98, mode=mode, history_size=4)


def test_delayed_pair_alignment_across_irregular_chunks():
    memory = TransitionMemory(2, 8, 2, 4, 2)
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)+1
    memory.commit(x[:3], nn.Identity())
    s, short, long = memory.pairs(x[3:5])
    torch.testing.assert_close(s, x[:1]); torch.testing.assert_close(short, x[2:3])
    torch.testing.assert_close(long, x[4:5])
    memory.commit(x[3:5], nn.Identity())
    s, short, long = memory.pairs(x[5:9])
    torch.testing.assert_close(s, x[1:5]); torch.testing.assert_close(short, x[3:7])
    torch.testing.assert_close(long, x[5:9])
    memory.commit(x[5:9], nn.Identity())
    assert memory.pending_count == 4
    torch.testing.assert_close(memory.pending, x[5:9])
    assert memory.clock == 9


def test_later_sources_and_successors_must_both_match_to_merge():
    memory = TransitionMemory(2, 4, 1, 2, 1, threshold=.99)
    # Identical repeating state: merge should preserve original birth time.
    memory.commit(torch.tensor([[1., 0.], [1., 0.]]), nn.Identity())
    memory.commit(torch.tensor([[1., 0.], [1., 0.]]), nn.Identity())
    assert memory.occupied.sum() == 2
    original_birth = memory.birth.clone()
    memory.commit(torch.tensor([[1., 0.], [1., 0.]]), nn.Identity())
    assert memory.count.max() >= 3
    assert memory.birth[0] == original_birth[0]
    assert memory.clock-memory.birth[0] >= 6
    # A source may recur while its long successor changes: don't merge it.
    memory.commit(torch.tensor([[0., 1.], [0., 1.]]), nn.Identity())
    assert memory.occupied.sum() > 2


def test_read_receipt_preserves_old_values_before_overwrite_and_ages():
    memory = TransitionMemory(2, 2, 1, 2, 1, mode='fifo')
    memory.commit(torch.tensor([[1., 0.], [1., 0.]]), nn.Identity())
    memory.commit(torch.tensor([[1., 0.], [1., 0.]]), nn.Identity())
    read = memory.read(torch.tensor([[1., 0.], [1., 0.]]), nn.Identity())
    expected = (read.weights[..., None]*memory.long_successor[read.slots]).sum(-2).clone()
    age_old, age_recent = memory.ages(read, 2)
    assert torch.all(age_old >= age_recent) and torch.all(age_recent >= 0)
    # This commit replaces every slot; pending view must refer to the old slots.
    memory.commit(torch.tensor([[0., 1.], [0., 1.]]), nn.Identity(), read)
    torch.testing.assert_close(memory.pending_view[-2:], expected)


def test_weighted_history_sigreg_has_only_current_gradient():
    current = torch.randn(7, 5, requires_grad=True)
    previous = torch.randn(4, 5, requires_grad=True)
    loss = weighted_sigreg(current, previous, torch.tensor([True, True, False, False]),
                           torch.randn(5, 9))
    loss.backward()
    assert current.grad.abs().sum() > 0 and previous.grad is None


def test_retained_raw_values_reencode_with_exact_frozen_weight_local_gradient():
    torch.manual_seed(13)
    encoder = nn.Linear(3, 2, bias=False, dtype=torch.float64)
    raw = torch.randn(5, 3, dtype=torch.float64)
    weights = torch.softmax(torch.randn(5, dtype=torch.float64), 0).detach()
    direction = torch.randn(2, dtype=torch.float64)
    value = (weights[:,None]*encoder(raw)).sum(0)
    gradient = torch.autograd.grad((value*direction).sum(), encoder.weight)[0]
    expected = direction[:,None] * (weights[:,None]*raw).sum(0)[None,:]
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
    # This isolates the conditional value path; it does not claim the old
    # feature generator, old parameters or attention weights are unchanged.


def test_v2_zero_gate_and_read_affects_next_state():
    model = body()
    ids = torch.randint(256, (1, 4)); logp = torch.randn(1, 4, 256).log_softmax(-1)
    state = model.initial_state(1)
    m = model.memory
    m.commit(torch.randn(4, 16), model.encoder)
    m.commit(torch.randn(4, 16), model.encoder)
    logits, s, _ = model.predict(ids, state, logp)
    reference, rs, _ = model.body.predict(ids, state, logp)
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
    torch.testing.assert_close(s, rs, rtol=0, atol=0)
    with torch.no_grad(): model.read_gain.fill_(.5)
    logits, s, _ = model.predict(ids, state, logp)
    assert (logits-reference).abs().max() > 1e-6
    assert (s-rs).abs().max() > 1e-6


def test_long_loss_uses_source_time_view_and_ignores_unavailable_sources():
    model = body()
    ids = torch.randint(256, (1, 4)); targets = torch.randint(256, (1, 4))
    logp = torch.randn(1, 4, 256).log_softmax(-1); state = model.initial_state(1)
    dirs = torch.randn(8, 6)
    first = model.objective(ids, targets, state, logp, dirs)
    assert first[3]['latent_long'] == 0
    model.commit_observed(first[2])
    second = model.objective(ids, targets, state, logp, dirs)
    assert second[3]['latent_long'] > 0
    second[0].backward()
    assert model.long_predictor[0].weight.grad.abs().sum() > 0
    saved = copy.deepcopy(model.state_dict())
    clone = body(); clone.load_state_dict(saved)
    again = clone.objective(ids, targets, state, logp, dirs)
    torch.testing.assert_close(second[0], again[0], rtol=0, atol=0)


def test_cuda_graph_v2_matches_eager_with_changing_history_and_merges():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from scripts.ib_local import gdn_fused
    gdn_fused.enable(); torch.set_num_threads(1)
    a, b = body('cuda'), body('cuda')
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    oa = torch.optim.AdamW(a.parameters(), lr=1e-3, weight_decay=0)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-3, weight_decay=0)
    graph = V2GraphTrainer(a, oa, tokens=4, projections=6, capture=True)
    eager = V2GraphTrainer(b, ob, tokens=4, projections=6, capture=False)
    ids = torch.tensor([[1, 2, 3, 4]], device='cuda'); targets = ids.roll(-1, 1)
    probs = torch.full((1, 4, 256), -5.54517744448, device='cuda')
    directions = torch.randn(8, 6, device='cuda')
    for _ in range(4):
        la, sa, ha, _ = graph.step(ids, targets, probs, directions)
        lb, sb, hb, _ = eager.step(ids, targets, probs, directions)
        torch.testing.assert_close(la, lb, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(sa, sb, atol=2e-5, rtol=2e-5)
        graph.finish(); eager.finish(); oa.step(); ob.step()
        for x, y in zip(a.state_dict().values(), b.state_dict().values()):
            if isinstance(x, torch.Tensor): torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)
