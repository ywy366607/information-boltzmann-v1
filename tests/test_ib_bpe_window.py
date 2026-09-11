import math

import pytest
import torch

from scripts.ib_bpe_window import BPEWindow, clock_inputs


def test_clock_preserves_substeps_after_million_events():
    start = 1_280_000
    actual = clock_inputs(start, 2, 4, 'cpu')
    for t in range(2):
        for s in range(4):
            for half, fraction in enumerate((.25, .75)):
                phase = start + t + (s + fraction) / 4
                expected = torch.tensor([math.sin(phase), math.cos(phase)])
                torch.testing.assert_close(actual[t, s, half], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Fused OU requires CUDA')
def test_window_equals_causal_single_event_execution_and_gradients():
    torch.manual_seed(21)
    torch.set_num_threads(1)
    model = BPEWindow(vocab=24, hidden=16, particles=8, channels=4, steps=1).cuda()
    x, v = torch.randn(8, 4, device='cuda'), torch.randn(8, 4, device='cuda')
    ids = torch.tensor([1, 2, 3], device='cuda')
    targets = torch.tensor([2, 3, 4], device='cuda')
    clocks = clock_inputs(17, 3, 1, 'cuda')
    noise = torch.randn(12, 8, 4, device='cuda')
    loss, end_x, end_v = model(x, v, ids, targets, clocks, noise)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    seq_x, seq_v = x, v
    losses = []
    for t in range(3):
        item, seq_x, seq_v = model(seq_x, seq_v, ids[t:t+1], targets[t:t+1],
                                  clocks[t:t+1], noise[t*4:(t+1)*4])
        losses.append(item)
    seq_loss = torch.stack(losses).mean()
    seq_grads = torch.autograd.grad(seq_loss, params, allow_unused=True)
    torch.testing.assert_close(loss, seq_loss, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(end_x, seq_x, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(end_v, seq_v, atol=2e-6, rtol=2e-5)
    for a, b in zip(grads, seq_grads):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-4)
