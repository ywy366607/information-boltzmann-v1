import copy
import os

import numpy as np
import pytest
import torch

from scripts.ib_bpe_window import clock_inputs
from scripts.ib_local.block_graph import BlockGraph
from scripts.ib_local.sampling import sample_window
from scripts.ib_local.window import LocalWindow


@pytest.mark.skipif(os.getenv('IB_TEST_CUDA') != '1', reason='requires an exclusive GPU window')
def test_reused_block_graph_preserves_full_window_gradients_and_updates():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    a = LocalWindow(vocab=32, hidden=16, particles=8, steps=1).cuda()
    b = copy.deepcopy(a)
    b.block_engine = BlockGraph(b, block_tokens=2)
    oa, ob = torch.optim.SGD(a.parameters(), lr=.001), torch.optim.SGD(b.parameters(), lr=.001)
    rng = np.random.default_rng(7)
    for iteration in range(2):
        xa = (torch.randn(8, 4, device='cuda') * .1).requires_grad_()
        va = torch.randn_like(xa).requires_grad_()
        xb, vb = xa.detach().clone().requires_grad_(), va.detach().clone().requires_grad_()
        tables, _ = sample_window(rng, 4, 1, 8, 'cuda', padding=True)
        if iteration:
            tables[1].append((torch.tensor([0], device='cuda'), torch.tensor([1], device='cuda'),
                              torch.tensor([[1., 0., 0., 0.]], device='cuda'),
                              torch.tensor([.5], device='cuda'), torch.tensor([False], device='cuda')))
        args = (torch.tensor([1, 2, 3, 4], device='cuda'), torch.tensor([2, 3, 4, 5], device='cuda'),
                clock_inputs(iteration * 4, 4, 1, 'cuda'), torch.randn(16, 8, 4, device='cuda'), tables)
        aa, bb = a(xa, va, *args), b(xb, vb, *args)
        for u, v in zip(aa[1:], bb[1:]):
            torch.testing.assert_close(u, v, atol=2e-5, rtol=2e-4)
        oa.zero_grad(set_to_none=True)
        ob.zero_grad(set_to_none=True)
        aa[0].backward()
        bb[0].backward()
        for p, q in zip((xa, va, *a.parameters()), (xb, vb, *b.parameters())):
            if p.grad is None:
                assert q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
        oa.step()
        ob.step()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=2e-5, rtol=2e-4)
