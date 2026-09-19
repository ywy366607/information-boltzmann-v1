import copy
import pytest
import torch
from scripts.ib_local.cbim_field import CBIMFieldModel
from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_graph_matches_two_eager_updates_and_preserves_initialization():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    eager = CBIMFieldModel(vocab_size=32, L=8, d=16).cuda()
    captured = copy.deepcopy(eager)
    runner = CBIMGraphTrainer(captured, tokens=4)
    for a, b in zip(eager.parameters(), captured.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    opt = torch.optim.AdamW(eager.parameters(), lr=3e-4, foreach=True, capturable=True)
    h = torch.zeros(1, 8, 16, device='cuda')
    for _ in range(2):
        ids = torch.randint(32, (1, 4), device='cuda')
        targets = torch.randint(32, (1, 4), device='cuda')
        opt.zero_grad(set_to_none=True)
        loss, nh, _ = eager(ids, targets, h)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eager.parameters(), 1., foreach=True)
        opt.step()
        h = nh.detach()
        other, state, _ = runner.step(ids, targets)
        torch.testing.assert_close(loss, other, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(h, state, atol=2e-6, rtol=2e-5)
        for a, b in zip(eager.parameters(), captured.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
