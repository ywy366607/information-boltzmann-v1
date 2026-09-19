import pytest
import torch
from torch.nn import functional as F
from scripts.ib_local.window import LocalWindow
from scripts.ib_bpe_window import clock_inputs


@pytest.mark.parametrize('mode', ['message_coupling', 'adaptive_force'])
def test_eager_forward_matches_coupling_block(mode):
    torch.manual_seed(41)
    model = LocalWindow(vocab=32, hidden=16, particles=8, coupling_mode=mode)
    model.recompute = False
    x, v = torch.randn(8, 4) * .1, torch.randn(8, 4) * .1
    ids, targets = torch.tensor([1, 2]), torch.tensor([2, 3])
    clocks = clock_inputs(0, 2, 4, 'cpu')
    noise = torch.randn(32, 8, 4)
    tables = [[] for _ in range(8)]
    module = getattr(model, mode)
    calls = []
    hook = module.register_forward_hook(lambda *args: calls.append(1))
    out = model(x, v, ids, targets, clocks, noise, tables)
    assert len(calls) == 2
    hook.remove()
    emb = model.core.force.embedding(ids)
    layer = model.core.force.net[0]
    shared = F.linear(emb, layer.weight[:, 4:-2], layer.bias)
    cp = F.linear(clocks, layer.weight[:, -2:])
    block = model.feature_block(x, v, shared, cp, noise, tables, tok_embs=emb)
    torch.testing.assert_close(out[2], block[0])
    torch.testing.assert_close(out[3], block[1])
    logits = F.linear(block[2], model.core.decoder.weight, model.core.decoder.bias)
    torch.testing.assert_close(out[1], F.cross_entropy(logits, targets))
