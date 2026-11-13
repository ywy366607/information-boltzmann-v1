"""Cache intervention mechanics only; no learned capability assertions."""
import copy
from types import SimpleNamespace

import torch

import scripts.ib_local.audit_pretrained_gdn_memory_location as audit


class CacheModel:
    config = SimpleNamespace(num_hidden_layers=1)

    def __init__(self):
        self.received = []

    def __call__(self, x, past_key_values, use_cache):
        cache = copy.deepcopy(past_key_values)
        self.received.append(copy.deepcopy(cache))
        cache.states[0]['recurrent_state'].add_(float(x.sum()))
        cache.states[0]['conv_state'][0].add_(float(x.shape[1]))
        return SimpleNamespace(logits=x.float()[..., None], past_key_values=cache)


def test_s_only_cut_keeps_conv_and_source_and_uses_matched_split(monkeypatch):
    monkeypatch.setattr(audit, 'clone_cache', copy.deepcopy)
    monkeypatch.setattr(audit, 'cache_nbytes', lambda cache: 8)
    source = SimpleNamespace(states=[dict(recurrent_state=torch.tensor([7.]),
                                         conv_state=(torch.tensor([2.]),))])
    outputs = []
    received = []
    for erase in (False, True):
        model = CacheModel()
        reader = audit.NativeOnly(model, split_at=3, erase=erase)
        reader.cache = source
        _, logits = reader.observe(torch.tensor([[1, 2, 3, 4]]), return_logits=True)
        outputs.append(logits)
        received.append(model.received)
        assert reader.position == 4
        assert reader.erase_audit['erased'] == erase
        assert torch.equal(source.states[0]['recurrent_state'], torch.tensor([7.]))
        assert torch.equal(source.states[0]['conv_state'][0], torch.tensor([2.]))
    assert torch.equal(outputs[0], outputs[1])
    assert len(received[0]) == len(received[1]) == 2
    assert torch.equal(received[0][1].states[0]['recurrent_state'], torch.tensor([13.]))
    assert torch.equal(received[1][1].states[0]['recurrent_state'], torch.tensor([0.]))
    assert torch.equal(received[0][1].states[0]['conv_state'][0],
                       received[1][1].states[0]['conv_state'][0])
