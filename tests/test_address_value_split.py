"""Value learning must not move an address sealed at write."""
import torch
from torch import nn

from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, LocalExperience


def populated():
    torch.manual_seed(7)
    memory = LayerExperienceRecall(4, capacity=2, latent_dim=3, topk=1)
    layer = memory.memory
    layer.clock.fill_(128)
    source = torch.tensor([1., 0., 0., 0.])
    slot, _ = layer.commit_episode(source, source, source, source, memory.encoder, source_time=0)
    memory.capsules[slot] = LocalExperience(torch.randn(1, 65, 4), None, int(layer.read_credit_version[slot]))
    return memory, slot


def test_a_later_encoder_update_leaves_the_sealed_address():
    memory, slot = populated()
    memory.remember_address(slot, memory.memory.source[slot])
    sealed = memory.addresses[slot].clone()
    with torch.no_grad():
        memory.encoder.weight.add_(1)
    memory.retrieve(torch.randn(1, 4), lambda capsule: capsule.inputs)
    assert torch.equal(memory.addresses[slot], sealed)


def test_value_loss_trains_the_value_map_and_not_the_sealed_address():
    memory, slot = populated()
    memory.remember_address(slot, memory.memory.source[slot])
    native = nn.Linear(4, 4)
    query = torch.randn(1, 4)
    value = memory.retrieve(query, lambda capsule: native(capsule.inputs))
    value.square().sum().backward()
    assert memory.value_map.weight.grad.norm() > 0
    assert native.weight.grad.norm() > 0
    assert memory.addresses[slot].grad_fn is None
    assert memory.addresses[slot].grad is None
