import pytest
import torch
from torch import nn

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.latent_navigation_v3 import ExperienceMemory, LatentNavigationV3


def test_byte_observations_do_not_write_episodes():
    memory = ExperienceMemory(3, capacity=4, short=1, long=2, topk=1)
    encoder = nn.Linear(3, 2, bias=False)
    memory.commit(torch.ones(2, 3), encoder)
    assert int(memory.clock) == 2
    assert int(memory.episodes) == 0 and not bool(memory.occupied.any())


def test_observed_episode_is_read_and_reencoded_with_gradient():
    memory = ExperienceMemory(3, capacity=4, short=1, long=2, topk=1)
    encoder = nn.Linear(3, 2, bias=False)
    x = torch.tensor([1., 0., 0.])
    action = torch.tensor([0., 1., 0.])
    short = torch.tensor([0., 0., 1.])
    long = torch.tensor([1., 1., 0.])
    memory.commit(torch.stack((x, action)), encoder)
    slot, merged = memory.commit_episode(x, action, short, long, encoder, progress=.2, source_time=0)
    assert slot == 0 and not merged and bool(memory.progress_valid[slot])
    out = memory.read(x[None], encoder)
    torch.testing.assert_close(out.value, encoder(action[None]))
    torch.testing.assert_close(out.long_value, encoder(long[None]))
    out.value.sum().backward()
    torch.testing.assert_close(encoder.weight.grad, action.expand_as(encoder.weight))
    slot2, merged = memory.commit_episode(x, action, short, long, encoder, source_time=0)
    assert merged and slot2 == slot and int(memory.count[slot]) == 2
    assert not bool(memory.progress_valid[slot])


def test_restore_and_invalid_source_time():
    encoder = nn.Linear(3, 2, bias=False)
    memory = ExperienceMemory(3, capacity=4, short=1, long=2, topk=1)
    x = torch.ones(3)
    memory.commit(torch.stack((x, x)), encoder)
    with pytest.raises(ValueError):
        memory.commit_episode(x, x, x, x, encoder, source_time=1)
    memory.commit_episode(x, x, x, x, encoder, progress=.1, source_time=0)
    clone = ExperienceMemory(3, capacity=4, short=1, long=2, topk=1)
    clone.load_state_dict(memory.state_dict())
    assert int(clone.episodes) == 1 and int(clone.clock) == 2
    torch.testing.assert_close(memory.read(x[None], encoder).value, clone.read(x[None], encoder).value)


def test_experience_recall_enters_same_gdn_and_observed_outcome_path():
    torch.manual_seed(19)
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32.,
                           decay_init='log_timescale')
    model = LatentNavigationV3(TableGDN(base, True), 8, 8, 1, 4, 2,
                               max_chunk=4, history_size=4)
    state = model.initial_state(1)
    ids = torch.tensor([[31, 32, 33, 34]])
    logp = torch.full((1, 4, 256), -5.545177)
    _, _, _, h, receipt = model.predict_observed(ids, state, logp)
    h = h.detach()
    model.commit_observed(h, receipt)
    model.memory.commit_episode(h[0], h[1], h[2], h[3], model.encoder,
                                progress=.1, source_time=0)
    plan = model.plan_from_state(h[:1])
    assert float(plan['progress_weight'][0].detach()) > 0
    torch.testing.assert_close(plan['progress'], torch.tensor([.1]))
    with torch.no_grad():
        model.read_gain.fill_(.2)
    logits, _, _, h2, _ = model.predict_observed(ids, state, logp)
    decision = model.plan_from_state(h2[:1])['decision_logits']
    loss = (logits.square().mean() + decision.square().mean() +
            model.observed_outcome_loss(h2[:1], h2[1:2], h2[2:3], h2[3:4]))
    loss.backward()
    assert model.read_gain.grad is not None and torch.isfinite(model.read_gain.grad)
    assert model.encoder.weight.grad is not None and model.encoder.weight.grad.norm() > 0
    assert model.action_decisions.weight.grad is not None and model.action_decisions.weight.grad.norm() > 0
