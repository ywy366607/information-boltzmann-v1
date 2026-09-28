import pytest
import torch
from torch import nn

from information_boltzmann.evaluation import WarmSiteSpec
from scripts.ib.evaluate_continuous_owt import _gdn2_step, evaluate_warm_sites
from scripts.ib_local.gated_deltanet_2 import GatedDeltaNet2LM


def test_warm_site_spec_requires_observed_local_history():
    with pytest.raises(ValueError, match="positive warm-in"):
        WarmSiteSpec(warm_in_tokens=0)


def test_warm_site_spec_requires_unique_local_contexts():
    with pytest.raises(ValueError, match="unique"):
        WarmSiteSpec(site_starts=(8, 8))


def test_warm_site_spec_counts_its_required_context_and_target():
    spec = WarmSiteSpec(site_starts=(8, 32), warm_in_tokens=256, score_tokens=128)
    assert spec.required_tokens_per_site == 385


def test_gdn2_one_event_adapter_matches_native_forward():
    torch.manual_seed(7)
    model = GatedDeltaNet2LM(vocab_size=17, d=8, layers=2, heads=2).eval()
    state = model.initial_state(1)
    current = torch.tensor([3])
    target = torch.tensor([4])

    native_loss, native_state, _ = model(
        current[:, None], target[:, None], state.clone()
    )
    logits, stepped_state, _ = _gdn2_step(
        model, state.clone(), current, micro_steps=1
    )

    stepped_loss = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(stepped_loss, native_loss)
    torch.testing.assert_close(stepped_state, native_state)


def test_evaluator_clones_only_mature_state_then_carries_each_site():
    class CounterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.incoming_states = []

        def step(self, state, current, *, micro_steps):
            assert micro_steps == 1
            self.incoming_states.append(float(state.item()))
            return torch.zeros(1, 3), state + 1, {"energy": state.mean()}

    model = CounterModel()
    terminal = torch.full((1, 1, 1, 1, 1), 5.0)
    spec = WarmSiteSpec(site_starts=(0, 4), warm_in_tokens=1, score_tokens=2)
    report = evaluate_warm_sites(
        model, terminal, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]).numpy(), spec, 1
    )

    assert model.incoming_states == [5.0, 6.0, 7.0, 5.0, 6.0, 7.0]
    assert float(terminal.item()) == 5.0
    assert report["cold_start"] is False
    assert report["resets_within_site"] == 0
    assert report["checkpoint_terminal_state_clones"] == 2
