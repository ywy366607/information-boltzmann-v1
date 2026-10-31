import math
import copy
from dataclasses import replace

import pandas as pd
import pytest
import torch
from torch import nn

from information_boltzmann.core.fly_amine_modulation import (
    AMINE_PARAMETER_NAMES,
    attach_amine_modulation,
    step_amine_modulation,
)
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState


class TinyAmineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.amine_enabled = True
        self.n_neurons = 4
        self.n_superclasses = 2
        self.register_buffer("amine_pre", torch.tensor([0, 1, 2], dtype=torch.long), persistent=False)
        self.register_buffer("amine_post", torch.tensor([2, 3, 2], dtype=torch.long), persistent=False)
        self.register_buffer("amine_channel", torch.tensor([0, 1, 0], dtype=torch.long), persistent=False)
        self.register_buffer("amine_target_flat", torch.tensor([2, 7, 2], dtype=torch.long), persistent=False)
        self.register_buffer("amine_contact_weight", torch.tensor([1.0, 1.0, 0.5]), persistent=False)
        self.superclass_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        self.amine_log_clearance_oa = nn.Parameter(torch.tensor(1.0))
        self.amine_log_clearance_5ht = nn.Parameter(torch.tensor(1.0))
        self.amine_log_affinity_oa = nn.Parameter(torch.tensor(0.0))
        self.amine_log_affinity_5ht = nn.Parameter(torch.tensor(0.0))
        self.amine_log_off_oa = nn.Parameter(torch.tensor(1.0))
        self.amine_log_off_5ht = nn.Parameter(torch.tensor(1.0))
        self.amine_response_oa = nn.Parameter(torch.tensor([0.2, -0.1]))
        self.amine_response_5ht = nn.Parameter(torch.tensor([-0.1, 0.3]))


def state():
    h = torch.zeros(1, 4)
    ring = (torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h))
    return FlyPhysicalState(h, ring, torch.zeros_like(h), torch.zeros_like(h),
                            torch.zeros_like(h), torch.ones_like(h), torch.zeros_like(h),
                            torch.zeros_like(h))


def test_amine_occupancy_is_bounded_and_uses_anatomical_contacts():
    model = TinyAmineModel()
    thresholds, concentration, receptor = step_amine_modulation(model, state(), torch.ones(1, 4))
    assert concentration[0, 2] > 0
    assert concentration[1].sum() == 0
    assert torch.all((receptor >= 0) & (receptor <= 1))
    assert thresholds[0, 2] != 1.0


def test_amine_response_and_kinetics_have_gradients():
    model = TinyAmineModel()
    thresholds, concentration, receptor = step_amine_modulation(model, state(), torch.ones(1, 4))
    loss = thresholds.square().sum() + concentration.sum() + receptor.sum()
    loss.backward()
    assert model.amine_response_oa.grad is not None
    assert model.amine_log_clearance_oa.grad is not None
    assert torch.isfinite(model.amine_response_oa.grad).all()
    assert set(AMINE_PARAMETER_NAMES) == {name for name, _ in model.named_parameters()}


def test_disabled_amine_is_exact_threshold_identity():
    model = TinyAmineModel()
    model.amine_enabled = False
    before = torch.randn(1, 4)
    after, concentration, receptor = step_amine_modulation(model, state(), before)
    assert torch.equal(before, after)
    assert concentration.numel() == 0
    assert receptor.numel() == 0


def test_amine_state_survives_generic_physical_state_roundtrip():
    model = TinyAmineModel()
    _, concentration, receptor = step_amine_modulation(model, state(), torch.ones(1, 4))
    saved = state()
    saved = FlyPhysicalState(**{**saved.state_dict(),
                                "amine_concentration": concentration,
                                "amine_receptor": receptor})
    restored = FlyPhysicalState(**{k: v.clone() if k != "ring" else tuple(t.clone() for t in v)
                                   for k, v in saved.state_dict().items()})
    assert torch.equal(saved.amine_concentration, restored.amine_concentration)
    assert torch.equal(saved.amine_receptor, restored.amine_receptor)


def test_clearance_has_the_requested_e_folding_time_across_saved_windows():
    model = TinyAmineModel()
    model.amine_log_clearance_oa.data.fill_(math.log(80.0))
    current = replace(state(), ring=tuple(torch.zeros(1, 4) for _ in range(4)),
                      amine_concentration=torch.ones(2, 4), amine_receptor=torch.zeros(2, 4))
    for tick in range(80):
        _, concentration, receptor = step_amine_modulation(model, current, torch.ones(1, 4))
        current = replace(current, amine_concentration=concentration, amine_receptor=receptor)
        if tick == 38:
            current = type(current)(**{k: tuple(x.detach().clone() for x in v) if k == 'ring'
                                     else v.detach().clone() for k, v in current.state_dict().items()})
    torch.testing.assert_close(current.amine_concentration[0], torch.full((4,), math.exp(-1)),
                               rtol=5e-6, atol=1e-7)


def test_receptor_off_time_and_constant_exposure_match_mass_action_solution():
    model = TinyAmineModel()
    model.amine_log_off_oa.data.fill_(math.log(12.0))
    model.amine_log_affinity_oa.data.fill_(math.log(.5))
    current = replace(state(), ring=tuple(torch.zeros(1, 4) for _ in range(4)),
                      amine_concentration=torch.zeros(2, 4), amine_receptor=torch.ones(2, 4))
    for _ in range(12):
        _, concentration, receptor = step_amine_modulation(model, current, torch.ones(1, 4))
        current = replace(current, amine_concentration=concentration, amine_receptor=receptor)
    torch.testing.assert_close(current.amine_receptor[0], torch.full((4,), math.exp(-1)),
                               rtol=5e-6, atol=1e-7)
    # A source at steady exposure 1, K=.5: b_inf=2/3 and effective rate=3/12.
    current = replace(state(), amine_concentration=torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.]]),
                      amine_receptor=torch.zeros(2, 4))
    _, concentration, receptor = step_amine_modulation(model, current, torch.ones(1, 4))
    assert concentration[0, 2] == 1
    torch.testing.assert_close(receptor[0, 2], torch.tensor((2/3)*(1-math.exp(-3/12))))


def test_anatomical_attach_named_optimizer_and_zero_response_equivalence(tmp_path):
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, advance_fly_input_event

    model = make_model(tmp_path)
    annotation = tmp_path/'amines.feather'
    pd.DataFrame({'bodyId': [0, 1], 'consensusNt': ['octopamine', 'serotonin']}).to_feather(annotation)
    before = torch.get_rng_state().clone()
    attach_amine_modulation(model, tmp_path/'graph.npz', annotation)
    assert torch.equal(before, torch.get_rng_state())
    assert model.amine_response_oa.numel() == model.n_superclasses
    learner = FlyBPTTLearner(model, physical(model))
    assert set(AMINE_PARAMETER_NAMES).issubset(learner.adam_names)
    parameter_ids = {id(p) for group in learner.optimizer.param_groups for p in group['params']}
    assert all(id(getattr(model, name)) in parameter_ids for name in AMINE_PARAMETER_NAMES)
    # Zero signed response leaves the real neural transition unchanged while
    # the previously missing chemical state starts integrating actual release.
    current = replace(learner.state, ring=tuple(torch.ones_like(learner.state.h) for _ in range(4)))
    active = advance_fly_input_event(model, current, torch.tensor([0]), settle_ticks=1)
    model.amine_enabled = False
    inactive = advance_fly_input_event(model, current, torch.tensor([0]), settle_ticks=1)
    for name in ('h', 'ge', 'gi'):
        torch.testing.assert_close(getattr(active, name), getattr(inactive, name), rtol=0, atol=0)
    assert active.amine_concentration.sum() > 0


@pytest.mark.parametrize('keyword', ['tau_clearance', 'tau_off', 'affinity'])
def test_invalid_kinetic_initialization_is_rejected_before_attachment(tmp_path, keyword):
    from test_fly_bptt_learning import make_model
    model = make_model(tmp_path)
    with pytest.raises(ValueError, match=keyword):
        attach_amine_modulation(model, tmp_path/'graph.npz', tmp_path/'missing.feather',
                                **{keyword: (0., 1.)})


def test_eager_and_checkpointed_learning_keep_chemical_state_and_kinetic_gradients(tmp_path):
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner

    torch.manual_seed(26)
    model = make_model(tmp_path)
    annotation = tmp_path/'amines.feather'
    pd.DataFrame({'bodyId': [0, 1], 'consensusNt': ['octopamine', 'serotonin']}).to_feather(annotation)
    attach_amine_modulation(model, tmp_path/'graph.npz', annotation, response_init=.2)
    initial = physical(model)
    initial.ring = tuple(torch.ones_like(initial.h) for _ in initial.ring)
    eager = FlyBPTTLearner(model, copy.deepcopy(initial), use_checkpointing=False, settle_ticks=2)
    checkpointed = FlyBPTTLearner(copy.deepcopy(model), copy.deepcopy(initial),
                                 use_checkpointing=True, settle_ticks=2)
    eager.previous_token = checkpointed.previous_token = 0
    for words in ([1, 2, 3, 4], [5, 6, 7, 8]):
        a, _ = eager.observe(words)
        b, _ = checkpointed.observe(words)
        torch.testing.assert_close(torch.tensor(a), torch.tensor(b))
        torch.testing.assert_close(eager.state.amine_concentration, checkpointed.state.amine_concentration)
        torch.testing.assert_close(eager.state.amine_receptor, checkpointed.state.amine_receptor)
        for name in AMINE_PARAMETER_NAMES:
            torch.testing.assert_close(getattr(eager.model, name), getattr(checkpointed.model, name))
    assert eager.state.amine_receptor.sum() > 0
    moments = eager.optimizer.state[eager.model.amine_log_clearance_oa]
    assert moments['exp_avg'].abs().sum() > 0
