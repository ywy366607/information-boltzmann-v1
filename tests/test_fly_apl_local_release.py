"""Numerical/interface checks, not language or biological capability studies."""
import copy
from dataclasses import replace

import pytest
import torch

from test_fly_apl_cable import setup
from information_boltzmann.core.fly_apl_cable import advance_apl_cable
from information_boltzmann.core.fly_apl_local_release import (
    APL_LOCAL_RELEASE_PARAMETER_NAMES, APL_LOCAL_RELEASE_STATE_FIELDS,
    attach_apl_local_release, advance_apl_local_release, local_release_state,
)
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def test_resume_inherits_release_and_rejects_silent_drop_or_unrecorded_activation():
    from test_fly_learning_repairs import native_credit_args
    from scripts.ib.train_fly_bptt_stream import resolve_learning_modes
    from information_boltzmann.core.fly_apl_cable import APL_CABLE_VERSION
    previous = {'apl_cable': True, 'apl_cable_version': APL_CABLE_VERSION,
                'apl_local_release': True, 'apl_local_release_version': 1}
    args = native_credit_args()
    resolve_learning_modes(args, previous, {})
    assert args.apl_local_release
    with pytest.raises(ValueError, match='matching enabled history'):
        resolve_learning_modes(native_credit_args(apl_local_release=False), previous, {})
    with pytest.raises(ValueError, match='explicit full-state'):
        resolve_learning_modes(native_credit_args(apl_local_release=True),
                               {'apl_cable': True, 'apl_cable_version': APL_CABLE_VERSION}, {})


def test_extension_rekeys_optimizer_by_name_without_replacing_existing_experience():
    from scripts.ib.fly_interface.migrate_native_apl_local_release import extend_payload
    from scripts.ib.fly_interface.migrate_native_kc_dopamine import named
    from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
    source = {'config': {'apl_cable': True}, 'model': {'apl_log_axial': torch.zeros(2)},
              'learner': {'physical': {'apl_voltage': torch.ones(1, 7), 'apl_packets': torch.rand(4, 7)},
                          'adam_names': ['first', 'apl_log_axial', 'last'],
                          'optimizer': {'state': {8: {'step': torch.tensor(9.), 'exp_avg': torch.tensor([.7])},
                                                  9: {'step': torch.tensor(4.), 'exp_avg': torch.ones(2)}},
                                        'param_groups': [{'params': [8, 9, 10],
                                                          'parameter_names': ['first', 'apl_log_axial', 'last']}]},
                          'interaction': {'pending': [3, 7]}},
              'mother_world': {'queue': [11, 12]}, 'rng_cpu': torch.get_rng_state()}
    before = state_digest(source)
    extended = extend_payload(source)
    assert state_digest(source) == before
    old_moments, new_moments = named(source['learner']['optimizer']), named(extended['learner']['optimizer'])
    for name, value in old_moments.items():
        assert state_digest(value) == state_digest(new_moments[name])
    names = extended['learner']['optimizer']['param_groups'][0]['parameter_names']
    assert names == ['first', 'apl_log_axial', *APL_LOCAL_RELEASE_PARAMETER_NAMES, 'last']
    assert all(new_moments[name] is None for name in APL_LOCAL_RELEASE_PARAMETER_NAMES)
    assert state_digest(source['mother_world']) == state_digest(extended['mother_world'])
    assert state_digest(source['learner']['interaction']) == state_digest(extended['learner']['interaction'])
    with pytest.raises(ValueError, match='Require native APL'):
        extend_payload(extended)


@pytest.mark.parametrize('device', DEVICES)
def test_local_chemistry_has_no_global_spike_broadcast_and_conserves_resources(tmp_path, device):
    model, state, _ = setup(tmp_path)
    attach_apl_local_release(model)
    model.to(device)
    state = FlyPhysicalState(**{k: tuple(p.to(device) for p in v) if k == 'ring'
                              else v.to(device) for k, v in state.state_dict().items()})
    # A node with no local voltage drive receives no new calcium merely because
    # the cell-wide compatibility spike packet is arbitrarily large.
    state = replace(state, ring=tuple(torch.full_like(p, 900.) for p in state.ring))
    drive = state.h.new_tensor([[2., 0., 0.]])
    for _ in range(12):
        old = local_release_state(model, state)
        emitted, gate, calcium, resources = advance_apl_local_release(model, state, drive)
        assert torch.count_nonzero(calcium[:, 1:]) == 0
        assert torch.count_nonzero(emitted[:, 1:]) == 0
        assert bool(((resources >= 0) & (resources <= 1)).all())
        recovery = model.apl_release_logit_resource_recovery.sigmoid()[model.aplc_owner][None]
        # Available = emitted + retained before the independent replenishment.
        retained = old[2] - emitted
        replenished = (1 - retained) * (1 - recovery)
        torch.testing.assert_close(resources, retained + replenished)
        state = replace(state, **dict(zip(APL_LOCAL_RELEASE_STATE_FIELDS, (gate, calcium, resources))))
    assert float(calcium[0, 0]) > 0
    assert float(resources[0, 0]) < 1
    before = calcium.clone()
    model.apl_calcium_entry_enabled = False
    _, _, calcium, _ = advance_apl_local_release(model, state, drive)
    torch.testing.assert_close(calcium, before * model.apl_release_logit_calcium_decay.sigmoid()[model.aplc_owner][None])


def test_cable_consumes_new_chemistry_and_preserves_old_delayed_packets(tmp_path):
    model, state, _ = setup(tmp_path)
    baseline = advance_apl_cable(model, state, torch.zeros_like(state.h))
    attach_apl_local_release(model)
    step = advance_apl_cable(model, state, torch.zeros_like(state.h))
    assert len(step) == 8
    # The new mechanism affects future emitted packets, not already emitted
    # packets or a pre-voltage synthetic cell mean.
    for a, b in zip(baseline[:4], step[:4]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(step[4][1:], state.apl_packets[:3], rtol=0, atol=0)
    assert not torch.equal(step[4][0], baseline[4][0])
    assert float(step[6].max()) > 0
    bad = replace(state, apl_release_calcium=torch.zeros(1, 2), apl_release_gate=torch.zeros(1, 3),
                  apl_release_resources=torch.ones(1, 3))
    with pytest.raises(ValueError, match='Malformed'):
        local_release_state(model, bad)
    with pytest.raises(ValueError, match='Incomplete'):
        local_release_state(model, replace(state, apl_release_calcium=torch.zeros(1, 3)))


@pytest.mark.parametrize('device', DEVICES)
def test_real_learner_checkpointing_and_continuation_keep_local_release_learning(tmp_path, device):
    torch.manual_seed(12)
    model, state, _ = setup(tmp_path)
    attach_apl_local_release(model)
    model.to(device)
    state = FlyPhysicalState(**{k: tuple(p.to(device) for p in v) if k == 'ring'
                              else v.to(device) for k, v in state.state_dict().items()})
    a = FlyBPTTLearner(model, copy.deepcopy(state), use_checkpointing=False, settle_ticks=3)
    b = FlyBPTTLearner(copy.deepcopy(model), copy.deepcopy(state), use_checkpointing=True, settle_ticks=3)
    a.previous_token = b.previous_token = 0
    for targets in ([1, 2, 3], [4, 5, 6]):
        p, _ = a.observe(targets)
        q, _ = b.observe(targets)
        torch.testing.assert_close(torch.tensor(p), torch.tensor(q))
        for name in APL_LOCAL_RELEASE_STATE_FIELDS:
            torch.testing.assert_close(getattr(a.state, name), getattr(b.state, name))
        for name in APL_LOCAL_RELEASE_PARAMETER_NAMES:
            torch.testing.assert_close(getattr(a.model, name), getattr(b.model, name))
            assert name in native_parameter_names(a.model)
            assert 'exp_avg' in a.optimizer.state[getattr(a.model, name)]
    assert any(a.optimizer.state[getattr(a.model, n)]['exp_avg'].abs().sum() > 0
               for n in APL_LOCAL_RELEASE_PARAMETER_NAMES)
    payload = checkpoint_model_weights(a.model)
    model2, _, _ = setup(tmp_path)
    attach_apl_local_release(model2, saved=payload)
    for name in APL_LOCAL_RELEASE_PARAMETER_NAMES:
        torch.testing.assert_close(getattr(model2, name), payload[name].cpu())
    restored = FlyPhysicalState(**a.state.state_dict()).detached()
    for name in APL_LOCAL_RELEASE_STATE_FIELDS:
        assert not getattr(restored, name).requires_grad
        torch.testing.assert_close(getattr(restored, name), getattr(a.state, name))
