"""Local receptor routing, dual effectors and real learner interfaces only."""
import copy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from information_boltzmann.core.fly_kc_lateral import (
    attach_kc_lateral, advance_kc_lateral, effector_gate,
    KC_LATERAL_STATE_FIELDS, KC_LATERAL_PARAMETER_NAMES,
)
from information_boltzmann.core.fly_kc_terminals import CellwiseGeometry


def receptor_setup(tmp_path):
    model = nn.Module()
    for name, value in {'kct_index':[0,1,2], 'kct_owner':[0,1,1,2],
                        'kct_site_owner':[0,1,2], 'kct_site_node':[0,2,3],
                        'edge_pre_e':[0,2], 'edge_post_e':[1,1],
                        'kc_terminal_digest':list(bytes(range(32)))}.items():
        model.register_buffer(name, torch.tensor(value, dtype=torch.uint8 if name.endswith('digest') else torch.long))
    model.kc_contact_memory_enabled = True
    model.kct_geometries = (CellwiseGeometry(np.ones(4), np.array([2]), np.array([1]),
                                            np.ones(1), np.array([0,1,1,2])),)
    path = tmp_path/'lateral.npz'
    np.savez(path, kc_index=np.array([0,1,2]), edge=np.array([0,1]), site=np.array([0,2]),
             node=np.array([1,2]), delay=np.array([2,1]), count=np.array([2,1]))
    path.with_suffix('.json').write_text(json.dumps({'kc_geometry_sha256':bytes(range(32)).hex()}))
    attach_kc_lateral(model, path)
    state = SimpleNamespace(h=torch.zeros(1,3), kc_lateral_receptor=torch.empty(0), kc_lateral_signal=torch.empty(0))
    return model, state, path


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_real_delay_local_transport_dual_inhibition_and_all_material_gradients(tmp_path, device):
    model, state, _ = receptor_setup(tmp_path)
    model.to(device)
    state = SimpleNamespace(**{k:v.to(device) for k,v in vars(state).items()})
    packet = torch.zeros(4,3,device=device)
    packet[1,0] = 1  # only the original two-tick delayed source slot
    receptor, signal = advance_kc_lateral(model, state, packet)
    assert receptor[0,1] > 0 and receptor[0,2] == 0
    assert signal[0,1] > signal[0,2] > 0
    assert signal[0,0] == signal[0,3] == 0
    sites = torch.arange(3,device=device)
    calcium, camp = [effector_gate(model, signal, sites, k) for k in ('calcium', 'camp')]
    assert calcium[0,1] < 1 and camp[0,1] < 1
    assert calcium[0,0] == calcium[0,2] == camp[0,2] == 1
    gradients = torch.autograd.grad((calcium*camp).sum(), tuple(model.parameters()))
    assert len(gradients) == 6
    assert all(bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0) for g in gradients)
    model.kc_lateral_delivery_enabled = False
    r, s = advance_kc_lateral(model, state, packet)
    assert torch.count_nonzero(r) == torch.count_nonzero(s) == 0
    model.kc_lateral_calcium_enabled = False
    torch.testing.assert_close(effector_gate(model, signal, sites, 'calcium'), torch.ones(1,3,device=device))
    model.kc_lateral_camp_enabled = False
    torch.testing.assert_close(effector_gate(model, signal, sites, 'camp'), torch.ones(1,3,device=device))


def test_contact_identity_and_incomplete_history_are_rejected(tmp_path):
    model, state, path = receptor_setup(tmp_path)
    from information_boltzmann.core.fly_kc_lateral import lateral_state
    state.kc_lateral_signal = torch.ones(1,4)
    with pytest.raises(ValueError, match='Incomplete'):
        lateral_state(model, state)
    model2, _, path = receptor_setup(tmp_path)
    payload = {**dict(model2.named_parameters()), 'kc_lateral_digest': model2.kc_lateral_digest ^ 1}
    with pytest.raises(ValueError, match='digest mismatch'):
        attach_kc_lateral(model2, path, saved=payload)


def test_archive_extension_preserves_named_optimizer_pending_credit_world_and_rng(tmp_path):
    from scripts.ib.fly_interface.migrate_native_kc_lateral import extend
    from scripts.ib.fly_interface.migrate_native_kc_dopamine import named
    from scripts.ib.fly_interface.verify_kc_lateral_archive import digest
    source = {'config': {'kc_contact_memory': True}, 'model': {'old':torch.ones(2)},
        'learner': {'physical': {'kc_terminal_gaba':torch.rand(4,7)},
            'adam_names':['old','kc_dopamine_log_transport','last'],
            'optimizer': {'state': {8:{'step':torch.tensor(9.),'exp_avg':torch.tensor([.7])},
                                    9:{'step':torch.tensor(4.),'exp_avg':torch.ones(2)}},
                'param_groups':[{'params':[8,9,10],
                    'parameter_names':['old','kc_dopamine_log_transport','last']}]},
            'interaction': {'pending':[3,7]}},
        'mother_world':{'queue':[11,12]},'rng_cpu':torch.get_rng_state()}
    geometry = tmp_path/'lateral.npz'
    np.savez(geometry,kc_index=np.arange(2))
    before = digest(source)
    payload = extend(source,geometry)
    assert digest(source) == before
    assert digest(payload['learner']['interaction']) == before['learner']['interaction']
    assert digest(payload['mother_world']) == before['mother_world']
    assert digest(payload['rng_cpu']) == before['rng_cpu']
    old,new = named(source['learner']['optimizer']), named(payload['learner']['optimizer'])
    for name,value in old.items():
        assert digest(new[name]) == digest(value)
    assert all(new[name] is None for name in KC_LATERAL_PARAMETER_NAMES)
    for name in KC_LATERAL_STATE_FIELDS:
        assert payload['learner']['physical'][name].shape == (1,7)
        assert torch.count_nonzero(payload['learner']['physical'][name]) == 0
    with pytest.raises(ValueError,match='reference-axis'):
        extend({**source,'config':{**source['config'],'mother_native_generative':True}},geometry)


def test_resume_requires_lateral_history_and_keeps_actual_contact_geometry():
    from test_fly_learning_repairs import native_credit_args
    from scripts.ib.train_fly_bptt_stream import resolve_learning_modes
    previous={'kc_contact_memory':True,'kc_contact_version':1,'kc_lateral':True,
              'kc_lateral_version':1,'kc_lateral_geometry':'actual.npz'}
    args=native_credit_args()
    resolve_learning_modes(args,previous,{})
    assert args.kc_lateral and str(args.kc_lateral_geometry) == 'actual.npz'
    with pytest.raises(ValueError,match='matching enabled history'):
        resolve_learning_modes(native_credit_args(kc_lateral=False),previous,{})
    with pytest.raises(ValueError,match='full-state KC lateral migration'):
        resolve_learning_modes(native_credit_args(kc_lateral=True),
                               {'kc_contact_memory':True,'kc_contact_version':1},{})


def contact_setup(tmp_path):
    from test_fly_kc_contact_memory import local_setup
    model, state = local_setup(tmp_path)
    path = tmp_path/'empty_lateral.npz'
    np.savez(path, kc_index=model.kct_index.numpy(), **{
        k:np.empty(0,dtype=np.int64) for k in ('edge','site','node','delay','count')})
    path.with_suffix('.json').write_text(json.dumps({'kc_geometry_sha256':bytes(model.kc_terminal_digest.tolist()).hex()}))
    attach_kc_lateral(model, path)
    # A represented opening receptor state is a numerical continuation fixture,
    # not new sensory experience or evidence of natural biological activation.
    state = replace(state, kc_lateral_receptor=torch.full((1,3), .7), kc_lateral_signal=torch.full((1,3), .5))
    return model, state


def test_camp_effector_changes_existing_contact_learning_without_rescaling_dopamine(tmp_path):
    from information_boltzmann.core.fly_kc_contact_memory import advance_contact_learning
    from information_boltzmann.core.fly_kc_terminals import terminal_state
    from information_boltzmann.core.fly_bptt_learning import advance_compartment_learning
    model, state = contact_setup(tmp_path)
    values = terminal_state(model, state) + [state.kc_lateral_receptor, state.kc_lateral_signal]
    values[3] = torch.ones_like(values[3])
    dopamine = torch.cat((torch.full_like(state.h, 2.), torch.full_like(state.h, .5)))
    def run():
        return advance_contact_learning(model, state, state.h, torch.zeros_like(state.h), state.ring,
            dopamine, state.credit_dopamine, state.dpm_trace, list(values), advance_compartment_learning)
    on = run()
    model.kc_lateral_camp_enabled = False
    off = run()
    assert on[0].abs().sum() < off[0].abs().sum()
    torch.testing.assert_close(state.dopamine, torch.full_like(state.dopamine, .5))
    assert torch.autograd.grad(on[0].sum(), model.kc_lateral_log_camp_gain)[0].abs().sum() > 0


def test_relative_burst_gate_enters_camp_integrator_before_burst_judgement(tmp_path):
    from information_boltzmann.core.fly_bptt_learning import (
        attach_comp_timing, attach_comp_relative_burst, attach_comp_camp,
        advance_compartment_learning,
    )
    from information_boltzmann.core.fly_kc_contact_memory import advance_contact_learning
    from information_boltzmann.core.fly_kc_terminals import terminal_state
    model, state = contact_setup(tmp_path)
    attach_comp_timing(model)
    attach_comp_relative_burst(model)
    attach_comp_camp(model, tau_ticks=4)
    state = replace(state, comp_camp=torch.full_like(state.comp_memory, .25))
    values = terminal_state(model, state) + [state.kc_lateral_receptor, state.kc_lateral_signal]
    dopamine = torch.cat((torch.full_like(state.h, 2.), torch.full_like(state.h, .5)))
    def run():
        return advance_contact_learning(model, state, state.h, torch.zeros_like(state.h), state.ring,
            dopamine, state.credit_dopamine, state.dpm_trace, list(values), advance_compartment_learning)
    on = run()
    model.kc_lateral_camp_enabled = False
    off = run()
    # Return order is the existing shared biochemical-state contract.
    assert on[4].shape == off[4].shape == (1, model.kcm_site.numel())
    assert bool((on[4] < off[4]).all())
    assert torch.autograd.grad(on[4].sum(), model.kc_lateral_log_camp_gain)[0].abs().sum() > 0


@pytest.mark.parametrize('checkpointing', [False, True])
def test_learner_carries_and_updates_local_states_with_checkpointing(tmp_path, checkpointing):
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
    from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
    model, state = contact_setup(tmp_path)
    learner = FlyBPTTLearner(model, state, use_checkpointing=checkpointing, settle_ticks=3)
    learner.previous_token = 0
    learner.observe([1,2,3])
    for name in KC_LATERAL_STATE_FIELDS:
        assert getattr(learner.state, name).shape == (1,3)
        assert not getattr(learner.state, name).requires_grad
    assert set(KC_LATERAL_PARAMETER_NAMES) <= set(native_parameter_names(model))
    assert 'kc_lateral_digest' in checkpoint_model_weights(model)
    assert model.kc_lateral_log_calcium_gain in learner.optimizer.state
