"""Unified execution/state contracts; no language or biological capability claim."""
from dataclasses import fields, replace
import io
import copy

import pytest
import torch
from torch import nn

from information_boltzmann.core.fly_native_runtime import (
    MECHANISMS, NativeStateLayout, NativeInstruction, NativeExecutionProgram,
    attach_execution_program, run_threshold_phase, state_owners,
    pack_component_updates, identity_names,
    attach_program_document,
    attachment_order,
    validate_program_bindings,
)
from information_boltzmann.core.fly_bptt_learning import step_fly_physical_tick, FlyPhysicalState
from test_fly_bptt_learning import make_model, physical


def program(count):
    layout = NativeStateLayout([('receptor', (1, count), 0), ('available', (1, count), 1)])
    code = [NativeInstruction('binding', 'ligand_binding',
                (('occupancy', 'old:receptor'), ('concentration', 'external:ligand'),
                 ('affinity', 'external:affinity'), ('off_rate', 'external:off')), ('receptor',)),
            NativeInstruction('resources', 'resource_balance',
                (('available', 'old:available'), ('probability', 'new:receptor'),
                 ('retained_fraction', 'external:recovery')), ('available', 'release'))]
    return NativeExecutionProgram(layout, code, ['ligand', 'affinity', 'off', 'recovery'],
        anatomy_digest='numerical-interface-fixture', evidence={'scope': 'unit interface only'},
        effectors=(('pre_membrane', 'extra_ge', 'release'),))


def test_unified_registry_has_one_state_owner_and_all_declared_fields_exist():
    owners = state_owners()
    assert len(owners) == sum(len(s.states) for s in MECHANISMS) + 8
    assert set(owners) <= {f.name for f in fields(FlyPhysicalState)}
    assert len(identity_names()) == len(set(identity_names()))
    complete = set()
    for spec in attachment_order():
        assert set(spec.attachment.dependencies) <= complete
        complete.add(spec.name)


def test_partial_continuing_component_is_rejected_before_any_initialization(tmp_path):
    from types import SimpleNamespace
    from information_boltzmann.core.fly_native_assembly import assemble_spatial_components
    from information_boltzmann.core.fly_mb_terminal import MB_TERMINAL_PARAMETER_NAMES
    model = nn.Linear(2, 2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    args = SimpleNamespace(graph=tmp_path / 'not_needed.npz', mb_terminal_projection=True)
    with pytest.raises(ValueError, match='Incomplete'):
        assemble_spatial_components(model, args, saved={MB_TERMINAL_PARAMETER_NAMES[0]: torch.ones(1)},
                                     previous_config={'mb_terminal_projection': True})
    assert torch.equal(rng, torch.get_rng_state())
    assert set(before) == set(model.state_dict())
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_compiler_rejects_future_inputs_duplicate_writers_and_unbound_states():
    p = program(6)
    for instructions in (p.instructions[::-1], (p.instructions[0], p.instructions[0])):
        with pytest.raises(ValueError):
            NativeExecutionProgram(p.layout, instructions, p.external_names,
                anatomy_digest='test', evidence={'scope': 'unit'})
    with pytest.raises(ValueError, match='every|Every'):
        NativeExecutionProgram(p.layout, p.instructions[:1], p.external_names,
            anatomy_digest='test', evidence={'scope': 'unit'})


def test_generated_fields_execute_in_real_tick_affect_brain_and_keep_gradients(tmp_path):
    model = make_model(tmp_path)
    p = program(model.n_neurons)
    attach_execution_program(model, p)
    model.binding_affinity = nn.Parameter(torch.tensor(.5))
    model.native_external_bindings = {
        'ligand': lambda model, state, frame: state.ring[1],
        'affinity': lambda model, state, frame: model.binding_affinity,
        'off': torch.tensor(.3), 'recovery': torch.tensor(.8)}
    state = physical(model)
    ring = [x.clone() for x in state.ring]
    ring[1][0, 2] = 1.
    state = replace(state, ring=tuple(ring), native_state=p.layout.initialize().requires_grad_())
    rng = torch.get_rng_state().clone()
    output = step_fly_physical_tick(model, state, None, torch.zeros_like(state.h), state.baseline, {})
    assert torch.equal(rng, torch.get_rng_state())
    model.native_program_enabled = False
    control = step_fly_physical_tick(model, state, None, torch.zeros_like(state.h), state.baseline, {})
    assert output.ge[0, 2] > control.ge[0, 2]
    grad = torch.autograd.grad(output.h.sum() + output.native_state.sum(),
                               (model.binding_affinity, state.native_state))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad)
    # One registry packs all legacy fields; an unmentioned field preserves the
    # exact object and history rather than allocating fresh zeros.
    assert pack_component_updates(state, {})['native_state'] is state.native_state


def test_generated_program_resume_and_phase_guard_preserve_all_physical_values(tmp_path):
    model = make_model(tmp_path)
    p = program(model.n_neurons)
    attach_execution_program(model, p)
    model.native_external_bindings = {'ligand': torch.ones(1, 6), 'affinity': torch.tensor(.5),
                                      'off': torch.tensor(.2), 'recovery': torch.tensor(.8)}
    initial = replace(physical(model), native_state=p.layout.initialize())
    def tick(state):
        return step_fly_physical_tick(model, state, None, torch.zeros_like(state.h), state.baseline, {})
    current = tick(initial).detached()
    stream = io.BytesIO()
    torch.save(current.state_dict(), stream)
    stream.seek(0)
    restored = FlyPhysicalState(**torch.load(stream, weights_only=False))
    expected, actual = tick(current), tick(restored)
    for key, value in expected.state_dict().items():
        if key == 'ring':
            for a, b in zip(value, actual.ring):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, getattr(actual, key), rtol=0, atol=0)
    context = p.begin(current.native_state)
    with pytest.raises(ValueError, match='repeated|skipped|reordered'):
        p.execute_phase('post_release', context, {})
    with pytest.raises(ValueError, match='Incomplete'):
        p.finish(current.native_state, context)
    with pytest.raises(ValueError, match='migration'):
        attach_execution_program(nn.Linear(1, 1), p, saved={'native_program_digest': torch.zeros(32, dtype=torch.uint8)})


def test_generated_program_never_invents_history_or_accepts_wrong_outputs(tmp_path):
    model = make_model(tmp_path)
    p = program(model.n_neurons)
    attach_execution_program(model, p)
    with pytest.raises(ValueError, match='history'):
        step_fly_physical_tick(model, physical(model), None, None, torch.empty(0), {})
    with pytest.raises(ValueError, match='outputs'):
        pack_component_updates(physical(model), {'kc_site_stp': [torch.empty(0)]})


@pytest.mark.parametrize('checkpointing', [False, True])
def test_declared_program_parameters_optimizer_and_history_roundtrip(tmp_path, checkpointing):
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
    from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
    model = make_model(tmp_path)
    p = program(model.n_neurons)
    document = {'program': p.state_dict(),
        'materials': {'affinity': {'shape': [], 'initial': 0.}},
        'bindings': {'ligand': {'kind': 'constant', 'value': [[1.] * model.n_neurons]},
                     'affinity': {'kind': 'material', 'name': 'affinity', 'transform': 'exp'},
                     'off': {'kind': 'constant', 'value': .2},
                     'recovery': {'kind': 'constant', 'value': .8}}}
    attach_program_document(model, document)
    state = replace(physical(model), native_state=p.layout.initialize())
    learner = FlyBPTTLearner(model, state, use_checkpointing=checkpointing, settle_ticks=1)
    learner.previous_token = 0
    learner.observe([1, 2])
    assert 'native_materials.affinity' in native_parameter_names(model)
    assert model.native_materials['affinity'] in learner.optimizer.state
    payload = {'model': checkpoint_model_weights(model), 'learner': learner.state_dict()}
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    restored = make_model(tmp_path)
    attach_program_document(restored, saved['learner']['native_runtime_document'], saved=saved['model'])
    restored.load_state_dict(saved['model'], strict=False)
    resumed_state = FlyPhysicalState(**saved['learner']['physical'])
    resumed = FlyBPTTLearner(restored, resumed_state, use_checkpointing=checkpointing, settle_ticks=1)
    resumed.load_adam_state(saved['learner']['optimizer'])
    parameter = restored.native_materials['affinity']
    for key, value in learner.optimizer.state[model.native_materials['affinity']].items():
        actual = resumed.optimizer.state[parameter][key]
        torch.testing.assert_close(value, actual, rtol=0, atol=0)
    torch.testing.assert_close(resumed.state.native_state, learner.state.native_state, rtol=0, atol=0)
    changed = copy.deepcopy(document)
    changed['bindings']['off']['value'] = .4
    with pytest.raises(ValueError, match='migration'):
        attach_program_document(make_model(tmp_path), changed, saved=saved['model'])


def test_local_port_changes_one_real_bouton_without_a_cell_average(tmp_path):
    from test_fly_kc_terminals import setup
    from information_boltzmann.core.fly_kc_terminals import advance_kc_terminals
    from information_boltzmann.core.fly_native_runtime import generated_effectors
    model, state, _ = setup(tmp_path)
    packet = torch.ones_like(state.h)
    reference = advance_kc_terminals(model, state, packet)
    gate = torch.ones(1, model.kct_site_node.numel())
    gate[0, 0] = 0
    changed = advance_kc_terminals(model, state, packet, local_inputs={'kc_calcium_gate': gate})
    assert reference[3][0, 0] > 0 and changed[3][0, 0] == 0
    torch.testing.assert_close(reference[3][0, 1:], changed[3][0, 1:], rtol=0, atol=0)
    # A malformed cell-sized vector must not pass as a site-sized gate.
    model.native_execution_program = type('Ports', (), {'effectors':
        (('pre_membrane', 'kc_calcium_gate', 'gate'),)})()
    with pytest.raises(ValueError, match='surface'):
        generated_effectors(model, {'new': {'gate': torch.ones_like(state.h)},
                                    'phases_done': ['pre_membrane']}, 'pre_membrane')


def test_external_local_dan_input_enters_physics_and_conditional_carried_credit(tmp_path):
    from test_fly_dan_credit import prepared, objective
    from information_boltzmann.core.fly_kc_terminals import terminal_state
    from information_boltzmann.core.fly_dan_release import advance_dan_release, DAN_RELEASE_STATE_FIELDS
    from information_boltzmann.core.fly_dan_credit import (
        advance_credit, carried_gradients, DAN_CREDIT_PARAMETERS, DAN_CREDIT_FIELDS)
    model, current = prepared(tmp_path)
    for i in range(3):
        kc = terminal_state(model, current)
        extra = {'dan_receptor_input': current.h.new_full((1, model.danl_owner.numel()), .17 + i * .02)}
        values = advance_dan_release(model, current, current.ring, kc, local_inputs=extra)
        history = advance_credit(model, current, current.ring, kc, values, local_inputs=extra)
        current = replace(current, **dict(zip(DAN_RELEASE_STATE_FIELDS, values)),
                          **dict(zip(DAN_CREDIT_FIELDS, history)))
    exact = torch.autograd.grad(objective(current), [getattr(model, name) for name in DAN_CREDIT_PARAMETERS])
    end = replace(current, **{name: getattr(current, name).detach().requires_grad_()
                              for name in DAN_RELEASE_STATE_FIELDS})
    gradients = torch.autograd.grad(objective(end), [getattr(end, name) for name in DAN_RELEASE_STATE_FIELDS])
    contracted = carried_gradients(model, dict(zip(DAN_RELEASE_STATE_FIELDS, gradients)), current.state_dict())
    for name, expected in zip(DAN_CREDIT_PARAMETERS, exact):
        torch.testing.assert_close(contracted[name], expected, rtol=2e-7, atol=2e-9)


def test_compiled_native_contact_families_preserve_every_physical_output_and_vjp(tmp_path):
    from test_fly_dan_release import setup
    from information_boltzmann.core.fly_native_templates import compile_existing_kc_routes
    from information_boltzmann.core.fly_kc_terminals import terminal_state
    model, initial, _ = setup(tmp_path)
    values = terminal_state(model, initial)
    values[3] = initial.h.new_tensor([[.3, .7, .1], [.4, .2, .8], [.6, .5, .2], [.1, .3, .4]])
    initial = replace(initial, kc_terminal_packets=values[3])
    document = compile_existing_kc_routes(model)
    compiled = attach_program_document(model, document)
    validate_program_bindings(model)
    initial = replace(initial, native_state=compiled.layout.initialize())
    rng = torch.get_rng_state().clone()
    results = []
    for enabled in (False, True):
        model.native_program_enabled = enabled
        current = copy.deepcopy(initial)
        for _ in range(3):
            current = step_fly_physical_tick(model, current, None,
                torch.zeros_like(current.h), current.baseline, {})
        results.append(current)
    assert torch.equal(rng, torch.get_rng_state())
    for name, expected in results[0].state_dict().items():
        actual = getattr(results[1], name)
        if name == 'ring':
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    leaves = [model.dan_local_log_affinity, model.kc_terminal_log_diffusion]
    gradients = [torch.autograd.grad(result.h.sum() + result.dan_local_receptor.sum()
                    + result.kc_terminal_gaba.sum(), leaves) for result in results]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
