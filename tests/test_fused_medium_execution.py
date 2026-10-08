"""Execution fusion equivalence, without architecture/capability claims."""
from dataclasses import replace
import os

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk


@pytest.mark.parametrize('diagnostics', [False, True])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_compiled_step_reuse_matches_native_refinement_and_credit(diagnostics, checkpointing):
    torch.set_num_threads(1)
    torch.manual_seed(971)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=4,
        hidden=4, bath_type='conductance', activity_adaptation=True,
        short_term_plasticity=True, anisotropic_transport=True).double()
    state = model.initial_belief().medium
    state = replace(state, field=torch.randn_like(state.field),
        flux=tuple(torch.randn_like(x) for x in state.flux),
        receptors=torch.rand_like(state.receptors),
        transmission=torch.rand_like(state.transmission))
    duration = torch.tensor(.023, dtype=torch.float64, requires_grad=True)
    prepared = model.medium.prepare_evolution()
    expected, expected_info = model.medium.native_advance(state, duration,
        substeps=4, prepared=prepared, diagnostics=diagnostics)
    actual, actual_info = model.medium._fused_substeps(model.medium.native_advance,
        state, duration, substeps=4, prepared=prepared, diagnostics=diagnostics,
        activation_checkpointing=checkpointing)
    for a, b in zip((actual.field, *actual.flux, actual.receptors,
                    actual.conduction, actual.transmission, actual.elapsed),
                   (expected.field, *expected.flux, expected.receptors,
                    expected.conduction, expected.transmission, expected.elapsed)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual_info.keys() == expected_info.keys()
    for key in actual_info:
        torch.testing.assert_close(actual_info[key], expected_info[key], rtol=1e-12, atol=1e-12)
    parameters = (duration, *tuple(p for p in model.parameters() if p.requires_grad))
    left = torch.autograd.grad(actual.field.square().sum() + actual.flux[0].sum(),
                               parameters, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(expected.field.square().sum() + expected.flux[0].sum(),
                                parameters, allow_unused=True)
    for a, b in zip(left, right):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1', reason='Opt-in CUDA allocation')
def test_fused_tensor_refinement_with_health_ledger_matches_native_cuda():
    torch.set_num_threads(1)
    torch.manual_seed(973)
    from information_boltzmann.runtime.execution_cache import configure_execution_cache
    configure_execution_cache('scratch/compiler_cache')
    args = dict(vocab_size=17, shape=(2, 2, 2), channels=4, hidden=4,
                bath_type='conductance', activity_adaptation=True,
                short_term_plasticity=True, anisotropic_transport=True)
    native = PlasticMediumPorts3D(**args).cuda()
    fused = PlasticMediumPorts3D(**args, medium_execution='fused').cuda()
    fused.load_state_dict(native.state_dict())
    state = native.initial_belief().medium
    state = replace(state, field=torch.randn_like(state.field),
        flux=tuple(torch.randn_like(x) for x in state.flux),
        receptors=torch.rand_like(state.receptors),
        transmission=torch.rand_like(state.transmission))
    duration = torch.tensor(.023, dtype=torch.float64, device='cuda', requires_grad=True)
    for model in (native, fused):
        outgoing, info = model.medium.advance(state, duration, substeps=4, diagnostics=True,
                                              activation_checkpointing=(model is fused))
        objective = outgoing.field.square().sum() + outgoing.flux[0].sum()
        gradients = torch.autograd.grad(objective, (duration, *tuple(model.parameters())), allow_unused=True)
        if model is native:
            expected, expected_info, expected_grad = outgoing, info, gradients
        else:
            assert hasattr(model.medium, '_compiled_advance')
            for a, b in zip((outgoing.field, *outgoing.flux, outgoing.receptors,
                            outgoing.conduction, outgoing.transmission, outgoing.elapsed),
                           (expected.field, *expected.flux, expected.receptors,
                            expected.conduction, expected.transmission, expected.elapsed)):
                torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-6)
            for key in info:
                torch.testing.assert_close(info[key], expected_info[key], rtol=5e-4, atol=1e-4, msg=key)
            for a, b in zip(gradients, expected_grad):
                assert (a is None) == (b is None)
                if a is not None:
                    torch.testing.assert_close(a, b, rtol=5e-4, atol=1e-4)


def test_fused_backend_preserves_weights_and_fp64_reference_flow():
    args = dict(vocab_size=17, shape=(2, 2, 2), channels=8, hidden=8,
                bath_type='conductance', activity_adaptation=True, short_term_plasticity=True)
    native = PlasticMediumPorts3D(**args).double()
    fused = PlasticMediumPorts3D(**args, medium_execution='fused', port_execution='fused').double()
    fused.load_state_dict(native.state_dict(), strict=True)
    assert native.architecture == fused.architecture
    initial = native.initial_belief()
    initial = replace(initial, medium=replace(initial.medium,
        field=torch.randn_like(initial.medium.field),
        flux=tuple(torch.randn_like(x) for x in initial.medium.flux)))
    for model in (native, fused):
        loss, final, _ = quiet_training_chunk(model, torch.tensor([[1, 2]]),
                                             torch.tensor([[2, 3]]), initial, event_duration=0.005)
        loss.backward()
        if model is native:
            expected_loss, expected = loss.detach(), final
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        else:
            torch.testing.assert_close(loss, expected_loss, atol=0, rtol=0)
            for got, wanted in zip(belief_tensors(final), belief_tensors(expected)):
                torch.testing.assert_close(got, wanted, atol=0, rtol=0)
            for name, p in model.named_parameters():
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, gradients[name], atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1', reason='Opt-in CUDA allocation')
@pytest.mark.parametrize('port_execution', ['native', 'fused'])
def test_fused_medium_matches_native_joint_ce_and_all_parameter_gradients(port_execution):
    torch.set_num_threads(1)
    torch.manual_seed(942)
    args = dict(vocab_size=17, shape=(4, 4, 4), channels=8, hidden=8, heads=2, queries=2,
                bath_type='conductance', activity_adaptation=True, short_term_plasticity=True)
    native = PlasticMediumPorts3D(**args).cuda()
    fused = PlasticMediumPorts3D(**args, medium_execution='fused', port_execution=port_execution).cuda()
    with torch.no_grad():
        native.medium.material.coefficients.normal_(std=0.01)
    fused.load_state_dict(native.state_dict(), strict=True)
    initial = native.initial_belief()
    initial = replace(initial, medium=replace(initial.medium,
        field=torch.randn_like(initial.medium.field),
        flux=tuple(torch.randn_like(x) for x in initial.medium.flux),
        receptors=torch.rand_like(initial.medium.receptors),
        transmission=torch.rand_like(initial.medium.transmission)))
    ids = torch.tensor([[1, 2]], device='cuda')
    targets = torch.tensor([[2, 3]], device='cuda')
    dt = torch.tensor(0.005, device='cuda', dtype=torch.float64)
    for model in (native, fused):
        loss, final, _ = quiet_training_chunk(model, ids, targets, initial, event_duration=dt)
        loss.backward()
        if model is native:
            expected_loss, expected = loss.detach(), final.detach()
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        else:
            torch.testing.assert_close(loss, expected_loss, atol=3e-6, rtol=3e-5)
            for got, wanted in zip(belief_tensors(final), belief_tensors(expected)):
                torch.testing.assert_close(got, wanted, atol=3e-6, rtol=3e-5)
            actual = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            assert actual.keys() == gradients.keys()
            for name, gradient in actual.items():
                assert gradient.isfinite().all()
                torch.testing.assert_close(gradient, gradients[name], atol=5e-6, rtol=5e-4, msg=name)
