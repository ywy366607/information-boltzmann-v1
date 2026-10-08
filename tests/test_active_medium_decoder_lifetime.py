"""Completed decoder storage is not pending learning credit or optimizer state."""
import copy

import numpy as np
import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.medium_health import MediumHealthAuditor
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.training import belief_tensors, clone_belief


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RetainCompletedDecoderZero(ActiveMediumTrainer):
    """Control restores the old resident zero before the next chunk begins."""

    def consume(self, *args, **kwargs):
        updates = self.optimizer_updates
        result = super().consume(*args, **kwargs)
        if self.pending == 0 and self.optimizer_updates > updates:
            self.model.decoder.weight.grad = torch.zeros_like(self.model.decoder.weight)
        return result


def numerical_model():
    torch.manual_seed(954)
    model = PlasticMediumPorts3D(
        vocab_size=13, shape=(2, 2, 2), channels=4, material_width=2,
        hidden=4, heads=1, queries=1, anisotropic_transport=True,
        bath_type='conductance', activity_adaptation=True,
        short_term_plasticity=True, material_reference_shape=None,
        read_mode='temporal', temporal_rates=[1., 4.],
        temporal_frequencies=[0., 3.], temporal_time_reference=.02,
        intrinsic_time_reference=.006, solver_max_step=.004,
        observer_max_step=.008,
        structure_options=dict(resource_density=4., speed_reference=3.,
            structure_time=1., prior_std=.2, initial_std=.2,
            maintenance_supply=2., initial_dual=.1)).double()
    # These two disconnected leaves make None-versus-zero semantics observable:
    # one remains skipped, the other retains real Adam momentum after update 1.
    model.register_parameter('unused_none', torch.nn.Parameter(torch.tensor(.4, dtype=torch.float64)))
    model.register_parameter('unused_momentum', torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64)))
    with torch.no_grad():
        model.intrinsic_time.head.weight.fill_(.03)
    return model


def learner(model, kind=ActiveMediumTrainer, saved=None):
    optimizer = make_medium_optimizer(model, lr=2e-4, weight_decay=.03,
        saved_state=None if saved is None else saved['optimizer'])
    health = MediumHealthAuditor(model, window_tokens=16, block_tokens=2)
    initial = model.initial_belief() if saved is None else clone_belief(saved['belief'])
    result = kind(model, optimizer, initial, carry_token=1, event_duration=.005,
        chunk_tokens=3, tokens_per_update=4, activation_checkpointing=True, health=health)
    if saved is not None:
        result.load_state_dict(saved['learner'])
    return result


def assert_tree_equal(left, right, path='root'):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        torch.testing.assert_close(left, right, atol=0, rtol=0, msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_tree_equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_tree_equal(a, b, f'{path}[{index}]')
    else:
        assert left == right, (path, left, right)


def assert_learning_equal(released, retained):
    assert_tree_equal(released.model.state_dict(), retained.model.state_dict(), 'model')
    assert_tree_equal(released.optimizer.state_dict(), retained.optimizer.state_dict(), 'AdamW')
    for index, (a, b) in enumerate(zip(belief_tensors(released.belief), belief_tensors(retained.belief))):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=f'physical_state[{index}]')
    left, right = released.state_dict(), retained.state_dict()
    a = left['pending_gradients'].pop('decoder.weight')
    b = right['pending_gradients'].pop('decoder.weight')
    if released.pending == 0:
        assert a is None
        assert b is not None and torch.count_nonzero(b) == 0
    else:
        assert a is not None and b is not None
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.count_nonzero(a) > 0
    # Every other gradient, including the disconnected leaves, stays exact.
    assert_tree_equal(left, right, 'learner')
    assert released.summary() == retained.summary()
    assert released.model.unused_none.grad is None
    assert retained.model.unused_none.grad is None


def test_three_adam_updates_release_only_completed_decoder_zero_with_partial_resume():
    model = numerical_model()
    released = learner(model)
    retained = learner(copy.deepcopy(model), RetainCompletedDecoderZero)
    for current in (released, retained):
        current.model.unused_momentum.grad = torch.full_like(current.model.unused_momentum, .02)
    prior = np.full(13, np.log(13))
    resumed_pair = None
    cursor = 0
    for count in (3, 1, 2, 2, 1, 3):
        targets = ((torch.arange(count) + cursor) % 12) + 1
        rng = torch.get_rng_state()
        expected = released.consume(targets, phase='numerical_contract', prior_nll=prior)
        torch.set_rng_state(rng)
        actual = retained.consume(targets, phase='numerical_contract', prior_nll=prior)
        assert actual == expected
        assert_learning_equal(released, retained)
        if resumed_pair is not None:
            for original, resumed in zip((released, retained), resumed_pair):
                torch.set_rng_state(rng)
                assert resumed.consume(targets, phase='numerical_contract', prior_nll=prior) == expected
                assert_tree_equal(original.model.state_dict(), resumed.model.state_dict(), 'resumed_model')
                assert_tree_equal(original.optimizer.state_dict(), resumed.optimizer.state_dict(), 'resumed_AdamW')
                assert_tree_equal(original.state_dict(), resumed.state_dict(), 'resumed_learner')
                for a, b in zip(belief_tensors(original.belief), belief_tensors(resumed.belief)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        cursor += count
        if cursor == 6:
            # Update 1 completed; two real events of update 2 are pending. The
            # decoder must now own nonzero credit, which resume cannot release.
            assert released.optimizer_updates == 1 and released.pending == 2
            resumed_pair = []
            for original, kind in ((released, ActiveMediumTrainer),
                                   (retained, RetainCompletedDecoderZero)):
                saved = dict(optimizer=copy.deepcopy(original.optimizer.state_dict()),
                    learner=copy.deepcopy(original.state_dict()), belief=clone_belief(original.belief))
                resumed = learner(copy.deepcopy(original.model), kind, saved)
                assert_tree_equal(original.state_dict(), resumed.state_dict(), 'pending_resume')
                resumed_pair.append(resumed)
    assert released.optimizer_updates == retained.optimizer_updates == 3
    assert released.pending == retained.pending == 0 and released.events == 12
    assert int(released.optimizer.state[released.model.decoder.weight]['step']) == 3
    assert released.model.medium.structural_posterior.windows_committed == 3
    assert released.summary()['loss_components']['component_events']['task_nll'] == 12
    assert released.model.unused_momentum.grad is not None
    assert torch.count_nonzero(released.model.unused_momentum.grad) == 0
    assert released.optimizer.state[released.model.unused_momentum]['exp_avg'].abs() > 0


def test_optimizer_host_staging_preserves_full_medium_state_and_adam(monkeypatch):
    import information_boltzmann.runtime.active_medium_training as training_module
    from information_boltzmann.runtime.optimizer_storage import optimizer_state_on_host

    monkeypatch.setattr(training_module, 'optimizer_state_on_host',
        lambda optimizer, *, enabled, release_cached_memory=False: optimizer_state_on_host(
            optimizer, enabled=enabled, device_types=('cpu',),
            release_cached_memory=release_cached_memory))
    model = numerical_model()
    reference = learner(model)
    staged = learner(copy.deepcopy(model))
    staged.optimizer_state_offload = True
    prior = np.full(13, np.log(13))
    for index in range(3):
        targets = (torch.arange(4) + index) % 12 + 1
        rng = torch.get_rng_state()
        expected = reference.consume(targets, phase='numerical_contract', prior_nll=prior)
        torch.set_rng_state(rng)
        assert staged.consume(targets, phase='numerical_contract', prior_nll=prior) == expected
        assert_tree_equal(reference.model.state_dict(), staged.model.state_dict(), 'model')
        assert_tree_equal(reference.optimizer.state_dict(), staged.optimizer.state_dict(), 'AdamW')
        assert_tree_equal(reference.state_dict(), staged.state_dict(), 'learner')
        for a, b in zip(belief_tensors(reference.belief), belief_tensors(staged.belief)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
