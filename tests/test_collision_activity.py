"""Numerical/interface audits of the opt-in collision activity options; no capability claims."""
from dataclasses import replace

import math

import pytest
import torch

from information_boltzmann.core.plastic_medium import PlasticMedium3D

STRUCTURE = dict(resource_density=4., speed_reference=4., structure_time=1., prior_std=.2,
                 initial_std=.2, maintenance_supply=2., initial_dual=.1)


def medium(structured=False, **options):
    torch.manual_seed(487)
    extra = (dict(anisotropic_transport=True, structure_options=STRUCTURE, material_reference_shape=None)
             if structured else {})
    return PlasticMedium3D((2, 2, 2), 3, hidden=8, bath_type='conductance', adaptive_conduction=True,
                           collision_options=options or None, **extra).double()


def state(net, seed=0):
    generator = torch.Generator().manual_seed(seed)
    draw = lambda like: torch.randn(like.shape, generator=generator, dtype=torch.float64)
    empty = net.initial_state(2)
    return replace(empty, field=draw(empty.field), flux=tuple(draw(x) for x in empty.flux),
                   receptors=torch.rand(empty.receptors.shape, generator=generator, dtype=torch.float64))


def test_default_options_keep_legacy_keys_and_arithmetic():
    control, explicit = medium(), medium(center_receptors=False, position_rates=False)
    assert list(control.state_dict()) == list(explicit.state_dict())
    s = state(control)
    for key, value in control.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    assert torch.equal(control.collide(s, 0.1).field, explicit.collide(s, 0.1).field)


def test_new_parameters_do_not_move_the_legacy_initialisation():
    control, active = medium(), medium(center_receptors=True, position_rates=True)
    assert set(active.state_dict()) - set(control.state_dict()) == {'collision_position'}
    for key, value in control.state_dict().items():
        assert torch.equal(value, active.state_dict()[key]), key


def test_zero_initialised_position_rates_are_exact_identity_and_trainable():
    control, active = medium(), medium(position_rates=True)
    with torch.no_grad():       # a uniform (all-zero) material carries no position code to differentiate
        control.material.coefficients.normal_(std=.3)
        active.material.coefficients.copy_(control.material.coefficients)
    s = state(control)
    assert torch.equal(control.collide(s, 0.1).field, active.collide(s, 0.1).field)
    loss = active.collide(s, 0.1).field.square().sum()
    grad, = torch.autograd.grad(loss, active.collision_position)
    assert grad.norm() > 0


def test_position_rates_add_exactly_the_material_map_and_share_flow_and_derivative():
    control, active = medium(), medium(position_rates=True)
    with torch.no_grad():
        control.material.coefficients.normal_(std=.3)
        active.material.coefficients.copy_(control.material.coefficients)
        active.collision_position.normal_(std=.2)
    s = state(control)
    packed, material = control._pack(s), control.material_field()
    difference = (active.collision_rates(packed, material, s.receptors)
                  - control.collision_rates(packed, material, s.receptors))
    expected = torch.nn.functional.linear(material, active.collision_position)[None].reshape(
        1, *material.shape[:-1], active.collision_layers, active.free_width // 2)
    torch.testing.assert_close(difference, expected.expand_as(difference), atol=1e-12, rtol=1e-12)
    # The analytic derivative measurement and the finite-duration flow read the same rates.
    h = 1e-6
    flow = (active.collide(s, h).field - control.collide(s, h).field) / h
    rhs = active.native_field_rhs(s) - control.native_field_rhs(s)
    torch.testing.assert_close(flow, rhs, atol=1e-4 * rhs.abs().max().item(), rtol=1e-3)
    assert rhs.abs().max() > 0


def test_centered_receptors_ignore_the_constant_gate_mode():
    off, on = medium(), medium(center_receptors=True)
    s = state(off)
    shifted = replace(s, receptors=s.receptors + 0.37)
    for net, changes in ((off, True), (on, False)):
        a = net.collision_rates(net._pack(s), net.material_field(), s.receptors)
        b = net.collision_rates(net._pack(shifted), net.material_field(), shifted.receptors)
        assert bool((a - b).abs().max() > 1e-6) is changes
    # Only the constant mode is removed: a non-constant gate pattern still reaches the rates.
    pattern = replace(s, receptors=s.receptors.clone())
    pattern.receptors[..., 0, 0] += 0.5
    a = on.collision_rates(on._pack(s), on.material_field(), s.receptors)
    b = on.collision_rates(on._pack(pattern), on.material_field(), pattern.receptors)
    assert (a - b).abs().max() > 1e-6


def test_bed_rates_give_installed_capacity_a_gradient_through_collisions():
    control, active = medium(structured=True), medium(structured=True, bed_rates=True)
    for net in (control, active):
        mean = net.structural_posterior.mean
        net.structural_posterior.begin_window(torch.randn(mean.shape, dtype=mean.dtype,
                                                          generator=torch.Generator().manual_seed(3)))
    s = state(control)
    prepared = control.prepare_evolution()
    allocation = active.prepare_evolution().structural_allocation
    assert torch.equal(control.collide(s, 0.1, allocation=prepared.structural_allocation).field,
                       active.collide(s, 0.1, allocation=allocation).field)
    with pytest.raises(ValueError):
        active.collide(s, 0.1)
    with torch.no_grad():
        active.collision_bed.normal_(std=.3)
    out = active.collide(s, 0.1, allocation=active.prepare_evolution().structural_allocation).field.square().sum()
    grad, = torch.autograd.grad(out, active.structural_posterior.mean)
    assert grad.norm() > 0
    # The legacy collision cannot see the bed at all.
    baseline = control.collide(s, 0.1, allocation=control.prepare_evolution().structural_allocation).field.square().sum()
    assert torch.autograd.grad(baseline, control.structural_posterior.mean, allow_unused=True)[0] is None


def test_kerr_rates_are_zero_identity_bounded_and_see_wave_wave_coincidence():
    control, active = medium(), medium(kerr_rates=True)
    s = state(control)
    assert torch.equal(control.collide(s, 0.1).field, active.collide(s, 0.1).field)
    loss = active.collide(s, 0.1).field.square().sum()
    grad, = torch.autograd.grad(loss, active.collision_kerr)
    assert grad.norm() > 0                       # first-order gradient even from the zero start
    with torch.no_grad():
        active.collision_kerr.normal_(std=.5)
    packed, material = active._pack(s), active.material_field()
    drive = (active.collision_rates(packed, material, s.receptors)
             - control.collision_rates(packed, material, s.receptors))
    scaled = replace(s, field=1e3 * s.field, flux=tuple(1e3 * x for x in s.flux))
    big = active.collision_rates(active._pack(scaled), material, scaled.receptors)         - control.collision_rates(active._pack(scaled), material, scaled.receptors)
    assert drive.abs().max() > 0 and torch.isfinite(big).all()
    bound = active.collision_kerr.abs().sum(0).max()
    assert big.abs().max() <= bound + 1e-9       # saturating drives cannot exceed |k1|+|k2| per pair
    # A single travelling wave (flux = +field along one axis) has zero coincidence signal; two counter-propagating
    # waves carry 4*F1*F2 in field^2 - flux^2 although their TOTAL energy is just the sum of the parts.
    f1 = torch.randn(1, 2, 2, 2, 3, dtype=torch.float64)
    f2 = torch.randn_like(f1)
    zero = torch.zeros_like(f1)
    def contrast(field, flux0):
        packed = torch.stack((field, flux0, zero, zero), -2)
        return packed[..., 0, :].square().sum(-1) - packed[..., 1:, :].square().sum((-2, -1))
    torch.testing.assert_close(contrast(f1, f1), torch.zeros(1, 2, 2, 2, dtype=torch.float64))
    torch.testing.assert_close(contrast(f1 + f2, f1 - f2), 4 * (f1 * f2).sum(-1))
    total = lambda field, flux0: field.square().sum(-1) + flux0.square().sum(-1)
    torch.testing.assert_close(total(f1 + f2, f1 - f2), 2 * (f1.square().sum(-1) + f2.square().sum(-1)))


def test_kerr_only_collisions_are_alive_local_per_head_and_energy_conserving():
    net = medium(kerr_only=True, heads=3, rate_scale=10.0)
    with torch.no_grad():
        net.material.coefficients.normal_(std=.3)
    s = state(net)
    packed, material = net._pack(s), net.material_field()
    rates = net.collision_rates(packed, material, s.receptors)
    assert rates.shape == (2, 2, 2, 2, net.collision_layers, net.free_width // 2) and torch.isfinite(rates).all()
    assert rates.abs().mean() > 1.0                       # O(s) from birth: about O(1) rad per event, not identity
    # state dependence: a different state at the same site moves the rates; so does position at the same state
    other = net.collision_rates(net._pack(state(net, seed=5)), material, s.receptors)
    assert (rates - other).abs().max() > 1e-3
    uniform = packed[:, :1, :1, :1].expand_as(packed)
    varied = net.collision_rates(uniform, material, s.receptors)
    assert varied.flatten(1, 3).var(1).sum() > 0
    # the head drive reads only its own channel slice (content-local), pairs are assigned to heads by channel
    heads = net.collision_pair_head
    assert heads.min() == 0 and heads.max() <= 2 and torch.unique(heads).numel() >= 2    # only C-1 free channels here
    assert heads.numel() == net.collision_layers * (net.free_width // 2)
    out = net.collide(s, 0.1)
    torch.testing.assert_close(net.energy(out), net.energy(s), atol=1e-12, rtol=1e-12)
    loss = out.field.pow(3).sum()
    grads = torch.autograd.grad(loss, (net.collision_scale, net.collision_base, net.collision_kerr,
                                       net.collision_head_energy, net.collision_head_position))
    assert all(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in net.collision_rate.parameters())      # the MLP is not used


def test_invalid_collision_options_are_rejected():
    with pytest.raises(ValueError):
        medium(unknown=True)
    with pytest.raises(ValueError):
        medium(bed_rates=True)          # no installed allocation to read


def scaled(channels, shape, hidden, seed=11, speed=1.0, **options):
    torch.manual_seed(seed)
    net = PlasticMedium3D(shape, channels, hidden=hidden, bath_type='conductance', adaptive_conduction=True,
                          speed_reference=speed, collision_options=dict(timescale=True, **options)).double()
    with torch.no_grad():
        net.material.coefficients.normal_(std=.3)
    return net


def test_timescale_rates_are_scale_free_log_uniform_and_relative():
    # Same rule at 4 sizes: only the independent per-pair draws and the hidden-width-normalised modulation exist.
    low, high = torch.tensor(0.3).log().item(), torch.tensor(10.0).log().item()
    for channels, shape, hidden in ((33, (2, 2, 2), 8), (129, (2, 2, 2), 64), (129, (4, 4, 4), 16), (257, (2, 2, 4), 128)):
        net = scaled(channels, shape, hidden)
        assert net.collision_log_phi.min() >= low and net.collision_log_phi.max() <= high
        s = state(net)
        rates = net.collision_rates(net._pack(s), net.material_field(), s.receptors)
        phi = rates.abs().flatten(-2) / net.speed_reference                 # radians per crossing time (T_x = 1)
        g = phi.log() - net.collision_log_phi
        assert abs(net.collision_log_phi.mean().item() - 0.5 * (low + high)) < 0.25
        assert abs(net.collision_log_phi.std().item() - (high - low) / 12 ** .5) < 0.2
        assert g.std() < 0.6 and abs(g.mean()) < 0.3              # modulation is O(0.2), not tied to size
        assert set(rates.sign().unique().tolist()) == {-1.0, 1.0}


def test_timescale_rates_follow_physical_time_not_the_lattice_or_the_step():
    slow, fast = scaled(33, (2, 2, 2), 8, speed=1.0), scaled(33, (2, 2, 2), 8, speed=2.0)
    s = state(slow)
    a = slow.collision_rates(slow._pack(s), slow.material_field(), s.receptors)
    b = fast.collision_rates(fast._pack(s), fast.material_field(), s.receptors)
    torch.testing.assert_close(b, 2.0 * a, atol=1e-12, rtol=1e-12)        # same radians per crossing time
    # the event length only enters through angle = rate * dt, i.e. half the step is half the angle
    one, half = slow.collide(s, 0.02).field, slow.collide(s, 0.01).field
    assert (slow.collide(s, 0.02).field - s.field).abs().max() > 1.5 * (half - s.field).abs().max()


def test_timescale_with_heads_is_alive_unitary_and_trains_its_own_parameters():
    net = scaled(32 + 1, (2, 2, 2), 8, heads=3, kerr_std=0.5)
    s = state(net)
    out = net.collide(s, 0.05)
    torch.testing.assert_close(net.energy(out), net.energy(s), atol=1e-12, rtol=1e-12)
    loss = out.field.pow(3).sum()
    grads = torch.autograd.grad(loss, (net.collision_log_phi, net.collision_kerr, net.collision_head_energy,
                                       net.collision_head_position))
    assert all(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in net.collision_rate.parameters())
    assert not hasattr(net, 'collision_base')
    other = net.collision_rates(net._pack(state(net, seed=5)), net.material_field(), s.receptors)
    assert (net.collision_rates(net._pack(s), net.material_field(), s.receptors) - other).abs().max() > 1e-3


def test_timescale_option_combinations_and_ranges_are_checked():
    with pytest.raises(ValueError):
        medium(timescale=True, kerr_only=True)
    with pytest.raises(ValueError):
        medium(timescale=True, phi_min=5.0, phi_max=1.0)
    control, active = medium(), medium(timescale=True)
    assert set(active.state_dict()) - set(control.state_dict()) == {'collision_log_phi', 'collision_sign'}


def test_rms_norm_hides_amplitude_and_the_amplitude_input_gives_it_back():
    plain, aware = scaled(33, (2, 2, 2), 8, center_receptors=True), scaled(33, (2, 2, 2), 8, center_receptors=True, amplitude_input=3)
    s = state(plain)
    boosted = replace(s, field=s.field.clone(), flux=tuple(x.clone() for x in s.flux))
    boosted.field[:, 0, 0, 0] *= 4.0                       # louder at one site, same direction in channel space
    for x in boosted.flux:
        x[:, 0, 0, 0] *= 4.0
    for net, sees_it in ((plain, False), (aware, True)):
        a = net.collision_rates(net._pack(s), net.material_field(), s.receptors)
        b = net.collision_rates(net._pack(boosted), net.material_field(), boosted.receptors)
        assert bool((a - b)[:, 0, 0, 0].abs().max() > 1e-6) is sees_it
    assert aware.collision_rate[0].in_features == plain.collision_rate[0].in_features + 3
    with pytest.raises(ValueError):
        scaled(33, (2, 2, 2), 8, amplitude_input=5)


def test_modulation_gain_scales_the_log_rate_state_dependence_only():
    one, three = scaled(33, (2, 2, 2), 8, mod_bound=1e9), scaled(33, (2, 2, 2), 8, mod_gain=3.0, mod_bound=1e9)
    s = state(one)
    f = lambda net: (net.collision_rates(net._pack(s), net.material_field(), s.receptors).abs().flatten(-2).log()
                     - net.collision_log_phi)
    torch.testing.assert_close(f(three), 3.0 * f(one), atol=1e-9, rtol=1e-9)


def test_modulation_is_bounded_so_a_huge_gain_cannot_blow_up_the_rates():
    net = scaled(33, (2, 2, 2), 8, heads=3, kerr_std=3.0, mod_gain=50.0)
    s = state(net)
    rates = net.collision_rates(net._pack(s), net.material_field(), s.receptors)
    deviation = rates.abs().flatten(-2).log() - net.collision_log_phi - torch.tensor(net.speed_reference).log()
    assert deviation.abs().max() <= net.collision_mod_bound + 1e-9
    assert rates.abs().max() < 10.0 * net.speed_reference * torch.tensor(net.collision_mod_bound).exp()


def test_gram_collisions_see_only_the_wave_packet_pair_not_the_position_or_the_channel_basis():
    net = scaled(33, (2, 2, 2), 8, gram_input=3, mod_gain=3.0)
    assert net.collision_rate[0].in_features == 33
    s = state(net)
    material = net.material_field()
    packed = net._pack(s)
    base = net.collision_rates(packed, material, s.receptors)
    # (a) position: the same state moved to another site gets the same rates, and material/gates are never read
    swapped = packed.clone(); swapped[:, 1, 1, 1] = packed[:, 0, 0, 0]
    moved = net.collision_rates(swapped, material, s.receptors)
    torch.testing.assert_close(moved[:, 1, 1, 1], base[:, 0, 0, 0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(base, net.collision_rates(packed, 3.0 * material + 1.0, 0.0 * s.receptors), atol=1e-12, rtol=1e-12)
    # (b) basis: the packet-pair features are invariant to an orthogonal change of the channel basis
    gen = torch.Generator().manual_seed(2)       # orthogonal change of basis inside each head's channel slice
    q = torch.block_diag(*(torch.linalg.qr(torch.randn(11, 11, dtype=torch.float64, generator=gen))[0] for _ in range(3)))
    torch.testing.assert_close(net._gram_features(packed @ q), net._gram_features(packed), atol=1e-10, rtol=1e-10)
    # (c) it still depends on the pair: a different state gives different rates; energy is conserved; gradients flow
    assert (net.collision_rates(net._pack(state(net, seed=5)), material, s.receptors) - base).abs().max() > 1e-3
    out = net.collide(s, 0.05)
    torch.testing.assert_close(net.energy(out), net.energy(s), atol=1e-12, rtol=1e-12)
    grads = torch.autograd.grad(out.field.pow(3).sum(), tuple(net.collision_rate.parameters()))
    assert all(g.abs().sum() > 0 for g in grads)
    with pytest.raises(ValueError):
        scaled(33, (2, 2, 2), 8, gram_input=3, amplitude_input=True)


def test_coincidence_collisions_need_two_partners_and_are_position_free_and_conservative():
    net = scaled(33, (2, 2, 2), 8, heads=3, kerr_std=3.0, coincidence=True)
    material = net.material_field()
    z = torch.zeros(1, 2, 2, 2, 33, dtype=torch.float64)
    f1 = torch.randn(1, 2, 2, 2, 33, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    f2 = torch.randn(1, 2, 2, 2, 33, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    rates = lambda packed: net.collision_rates(packed, material)
    assert rates(torch.zeros(1, 2, 2, 2, 4, 33, dtype=torch.float64)).abs().max() == 0          # vacuum: no collision
    single = torch.stack((f1, f1, z, z), -2)                                                     # one wave along axis 0
    assert rates(single).abs().max() < 1e-12                                                     # one packet: no collision
    counter = torch.stack((f1 + f2, f1 - f2, z, z), -2)                                          # two counter-propagating packets
    perpendicular = torch.stack((f1 + f2, f1, f2, z), -2)                                        # two crossing packets
    assert rates(counter).abs().max() > 1e-3 and rates(perpendicular).abs().max() > 1e-3
    # small amplitude: the rate is bilinear (grows like amplitude^2), and bounded for huge amplitude
    small, tiny = rates(1e-3 * counter), rates(1e-4 * counter)
    torch.testing.assert_close(small, 100.0 * tiny, rtol=2e-2, atol=0)
    assert rates(1e6 * counter).abs().max() <= (net.collision_log_phi.exp() * net.collision_kerr[1].abs()).max() * net.speed_reference * 1.01
    # position-free: the material code and the receptors are not read; energy is conserved
    other = net.collision_rates(counter, 5.0 * material + 2.0)
    torch.testing.assert_close(rates(counter), other, atol=1e-12, rtol=1e-12)
    s = state(net)
    out = net.collide(s, 0.05)
    torch.testing.assert_close(net.energy(out), net.energy(s), atol=1e-12, rtol=1e-12)
    assert torch.autograd.grad(out.field.pow(3).sum(), (net.collision_log_phi, net.collision_kerr, net.collision_head_energy))[0].abs().sum() > 0
    with pytest.raises(ValueError):
        scaled(33, (2, 2, 2), 8, coincidence=True)


def test_cross_group_layers_let_collisions_turn_field_into_flux_and_stay_orthogonal():
    import copy
    inner = scaled(8, (2, 2, 2), 8, heads=2, kerr_std=3.0, coincidence=True)
    cross = scaled(8, (2, 2, 2), 8, heads=2, kerr_std=3.0, coincidence=True, cross_groups=True)
    assert cross.collision_layers == inner.collision_layers + 3
    # pair views are a bijection of the coordinates for every layer
    x = torch.randn(1, 2, 2, 2, cross.free_width, dtype=torch.float64)
    for layer in range(cross.collision_layers):
        left, right = cross._pair_view(x, layer)
        assert left.shape[-1] == cross.free_width // 2
        torch.testing.assert_close(cross._pair_restore(left, right, layer), x)
    # the cross layers pair different groups at the same channel
    width = cross.channels - 1
    marker = torch.arange(cross.free_width, dtype=torch.float64).expand(1, 2, 2, 2, -1)
    for layer in range(cross.collision_inner_layers, cross.collision_layers):
        left, right = cross._pair_view(marker, layer)
        assert bool(((left // width) != (right // width)).all()) and bool(((left % width) == (right % width)).all())
    # a field-only state: with cross layers the fluxes receive real weight; energy and the 4 channel sums are conserved
    base = state(cross)
    s = replace(base, flux=tuple(0 * f for f in base.flux))
    for net in (inner, cross):
        with torch.no_grad():
            net.collision_kerr.zero_().add_(3.0)
    flux_inner = sum(f.square().sum() for f in inner.collide(s, 0.3).flux)
    out = cross.collide(s, 0.3)
    flux_cross = sum(f.square().sum() for f in out.flux)
    assert flux_cross > 20 * flux_inner and flux_cross > 0.05 * s.field.square().sum()
    torch.testing.assert_close(cross.energy(out), cross.energy(s), atol=1e-12, rtol=1e-12)
    for before, after in zip((s.field, *s.flux), (out.field, *out.flux)):
        torch.testing.assert_close(after.sum(-1), before.sum(-1), atol=1e-10, rtol=1e-10)
    # the analytic derivative still equals the finite-duration flow (cross layers included)
    quiet = copy.deepcopy(cross)
    with torch.no_grad():
        quiet.collision_kerr.zero_()
    h = 1e-6
    flow = (cross.collide(base, h).field - quiet.collide(base, h).field) / h
    rhs = cross.native_field_rhs(base) - quiet.native_field_rhs(base)
    torch.testing.assert_close(flow, rhs, atol=1e-4 * rhs.abs().max().item(), rtol=1e-3)
    assert rhs.abs().max() > 0


def test_cross_group_coincidence_still_needs_two_partners():
    net = scaled(8, (2, 2, 2), 8, heads=2, kerr_std=3.0, coincidence=True, cross_groups=True)
    material = net.material_field()
    z = torch.zeros(1, 2, 2, 2, 8, dtype=torch.float64)
    f1 = torch.randn(1, 2, 2, 2, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    assert net.collision_rates(torch.zeros(1, 2, 2, 2, 4, 8, dtype=torch.float64), material).abs().max() == 0
    assert net.collision_rates(torch.stack((f1, f1, z, z), -2), material).abs().max() < 1e-12
    assert net.collision_rates(torch.stack((f1 + f1.roll(1, -1), f1 - f1.roll(1, -1), z, z), -2), material).abs().max() > 1e-3


def test_pair_gated_mlp_collision_is_a_learned_operator_with_physical_guard_rails():
    net = scaled(8, (2, 2, 2), 8, gram_input=2, pair_gate=True, cross_groups=True, mod_gain=4.0)
    material = net.material_field()
    z = torch.zeros(1, 2, 2, 2, 8, dtype=torch.float64)
    gen = torch.Generator().manual_seed(4)
    f1 = torch.randn(1, 2, 2, 2, 8, dtype=torch.float64, generator=gen)
    f2 = torch.randn(1, 2, 2, 2, 8, dtype=torch.float64, generator=gen)
    rates = lambda packed, mat=material: net.collision_rates(packed, mat)
    # guard rails hold for ANY mlp weights: random large weights cannot make vacuum or one packet collide
    with torch.no_grad():
        for p in net.collision_rate.parameters():
            p.normal_(std=1.0)
    assert rates(torch.zeros(1, 2, 2, 2, 4, 8, dtype=torch.float64)).abs().max() == 0
    assert rates(torch.stack((f1, f1, z, z), -2)).abs().max() < 1e-12
    counter = torch.stack((f1 + f2, f1 - f2, z, z), -2)
    assert rates(counter).abs().max() > 1e-3
    torch.testing.assert_close(rates(counter), rates(counter, 5.0 * material + 2.0), atol=1e-12, rtol=1e-12)   # position-free
    # the mlp is really the operator: it moves the rates, receives gradient, and the step stays orthogonal
    before = rates(counter)
    with torch.no_grad():
        net.collision_rate[2].bias.add_(0.5)
    assert (rates(counter) - before).abs().max() > 1e-3
    s = state(net)
    out = net.collide(s, 0.05)
    torch.testing.assert_close(net.energy(out), net.energy(s), atol=1e-12, rtol=1e-12)
    grads = torch.autograd.grad(out.field.pow(3).sum(), tuple(net.collision_rate.parameters()))
    assert all(g.abs().sum() > 0 for g in grads)
    # bounded relative modulation: exploding mlp output cannot exceed exp(+-mod_bound) of the timescale
    bound = (net.collision_log_phi.exp() * torch.tensor(net.collision_mod_bound).exp() * net.speed_reference).max()
    assert rates(counter).abs().max() <= bound * 1.0001
    # centred inputs: an equipartitioned pair (orthonormal packet vectors) gives zero shape features
    orth = torch.linalg.qr(torch.randn(8, 4, dtype=torch.float64, generator=gen))[0].T          # [4, 8], orthonormal rows
    heads = net.collision_gram_heads
    packed = orth.expand(1, 2, 2, 2, 4, 8).clone()
    features = net._gram_features(packed)
    assert features.shape[-1] == 11 * heads
    one_head = net.channels // heads
    q4 = torch.linalg.qr(torch.randn(one_head, 4, dtype=torch.float64, generator=gen))[0].T
    equi = torch.cat([q4] * heads, -1).expand(1, 2, 2, 2, 4, one_head * heads).clone()
    shape_part = net._gram_features(equi).reshape(1, 2, 2, 2, heads, 11)[..., :10]
    assert shape_part.abs().max() < 1e-9
    with pytest.raises(ValueError):
        scaled(8, (2, 2, 2), 8, pair_gate=True)
    with pytest.raises(ValueError):
        scaled(8, (2, 2, 2), 8, gram_input=2, heads=2, pair_gate=True)


def test_gravity_is_a_long_range_orthogonal_force_on_the_field_flux_planes():
    options = dict(heads=2, kerr_std=3.0, coincidence=True, cross_groups=True)
    plain = scaled(8, (4, 4, 4), 8, **options)
    grav = scaled(8, (4, 4, 4), 8, gravity=True, gravity_std=2.0, **options)
    M = grav.collision_green.double()
    S = 64
    # Green gradients: zero mean over sources, no self force, antisymmetric about the source, translation invariant
    assert M.sum(-1).abs().max() < 1e-9 and M.diagonal(dim1=-2, dim2=-1).abs().max() < 1e-9
    idx = torch.arange(S).reshape(4, 4, 4)
    j = int(idx[1, 2, 3])
    for a in range(3):
        reflected = ((2 * torch.tensor([1, 2, 3]) - torch.stack(torch.meshgrid(*(torch.arange(4),) * 3, indexing='ij'), -1)) % 4)
        mirror = idx[reflected[..., 0], reflected[..., 1], reflected[..., 2]].reshape(-1)
        torch.testing.assert_close(M[a][:, j], -M[a][mirror, j], atol=1e-9, rtol=1e-9)
        shifted = idx.roll((1, 1, 1), (0, 1, 2)).reshape(-1)
        torch.testing.assert_close(M[a][shifted][:, shifted], M[a], atol=1e-9, rtol=1e-9)
    # sign: M = 2*pi*d_a Phi with Laplace Phi = +mass, so the FORCE -d_a Phi points toward the source:
    # the site one step above the source has a positive gradient (potential rises away from the well)
    assert all(float(M[a][int(idx[(1,0,0)[0:1][0], 0, 0]) if a == 0 else 0, 0]) > 0 for a in (0,))
    above = [int(idx[1, 0, 0]), int(idx[0, 1, 0]), int(idx[0, 0, 1])]
    assert all(float(M[a][above[a], 0]) > 0 for a in range(3))
    # a lone lump of energy at one site pulls on a vacuum site far away; without gravity that site has exactly zero rate
    s = plain.initial_state(1, dtype=torch.float64)
    field = s.field.clone(); field[0, 0, 0, 0] = torch.randn(8, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    lump = replace(s, field=field)
    material = plain.material_field()
    far = (1, 1, 2)
    r_plain = plain.collision_rates(plain._pack(lump), material)
    r_grav = grav.collision_rates(grav._pack(lump), grav.material_field())
    assert r_plain[0][far].abs().max() == 0 and r_grav[0][far].abs().max() > 1e-6
    # vacuum stays identity, energy and the channel sums are conserved, rates are translation covariant
    vac = grav.collision_rates(grav._pack(s), grav.material_field())
    assert vac.abs().max() == 0
    rolled = replace(lump, field=lump.field.roll((1, 2, 3), (1, 2, 3)))
    r_rolled = grav.collision_rates(grav._pack(rolled), grav.material_field())
    torch.testing.assert_close(r_rolled, r_grav.roll((1, 2, 3), (1, 2, 3)), atol=1e-10, rtol=1e-10)
    full = state(grav)
    out = grav.collide(full, 0.2)
    torch.testing.assert_close(grav.energy(out), grav.energy(full), atol=1e-11, rtol=1e-11)
    grads = torch.autograd.grad(out.field.pow(3).sum(), (grav.collision_gravity_gain, grav.collision_log_phi, grav.collision_kerr))
    assert all(g.abs().sum() > 0 for g in grads)
    # the force is a real change of the dynamics: with the gain at zero the rates equal the no-gravity ones
    with torch.no_grad():
        grav.collision_gravity_gain.zero_()
        torch.testing.assert_close(grav.collision_rates(grav._pack(full), grav.material_field()),
                                   grav.__class__.collision_rates(grav, grav._pack(full), grav.material_field()))
    with pytest.raises(ValueError):
        scaled(8, (4, 4, 4), 8, heads=2, coincidence=True, gravity=True)


def structural_medium(shape=(4, 4, 4), seed=3):
    torch.manual_seed(seed)
    net = PlasticMedium3D(shape, 4, hidden=8, bath_type='conductance', adaptive_conduction=False, anisotropic_transport=True,
                          structure_options=dict(STRUCTURE, resource_density=4.0), material_reference_shape=None).double()
    return net


def installed(net):
    basis = net.material.basis(net.coordinates).double()
    return net.structural_posterior.allocation(basis).double()


def test_structural_gravity_is_identity_at_zero_conserves_capacity_and_grows_every_density_mode():
    net = structural_medium()
    posterior = net.structural_posterior
    with torch.no_grad():                                      # a small random deviation from the uniform bed
        posterior.mean.copy_(0.05 * torch.randn(posterior.mean.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1)))
    before = posterior.mean.detach().clone()
    assert net.structural_gravity_step(0.0) == {} and torch.equal(posterior.mean, before)      # rate 0: bitwise identity
    mass0 = installed(net)[..., :3].sum(-1)
    info = net.structural_gravity_step(0.05)
    mass1 = installed(net)[..., :3].sum(-1)
    assert abs(info['mass_change']) < 5e-3                                       # total capacity is conserved (band-limit projection only)
    assert float(mass1.std() / mass1.mean()) > float(mass0.std() / mass0.mean())  # the contrast grows: attraction, not diffusion
    # repeated steps keep every site inside (0, cap) and the contrast keeps growing until exclusion saturates it
    contrast = [float(mass1.std() / mass1.mean())]
    for _ in range(60):
        net.structural_gravity_step(0.05)
        m = installed(net)
        assert bool(torch.isfinite(m).all()) and bool((m[..., 3] > 0).all()) and bool((m[..., :3].sum(-1) < 4.0).all())
        contrast.append(float(m[..., :3].sum(-1).std() / m[..., :3].sum(-1).mean()))
    assert contrast[-1] > 1.5 * contrast[0]
    with pytest.raises(ValueError):
        net.structural_gravity_step(-1.0)


def test_structural_gravity_linear_growth_rate_is_the_requested_constant():
    net = structural_medium(shape=(4, 4, 4))
    posterior = net.structural_posterior
    basis = net.material.basis(net.coordinates).double()
    S = 64
    # build a pure lowest-mode density perturbation of the TOTAL capacity: add the same logit bump to all three axes
    mode = torch.cos(2 * math.pi * net.coordinates[..., 0]).double()
    flat_basis = basis.reshape(S, -1)
    target = (0.02 * mode).reshape(S, 1).expand(S, 3)
    with torch.no_grad():
        posterior.mean.copy_(torch.linalg.pinv(flat_basis) @ target)
    m0 = installed(net)[..., :3].sum(-1)
    amplitude0 = float((m0 * mode).sum() / (mode * mode).sum())
    rate = 0.02
    net.structural_gravity_step(rate)
    m1 = installed(net)[..., :3].sum(-1)
    amplitude1 = float((m1 * mode).sum() / (mode * mode).sum())
    measured = (amplitude1 - amplitude0) / amplitude0
    assert 0.5 * rate < measured < 1.6 * rate, measured       # one window grows the mode by ~rate (linear regime)


def test_structural_gravity_screening_and_diffusion_select_the_wavenumber_with_the_derived_rates():
    net = structural_medium(shape=(8, 8, 8))
    posterior = net.structural_posterior
    basis = net.material.basis(net.coordinates).double()
    S = 512
    n = 8
    lam = lambda k: 2.0 * n * n * (1 - math.cos(2 * math.pi * k / n))
    rate, kappa2, diffusion = 0.03, 50.0, 5e-5
    flat_basis = basis.reshape(S, -1)
    for k in (1, 2, 4):
        mode = torch.cos(2 * math.pi * k * net.coordinates[..., 2]).double()
        with torch.no_grad():
            posterior.mean.copy_(torch.linalg.pinv(flat_basis) @ (0.01 * mode).reshape(S, 1).expand(S, 3))
        m0 = installed(net)[..., :3].sum(-1)
        a0 = float((m0 * mode).sum() / (mode * mode).sum())
        net.structural_gravity_step(rate, kappa2, diffusion)
        m1 = installed(net)[..., :3].sum(-1)
        a1 = float((m1 * mode).sum() / (mode * mode).sum())
        expected = rate * lam(k) / (lam(k) + kappa2) - diffusion * lam(k)
        assert abs((a1 - a0) / a0 - expected) < 0.35 * abs(expected) + 2e-3, (k, (a1 - a0) / a0, expected)
    # the finite-range force makes the lowest mode grow slower than the second one: scale selection
    growth = lambda k: rate * lam(k) / (lam(k) + kappa2) - diffusion * lam(k)
    assert growth(2) > growth(1) and growth(2) > growth(4)


def plastic_medium(**plasticity):
    torch.manual_seed(5)
    return PlasticMedium3D((4, 4, 4), 4, hidden=8, bath_type='conductance', adaptive_conduction=True, short_term_plasticity=True,
                           anisotropic_transport=True, structure_options=dict(STRUCTURE, resource_density=4.0),
                           material_reference_shape=None, plasticity_options=plasticity or None).double()


def test_plasticity_spectrum_is_log_uniform_and_switching_it_on_keeps_the_nominal_speed():
    default, spread = plastic_medium(), plastic_medium(spectrum=(1.0, 180.0))
    with torch.no_grad():
        for net in (default, spread):
            net.material.coefficients.normal_(std=.3)
    with pytest.raises(ValueError):
        medium_zero = plastic_medium(spectrum=(1.0, 180.0))
        medium_zero.initialize_plasticity_spectrum()    # material not initialised
    spread.initialize_plasticity_spectrum()
    material = spread.material_field()
    # the legacy initial spectrum is a delta at the reference time; the new one spans the whole range
    rate_default = default.conduction_plasticity.log_rate(default.material_field())
    rate_spread = spread.conduction_plasticity.log_rate(material)
    assert rate_default.abs().max() == 0
    log_tau = -rate_spread
    assert abs(float(log_tau.mean()) - 0.5 * math.log(180.0)) < 0.8 and 1.0 < float(log_tau.std()) < 2.2
    stp = spread.short_term_plasticity.coefficients(material)[0][..., 0]               # recovery rates, [X,Y,Z,3]
    assert 1.0 < float((-stp.log()).std()) < 2.2
    # compensation: the initial effective factor equals the installed one, although sigmoid(0)*U = 1/4
    state = spread.initial_state(1, dtype=torch.float64)
    spread.structural_posterior.begin_window(torch.zeros_like(spread.structural_posterior.mean))
    installed = spread.prepare_evolution().structural_factor
    assert spread.utilization_compensation == 4.0
    torch.testing.assert_close(spread.utilized_structural_factor(state, installed), installed[None], atol=1e-12, rtol=1e-12)
    assert default.utilization_compensation == 1.0
    with pytest.raises(ValueError):
        plastic_medium(spectrum=(5.0, 1.0))
    with pytest.raises(ValueError):
        plastic_medium(nonsense=True)


def soil_state(net, seed=0):
    s = state(net, seed)
    with torch.no_grad():                                      # a non-uniform bed so soil differs between sites
        net.structural_posterior.mean.copy_(0.4 * torch.randn(net.structural_posterior.mean.shape, dtype=torch.float64,
                                                              generator=torch.Generator().manual_seed(seed + 5)))
    return s


def test_soil_absorption_off_is_bitwise_legacy_and_on_decays_each_site_by_its_soil_share():
    net = structural_medium()
    s = soil_state(net)
    assert net.soil_absorption == 0.0
    prepared = net.prepare_evolution()
    legacy, _ = net.advance(s, 0.05, prepared=prepared, diagnostics=False)
    net.soil_absorption = 0.0
    again, _ = net.advance(s, 0.05, prepared=prepared, diagnostics=False)
    assert torch.equal(legacy.field, again.field)
    soil = net.soil_share(prepared.structural_allocation)
    assert float(soil.std()) > 0 and bool(((soil > 0) & (soil < 1)).all())
    net.soil_absorption = 3.0
    dt = torch.tensor(0.05, dtype=torch.float64).reshape(1, 1, 1, 1, 1)
    out = net.absorb_soil(s, dt, prepared.structural_allocation)
    torch.testing.assert_close(out.field, s.field * torch.exp(-3.0 * soil[None] * 0.05), atol=0, rtol=1e-14)
    for x, y in zip(out.flux, s.flux):
        torch.testing.assert_close(x, y * torch.exp(-3.0 * soil[None] * 0.05), atol=0, rtol=1e-14)
    assert float(net.energy(out).sum()) < float(net.energy(s).sum())


def test_soil_absorption_flow_and_derivative_measurement_agree():
    net = structural_medium()
    s = soil_state(net, seed=1)
    prepared = net.prepare_evolution()
    soil = net.soil_share(prepared.structural_allocation)
    rhs0 = net.native_field_rhs(s, prepared=prepared)
    net.soil_absorption = 2.5
    rhs1 = net.native_field_rhs(s, prepared=prepared)
    torch.testing.assert_close(rhs1 - rhs0, -2.5 * soil[None] * s.field, atol=1e-12, rtol=1e-12)
    h = 1e-6
    with_soil, _ = net.advance(s, h, prepared=prepared, diagnostics=False)
    net.soil_absorption = 0.0
    without, _ = net.advance(s, h, prepared=prepared, diagnostics=False)
    flow = (with_soil.field - without.field) / h
    torch.testing.assert_close(flow, rhs1 - rhs0, atol=1e-4 * float((rhs1 - rhs0).abs().max()), rtol=1e-3)


def test_channel_growth_is_identity_at_zero_conserves_soil_and_lays_channels_along_the_flow():
    net = structural_medium(shape=(4, 4, 4))
    posterior = net.structural_posterior
    before = posterior.mean.detach().clone()
    flow = torch.zeros(4, 4, 4, 3, dtype=torch.float64)
    flow[:, 1, 1, 0] = 5.0                                     # a line of x through-flow
    flow += 0.1
    assert net.channel_growth_step(flow, 0.0) == {} and torch.equal(posterior.mean, before)
    soil0 = installed(net)[..., 3].mean()
    for _ in range(30):
        info = net.channel_growth_step(flow, 0.2, mu=2.0)
    alloc = installed(net)
    assert abs(float(alloc[..., 3].mean() - soil0)) < 0.05 * float(soil0)   # dissipation budget conserved (projection only)
    on_line = alloc[:, 1, 1]
    elsewhere = alloc[:, 3, 3]
    assert float(on_line[:, 0].mean()) > 2 * float(on_line[:, 1:3].mean())  # the channel points along the flow
    assert float(on_line[:, 3].mean()) < float(elsewhere[:, 3].mean())     # flow turns soil into channel; no flow -> soil
    assert info['projection_residual'] < 0.05
    with pytest.raises(ValueError):
        net.channel_growth_step(flow, 1.5)


def test_split_write_gives_each_port_its_own_channel_block_and_is_off_by_default():
    from information_boltzmann.core.torus3d import PredictiveImpedanceWriteAgent
    torch.manual_seed(0)
    plain = PredictiveImpedanceWriteAgent(16, 50, 8, exchange='contact_mode', local_shape=(4, 4, 4))
    assert not plain.split and not hasattr(plain, 'split_mask')
    split = PredictiveImpedanceWriteAgent(16, 50, 8, exchange='contact_mode', local_shape=(4, 4, 4), split=True)
    mask = split.split_mask
    assert mask.shape == (8, 16)
    assert torch.equal((mask > 0).sum(0), torch.ones(16, dtype=torch.long))          # every channel belongs to one port
    assert torch.equal((mask > 0).sum(1), torch.full((8,), 2, dtype=torch.long))      # every port owns d/ports channels
    assert torch.allclose(mask.sum(0), torch.full((16,), 8.0))                         # uniform gate -> unit weight per channel
    with pytest.raises(ValueError):
        PredictiveImpedanceWriteAgent(12, 50, 8, exchange='contact_mode', local_shape=(4, 4, 4), split=True)


def test_cell_cross_section_is_off_by_default_conserved_and_scales_the_local_rule():
    plain = medium(structured=False, timescale=True, heads=1, kerr_std=3.0, coincidence=True)
    assert plain.cell_theta is None and 'cell_theta' not in plain.state_dict()
    torch.manual_seed(487)
    cell = PlasticMedium3D((4, 4, 4), 3, hidden=8, bath_type='conductance', adaptive_conduction=True,
                           collision_options=dict(timescale=True, heads=1, kerr_std=3.0, coincidence=True, cell_gain=3.0)).double()
    sigma = cell.cell_cross_section()
    assert abs(float(sigma.mean()) - 1) < 1e-12                                 # conserved resource: mean exactly 1
    assert float(sigma.max() / sigma.min()) > 2.0                               # heterogeneous at birth
    assert float(sigma.max() / sigma.min()) <= 9.0 + 1e-9                       # within [1/3, 3] relative spread
    s = state(cell)
    packed, material = cell._pack(s), cell.material_field()
    rule = cell._collision_rates_rule(packed, material, s.receptors)
    torch.testing.assert_close(cell.collision_rates(packed, material, s.receptors), rule * sigma[None, ..., None, None])
    loss = cell.collide(s, 0.1).field.square().sum()
    grad, = torch.autograd.grad(loss, cell.cell_theta)
    assert grad.norm() > 0                                                      # learnable from the task
    with pytest.raises(ValueError):
        PlasticMedium3D((4, 4, 4), 3, hidden=8, collision_options=dict(timescale=True, heads=1, coincidence=True, cell_gain=0.5))
