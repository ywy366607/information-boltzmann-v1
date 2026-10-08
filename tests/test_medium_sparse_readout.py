"""Finite read workspace equivalence, independent of training capability."""
import copy

import pytest
import torch

from information_boltzmann.core.readout_probes import PredictivePhysicalReadAgent


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_reader(dtype=torch.float64, *, dynamic=True, execution='support',
                radius=(.0866701528, .1300052255, .1733403057), generic=False):
    d = 16
    w = -torch.ones(d, dtype=torch.float64) / d ** .5
    w[0] += 1
    w /= w.norm()
    basis = (torch.eye(d, dtype=w.dtype) - 2 * w[:, None] * w[None, :])[:, 1:]
    if generic:
        basis = torch.linalg.qr(torch.randn(d, d, dtype=w.dtype))[0][:, 1:]
    reader = PredictivePhysicalReadAgent(
        (8, 8, 8), d, basis, heads=4, queries=4,
        aperture_type='compact_probes', port_radius=radius, dynamic=dynamic,
        coordinate_reflector=None if generic else w,
        compact_key_execution=execution).to(dtype=dtype)
    # Exercise the entire learned graph instead of fresh zero output branches.
    with torch.no_grad():
        for name, parameter in reader.named_parameters():
            if name == 'probe_coords':
                continue
            if parameter.ndim >= 2:
                parameter.normal_(std=.4 / parameter.shape[-1] ** .5)
            elif name.endswith('weight'):
                parameter.fill_(1.)
            else:
                parameter.normal_(std=.08)
    return reader


def tolerances(dtype):
    return dict(atol=3e-5, rtol=3e-5) if dtype == torch.float32 else dict(atol=3e-12, rtol=3e-12)


def execute(reader, source, upstream, motion, precision):
    field = source @ upstream
    feature, info = reader(field, precision, motion=motion if reader.dynamic else None,
                           return_diag=True)
    direction = torch.linspace(-.7, .9, feature.numel(), dtype=feature.dtype).reshape_as(feature)
    loss = (feature * direction).sum() + .17 * info['_read_action_complexity']
    gradients = torch.autograd.grad(
        loss, (field, source, upstream, motion, precision, *reader.parameters()),
        allow_unused=True)
    return feature, info, gradients


def assert_equivalent(fast, dense, dtype):
    source = torch.randn(2, 8, 8, 8, 16, dtype=dtype, requires_grad=True)
    upstream = torch.randn(16, 16, dtype=dtype, requires_grad=True)
    motion = torch.randn_like(source, requires_grad=True)
    precision = torch.rand(2, 16, dtype=dtype, requires_grad=True)
    reference_inputs = tuple(x.detach().clone().requires_grad_()
                             for x in (source, upstream, motion, precision))
    actual, ai, ag = execute(fast, source, upstream, motion, precision)
    expected, ei, eg = execute(dense, *reference_inputs)
    torch.testing.assert_close(actual, expected, **tolerances(dtype))
    assert ai.keys() == ei.keys()
    for key in ai:
        torch.testing.assert_close(ai[key], ei[key], **tolerances(dtype))
    assert len(ag) == len(eg)
    for a, e in zip(ag, eg):
        assert (a is None) == (e is None)
        if a is not None:
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, e, **tolerances(dtype))
    assert ag[3] is not None if fast.dynamic else ag[3] is None
    # Compact readers do not use shared precision as an information side path.
    assert ag[4] is None
    return ag


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('layout', ['birth', 'overlap', 'periodic'])
def test_sparse_reader_outputs_diagnostics_and_all_gradients(dtype, dynamic, layout):
    torch.manual_seed(1201)
    fast = make_reader(dtype, dynamic=dynamic)
    with torch.no_grad():
        if layout == 'overlap':
            fast.probe_coords.copy_(fast.probe_coords.new_tensor([.017, .499, .989]))
        elif layout == 'periodic':
            fast.probe_coords.add_(fast.probe_coords.new_tensor([-.216, .687, 1.119]))
    dense = copy.deepcopy(fast)
    dense.compact_key_execution = 'dense'
    gradients = assert_equivalent(fast, dense, dtype)
    names = tuple(dict(fast.named_parameters()))
    assert gradients[5 + names.index('probe_coords')].norm() > 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_exact_support_edges_follow_original_logeps_and_zero_mask(dtype, offset):
    torch.manual_seed(1202)
    fast = make_reader(dtype, radius=(.125, .1875, .25))
    with torch.no_grad():
        fast.probe_coords.copy_(fast.probe_coords.new_tensor([0., .0625, .25]))
        fast.probe_coords[..., 0].add_(offset * 4 * torch.finfo(dtype).eps)
    footprint = fast.footprint()
    if offset == 0:
        assert (footprint == 0).any()
    dense = copy.deepcopy(fast)
    dense.compact_key_execution = 'dense'
    assert_equivalent(fast, dense, dtype)


def test_support_is_recomputed_after_learned_coordinates_move():
    torch.manual_seed(1203)
    fast = make_reader()
    dense = copy.deepcopy(fast)
    dense.compact_key_execution = 'dense'
    original = (fast.footprint() > 0).clone()
    with torch.no_grad():
        change = torch.randn_like(fast.probe_coords) * .27
        fast.probe_coords.add_(change)
        dense.probe_coords.copy_(fast.probe_coords)
    assert not torch.equal(fast.footprint() > 0, original)
    assert_equivalent(fast, dense, torch.float64)


def test_generic_dense_coordinate_basis_is_compatible_with_sparse_keys():
    torch.manual_seed(1204)
    fast = make_reader(generic=True)
    assert fast.coordinate_reflector is None
    dense = copy.deepcopy(fast)
    dense.compact_key_execution = 'dense'
    assert_equivalent(fast, dense, torch.float64)


def test_execution_option_preserves_parameters_and_strict_checkpoint_loading():
    torch.manual_seed(1205)
    dense, sparse = make_reader(execution='dense'), make_reader()
    assert dense.state_dict().keys() == sparse.state_dict().keys()
    assert not any('capacity' in name or 'support' in name for name in sparse.state_dict())
    sparse.load_state_dict(dense.state_dict(), strict=True)
    dense.load_state_dict(sparse.state_dict(), strict=True)
    assert dense.compact_key_execution == 'dense'
    assert sparse.compact_key_execution == 'support'
    assert sparse.compact_key_capacity == 72
    footprint = sparse.footprint().reshape(4, 4, 512)
    assert (footprint > 0).any(1).sum(-1).tolist() == [32, 32, 32, 32]
    for key, value in sparse.state_dict().items():
        torch.testing.assert_close(value, dense.state_dict()[key], atol=0, rtol=0)


def test_dense_fallback_when_support_capacity_spans_the_grid():
    torch.manual_seed(1206)
    fast = make_reader(radius=(.5, .5, .5))
    assert fast.compact_key_capacity == fast.nodes
    dense = copy.deepcopy(fast)
    dense.compact_key_execution = 'dense'
    assert_equivalent(fast, dense, torch.float64)


@pytest.mark.parametrize('dynamic', [False, True])
def test_support_path_captures_fullgraph_and_preserves_reverse_gradients(dynamic):
    torch.manual_seed(1207)
    fast = make_reader(torch.float32, dynamic=dynamic)
    dense = copy.deepcopy(fast)
    dense.compact_key_execution = 'dense'
    compiled = torch.compile(fast, backend='aot_eager', fullgraph=True, dynamic=False)
    assert_equivalent(compiled, dense, torch.float32)


def test_invalid_execution_configuration_is_rejected():
    with pytest.raises(ValueError, match='dense or support'):
        make_reader(execution='unknown')
    with pytest.raises(ValueError, match='compact probes'):
        PredictivePhysicalReadAgent((2, 2, 2), 4, torch.eye(4)[:, 1:],
                                   compact_key_execution='support')


def test_native_support_path_preserves_forward_mode_field_and_motion_response():
    torch.manual_seed(1208)
    sparse = make_reader()
    dense = copy.deepcopy(sparse)
    dense.compact_key_execution = 'dense'
    field = torch.randn(1, 8, 8, 8, 16, dtype=torch.float64)
    motion = torch.randn_like(field)
    directions = (torch.randn_like(field), torch.randn_like(motion))
    results = []
    for reader in (sparse, dense):
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            feature, _ = reader(
                torch.autograd.forward_ad.make_dual(field, directions[0]),
                torch.ones(1, 16, dtype=field.dtype),
                motion=torch.autograd.forward_ad.make_dual(motion, directions[1]))
            results.append(torch.autograd.forward_ad.unpack_dual(feature))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, **tolerances(torch.float64))
