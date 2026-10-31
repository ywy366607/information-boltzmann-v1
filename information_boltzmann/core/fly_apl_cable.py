"""Full supplied APL cable forest with local synaptic input and graded output.

No branch/ROI pooling is used in the physical state. The old two neuron slots
are area-weighted readout coordinates only. KC bouton cable state remains a
separate, unresolved mechanism; this module does not claim to reconstruct it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd.function import once_differentiable


APL_CABLE_PARAMETER_NAMES = ('apl_log_axial',)
APL_CABLE_ID_NAMES = ('apl_cable_node_ids', 'apl_cable_digest')
APL_CABLE_STATE_FIELDS = ('apl_voltage', 'apl_ge', 'apl_gi', 'apl_packets')
APL_CABLE_VERSION = 1


class CableGeometry:
    """Sparse finite-volume diffusion, exact direct solve to FP64 residual tolerance."""

    def __init__(self, area, first, second, conductance, owner):
        self.area = np.asarray(area, dtype=np.float64)
        self.first = np.asarray(first, dtype=np.int64)
        self.second = np.asarray(second, dtype=np.int64)
        self.conductance = np.asarray(conductance, dtype=np.float64)
        self.owner = np.asarray(owner, dtype=np.int64)
        if np.any(self.area <= 0) or np.any(self.conductance <= 0):
            raise ValueError('Cable capacities and axial conductances must be positive')
        if np.any(self.owner[self.first] != self.owner[self.second]):
            raise ValueError('Axial connection crosses distinct native cells')
        self._cached = None

    def factor(self, log_axial):
        from scipy import sparse
        from scipy.sparse.linalg import splu
        values = tuple(log_axial.detach().double().cpu().tolist())
        if not all(np.isfinite(values)):
            raise ValueError('Nonfinite native axial material parameter')
        if self._cached is not None and self._cached[0] == values:
            return self._cached[1:]
        weight = self.conductance * np.exp(values)[self.owner[self.first]]
        diagonal = self.area + np.bincount(np.r_[self.first, self.second],
            weights=np.r_[weight, weight], minlength=len(self.area))
        index = np.arange(len(self.area))
        matrix = sparse.csc_matrix((np.r_[diagonal, -weight, -weight],
            (np.r_[index, self.first, self.second], np.r_[index, self.second, self.first])),
            shape=(len(index), len(index)))
        factor = splu(matrix, permc_spec='MMD_AT_PLUS_A')
        self._cached = values, matrix, factor, weight
        return matrix, factor, weight

    def __deepcopy__(self, memo):
        # Sparse LU is an execution cache, never an individual's state.
        result = type(self)(self.area.copy(), self.first.copy(), self.second.copy(),
                            self.conductance.copy(), self.owner.copy())
        memo[id(self)] = result
        return result

    @staticmethod
    def solve_checked(matrix, factor, rhs):
        result = factor.solve(rhs)
        error = matrix @ result
        error -= rhs
        np.abs(error, out=error)
        residual = float(error.max(initial=0.))
        scale = max(1., float(rhs.max(initial=0.)), -float(rhs.min(initial=0.)))
        if not np.isfinite(result).all() or residual > 1e-9 * scale:
            raise FloatingPointError('Native cable direct solve did not meet residual contract')
        return result

    def solve_owned_rhs(self, matrix, factor, rhs):
        """Solve a private RHS that a specialized geometry may reuse as scratch."""
        return self.solve_checked(matrix, factor, rhs)


def _release_idle_solver_cache(voltage):
    """Release unused CUDA reservations at a memory-constrained host-solve boundary.

    In particular WDDM can consume host commit for device reservations. Keeping
    idle allocator blocks while requesting a large CPU solve can then fail a
    NumPy allocation before CUDA's own OOM/retry path ever runs. Live tensors,
    optimizer state and physical history are never released here.
    """
    if not voltage.is_cuda or torch.cuda.is_current_stream_capturing():
        return
    idle = torch.cuda.memory_reserved(voltage.device) - torch.cuda.memory_allocated(voltage.device)
    free, total = torch.cuda.mem_get_info(voltage.device)
    if idle > free and free < total // 8:
        torch.cuda.empty_cache()


class _Diffuse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, log_axial, geometry, record_grad):
        _release_idle_solver_cache(voltage)
        matrix, factor, weight = geometry.factor(log_axial)
        # Each RHS is independent. Transfer one row before converting to FP64:
        # converting on CUDA first needlessly duplicates the entire field there.
        keep_result = record_grad and ctx.needs_input_grad[1]
        fused = (not keep_result and voltage.dtype == torch.float32 and hasattr(geometry, 'diffuse_rows'))
        # (the fused path computes in the geometry's own precision, float32 for the cell-wise forests)
        if fused:
            # bitwise the same rows as the loop below, in one parallel pass per row
            solved = geometry.diffuse_rows(matrix, factor, voltage.detach().cpu().numpy())
            output = torch.from_numpy(solved).to(voltage.device)
        else:
            output = torch.empty_like(voltage)
        result = np.empty((voltage.shape[1], voltage.shape[0]), np.float64) if keep_result else None
        for row in (() if fused else range(voltage.shape[0])):
            rhs = voltage[row].detach().cpu().numpy().astype(np.float64, copy=True)[:, None]
            rhs *= geometry.area[:, None]
            solved = geometry.solve_owned_rhs(matrix, factor, rhs)
            output[row].copy_(torch.as_tensor(solved[:, 0], device=voltage.device, dtype=voltage.dtype))
            if keep_result:
                result[:, row] = solved[:, 0]
        ctx.geometry, ctx.matrix, ctx.factor, ctx.weight = geometry, matrix, factor, weight
        # Use the autograd storage contract: a raw NumPy attribute would escape
        # saved-tensor/checkpoint hooks and retain every spatial activation.
        ctx.save_for_backward(*((torch.from_numpy(result),) if keep_result else ()))
        ctx.log_device, ctx.log_dtype, ctx.count = log_axial.device, log_axial.dtype, log_axial.numel()
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        _release_idle_solver_cache(gradient)
        g = ctx.geometry
        dv = torch.empty_like(gradient) if ctx.needs_input_grad[0] else None
        material = np.zeros(ctx.count, np.float64) if ctx.needs_input_grad[1] else None
        saved_result = ctx.saved_tensors[0].numpy() if material is not None else None
        for row in range(gradient.shape[0]):
            rhs = gradient[row].detach().cpu().numpy().astype(np.float64, copy=True)[:, None]
            adjoint = g.solve_owned_rhs(ctx.matrix, ctx.factor, rhs)[:, 0]
            if material is not None:
                # Bound gather temporaries without dropping edges or changing
                # the physical solve. This block size is execution metadata.
                for start in range(0, len(g.first), 65536):
                    block = slice(start, start + 65536)
                    first, second = g.first[block], g.second[block]
                    local = -ctx.weight[block] * (adjoint[first] - adjoint[second])
                    local *= saved_result[first, row] - saved_result[second, row]
                    material += np.bincount(g.owner[first], weights=local, minlength=ctx.count)
            if dv is not None:
                adjoint *= g.area
                dv[row].copy_(torch.as_tensor(adjoint, device=gradient.device, dtype=gradient.dtype))
        dm = (torch.as_tensor(material, device=ctx.log_device, dtype=ctx.log_dtype)
              if material is not None else None)
        return dv, dm, None, None


def diffuse(voltage, log_axial, geometry):
    return _Diffuse.apply(voltage, log_axial, geometry, torch.is_grad_enabled())


def attach_apl_cable(model, geometry_path, *, saved=None, metadata_only=False):
    path = Path(geometry_path)
    # Route construction indexes these arrays per contact. Materialize once:
    # repeated NpzFile indexing would re-inflate a whole array for every site.
    with np.load(path, allow_pickle=False) as packed:
        data = {key: packed[key] for key in packed.files}
    digest = np.frombuffer(hashlib.sha256(path.read_bytes()).digest(), dtype=np.uint8).copy()
    device = model.edge_pre_i.device

    def buffer(name, value, dtype=torch.long, persistent=False):
        model.register_buffer(name, torch.as_tensor(value, dtype=dtype, device=device), persistent=persistent)

    def segments(prefix, lag, target):
        order = np.argsort(target, kind='stable')
        unique, counts = np.unique(np.asarray(target)[order], return_counts=True)
        for key, value in [('order', order), ('target', unique), ('counts', counts)]:
            buffer(f'{prefix}_segment_{key}_{lag}', value)

    buffer('apl_cable_node_ids', data['node_ids'], persistent=True)
    buffer('apl_cable_digest', digest, torch.uint8, persistent=True)
    for name in APL_CABLE_ID_NAMES:
        if saved is not None and 'apl_log_axial' in saved:
            if name not in saved or not torch.equal(saved[name].cpu(), getattr(model, name).cpu()):
                raise ValueError('APL cable anatomical identity/digest mismatch: ' + name)
    buffer('apl_cable_index', data['apl_index'])
    buffer('aplc_owner', data['node_owner'])
    buffer('aplc_area', data['area'], torch.float32)
    totals = np.bincount(data['node_owner'], weights=data['area'], minlength=len(data['apl_index']))
    buffer('aplc_area_fraction', data['area'] / totals[data['node_owner']], torch.float32)
    buffer('aplc_input_density', totals[data['node_owner']] / data['area'], torch.float32)
    # The supplied morphology is a rooted forest. Use the same exact bounded
    # tree elimination as KC/DAN instead of general sparse-LU workspace.
    # Import at attachment time: the shared forest class derives from the
    # CableGeometry interface defined above.
    from .fly_kc_terminals import CellwiseGeometry
    model.apl_geometry = CellwiseGeometry(data['area'], data['cable_u'], data['cable_v'], data['cable_g'], data['node_owner'])
    for route in ('in_e', 'in_i', 'out_i', 'out_mb'):
        for lag in range(1, 5):
            selected = data[route + '_delay'] == lag
            for key in ('edge', 'pre', 'post', 'node', 'pre_node', 'fraction'):
                buffer(f'aplc_{route}_{key}_{lag}', data[route + '_' + key][selected],
                       torch.float32 if key == 'fraction' else torch.long)
            if route != 'out_mb':
                segments('aplc_' + route, lag, data[route + ('_node' if route.startswith('in') else '_post')][selected])
            if route.startswith('in'):
                pre_node = data[route + '_pre_node'][selected]
                slots = np.flatnonzero(pre_node >= 0)
                buffer(f'aplc_{route}_local_slots_{lag}', slots)
                buffer(f'aplc_{route}_local_nodes_{lag}', pre_node[slots])
    buffer('aplc_body_counts', np.bincount(data['node_owner']))
    model.apl_log_axial = nn.Parameter(torch.zeros(len(data['apl_index']), device=device))
    if saved is not None and 'apl_log_axial' in saved:
        model.load_state_dict({'apl_log_axial': saved['apl_log_axial']}, strict=False)
    if not metadata_only:
        if not getattr(model, 'mb_terminal_enabled', False):
            raise ValueError('APL cable needs the measured MB axonal projection')
        for channel in ('e', 'i'):
            edge = torch.as_tensor(data[f'verify_{channel}_edge'], device=device)
            for end in ('pre', 'post'):
                actual = getattr(model, f'edge_{end}_{channel}').index_select(0, edge).cpu().numpy()
                if not np.array_equal(actual, data[f'verify_{channel}_{end}']):
                    raise ValueError('APL contact cache does not match actual edge order')
            # Every nonzero incoming/outgoing edge needs measured location.
            pre, post = data[f'verify_{channel}_pre'], data[f'verify_{channel}_post']
            used_in = np.unique(data[f'in_{channel}_edge'])
            incoming = data[f'verify_{channel}_edge'][np.isin(post, data['apl_index'])]
            outgoing = data[f'verify_{channel}_edge'][np.isin(pre, data['apl_index'])]
            missing = set(incoming) - set(used_in)
            if channel == 'i':
                missing |= set(outgoing) - set(np.unique(data['out_i_edge']))
            elif len(outgoing):
                raise ValueError('Unexpected excitatory APL output')
            missing = torch.tensor(sorted(missing), device=device, dtype=torch.long)
            if missing.numel() and bool((getattr(model, f'edge_weight_{channel}')[missing] != 0).any()):
                raise ValueError('Unlocated nonzero native APL edge; cannot invent a spatial route')
        gain_lookup = {int(index): i for i, index in enumerate(model.graded_index.cpu())}
        buffer('aplc_gain_index', [gain_lookup[int(index)] for index in data['apl_index']])
        kc_lookup = {int(index): i for i, index in enumerate(model.mb_terminal_kc_index.cpu())}
        kc_indices = model.mb_terminal_kc_index.cpu().tolist()
        source_ids = model.mb_terminal_source_body_ids.cpu().tolist()
        mb_slots = [source_ids.index(int(body)) for body in data['node_ids'][np.r_[0, np.flatnonzero(np.diff(data['node_owner'])) + 1], 0]]
        buffer('aplc_mb_source_slot', mb_slots)
        for lag in range(1, 5):
            # Recover the exact previous per-edge MB coupling, distributing it
            # over measured local APL output sites without changing pair totals.
            weight_by_pair = {}
            for p, q, w in zip(getattr(model, f'mbt_pre_{lag}').cpu().tolist(),
                               getattr(model, f'mbt_post_{lag}').cpu().tolist(),
                               getattr(model, f'mbt_gaba_weight_{lag}').cpu().tolist()):
                if p in data['apl_index']:
                    weight_by_pair[p, kc_indices[q % len(kc_lookup)]] = w
            pre = getattr(model, f'aplc_out_mb_pre_{lag}').cpu().tolist()
            post = getattr(model, f'aplc_out_mb_post_{lag}').cpu().tolist()
            node = getattr(model, f'aplc_out_mb_node_{lag}').cpu().numpy()
            mapped = [(mb_slots[int(data['node_owner'][n])] * len(kc_lookup) + kc_lookup[q]) for n, q in zip(node, post)]
            weights = [weight_by_pair.get((p, q), 0.) for p, q in zip(pre, post)]
            covered = set(zip(pre, post))
            if any(w != 0 and key not in covered for key, w in weight_by_pair.items()):
                raise ValueError('Unlocated nonzero APL axonal edge')
            buffer(f'aplc_mb_post_{lag}', mapped)
            buffer(f'aplc_mb_weight_{lag}', weights, torch.float32)
            segments('aplc_mb', lag, np.asarray(mapped, dtype=np.int64))
    model.apl_output_enabled = model.apl_axial_enabled = True
    model.apl_cable_manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
    return dict(model.apl_cable_manifest)


def cable_state(model, current):
    owner, index = model.aplc_owner, model.apl_cable_index
    values = []
    for field, old in zip(APL_CABLE_STATE_FIELDS[:3], (current.h, current.ge, current.gi)):
        value = getattr(current, field, old.new_empty(0))
        if not value.numel():
            value = old.index_select(1, index).index_select(1, owner)
        if value.shape != (current.h.shape[0], owner.numel()):
            raise ValueError('Nonempty APL cable state has incompatible shape')
        values.append(value)
    packets = getattr(current, 'apl_packets', current.h.new_empty(0))
    if not packets.numel():
        packets = torch.cat([p.index_select(1, index).index_select(1, owner) for p in current.ring], 0)
    if packets.shape != (4, owner.numel()):
        raise ValueError('APL cable delayed packet shape mismatch')
    return *values, packets


def _arrival(model, current, packets, route, *, destination_nodes=False, local_inputs=None):
    result = current.h.new_zeros((1, model.aplc_owner.numel())) if destination_nodes else torch.zeros_like(current.h)
    for lag in range(1, 5):
        pre = getattr(model, f'aplc_{route}_pre_{lag}')
        if not pre.numel():
            continue
        node = getattr(model, f'aplc_{route}_node_{lag}')
        if destination_nodes:
            values = current.ring[lag - 1].index_select(1, pre)[0]
            if hasattr(model, 'kct_index'):
                values = values * (~torch.isin(pre, model.kct_index)).to(values.dtype)
            slots = getattr(model, f'aplc_{route}_local_slots_{lag}')
            nodes = getattr(model, f'aplc_{route}_local_nodes_{lag}')
            values = values.index_copy(0, slots, packets[lag - 1].index_select(0, nodes))
            target = node
        else:
            values = packets[lag - 1].index_select(0, node)
            target = getattr(model, f'aplc_{route}_post_{lag}')
        channel = route[-1]
        weight = getattr(model, 'edge_weight_' + channel).index_select(0, getattr(model, f'aplc_{route}_edge_{lag}'))
        value = values * weight * getattr(model, f'aplc_{route}_fraction_{lag}')
        result = result + _segmented(model, 'aplc_' + route, lag, value, result.shape[1]).unsqueeze(0)
    from .fly_native_runtime import owns_route, local_input
    if destination_nodes and route == 'in_e' and hasattr(model, 'kct_index') and not owns_route(model, 'kc_apl'):
        from .fly_kc_terminals import kc_arrival
        result = result + kc_arrival(model, current, 'apl')
    if destination_nodes and local_inputs:
        extra = local_input(local_inputs, 'apl_ge' if route == 'in_e' else 'apl_gi', result)
        if extra is not None:
            result = result + extra
    return result * model.aplc_input_density if destination_nodes else result


def _segmented(model, prefix, lag, values, size):
    """Fixed-order local sums, avoiding nondeterministic repeated-node atomics."""
    order = getattr(model, f'{prefix}_segment_order_{lag}')
    if not order.numel():
        return values.new_zeros(size)
    summed = torch.segment_reduce(values.index_select(0, order), 'sum',
        lengths=getattr(model, f'{prefix}_segment_counts_{lag}'))
    return values.new_zeros(size).index_copy(0, getattr(model, f'{prefix}_segment_target_{lag}'), summed)


def advance_apl_cable(model, current, source, *, base_rates=None, thresholds=None, conductance_gains=None,
                      local_inputs=None):
    """Advance local membrane physics; all arrivals consume OLD delayed packets."""
    if not hasattr(model, 'apl_cable_index'):
        return None
    if current.h.shape[0] != 1:
        raise ValueError('Current native cable interface requires one continuous individual')
    voltage, ge, gi, packets = cable_state(model, current)
    index, owner = model.apl_cable_index, model.aplc_owner

    def spread(value):
        return value.expand_as(current.h).index_select(1, index).index_select(1, owner)

    rates = base_rates if base_rates is not None else model.get_decay_rates()
    gains = conductance_gains if conductance_gains is not None else model.get_conductance_gains()
    ge = spread(rates[1]) * ge + (1 - spread(rates[1])) * _arrival(model, current, packets, 'in_e', destination_nodes=True, local_inputs=local_inputs)
    gi = spread(rates[2]) * gi + (1 - spread(rates[2])) * _arrival(model, current, packets, 'in_i', destination_nodes=True, local_inputs=local_inputs)
    excitatory, inhibitory = spread(gains[0]) * ge, spread(gains[1]) * gi
    total = -torch.log(spread(rates[0]).clamp(1e-5, 1 - 1e-7)) + excitatory + inhibitory
    alpha = torch.exp(-total.clamp(1e-5, 20.))
    drive = torch.zeros_like(voltage) if source is None else spread(source)
    local = alpha * voltage + (1 - alpha) / total.clamp_min(1e-5) * (excitatory * model.E_E + inhibitory * model.E_I + drive)
    voltage = diffuse(local, model.apl_log_axial, model.apl_geometry) if model.apl_axial_enabled else local
    threshold = thresholds if thresholds is not None else model.get_thresholds()
    floor = spread(model.graded_floor) if getattr(model, 'graded_floor', None) is not None else 0.
    gain = torch.exp(model.log_graded_gain.index_select(0, model.aplc_gain_index)).index_select(0, owner)
    voltage_drive = torch.relu((voltage - floor) / spread(threshold))
    extra = ()
    if getattr(model, 'apl_local_release_enabled', False):
        from .fly_apl_local_release import advance_apl_local_release
        released, *chemistry = advance_apl_local_release(model, current, voltage_drive)
        release = gain * released
        extra = tuple(chemistry)
    else:
        release = gain * voltage_drive
    if not model.apl_output_enabled:
        release = release * 0.
    out = _arrival(model, current, packets, 'out_i') if model.apl_output_enabled else torch.zeros_like(current.h)
    return (out, voltage, ge, gi, torch.cat((release, packets[:3]), 0), *extra)


def apl_mb_arrival(model, current):
    if not hasattr(model, 'apl_cable_index'):
        return None
    size = model.mb_terminal_source_index.numel() * model.mb_terminal_kc_index.numel()
    result = current.h.new_zeros(size)
    if not model.apl_output_enabled:
        return result
    packets = cable_state(model, current)[-1]
    for lag in range(1, 5):
        node = getattr(model, f'aplc_out_mb_node_{lag}')
        values = packets[lag - 1].index_select(0, node) * getattr(model, f'aplc_mb_weight_{lag}') * getattr(model, f'aplc_out_mb_fraction_{lag}')
        result = result + _segmented(model, 'aplc_mb', lag, values, size)
    return result


def apply_apl_readout(model, h, ge, gi, ring, step):
    if step is None:
        return h, ge, gi, ring
    _, voltage, local_ge, local_gi, packets = step[:5]

    def read(value):
        return torch.segment_reduce((value * model.aplc_area_fraction)[0], 'sum',
                                    lengths=model.aplc_body_counts).unsqueeze(0)

    index = model.apl_cable_index
    return (h.index_copy(1, index, read(voltage)), ge.index_copy(1, index, read(local_ge)),
            gi.index_copy(1, index, read(local_gi)),
            (ring[0].index_copy(1, index, read(packets[:1])), *ring[1:]))
