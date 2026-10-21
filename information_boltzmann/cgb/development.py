"""Executable candidate spatial development, with explicit unfitted laws.

Identity-conditioned gradient receptors guide persistent noisy growth cones.
No measured edges or adult coordinates are read. Geometry and occurrence are
developmental hypotheses, not inferred biological laws. Strength uses V3d only
after geometry has produced opportunities. Memory is finite cone slots plus
funded segments; each segment consumes the same local material as synapses.

The fixed fields, path mixtures and target-Gaussian dynamics have NOT passed
the framework's first-principles mechanism admission. Retained for inverse-
morphology comparisons and visualization, not as a qualified birth law.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp

from .structural_budget import MaterialParams, StructuralBudget


@dataclass(frozen=True)
class DevelopmentParams:
    step_um: float = 4.
    field_scale_um: float = 65.
    soma_spread_um: float = 30.
    branching_per_um: float = .007
    cone_slots_per_cell: int = 8
    material_per_um: float = .0015
    initial_material: float = 2.
    seed: int = 21

    def __post_init__(self):
        if min(self.step_um, self.field_scale_um, self.material_per_um) <= 0:
            raise ValueError('Positive spatial scales and material cost required')
        if self.cone_slots_per_cell < 3 or self.branching_per_um < 0 or self.initial_material < 0:
            raise ValueError('Invalid development budget or growth rates')


def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


class SpatialDevelopment:
    def __init__(self, type_code, cell_type, params=None, spatial_plan=None):
        self.params = params or DevelopmentParams()
        self.cell_type = np.asarray(cell_type, dtype=np.int64).copy()
        self.n = len(self.cell_type)
        self.rng = np.random.default_rng(self.params.seed)
        self.soma = self.rng.normal(0, self.params.soma_spread_um, (self.n, 3))
        # Fixed candidate morphogen fields. Their receptors are a seeded
        # projection of the fitted identity code, NOT a fitted morphology map.
        self.fields = np.array([[-60, 0, 0], [60, 0, 0], [0, -55, 15],
                                [0, 55, 15], [0, 0, -50], [0, 0, 55]], dtype=float)
        projection = self.rng.normal(size=(type_code.shape[1], len(self.fields)))
        code = type_code[self.cell_type]
        code = code/np.maximum(np.linalg.norm(code, axis=1, keepdims=True), 1e-30)
        self.receptors = np.tanh(code @ projection)
        self.spatial_plan = None
        if spatial_plan is not None:
            self.spatial_plan = {k: np.asarray(v).copy() for k, v in spatial_plan.items()}
            self.soma = self.spatial_plan['soma_mean'].copy()
            self.centers = self.soma+self.spatial_plan['centroid_offset']
            self.spatial_root = np.linalg.cholesky(self.spatial_plan['covariance'])
        self.cone_cell = np.repeat(np.arange(self.n), 3)
        self.cone_side = np.tile([0, 1, 1], self.n)  # output, input, input
        self.cone_xyz = self.soma[self.cone_cell].copy()
        self.cone_dir = unit(self.rng.normal(size=self.cone_xyz.shape))
        self.cone_parent = np.full(len(self.cone_cell), -1, dtype=int)
        self.cone_active = np.ones(len(self.cone_cell), bool)
        self.start, self.end, self.cell, self.side, self.parent, self.born = [], [], [], [], [], []
        self.generation = 0
        initial = self.params.initial_material
        if self.spatial_plan is not None:
            # Fitted finite cable capacities plus an explicit synaptic reserve.
            initial = initial+self.params.material_per_um*self.spatial_plan['length']
        self.budget = StructuralBudget(self.n, np.arange(self.n), np.zeros(self.n),
                                       MaterialParams(), initial)

    def step(self):
        p = self.params
        active = np.flatnonzero(self.cone_active)
        if not len(active):
            self.generation += 1
            return
        pos = self.cone_xyz[active]
        cell = self.cone_cell[active]
        if self.spatial_plan is None:
            delta = self.fields[None, :, :] - pos[:, None, :]
            amplitude = np.exp(-np.sum(delta*delta, axis=-1)/(2*p.field_scale_um**2))
            gradient = np.sum(delta*amplitude[..., None]*self.receptors[cell, :, None], axis=1)/p.field_scale_um
            incoming = self.cone_side[active] == 1
            gradient[incoming] = .4*gradient[incoming]+(self.soma[cell[incoming]]-pos[incoming])/35
            radius = np.linalg.norm(pos, axis=1, keepdims=True)
            boundary = -.6*(radius/100)**3*unit(pos)
            direction = unit(.85*self.cone_dir[active] + .35*gradient + boundary
                             + .25*self.rng.normal(size=pos.shape))
            requested = np.full(len(cell), p.step_um)
        else:
            # Local Gaussian target field. Endpoints have the requested stationary
            # covariance absent finite length/material truncation. The shared
            # correlation time is a candidate rule, not inferred molecular data.
            rho = np.exp(-1/8)
            noise = np.einsum('nij,nj->ni', self.spatial_root[cell], self.rng.normal(size=pos.shape))
            delta = (1-rho)*(self.centers[cell]-pos)+np.sqrt(1-rho*rho)*noise
            requested = np.linalg.norm(delta, axis=1)
            direction = unit(delta)
            accumulated = self.budget.reserved/p.material_per_um
            total = np.bincount(cell, requested, minlength=self.n)
            remaining = np.maximum(0, self.spatial_plan['length']-accumulated)
            factor = np.minimum(1, remaining/np.maximum(total, 1e-30))
            requested *= factor[cell]
        wanted_length = np.bincount(cell, requested, minlength=self.n)
        wanted_material = wanted_length*p.material_per_um
        paid = self.budget.reserve(wanted_material)
        share = np.zeros(self.n)
        np.divide(paid, wanted_material, out=share, where=wanted_material > 0)
        length = requested*share[cell]
        new_pos = pos + direction*length[:, None]
        for j, cone in enumerate(active):
            if length[j] <= 1e-12:
                self.cone_active[cone] = False
                continue
            self.start.append(pos[j].copy())
            self.end.append(new_pos[j].copy())
            self.cell.append(int(cell[j]))
            self.side.append(int(self.cone_side[cone]))
            self.parent.append(int(self.cone_parent[cone]))
            self.born.append(self.generation+1)
            self.cone_parent[cone] = len(self.start)-1
        self.cone_xyz[active] = new_pos
        self.cone_dir[active] = direction
        counts = np.bincount(self.cone_cell, minlength=self.n)
        # Branching is a Poisson candidate rate per funded cable length.
        rate = p.branching_per_um if self.spatial_plan is None else self.spatial_plan['branch_density'][cell]
        branch = self.rng.random(len(active)) < -np.expm1(-rate*length)
        additions = []
        for j in np.flatnonzero(branch):
            cone, c = active[j], int(cell[j])
            if counts[c] < p.cone_slots_per_cell and length[j] > 1e-12:
                counts[c] += 1
                additions.append(cone)
        if additions:
            selected = np.array(additions)
            self.cone_cell = np.r_[self.cone_cell, self.cone_cell[selected]]
            self.cone_side = np.r_[self.cone_side, self.cone_side[selected]]
            self.cone_xyz = np.concatenate([self.cone_xyz, self.cone_xyz[selected]])
            self.cone_dir = np.concatenate([self.cone_dir,
                unit(self.cone_dir[selected]+self.rng.normal(0, .6, (len(selected), 3)))])
            self.cone_parent = np.r_[self.cone_parent, self.cone_parent[selected]]
            self.cone_active = np.r_[self.cone_active, np.ones(len(selected), bool)]
        self.generation += 1

    def segments(self):
        return {k: np.asarray(getattr(self, k), dtype=(float if k in ('start', 'end') else int))
                .reshape((-1, 3) if k in ('start', 'end') else (-1,))
                for k in ('start', 'end', 'cell', 'side', 'parent', 'born')}

    def state_dict(self):
        keys = ('cell_type', 'soma', 'fields', 'receptors', 'cone_cell', 'cone_side',
                'cone_xyz', 'cone_dir', 'cone_parent', 'cone_active')
        return {'params': asdict(self.params), 'generation': self.generation,
                'arrays': {k: getattr(self, k).copy() for k in keys},
                'segments': self.segments(), 'budget': self.budget.state_dict(),
                'rng': copy.deepcopy(self.rng.bit_generator.state),
                'spatial_plan': copy.deepcopy(self.spatial_plan)}

    @classmethod
    def from_state_dict(cls, state):
        if cls is SpatialDevelopment and state.get('kind')=='conditional_tree':
            from .tree_development import ConditionalTreeDevelopment
            return ConditionalTreeDevelopment.from_state_dict(state)
        obj = cls.__new__(cls)
        obj.params = DevelopmentParams(**state['params'])
        integral = {'cell_type', 'cone_cell', 'cone_side', 'cone_parent',
                    'remaining_branches', 'edge_offsets', 'edge_cursor'}
        for k, value in state['arrays'].items():
            dtype = bool if k == 'cone_active' else int if k in integral else float
            setattr(obj, k, np.asarray(value, dtype=dtype).copy())
        obj.n = len(obj.cell_type)
        obj.spatial_plan = copy.deepcopy(state.get('spatial_plan'))
        if obj.spatial_plan is not None:
            obj.spatial_plan = {k: np.asarray(v) for k, v in obj.spatial_plan.items()}
            obj.centers = obj.soma+obj.spatial_plan['centroid_offset']
            obj.spatial_root = np.linalg.cholesky(obj.spatial_plan['covariance'])
        for k, value in state['segments'].items():
            rows = np.asarray(value, dtype=float if k in ('start','end') else int)
            setattr(obj, k, [row.copy() if isinstance(row, np.ndarray) else int(row) for row in rows])
        obj.generation = state['generation']
        obj.budget = StructuralBudget.from_state_dict(state['budget'])
        obj.rng = np.random.default_rng()
        obj.rng.bit_generator.state = copy.deepcopy(state['rng'])
        return obj


def encounter(segments, n_cells, scale, polarized=True):
    """Sparse cable-length overlap, explicitly separated output/input arbors.

    Subdivide segments for voxel quadrature. This is a 2/4um encounter proxy,
    not submicron membrane contact. No dense voxel volume is allocated.
    """
    xyz, cells, sides, weights = [], [], [], []
    for a, b, cell, side in zip(segments['start'], segments['end'], segments['cell'], segments['side']):
        length = np.linalg.norm(b-a)
        count = max(1, int(np.ceil(length/(scale*.5))))
        xyz.append(a+(b-a)*(np.arange(count)[:, None]+.5)/count)
        cells.append(np.full(count, cell))
        sides.append(np.full(count, side))
        weights.append(np.full(count, length/count))
    if not xyz:
        return sp.csr_matrix((n_cells, n_cells))
    voxel = np.floor(np.concatenate(xyz)/scale).astype(np.int64)
    _, column = np.unique(voxel, axis=0, return_inverse=True)
    c, side, w = np.concatenate(cells), np.concatenate(sides), np.concatenate(weights)
    shape = (n_cells, int(column.max())+1)
    if not polarized:
        density = sp.coo_matrix((w, (c, column)), shape=shape).tocsr()
        return (density @ density.T).tocsr()
    out = sp.coo_matrix((w[side == 0], (c[side == 0], column[side == 0])), shape=shape).tocsr()
    inc = sp.coo_matrix((w[side == 1], (c[side == 1], column[side == 1])), shape=shape).tocsr()
    return (out @ inc.T).tocsr()


def geometry_features(overlaps, cell_type, pre, post, floors):
    """Use the fitted model's 4um/2um forward/reverse feature conventions."""
    _, codes = np.unique(cell_type, return_inverse=True)
    n_types = int(codes.max())+1
    membership = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                               shape=(n_types, len(codes)))
    counts = np.bincount(codes)
    features = []
    for matrix, floor in zip(overlaps, floors):
        typed = (membership @ sp.csr_matrix(matrix) @ membership.T).toarray()
        typed /= counts[:, None]*counts[None, :]
        for a, b in ((codes[pre], codes[post]), (codes[post], codes[pre])):
            e = typed[a, b]
            features.extend([np.log(np.maximum(e, floor)), (e == 0).astype(float)])
    return np.stack(features, axis=1)
