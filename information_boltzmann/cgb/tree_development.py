"""Finite statistical tree birth with bounded execution workspace.

This is a conditional inverse-morphology model, not a molecular growth law.
The finite branch stock and arc-length reservoir are type-conditioned; every
physical tip survives in state even when not selected in the current batch.
Exact conditional forest probabilities do not derive a physical branching
process. Prescribed branch stocks and Gamma lengths remain statistical choices;
this candidate has not passed first-principles birth-law admission.
"""
from __future__ import annotations

import numpy as np

from .development import SpatialDevelopment, unit
from .tree_geometry import forest_branch_probability


class ConditionalTreeDevelopment(SpatialDevelopment):
    def __init__(self, type_code, cell_type, spatial_plan, tree_program, params=None):
        super().__init__(type_code,cell_type,params,spatial_plan)
        t=self.cell_type
        if np.any(~tree_program['known'][t]): raise ValueError('Tree type lacks training support')
        self.remaining_branches=np.maximum(0,np.rint(self.spatial_plan['length']*
            self.spatial_plan['branch_density']-1)).astype(int)
        roots=tree_program['root_count'][t].astype(int)
        self.cone_cell=np.repeat(np.arange(self.n),roots)
        self.cone_xyz=self.soma[self.cone_cell].copy()
        self.cone_dir=unit(self.centers[self.cone_cell]-self.cone_xyz)
        self.cone_parent=np.full(len(self.cone_cell),-1,int)
        self.cone_active=np.ones(len(self.cone_cell),bool)
        self.output_share=tree_program['output_share'][t].copy()
        self.cone_side=(self.rng.random(len(self.cone_cell))>=self.output_share[self.cone_cell]).astype(int)
        self.edge_offsets=np.r_[0,np.cumsum(2*self.remaining_branches+roots)]
        self.edge_lengths=np.empty(self.edge_offsets[-1])
        self.edge_cursor=np.zeros(self.n,int)
        for c in range(self.n):
            n=self.edge_offsets[c+1]-self.edge_offsets[c]
            shape=float(tree_program['gamma_shape'][t[c]])
            weights=self.rng.gamma(shape,1.,n)
            # Root shafts have their own training estimate. Allocate remaining
            # finite cable among non-root chains, then condition on total length.
            primary=float(tree_program['primary_length'][t[c]])
            if n>roots[c] and primary>0:
                weights[roots[c]:]*=max(self.spatial_plan['length'][c]-primary*roots[c],1e-8)/max(weights[roots[c]:].sum(),1e-30)
                weights[:roots[c]]=primary
            weights=np.maximum(weights,1e-12)
            self.edge_lengths[self.edge_offsets[c]:self.edge_offsets[c+1]]=weights/weights.sum()*self.spatial_plan['length'][c]
        self.cone_remaining=np.zeros(len(self.cone_cell))
        for i,c in enumerate(self.cone_cell): self.cone_remaining[i]=self._take_length(int(c))

    def _take_length(self,c):
        idx=int(self.edge_offsets[c]+self.edge_cursor[c])
        if idx>=self.edge_offsets[c+1]: raise RuntimeError('Finite chain reservoir exhausted')
        self.edge_cursor[c]+=1
        return self.edge_lengths[idx]

    def step(self):
        active=np.flatnonzero(self.cone_active)
        if not len(active): self.generation+=1;return
        # The limit is workspace only. Pending physical tips remain in state.
        used=np.zeros(self.n,int);selected=[]
        for i in active:
            c=self.cone_cell[i]
            if used[c]<self.params.cone_slots_per_cell: selected.append(i);used[c]+=1
        active=np.asarray(selected,int);cell=self.cone_cell[active];pos=self.cone_xyz[active]
        rho=np.exp(-1/8)
        noise=np.einsum('nij,nj->ni',self.spatial_root[cell],self.rng.normal(size=pos.shape))
        field=(1-rho)*(self.centers[cell]-pos)+np.sqrt(1-rho*rho)*noise
        # Existing persistent-cone law, now advanced in funded arc-length steps.
        direction=unit(.85*self.cone_dir[active]+.35*unit(field)+.25*self.rng.normal(size=pos.shape))
        requested=np.minimum(self.params.step_um,self.cone_remaining[active])
        wanted=np.bincount(cell,requested,minlength=self.n)*self.params.material_per_um
        paid=self.budget.reserve(wanted)
        share=np.divide(paid,wanted,out=np.zeros(self.n),where=wanted>0)
        length=requested*share[cell]
        frontier=np.bincount(self.cone_cell[self.cone_active],minlength=self.n)
        additions=[]
        for j,i in enumerate(active):
            c=int(cell[j]);new=pos[j]+direction[j]*length[j]
            if length[j]>1e-12:
                self.start.append(pos[j].copy());self.end.append(new.copy());self.cell.append(c)
                self.side.append(int(self.cone_side[i]));self.parent.append(int(self.cone_parent[i]));self.born.append(self.generation+1)
                self.cone_parent[i]=len(self.start)-1
            self.cone_xyz[i]=new;self.cone_dir[i]=direction[j];self.cone_remaining[i]-=length[j]
            if self.cone_remaining[i]<=1e-10:
                self.cone_active[i]=False
                branch=self.rng.random()<forest_branch_probability(self.remaining_branches[c],frontier[c])
                frontier[c]-=1
                if branch:
                    self.remaining_branches[c]-=1;frontier[c]+=2
                    for _ in range(2):
                        additions.append((c,new.copy(),unit(direction[j]+self.rng.normal(size=3)),
                            int(self.cone_parent[i]),int(self.rng.random()>=self.output_share[c]),self._take_length(c)))
        if additions:
            self.cone_cell=np.r_[self.cone_cell,[a[0] for a in additions]]
            self.cone_xyz=np.concatenate([self.cone_xyz,np.stack([a[1] for a in additions])])
            self.cone_dir=np.concatenate([self.cone_dir,np.stack([a[2] for a in additions])])
            self.cone_parent=np.r_[self.cone_parent,[a[3] for a in additions]]
            self.cone_side=np.r_[self.cone_side,[a[4] for a in additions]]
            self.cone_remaining=np.r_[self.cone_remaining,[a[5] for a in additions]]
            self.cone_active=np.r_[self.cone_active,np.ones(len(additions),bool)]
        self.generation+=1

    def state_dict(self):
        state=super().state_dict();state['kind']='conditional_tree'
        for key in ('remaining_branches','output_share','edge_offsets','edge_lengths','edge_cursor','cone_remaining'):
            state['arrays'][key]=getattr(self,key).copy()
        return state
