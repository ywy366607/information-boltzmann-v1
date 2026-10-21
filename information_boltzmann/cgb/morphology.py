"""A compact identity/side decoder for real morphology moments.

This estimates distributions and cable budgets, not a molecular growth law or
an exact cell skeleton. Decoding log covariance preserves positive definiteness.
"""
from __future__ import annotations

import numpy as np

TRI=np.triu_indices(3)


def symmetric_log(covariance):
    eigen,vectors=np.linalg.eigh(covariance)
    matrix=(vectors*np.log(np.maximum(eigen,1e-6))[...,None,:])@np.swapaxes(vectors,-1,-2)
    return matrix[...,TRI[0],TRI[1]]


def symmetric_exp(values):
    matrix=np.zeros((*values.shape[:-1],3,3))
    matrix[...,TRI[0],TRI[1]]=values
    matrix[...,TRI[1],TRI[0]]=values
    eigen,vectors=np.linalg.eigh(matrix)
    if np.max(eigen)>np.log(np.finfo(float).max)-10:
        raise ValueError('Predicted covariance cannot be represented finitely')
    return (vectors*np.exp(eigen)[...,None,:])@np.swapaxes(vectors,-1,-2)


def identity_features(code,cell_type,side):
    c=np.asarray(code,dtype=float)[np.asarray(cell_type,dtype=int)]
    c=c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-30)
    side=np.asarray(side).astype(str)
    sign=(side=='L').astype(float)-(side=='R').astype(float)
    return np.column_stack([np.ones(len(c)),side=='L',side=='R',c,c*sign[:,None]])


class MorphologyProgram:
    def __init__(self,values):
        self.values={k:np.asarray(v).copy() for k,v in values.items()}

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls({k:z[k] for k in z.files})

    def describe(self,cell_type,side):
        v=self.values
        x=identity_features(v['code'],cell_type,side)
        x=(x-v['feature_mean'])/v['feature_scale']
        y=(x@v['weight'])*v['target_scale']+v['target_mean']
        return {'soma_mean':y[:,:3], 'centroid_offset':y[:,3:6],
                'length':np.exp(y[:,6]), 'covariance':symmetric_exp(y[:,7:13]),
                'branch_density':np.exp(y[:,13]), 'side':np.asarray(side).astype(str)}


class SpatialIdentityProgram(MorphologyProgram):
    """Known-type spatial phenotype, with a shared bilateral canonical frame."""

    def describe(self, cell_type, side):
        cell_type = np.asarray(cell_type, dtype=int)
        side = np.asarray(side).astype(str)
        if np.any(cell_type < 0) or np.any(cell_type >= len(self.values['known'])):
            raise IndexError('Unknown spatial type index')
        if side.shape != cell_type.shape or np.any(~np.isin(side, ['L', 'R'])):
            raise ValueError('A left/right side is required for each cell')
        if np.any(~self.values['known'][cell_type]):
            raise ValueError('Spatial identity has no training soma for this type')
        y = self.values['spatial_code'][cell_type].copy()
        sign = np.where(side == 'L', 1., -1.)
        soma, offset = y[:, :3].copy(), y[:, 3:6].copy()
        soma[:, 0] = self.values['midline'] + sign*soma[:, 0]
        offset[:, 0] *= sign
        reflection = np.column_stack([sign, np.ones((len(sign), 2))])
        covariance = symmetric_exp(y[:, 7:13])
        covariance *= reflection[:, :, None]*reflection[:, None, :]
        return {'soma_mean': soma, 'centroid_offset': offset,
                'length': np.exp(y[:, 6]), 'covariance': covariance,
                'branch_density': np.exp(y[:, 13]), 'side': side}
