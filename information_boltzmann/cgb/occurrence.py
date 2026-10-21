"""Independently calibrated cell-pair connection probability, not strength."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class OccurrenceProgram:
    def __init__(self,values):
        self.values={k:np.asarray(v).copy() for k,v in values.items()}
        self.code=self.values['code'].astype(float)
        self.code/=np.maximum(np.linalg.norm(self.code,axis=1,keepdims=True),1e-30)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls({k:z[k] for k in z.files})

    def raw_features(self,overlap4,overlap2):
        result=[]
        for e,floor in zip((overlap4,overlap2),self.values['floors']):
            e=np.asarray(e)
            if not np.isfinite(e).all() or (e<0).any(): raise ValueError('Invalid cable overlap')
            result.extend([np.log(np.maximum(e,floor)),(e==0).astype(float)])
        return np.stack(result,axis=-1)

    def probability(self,pre_type,post_type,overlap4,overlap2,block_size=8192):
        pre_type,post_type=np.asarray(pre_type,int),np.asarray(post_type,int)
        if block_size < 1:
            raise ValueError('Positive deployment block size required')
        if np.any(pre_type < 0) or np.any(post_type < 0) or np.any(pre_type >= len(self.code)) or np.any(post_type >= len(self.code)):
            raise IndexError('Unknown type index')
        raw=self.raw_features(overlap4,overlap2)
        v=self.values
        if pre_type.shape!=post_type.shape or raw.shape!=(len(pre_type),4):
            raise ValueError('One full geometry vector per directed cell pair required')
        out=np.empty(len(pre_type))
        for lo in range(0,len(out),block_size):
            hi=lo+block_size;a=self.code[pre_type[lo:hi]];b=self.code[post_type[lo:hi]]
            x=(raw[lo:hi]-v['feature_mean'])/v['feature_scale']
            score=v['bias']+x@v['geometry']+a@v['out']+b@v['in']+np.sum((a@v['affinity'])*b,axis=1)
            out[lo:hi]=expit(score)
        return out
