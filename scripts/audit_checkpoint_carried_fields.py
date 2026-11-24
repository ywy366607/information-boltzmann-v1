"""Hash selected actual carried state without loading all weights/optimizer.

FakeTensorMode indexes ZIP storage offsets; only requested logical tensor
views are read from the decompressed checkpoint. Unused storage bytes do not
become evidence. This is an audit reader, never a model/training operation.
"""
import argparse
import dataclasses
from enum import Enum
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from scripts.ib_local.audit_runtime_checkpoint_pair import digest


def read(path):
    with gzip.open(path,'rb') as stream:
        with FakeTensorMode():
            checkpoint = torch.load(stream,map_location='cpu',weights_only=False)
        cache = {}
        def concrete(v):
            if torch.is_tensor(v):
                storage = v.untyped_storage()
                offset = getattr(storage,'_checkpoint_offset',None)
                if offset is None: raise ValueError('checkpoint reader cannot index requested tensor')
                key = (offset,storage.nbytes())
                if key not in cache:
                    stream.seek(offset)
                    raw = stream.read(storage.nbytes())
                    if len(raw) != storage.nbytes(): raise EOFError('incomplete checkpoint tensor')
                    cache[key] = bytearray(raw)
                whole = torch.frombuffer(cache[key],dtype=v.dtype)
                return whole.as_strided(tuple(v.shape),tuple(v.stride()),v.storage_offset()).contiguous()
            if isinstance(v,np.random.Generator): return v.bit_generator.state
            if isinstance(v,Enum): return (type(v).__name__,v.value)
            if dataclasses.is_dataclass(v):
                return {f.name:concrete(getattr(v,f.name)) for f in dataclasses.fields(v)}
            if isinstance(v,dict): return {k:concrete(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)): return type(v)(concrete(x) for x in v)
            return v
        fields = ('bfss_momenta','internal_pending','joint_internal_ticks','rng',
                  'native_optimizer_updates','native_external_bytes','native_self_bytes')
        engine = checkpoint['engine']
        result = {k:digest(concrete(engine[k])) for k in fields if k in engine}
        result['cycle'] = checkpoint['cycle']
        return result,sum(n for _,n in cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a = p.parse_args()
    before,n = read(a.before); after,m = read(a.after)
    result = dict(schema=1,equal={k:before[k]==after.get(k) for k in before},
                  selected_storage_bytes=[n,m],whole_body_loaded=False,model_updates=0,world_calls=0)
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    if not all(result['equal'].values()): raise RuntimeError(result)
    print(json.dumps(result))


if __name__=='__main__': main()
