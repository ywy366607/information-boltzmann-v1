"""Bound independent JVP workspace without changing any eligibility direction.

The production loop already processes directions in groups of four. This
external adapter runs those same directions in groups of two and joins the
same outputs. It must pass the complete actual-event equivalence gate.
"""
import torch
from torch.utils._pytree import tree_map


class _TorchView:
    def __getattr__(self,name):return getattr(torch,name)

    def vmap(self,fn,*args,**kwargs):
        mapped=torch.vmap(fn,*args,**kwargs)
        if not fn.__qualname__.endswith('AdaptiveHamiltonian.quiet.<locals>.one'):
            return mapped
        def bounded(*values):
            parts=[mapped(*(v[start:start+2] for v in values))
                   for start in range(0,len(values[0]),2)]
            if len(parts)==1:return parts[0]
            return tree_map(lambda *xs:torch.cat(xs,0),*parts)
        return bounded


def install():
    from scripts.ib_local import adaptive_hamiltonian
    old=adaptive_hamiltonian.torch
    adaptive_hamiltonian.torch=_TorchView()
    return lambda:setattr(adaptive_hamiltonian,'torch',old)
