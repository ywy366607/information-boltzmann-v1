"""Two learned potentials on the existing full per-head memory matrices.

The weighted potential contains the reference coupling exactly. The free
potential is a positive attention/MLP energy; its common-basis rotation charge
is not assumed conserved. Both retain S/P and the same split integrator.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.bfss_gdn_capture import matrix_view
from scripts.ib_local.bfss_gdn_reference import coupling_potential_and_gradient


class HeadPotential(nn.Module):
    def __init__(self, heads, dk, dv, kind, coupling=.01, width=32):
        super().__init__()
        if kind not in ('weighted', 'free'):
            raise ValueError('unknown learned potential')
        self.kind, self.heads, self.coupling = kind, heads, coupling
        self.input = nn.Linear(dk*dv, width, bias=False)
        self.norm = nn.RMSNorm(width)
        self.query = nn.Linear(width, width//2, bias=False)
        self.key = nn.Linear(width, width//2, bias=False)
        self.log_strength = nn.Parameter(torch.zeros(()))
        self.register_buffer('force_scale', torch.ones(()))
        if kind == 'weighted':
            # Constant attention exactly recovers the inherited potential.
            # Q is nonzero, so K has an immediate first-update gradient.
            nn.init.zeros_(self.key.weight)
        else:
            self.value = nn.Linear(width, width, bias=False)
            self.readout = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(),
                                         nn.Linear(width, 1))

    def forward(self, s):
        h = self.norm(self.input(s.flatten(-2)))
        q, k = self.query(h), self.key(h)
        a = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
        strength = self.coupling*self.log_strength.exp()*self.force_scale
        if self.kind == 'free':
            context = a @ self.value(h)
            return strength*F.softplus(self.readout(torch.cat((h,context),-1))).sum((-2,-1))
        weights = (a+a.transpose(-1,-2))*self.heads/2
        energy = s.new_zeros(s.shape[:-3])
        for i in range(self.heads):
            for j in range(i+1,self.heads):
                si,sj=s[...,i,:,:],s[...,j,:,:]
                left=si@sj.transpose(-1,-2)-sj@si.transpose(-1,-2)
                right=si.transpose(-1,-2)@sj-sj.transpose(-1,-2)@si
                energy=energy+weights[...,i,j]*(left.square().sum((-2,-1))+
                                                right.square().sum((-2,-1)))
        return strength*energy/8

    def force(self, s, *, create_graph=False):
        # Force evaluation is also required during no-grad live planning.
        with torch.enable_grad():
            state=s if s.requires_grad else s.detach().requires_grad_(True)
            gradient=torch.autograd.grad(self(state).sum(),state,create_graph=create_graph)[0]
        return gradient if create_graph else gradient.detach()

    @torch.no_grad()
    def calibrate(self,s):
        """Use only the inherited state to match initial force magnitude."""
        if self.kind=='free':
            target=coupling_potential_and_gradient(s,self.coupling)[1].norm()
            actual=self.force(s).norm()
            self.force_scale.copy_(target/actual.clamp_min(torch.finfo(s.dtype).eps))


def quiet_step(potential,s,p,*,dt,frequency,create_graph=False):
    half=p-dt/2*potential.force(s,create_graph=create_graph)
    angle=frequency*dt
    c, sine=math.cos(angle),math.sin(angle)
    next_s=c*s+(sine/frequency)*half
    next_p=c*half-frequency*sine*s
    return next_s,next_p-dt/2*potential.force(next_s,create_graph=create_graph)


class LearnedHeadCoupling(nn.Module):
    def __init__(self,base,kind,coupling):
        super().__init__()
        self.potentials=nn.ModuleList([HeadPotential(l.heads,l.dk,l.dv,kind,coupling)
                                      for l in base.layers])

    def quiet(self,base,recurrent,momenta,*,dt,frequency,steps,create_graph=False):
        states,ps=[],[]
        for i,(layer,potential,p) in enumerate(zip(base.layers,self.potentials,momenta)):
            s=matrix_view(layer,recurrent[i])
            for _ in range(steps):
                s,p=quiet_step(potential,s,p,dt=dt,frequency=frequency,create_graph=create_graph)
            states.append(torch.cat((s.flatten(1),recurrent[i,:,layer.matrix_size:]),-1)
                          if layer.conv_kernel else s)
            ps.append(p)
        return torch.stack(states),ps
