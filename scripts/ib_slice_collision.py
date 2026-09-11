"""Conservative continuous-particle Slice collision candidate (not an LBM solver).

Frozen substep generator A=P S P^T is skew. P[:,a]=U[:,a] outer B[a,:], with each U column centered. Its Cayley
transform preserves total momentum and kinetic energy without norm rescaling.
State-dependent channels do not imply detailed balance or an H theorem.
"""
import math
import torch
from torch import nn
from fine_grain.information_boltzmann.state import PhaseState

class SliceCayleyCollision(nn.Module):
    def __init__(self,dim,channels=16,width=32,rate=1.):
        super().__init__()
        if channels<3 or width<1 or rate<0:raise ValueError('Invalid channel configuration')
        self.channels,self.max_rate=channels,rate
        self.assignment=nn.Linear(dim,channels)
        self.channel_mixing=nn.Parameter(torch.randn(channels,channels)*.1)
        self.velocity_axes=nn.Parameter(torch.randn(channels,dim))
        self.features=nn.Sequential(nn.Linear(2*dim,width),nn.SiLU())
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)

    def forward(self,state,duration,generator=None):
        if duration<0:raise ValueError('Nonnegative duration required')
        zero=state.v.new_zeros(())
        if duration==0 or self.max_rate==0:
            return state,zero,{'candidates':0,'accepted':0,'cross_moment_change':0.,'pairs':[]}
        n=state.x.shape[0]
        w=self.assignment(state.x).softmax(-1)
        centered=w-w.mean(0,keepdim=True)
        u=centered/torch.sqrt(centered.square().sum(0,keepdim=True)+1e-8)/math.sqrt(self.channels)
        mass=w.sum(0).clamp_min(torch.finfo(w.dtype).eps)
        channel=(w.T@torch.cat((state.x,state.v),-1))/mass[:,None]
        f=self.features(channel);q,k=self.query(f),self.key(f)
        pair=q@k.T+math.sqrt(q.shape[-1])*self.channel_mixing
        raw=(pair-pair.T)/math.sqrt(q.shape[-1])
        skew=raw.tanh()*(self.max_rate/self.channels)
        axes=torch.nn.functional.normalize(self.velocity_axes,dim=-1,eps=1e-8)
        gram=(u.T@u)*(axes@axes.T)
        eye=torch.eye(self.channels,device=w.device,dtype=w.dtype)
        projected=((u.T@state.v)*axes).sum(-1)
        rhs=skew@projected
        coefficients=torch.linalg.solve(eye-.5*duration*(skew@gram),rhs)
        delta=duration*((u*coefficients[None,:])@axes)
        velocity=state.v+delta
        cross=(delta*state.x).sum(-1).mean().detach()
        return PhaseState(state.x,velocity,state.time),zero,{'candidates':0,'accepted':0,'cross_moment_change':float(cross),'pairs':[]}


def install_slice_collision(model,channels=16,width=32,rate=1.):
    reference=next(model.parameters())
    model.collision=SliceCayleyCollision(model.force.net[-1].out_features,channels,width,rate).to(reference)
    return model
