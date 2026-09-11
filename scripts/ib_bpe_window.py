"""Fixed-window BPE execution: batched token projection/readout and device-only dynamics."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from fine_grain.information_boltzmann import InformationBoltzmann
from scripts.ib_slice_collision import SliceCayleyCollision
from scripts.ib_fused_ou import FusedOU

class BPEWindow(nn.Module):
    def __init__(self,vocab=50257,hidden=128,particles=512,channels=16,steps=4,recompute=True,recompute_stride=1):
        super().__init__()
        self.core=InformationBoltzmann(vocab_size=vocab,phase_dim=4,particles=particles,hidden_dim=hidden,flow_layers=4,steps=steps,gamma=.3,gamma_mode='local',temperature=.1)
        # One vocabulary table, used for drive, initialization and readout.
        self.core.initial.embedding=self.core.force.embedding
        self.core.decoder.weight=self.core.force.embedding.weight
        self.collision=SliceCayleyCollision(4,channels,32)
        del self.core.collision
        self.steps,self.recompute=steps,recompute
        self.recompute_stride = recompute_stride
        self.register_buffer('eye',torch.eye(channels))

    def collision_step(self,x,v,dt):
        c=self.collision;w=c.assignment(x).softmax(-1)
        centered=w-w.mean(0,keepdim=True)
        u=centered/torch.sqrt(centered.square().sum(0,keepdim=True)+1e-8)/math.sqrt(c.channels)
        channel=(w.T@torch.cat((x,v),-1))/w.sum(0).clamp_min(torch.finfo(w.dtype).eps)[:,None]
        f=c.features(channel);q,k=c.query(f),c.key(f)
        pair=q@k.T+math.sqrt(q.shape[-1])*c.channel_mixing
        skew=((pair-pair.T)/math.sqrt(q.shape[-1])).tanh()*(c.max_rate/c.channels)
        axes=F.normalize(c.velocity_axes,dim=-1,eps=1e-8)
        gram=(u.T@u)*(axes@axes.T)
        rhs=skew@((u.T@v)*axes).sum(-1)
        # No per-substep host error check. Numerical tests audit finite results.
        coeff,info=torch.linalg.solve_ex(self.eye-.5*dt*(skew@gram),rhs,check_errors=False)
        return v+dt*((u*coeff[None,:])@axes)

    def drive(self,x,shared):
        force=self.core.force
        z=F.linear(x,force.net[0].weight[:,:4])+shared
        return .5*force.net[2](force.net[1](z)).tanh()

    def kick(self,x,v,shared,noise,h,recompute=True):
        drive=checkpoint(self.drive,x,shared,use_reentrant=False,preserve_rng_state=False) if self.recompute and recompute and torch.is_grad_enabled() else self.drive(x,shared)
        gamma=self.core.force.damping(x)
        return FusedOU.apply(v,-x+drive,gamma,noise,h,.1)

    def forward(self,x,v,input_ids,targets,clocks,noise):
        force=self.core.force;layer=force.net[0];dt=1./self.steps
        shared=F.linear(force.embedding(input_ids),layer.weight[:,4:-2],layer.bias)
        clock_proj=F.linear(clocks,layer.weight[:,-2:])
        observations=[]
        for t in range(input_ids.shape[0]):
            for s in range(self.steps):
                base=(t*self.steps+s)*4
                for half in range(2):
                    drive_input=shared[t]+clock_proj[t,s,half]
                    v=self.kick(x,v,drive_input,noise[base+half*2],dt/4,(base+half*2)%self.recompute_stride==0)
                    x=x+(dt/2)*v
                    v=self.kick(x,v,drive_input,noise[base+half*2+1],dt/4,(base+half*2+1)%self.recompute_stride==0)
                    if half==0:v=self.collision_step(x,v,dt)
            observations.append(torch.cat((x,v),-1))
        states=torch.stack(observations)
        # Readout after each causal event, batched only because weights stay fixed
        # inside the optimizer window. No future input enters an earlier state.
        def readout(z):return self.core.features(z).mean(1)
        features=checkpoint(readout,states,use_reentrant=False,preserve_rng_state=False) if self.recompute and torch.is_grad_enabled() else readout(states)
        logits=F.linear(features,self.core.decoder.weight,self.core.decoder.bias)
        loss=F.cross_entropy(logits,targets)
        return loss,x,v


def clock_inputs(start,length,steps,device):
    t=torch.arange(length,device=device,dtype=torch.float64)[:,None,None]+start
    offset=(torch.arange(steps,device=device,dtype=torch.float64)[None,:,None]+torch.tensor([.25,.75],device=device)[None,None,:])/steps
    phase=t+offset
    return torch.stack((phase.sin(),phase.cos()),-1).float()
