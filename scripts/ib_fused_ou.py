"""Experimental fused positive-gamma OU kick; noise remains externally sampled."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _forward(V,C,G,Z,Y,N:tl.constexpr,D:tl.constexpr,H:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
 v=tl.load(V+i,mask,0);c=tl.load(C+i,mask,0);g=tl.load(G+i//D,mask,1);z=tl.load(Z+i,mask,0)
 decay=tl.exp(-g*H);integ=-libdevice.expm1(-g*H)/g;std=tl.sqrt(-T*libdevice.expm1(-2*g*H))
 tl.store(Y+i,decay*v+integ*c+std*z,mask)

@triton.jit
def _backward(V,C,G,Z,DY,DV,DC,DG,N:tl.constexpr,D:tl.constexpr,H:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
 v=tl.load(V+i,mask,0);c=tl.load(C+i,mask,0);g=tl.load(G+i//D,mask,1);z=tl.load(Z+i,mask,0);dy=tl.load(DY+i,mask,0)
 decay=tl.exp(-g*H);integ=-libdevice.expm1(-g*H)/g;std=tl.sqrt(-T*libdevice.expm1(-2*g*H))
 dg=-H*decay*v+((H*decay*g+libdevice.expm1(-g*H))/(g*g))*c
 if T>0:dg+=T*H*decay*decay/tl.maximum(std,1.e-30)*z
 tl.store(DV+i,dy*decay,mask);tl.store(DC+i,dy*integ,mask);tl.store(DG+i,dy*dg,mask)

class FusedOU(torch.autograd.Function):
 @staticmethod
 def forward(ctx,v,c,g,z,h,temperature):
  v,c,g,z=[a.contiguous() for a in (v,c,g,z)];y=torch.empty_like(v)
  _forward[(triton.cdiv(v.numel(),128),)](v,c,g,z,y,v.numel(),v.shape[-1],h,temperature,128,enable_fp_fusion=False)
  ctx.save_for_backward(v,c,g,z);ctx.h=h;ctx.temperature=temperature
  return y
 @staticmethod
 def backward(ctx,dy):
  v,c,g,z=ctx.saved_tensors;dv=torch.empty_like(v);dc=torch.empty_like(c);dg=torch.empty_like(v)
  _backward[(triton.cdiv(v.numel(),128),)](v,c,g,z,dy.contiguous(),dv,dc,dg,v.numel(),v.shape[-1],ctx.h,ctx.temperature,128,enable_fp_fusion=False)
  return dv,dc,dg.sum(-1,keepdim=True),None,None,None


def install_fused_ou(model):
    from types import MethodType
    def kick(self,x,v,token,time,duration,generator=None,budget=None):
        gamma=self.damping(x)
        drive=self.drive(x,token,time)
        c=-self.kappa*x+drive
        noise_device=generator.device if generator is not None else v.device
        noise=torch.randn(v.shape,device=noise_device,dtype=v.dtype,generator=generator).to(v.device) if self.temperature>0 and duration>0 else torch.zeros_like(v)
        result=FusedOU.apply(v,c,gamma,noise,duration,self.temperature)
        if budget is not None:
            # Bookkeeping is detached and never contributes to the training loss.
            with torch.no_grad():
                integral=-torch.expm1(-gamma*duration)/gamma
                mean=torch.exp(-gamma*duration)*v+integral*c
                terminal=c/gamma;transient=v-terminal
                integral_v=terminal*duration+transient*integral
                integral_v2=(terminal.square().sum(-1)*duration+2*(terminal*transient).sum(-1)*integral.squeeze(-1)+transient.square().sum(-1)*(-torch.expm1(-2*gamma.squeeze(-1)*duration))/(2*gamma.squeeze(-1)))
                values={'drive_work':(drive*integral_v).sum(-1).mean(),'trap_work':(-self.kappa*x*integral_v).sum(-1).mean(),'deterministic_damping_loss':(gamma.squeeze(-1)*integral_v2).mean(),'ou_fluctuation_energy':.5*(result.square()-mean.square()).sum(-1).mean(),'ou_expected_fluctuation_energy':.5*v.shape[-1]*self.temperature*(-torch.expm1(-2*gamma*duration)).mean()}
                for key,value in values.items():budget[key]=budget.get(key,0.)+(value if getattr(budget,'defer_cpu',False) else float(value))
        return result
    if next(model.parameters()).device.type!='cuda' or next(model.parameters()).dtype!=torch.float32:
        raise ValueError('Experimental fused OU currently requires CUDA FP32')
    if model.force.gamma_field is None and model.force.gamma<=0:
        raise ValueError('Fused OU requires strictly positive gamma')
    if model.force.gamma_field is not None and model.force.gamma_bounds[0]<=0:
        raise ValueError('Fused OU requires strictly positive gamma bounds')
    model.force.kick=MethodType(kick,model.force)
    return model
