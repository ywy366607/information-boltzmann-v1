"""One fused temporal scan per head and local gate direction; exact local JVP."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def _scan(K,V,A,B,S,P,ES,EP,Z,PHI,OS,OP,OES,OEP,C,T,
          NH:tl.constexpr,D:tl.constexpr,BUDGET:tl.constexpr):
    head=tl.program_id(0); direction=tl.program_id(1)
    d=tl.arange(0,D); ix=d[:,None]*D+d[None,:]
    s=tl.load(S+head*D*D+ix); p=tl.load(P+head*D*D+ix)
    es=tl.load(ES+(direction*NH+head)*D*D+ix)
    ep=tl.load(EP+(direction*NH+head)*D*D+ix)
    z=tl.load(Z+head); offset=tl.load(PHI); alignment=tl.load(PHI+1)
    correction=tl.full((),0.,tl.float32)
    for t in range(T):
        k=tl.load(K+(head*T+t)*D+d); v=tl.load(V+(head*T+t)*D+d)
        a=tl.load(A+head*T+t); beta=tl.load(B+head*T+t)
        prior=a*s
        r=v-tl.sum(prior*k[:,None],axis=0)
        u=tl.sum((p/z)*k[:,None],axis=0)
        dr=-a*tl.sum(es*k[:,None],axis=0)
        du=tl.sum((ep/z)*k[:,None],axis=0)
        nr0=tl.sqrt(tl.sum(r*r,axis=0)); nu0=tl.sqrt(tl.sum(u*u,axis=0))
        nr=tl.maximum(nr0,1.e-8); nu=tl.maximum(nu0,1.e-8)
        cosine=tl.sum(r*u,axis=0)/(nr*nu)
        dc=tl.sum(dr*u+r*du,axis=0)/(nr*nu)-cosine*(
            tl.where(nr0>1.e-8,tl.sum(r*dr,axis=0)/(nr*nr),0.)+
            tl.where(nu0>1.e-8,tl.sum(u*du,axis=0)/(nu*nu),0.))
        clipped=tl.minimum(tl.maximum(beta,1.e-7),1.-1.e-7)
        logit=tl.log(clipped)-tl.log(1.-clipped)+offset+alignment*cosine
        g=tl.sigmoid(logit)
        dg=g*(1.-g)*(tl.where(direction==0,1.,cosine)+alignment*dc)
        ep=ep+z*k[:,None]*(dg*(r-u)[None,:]+g*(dr-du)[None,:])
        p=p+z*g*k[:,None]*(r-u)[None,:]
        proposed=prior+beta*k[:,None]*r[None,:]
        des=a*es+beta*k[:,None]*dr[None,:]
        s=proposed; es=des
        if BUDGET>0:
            mag=tl.abs(proposed); sign=tl.where(proposed>0,1.,tl.where(proposed<0,-1.,0.))
            row=tl.sum(mag,axis=1)[:,None]; col=tl.sum(mag,axis=0)[None,:]
            drow=tl.sum(sign*des,axis=1)[:,None]; dcol=tl.sum(sign*des,axis=0)[None,:]
            maximum=tl.maximum(row,col)
            dm=tl.where(row>col,drow,tl.where(col>row,dcol,.5*(drow+dcol)))
            den=tl.maximum(maximum,BUDGET)
            dd=tl.where(maximum>BUDGET,dm,tl.where(maximum<BUDGET,0.,.5*dm))
            s=proposed*(BUDGET/den)
            es=(BUDGET/den)*(des-proposed*dd/den)
            diff=s-proposed; correction+=tl.sum(tl.sum(diff*diff,axis=1),axis=0)
    tl.store(OES+(direction*NH+head)*D*D+ix,es)
    tl.store(OEP+(direction*NH+head)*D*D+ix,ep)
    if direction==0:
        tl.store(OS+head*D*D+ix,s); tl.store(OP+head*D*D+ix,p)
        tl.store(C+head,correction)


@torch.no_grad()
def capture(k,v,alpha,beta,s,p,es,ep,impedance,phi,budget):
    batch,length,heads,d=k.shape
    if s.dtype!=torch.float32 or s.shape[-1]!=d:
        raise ValueError('contact fused capture requires square FP32 states')
    k,v=[x.transpose(1,2).contiguous() for x in (k,v)]
    alpha,beta=[x.transpose(1,2).contiguous() for x in (alpha,beta)]
    s,p,es,ep=[x.contiguous() for x in (s,p,es,ep)]
    z=impedance.expand(batch,heads).contiguous()
    sn,pn,dsn,dpn=[torch.empty_like(x) for x in (s,p,es,ep)]
    correction=torch.empty(batch*heads,device=s.device)
    _scan[(batch*heads,2)](k,v,alpha,beta,s,p,es,ep,z,phi,sn,pn,dsn,dpn,correction,
        T=length,NH=batch*heads,D=d,BUDGET=float(budget),num_warps=4,enable_fp_fusion=False)
    return sn,pn,dsn,dpn,correction.sum().sqrt()
