"""Fused, algebraically identical local-credit recurrences; PyTorch fallbacks in learner."""
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _gates(H,B,C,PACKET,JAC,MOD,X,A,I,V,S,PSI,BETA,RHO,N:tl.constexpr,D:tl.constexpr,DECAY:tl.constexpr,BLOCK:tl.constexpr):
        ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        cols=3*(D+1)
        row=ix//cols; col=ix%cols; head=col//(D+1); feature=col%(D+1)
        valid=row<N
        modality=tl.load(MOD+row,valid,0)
        x=tl.load(X+feature,valid&(feature<D),0)
        x=tl.where(feature==D,1.,x)
        raw=tl.load(PACKET+row,valid,0)*tl.load(JAC+modality*3+head,valid,0)*x
        oldc=tl.load(C+ix,valid,0)
        a=tl.load(A+row,valid,0); integ=tl.load(I+row,valid,0)
        v=tl.load(V+row,valid,0); s=tl.load(S+row,valid,0)
        psi=tl.load(PSI+row,valid,0); beta=tl.load(BETA+row,valid,0); rho=tl.load(RHO+row,valid,0)
        eh=tl.load(H+ix,valid,0); eb=tl.load(B+ix,valid,0)
        ev=a*eh+integ*(raw-oldc)
        ds=psi*(ev-beta*eb)
        tl.store(H+ix,(1-s)*ev-v*ds,valid)
        tl.store(B+ix,rho*eb+(1-rho)*ds,valid)
        tl.store(C+ix,DECAY*oldc+(1-DECAY)*raw,valid)

    @triton.jit
    def _projection(H,B,C,G,MOD,X,A,I,V,S,PSI,BETA,RHO,N:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
        ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        row=ix//D; col=ix%D; valid=row<N
        modality=tl.load(MOD+row,valid,0)
        drive=tl.load(G+modality,valid,0)*tl.load(X+col,valid,0)-tl.load(C+modality*D+col,valid,0)
        ev=tl.load(A+row,valid,0)*tl.load(H+ix,valid,0)+tl.load(I+row,valid,0)*drive
        eb=tl.load(B+ix,valid,0)
        ds=tl.load(PSI+row,valid,0)*(ev-tl.load(BETA+row,valid,0)*eb)
        rho=tl.load(RHO+row,valid,0)
        tl.store(H+ix,(1-tl.load(S+row,valid,0))*ev-tl.load(V+row,valid,0)*ds,valid)
        tl.store(B+ix,rho*eb+(1-rho)*ds,valid)

    @triton.jit
    def _edges(W,P,PRE,POST,FACTOR,Z,E:tl.constexpr,N:tl.constexpr,
               S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,LR:tl.constexpr,APPLY:tl.constexpr,BLOCK:tl.constexpr):
        ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); valid=ix<E
        pre=tl.load(PRE+ix,valid,0,cache_modifier=".cg"); post=tl.load(POST+ix,valid,0,cache_modifier=".cg")
        tier=tl.where(ix<S1,0,tl.where(ix<S2,1,tl.where(ix<S3,2,3)))
        delta=-LR*tl.load(FACTOR+post,valid,0)*tl.load(Z+tier*N+pre,valid,0)
        pending=tl.load(P+ix,valid,0,cache_modifier=".cg")+delta
        if APPLY:
            value=tl.maximum(0.,tl.minimum(5.,tl.load(W+ix,valid,0,cache_modifier=".cg")+pending))
            tl.store(W+ix,value,valid); tl.store(P+ix,0.,valid)
        else:
            tl.store(P+ix,pending,valid)

    @triton.jit
    def _physical(H,B,DE,DI,OUT,L,V,A,INT,G,HP,BP,GEP,GIP,GEN,GIN,S,PSI,
                  RHO,BETA,LEAKM,LEAKE,LEAKI,THETA,MASKS,GEGAIN,GIGAIN,
                  EE:tl.constexpr,EI:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
        ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        row=ix//8; col=ix%8; valid=row<N
        ge_gain=tl.load(GEGAIN);gi_gain=tl.load(GIGAIN)
        v=tl.load(V+row,valid,0);a=tl.load(A+row,valid,0);integ=tl.load(INT+row,valid,1)
        g=tl.load(G+row,valid,1);hp=tl.load(HP+row,valid,0);bp=tl.load(BP+row,valid,0)
        psi=tl.load(PSI+row,valid,0);rho=tl.load(RHO+row,valid,0);beta=tl.load(BETA+row,valid,0)
        spikes=tl.load(S+row,valid,0)
        da=-a*tl.cast((g>=1e-5)&(g<=20),tl.float32)
        gb=tl.maximum(g,1e-5)
        dbeta=(-da*gb-(1-a)*tl.cast(g>=1e-5,tl.float32))/(gb*gb)
        force=(v-a*hp)/integ
        dvG=da*hp+dbeta*force
        mask=tl.load(MASKS+ix,valid,0)
        leak=tl.where(col==4,tl.load(LEAKE+row,valid,1),tl.load(LEAKI+row,valid,1))
        oldg=tl.where(col==4,tl.load(GEP+row,valid,0),tl.load(GIP+row,valid,0))
        newg=tl.where(col==4,tl.load(GEN+row,valid,0),tl.load(GIN+row,valid,0))
        oldd=tl.where(col==4,tl.load(DE+row,valid,0),tl.load(DI+row,valid,0))
        arrival=(newg-leak*oldg)/(1-leak)
        newd=leak*oldd+(oldg-arrival)*leak*(-tl.log(leak))*mask
        tl.store(DE+row,newd,valid&(col==4));tl.store(DI+row,newd,valid&(col==5))
        direct=tl.where(col==1,tl.log(tl.load(LEAKM+row,valid,1))*dvG*mask,0.)
        direct=tl.where(col==4,ge_gain*newd*(dvG+integ*EE),direct)
        direct=tl.where(col==5,gi_gain*newd*(dvG+integ*EI),direct)
        direct=tl.where(col==6,ge_gain*tl.load(GEN+row,valid,0)*(dvG+integ*EE)*mask,direct)
        direct=tl.where(col==7,gi_gain*tl.load(GIN+row,valid,0)*(dvG+integ*EI)*mask,direct)
        dtheta=tl.where(col==0,tl.load(THETA+row,valid,0)*mask,0.)
        dtheta=tl.where(col==2,beta*bp*mask,dtheta)
        db=tl.where(col==3,rho*(-tl.log(rho))*(bp-spikes)*mask,0.)
        ev=a*tl.load(H+ix,valid,0)+direct
        ds=psi*(ev-beta*tl.load(B+ix,valid,0)-dtheta)
        hn=(1-spikes)*ev-v*ds
        bn=rho*tl.load(B+ix,valid,0)+(1-rho)*ds+db
        tl.store(H+ix,hn,valid);tl.store(B+ix,bn,valid)
        tl.store(OUT+ix,tl.load(L+row,valid,0)*hn,valid)

    @triton.jit
    def _projection_weights(W,H,F,N:tl.constexpr,D:tl.constexpr,LR:tl.constexpr,WD:tl.constexpr,BLOCK:tl.constexpr):
        ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=ix<N*D
        value=tl.load(W+ix,valid,0)
        eligibility=tl.load(H+ix,valid,0)
        factor=tl.load(F+ix//D,valid,0)
        tl.store(W+ix,value*(1-LR*WD)-LR*factor*eligibility,valid)

    @triton.jit
    def _projection_compensated(W,R,H,F,N:tl.constexpr,D:tl.constexpr,LR:tl.constexpr,WD:tl.constexpr,BLOCK:tl.constexpr):
        ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); valid=ix<N*D
        old=tl.load(W+ix,valid,0)
        change=-LR*(tl.load(F+ix//D,valid,0)*tl.load(H+ix,valid,0)+WD*old)
        total=tl.load(R+ix,valid,0)+change
        new=old+total
        tl.store(R+ix,total-(new-old),valid)
        tl.store(W+ix,new,valid)

    @triton.jit
    def _local_coba_edges(W,P,R,EG,EH,EB,PRE,POST,PULSES,COEFFICIENTS,
                         E:tl.constexpr,N:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,
                         LR:tl.constexpr,APPLY:tl.constexpr,BLOCK:tl.constexpr,
                         MASK_DORMANT:tl.constexpr=False):
        ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); valid=ix<E
        pre=tl.load(PRE+ix,valid,0,cache_modifier=".cg")
        post=tl.load(POST+ix,valid,0,cache_modifier=".cg")
        tier=tl.where(ix<S1,0,tl.where(ix<S2,1,tl.where(ix<S3,2,3)))
        oldg=tl.load(EG+ix,valid,0,cache_modifier=".cg")
        oldh=tl.load(EH+ix,valid,0,cache_modifier=".cg")
        eb=tl.load(EB+ix,valid,0,cache_modifier=".cg")
        arriving=tl.load(PULSES+tier*N+pre,valid,0)
        active=(oldg!=0.)|(oldh!=0.)|(eb!=0.)|(arriving!=0.)
        coefficient_mask=valid & active if MASK_DORMANT else valid
        columns=tl.arange(0,8)
        packed=tl.load(COEFFICIENTS+post[:,None]*8+columns[None,:],coefficient_mask[:,None],0)
        leak=tl.sum(tl.where(columns[None,:]==0,packed,0),1)
        jhh=tl.sum(tl.where(columns[None,:]==1,packed,0),1)
        jhb=tl.sum(tl.where(columns[None,:]==2,packed,0),1)
        jhg=tl.sum(tl.where(columns[None,:]==3,packed,0),1)
        jbh=tl.sum(tl.where(columns[None,:]==4,packed,0),1)
        jbb=tl.sum(tl.where(columns[None,:]==5,packed,0),1)
        jbg=tl.sum(tl.where(columns[None,:]==6,packed,0),1)
        signal=tl.sum(tl.where(columns[None,:]==7,packed,0),1)
        eg=leak*oldg+(1-leak)*arriving
        eh=jhh*oldh+jhb*eb+jhg*eg
        eb=jbh*oldh+jbb*eb+jbg*eg
        tl.store(EG+ix,eg,valid);tl.store(EH+ix,eh,valid);tl.store(EB+ix,eb,valid)
        pending=tl.load(P+ix,valid,0,cache_modifier=".cg")-LR*signal*eh
        if APPLY:
            old=tl.load(W+ix,valid,0,cache_modifier=".cg")
            total=tl.load(R+ix,valid,0,cache_modifier=".cg")+pending
            candidate=old+total
            new=tl.maximum(0.,tl.minimum(5.,candidate))
            remainder=tl.where((candidate>=0.)&(candidate<=5.),total-(new-old),0.)
            tl.store(W+ix,new,valid);tl.store(R+ix,remainder,valid);tl.store(P+ix,0.,valid)
        else:
            tl.store(P+ix,pending,valid)


def supported(tensor):
    return triton is not None and tensor.is_cuda and tensor.dtype==torch.float32


def update_gate_traces(h,b,c,packet,jac,modality,x,a,integ,v,spikes,psi,beta,rho,decay):
    n=h.shape[0]; d=x.numel()
    _gates[(triton.cdiv(h.numel(),1024),)](h,b,c,packet,jac,modality,x,a,integ,v,spikes,psi,beta,rho,n,d,decay,1024)


def update_projection_traces(state,gates,modality,x,a,integ,v,spikes,psi,beta,rho,decay):
    n,d=state.e_proj.shape
    _projection[(triton.cdiv(n*d,1024),)](state.e_proj,state.e_adaptation,state.c_gate,gates,modality,x,
                                       a,integ,v,spikes,psi,beta,rho,n,d,1024)
    state.c_gate.mul_(decay).add_(gates[:,None]*x[None,:],alpha=1-decay)


def update_edge_traces(weight,pending,pre,post,L,phi,q,z,splits,lr,apply,block=1024):
    factor=(L.reshape(-1)*phi.reshape(-1)*q.reshape(-1)).contiguous()
    if weight.numel()==0: return
    _edges[(triton.cdiv(weight.numel(),block),)](weight,pending,pre,post,factor,z,weight.numel(),
                                              L.numel(),splits[1],splits[2],splits[3],lr,apply,block)


def update_physical_traces(h,b,de,di,out,L,bio,hprev,bprev,geprev,giprev,gen,gin,
                           spikes,psi,rho,beta,rates,threshold,masks,gains,EE,EI):
    n=h.shape[0]
    _physical[(triton.cdiv(n*8,256),)](h,b,de,di,out,L,bio['v_pre'],bio['alpha_eff'],bio['beta_int'],
        bio['g_total'],hprev,bprev,geprev,giprev,gen,gin,spikes,psi,rho,beta,*rates,threshold,masks,
        gains[0],gains[1],EE,EI,n,256)


def update_projection_weights(weight,eligibility,factor,lr,decay,residual=None):
    n,d=weight.shape
    if residual is None:
        _projection_weights[(triton.cdiv(n*d,1024),)](weight,eligibility,factor,n,d,lr,decay,1024)
    else:
        _projection_compensated[(triton.cdiv(n*d,1024),)](weight,residual,eligibility,factor,n,d,lr,decay,1024, enable_fp_fusion=False)


def update_local_coba_edges(state,weight,pending,pre,post,pulses,signal,bio,direct,synaptic_decay,
                           spikes,psi,beta,rho,splits,lr,apply,*,block=256,mask_dormant=True,num_warps=4):
    # A row packs the destination's 2x2 local Jacobian and conductance
    # coefficients into one cache line; edge trace streams bypass L1.
    # If all three histories and the arriving pulse are exactly zero, their
    # next histories and instantaneous gradient remain zero. Mask only the
    # unnecessary coefficient gather; pending updates/residuals still apply.
    a=bio['alpha_eff']; v=bio['v_pre']
    reset=(1-spikes)-v*psi
    adapt=(1-rho)*psi
    coefficients=torch.stack((synaptic_decay,reset*a,v*psi*beta,reset*direct,
                               adapt*a,rho-adapt*beta,adapt*direct,signal),-1).contiguous()
    _local_coba_edges[(triton.cdiv(weight.numel(),block),)](weight,pending,state.rounding_residual,
        state.conductance,state.voltage,state.adaptation,pre,post,pulses,coefficients,weight.numel(),signal.numel(),
        splits[1],splits[2],splits[3],lr,apply,block,mask_dormant,
        num_warps=num_warps,enable_fp_fusion=False)
