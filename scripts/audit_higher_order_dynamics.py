"""Passive actual-checkpoint higher-order/energy-flow/Hodge audit, no Torch.

The graph is the existing pair-potential head graph, not an invented simplicial
body. Node signals are local Hamiltonian energies; oriented edge signals are
the actual conservative pair energy flux. No triangle is filled by convention.
"""
from __future__ import annotations

import argparse
import gzip
import io
import itertools
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np

from scripts.audit_live_matrix_dynamics import DataOnly, View


def extract(path):
    with Path(path).open('rb') as raw, gzip.GzipFile(fileobj=raw) as stream, zipfile.ZipFile(stream) as archive:
        info=next(i for i in archive.infolist() if i.filename.endswith('/data.pkl'))
        state=DataOnly(io.BytesIO(archive.read(info))).load()
        model,engine=state['model'],state['engine']
        requested={'recurrent':engine['recurrent']}
        for i,p in enumerate(engine['bfss_momenta']):requested[f'p{i}']=p
        for name in ('theta','context','frequency_scale','impedance','spectrum_revision'):
            requested[name]=model['head_coupling.'+name]
        requested['real_gradient']=model['head_coupling.last_gradient']
        matching=[(index,value) for index,value in state['optimizer']['state'].items()
            if isinstance(value.get('exp_avg'),View) and value['exp_avg'].shape==requested['theta'].shape]
        if len(matching)!=1:raise ValueError('ambiguous actual dynamics optimizer state')
        identity,optimizer=matching[0]
        optimizer_scalars={}
        for name,value in optimizer.items():
            if isinstance(value,View):requested['optimizer_'+name]=value
            else:optimizer_scalars[name]=value
        optimizer_group=next(group for group in state['optimizer']['param_groups'] if identity in group['params'])
        # Actual trained MLP fan-in, not an assertion of functional magnitude.
        for name,value in model.items():
            if name.endswith('mlp.gate.weight') and '.base.layers.' in name:
                requested[name]=value
        storages={view.storage.key:view.storage for view in requested.values()}
        prefix=info.filename[:-len('data.pkl')]
        entries=[
            (archive.getinfo(prefix+'data/'+key),key) for key in storages]
        cache={key:archive.read(info) for info,key in sorted(entries,key=lambda x:x[0].header_offset)}
        def array(v):
            return np.ndarray(v.shape,dtype=v.dtype,buffer=cache[v.storage.key],
                offset=v.offset*v.dtype.itemsize,strides=tuple(s*v.dtype.itemsize for s in v.stride)).copy()
        result={key:array(view) for key,view in requested.items()}
        result.update(config=state['config'],cycle=state['cycle'],
                      requested_bytes=sum(map(len,cache.values())),
                      optimizer_scalars=optimizer_scalars,optimizer_group=optimizer_group)
        return result


def sigmoid(x):return np.exp(-np.logaddexp(0,-x))


def omega(saved,s,refresh=False):
    theta=saved['theta'];h=s.shape[1]
    if refresh:
        flat=s.reshape(len(s),h,-1);norm=(flat*flat).sum(-1)
        corr=(flat@flat.swapaxes(-1,-2))/np.sqrt((1+norm[...,None])*(1+norm[:,None,:]))
        context=corr.mean(-1)
    else:context=saved['context'].reshape(len(s),h)
    scale=saved['frequency_scale'].reshape(len(s),h)
    return np.logaddexp(0,theta[:,:h]+theta[:,h:h+1]*context)*scale


def pairs(s,theta):
    """Unordered V_ij and both exact gradients, including content weights."""
    l,h,k,v=s.shape
    ii,jj=np.triu_indices(h,1)
    x,y=s[:,ii],s[:,jj]
    a=x@y.swapaxes(-1,-2)-y@x.swapaxes(-1,-2)
    b=x.swapaxes(-1,-2)@y-y.swapaxes(-1,-2)@x
    area=(a*a).sum((-2,-1))+(b*b).sum((-2,-1))
    nx,ny=(x*x).sum((-2,-1)),(y*y).sum((-2,-1))
    denominator=np.sqrt((1+nx)*(1+ny))
    corr=(x*y).sum((-2,-1))/denominator
    logits=theta[:,h+2:h+3]*corr+theta[:,h+3:h+4]*(np.log1p(nx)+np.log1p(ny))
    sig=sigmoid(logits);weight=2*sig
    strength=np.logaddexp(0,theta[:,h+1:h+2])/8
    dx=theta[:,h+2:h+3,None,None]*(y/denominator[...,None,None]-corr[...,None,None]*x/(1+nx[...,None,None]))
    dy=theta[:,h+2:h+3,None,None]*(x/denominator[...,None,None]-corr[...,None,None]*y/(1+ny[...,None,None]))
    dx+=2*theta[:,h+3:h+4,None,None]*x/(1+nx[...,None,None])
    dy+=2*theta[:,h+3:h+4,None,None]*y/(1+ny[...,None,None])
    wx=(2*sig*(1-sig))[...,None,None]*dx
    wy=(2*sig*(1-sig))[...,None,None]*dy
    gx=4*(a@y-y@b)
    gy=4*(-a@x+x@b)
    factor=strength[...,None,None]
    gx=factor*(weight[...,None,None]*gx+area[...,None,None]*wx)
    gy=factor*(weight[...,None,None]*gy+area[...,None,None]*wy)
    return ii,jj,strength*weight*area,gx,gy


def gradient(s,theta):
    ii,jj,v,gx,gy=pairs(s,theta)
    g=np.zeros_like(s)
    for n,(i,j) in enumerate(zip(ii,jj)):
        g[:,i]+=gx[:,n];g[:,j]+=gy[:,n]
    return g


def quiet(s,p,saved,dt,refresh=False):
    w=omega(saved,s,refresh)[...,None,None]
    z=saved['impedance'].reshape(w.shape)
    half=p-dt/2*gradient(s,saved['theta'])
    a,b=np.cos(w*dt),np.sin(w*dt)
    ns=a*s+b/z*half
    np_=a*half-z*b*s-dt/2*gradient(ns,saved['theta'])
    return ns,np_


def incidence(h,ii,jj):
    b=np.zeros((h,len(ii)))
    b[ii,np.arange(len(ii))]=-1;b[jj,np.arange(len(ii))]=1
    return b


def signals(s,p,saved):
    ii,jj,v,gx,gy=pairs(s,saved['theta'])
    w=omega(saved,s);z=saved['impedance'].reshape(w.shape)
    velocity=w[...,None,None]/z[...,None,None]*p
    flux=((velocity[:,ii]*gx).sum((-2,-1))-(velocity[:,jj]*gy).sum((-2,-1)))/2
    energy=(w/z*(p*p).sum((-2,-1))+w*z*(s*s).sum((-2,-1)))/2
    for n,(i,j) in enumerate(zip(ii,jj)):
        energy[:,i]+=v[:,n]/2;energy[:,j]+=v[:,n]/2
    b=incidence(s.shape[1],ii,jj)
    harmonic=np.eye(len(ii))-b.T@np.linalg.pinv(b@b.T)@b
    return energy,flux,flux@harmonic,b


def mixed_contrast(fn,s,target=0,inputs=(1,2),relative=.05):
    directions=[]
    for i in inputs:
        v=np.zeros_like(s);v[:,i]=s[:,i]*relative;directions.append(v)
    total=None;single=[];base=fn(s)[:,target]
    for bits in itertools.product((0,1),repeat=len(inputs)):
        y=fn(s+sum((d*b for d,b in zip(directions,bits)),np.zeros_like(s)))[:,target]
        signed=(-1)**(len(inputs)-sum(bits))*y
        total=signed.copy() if total is None else total+signed
    for d in directions:single.append(fn(s+d)[:,target]-base)
    norms=lambda a:np.linalg.norm(a.reshape(len(s),-1),axis=-1)
    reference=sum(norms(v) for v in single)
    return dict(absolute=norms(total).tolist(),first_order_scale=reference.tolist(),
        relative=(norms(total)/np.maximum(reference,1e-30)).tolist(),
        intervention=f'heads {inputs}, multiplicative {relative}; target head {target}')


def archive_topology(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    names={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    families={name:db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
              for name in ('episodes','relations','unknown_bindings','evidence','relation_variables') if name in names}
    rows=[]
    if 'relation_variables' in names:
        rows=[('r:'+str(a),'u:'+str(b)) for a,b in db.execute('SELECT relation_id,unknown_id FROM relation_variables')]
    parents={}
    def find(x):
        parents.setdefault(x,x)
        while x!=parents[x]:parents[x]=parents[parents[x]];x=parents[x]
        return x
    for a,b in rows:parents[find(a)]=find(b)
    components=len({find(x) for x in list(parents)})
    rank=len(rows)-len(parents)+components
    db.close()
    return dict(counts=families,variable_use_graph=dict(nodes=len(parents),edges=len(rows),
        components=components,beta1=rank,faces=0),
        interpretation='Actual typed relation-variable incidence graph only; no invented triangles or causal-validation claim')


def policy_regulation(checkpoint):
    path=Path(checkpoint).with_name('metrics.jsonl')
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()][-128:]
    values=[]
    for row in rows:
        p=np.asarray(row.get('choice_probability',[]),dtype=float)
        if p.shape!=(3,) or not np.all(p>0):continue
        logits=np.log(p)
        soft=lambda a:np.exp(a-a.max())/np.exp(a-a.max()).sum()
        d1=np.array([0.,.1,0.]);d2=np.array([0.,0.,.1])
        mixed=(soft(logits+d1+d2)-soft(logits+d1)-soft(logits+d2)+soft(logits))[0]
        values.append(dict(cycle=row['cycle'],mixed_probability=float(mixed),
            local_mixed_derivative=float(2*p.prod())))
    return dict(samples=len(values),events=values,perturbation_logit=.1,
        definition='Actual three-action policy distribution; vary logits 1 and 2, measure mixed change of probability 0',
        scope='Conditional softmax competition, also present in ordinary attention; no whole-world replay or learning-utility claim')


def optimizer_regulation(saved):
    """Exact scalar-observer production formula on cloned actual state."""
    group=saved['optimizer_group'];beta1,beta2=group['betas']
    count=float(saved['optimizer_step'])+1
    observer=saved['optimizer_scalars']['budget_count']+1
    power=beta1**observer
    noise=(1-beta1)/(1+beta1)*(1+power)/(1-power)
    m=saved['optimizer_exp_avg'].astype('f8');v=saved['optimizer_exp_avg_sq'].astype('f8')
    def execute(g):
        nm=beta1*m+(1-beta1)*g;nv=beta2*v+(1-beta2)*g*g
        energy=(beta1*float(saved['optimizer_budget_energy'])+(1-beta1)*np.mean(g*g))/(1-power)
        upper=(np.sqrt(np.mean(nm*nm))+power*float(saved['optimizer_budget_prior_rms']))/(1-power)
        gate=float(np.clip((upper*upper-noise*energy)/max((1-noise)*energy,1e-300),0,1))
        direction=(nm/(1-beta1**count))/(np.sqrt(nv/(1-beta2**count))+group['eps'])
        return -group['lr']*gate*(direction+group['weight_decay']*saved['theta']),gate
    g=saved['real_gradient'].astype('f8')
    selected=np.argsort(np.abs(g).flatten())[-3:][::-1]
    target,a,b=selected
    d1=np.zeros_like(g);d2=np.zeros_like(g)
    d1.flat[a]=.05*g.flat[a];d2.flat[b]=.05*g.flat[b]
    base,gate=execute(g)
    a1=execute(g+d1)[0];a2=execute(g+d2)[0];both=execute(g+d1+d2)[0]
    # Fixing the original scalar gate leaves coordinatewise AdamW, a control
    # with exactly zero other-coordinate effect on the target coordinate.
    return dict(target_coordinate=int(target),perturbed_coordinates=[int(a),int(b)],
        cloned_next_gate=gate,stored_last_gate=float(saved['optimizer_budget_gate']),
        target_step=float(base.flat[target]),mixed_target_step=float((both-a1-a2+base).flat[target]),
        fixed_gate_mixed_target_step=0.,
        definition='Two other actual gradient coordinates change the target executed update through the existing shared budget observer',
        scope='One hypothetical next optimizer proposal using saved actual gradient and state; no live update or invented learning task')


def diagnose(saved,steps):
    l,_,h,k,v=np.stack([saved[f'p{i}'] for i in range(len(saved['context']))]).shape
    p=np.stack([saved[f'p{i}'] for i in range(l)]).reshape(l,h,k,v).astype('f8')
    s=saved['recurrent'][...,:h*k*v].reshape(l,h,k,v).astype('f8')
    for key in ('theta','context','frequency_scale','impedance'):saved[key]=saved[key].astype('f8')
    energy,flow,harm,b=signals(s,p,saved)
    g=gradient(s,saved['theta'])
    direction=s/np.linalg.norm(s)
    eps=1e-5
    numerical=(pairs(s+eps*direction,saved['theta'])[2].sum()-pairs(s-eps*direction,saved['theta'])[2].sum())/(2*eps)
    analytical=(g*direction).sum()
    d=np.zeros((h+len(flow[0]),h+len(flow[0])))
    d[:h,h:]=b;d[h:,:h]=b.T
    lap=np.zeros_like(d);lap[:h,:h]=b@b.T;lap[h:,h:]=b.T@b
    dt=saved['config']['bfss_dynamics']['dt']
    w=omega(saved,s)[...,None,None];z=saved['impedance'].reshape(w.shape)
    velocity=w/z*p;acceleration=-w*z*s-g
    eps_energy=1e-6
    derivative=(signals(s+eps_energy*velocity,p+eps_energy*acceleration,saved)[0]-
                signals(s-eps_energy*velocity,p-eps_energy*acceleration,saved)[0])/(2*eps_energy)
    ii,jj,pair_energy,_,_=pairs(s,saved['theta'])
    signless=np.abs(b)
    metric=np.einsum('ie,le,je->lij',signless,2*pair_energy,signless)
    eigen=np.linalg.eigvalsh(metric)
    result=dict(schema=1,cycle=saved['cycle'],requested_bytes=saved['requested_bytes'],torch_imported=False,
        world_calls=0,optimizer_updates=0,finite_difference_gradient_error=abs(numerical-analytical),
        energy_continuity_relative_error=float(np.linalg.norm(derivative-flow@b.T)/
            max(np.linalg.norm(flow@b.T),1e-30)),
        amplitude_compatibility_metric=dict(
            definition='Frozen pair-weight Gauss-Newton metric of commutator constraints for delta S_h=a_h*S_h; signless incidence, not Hodge',
            smallest_eigenvalue=eigen[:,0].tolist(),largest_eigenvalue=eigen[:,-1].tolist(),
            nullity=(eigen<=eigen[:,-1:]*1e-10).sum(-1).tolist()),
        units='heads are units for interaction order; energy polynomial degree is not body order',
        instant_force=mixed_contrast(lambda x:gradient(x,saved['theta']),s),
        finite_quiet_map=mixed_contrast(lambda x:quiet(x,p,saved,dt)[0],s),
        refreshed_context_map=mixed_contrast(lambda x:quiet(x,p,saved,dt,True)[0],s),
        hodge=dict(nodes=h,edges=flow.shape[1],faces=0,beta0=1,beta1=flow.shape[1]-h+1,
            isotropic_edge_flow_expected_harmonic_fraction=(flow.shape[1]-h+1)/flow.shape[1],
            dirac_square_error=float(np.linalg.norm(d@d-lap)),
            divergence_of_harmonic_error=float(np.abs(harm@b.T).max()),
            instantaneous_harmonic_flow_fraction=((harm*harm).sum(-1)/np.maximum((flow*flow).sum(-1),1e-300)).tolist()),
        layers=[dict(layer=i+1,local_energy=energy[i].tolist(),oriented_pair_flux=flow[i].tolist(),
            coupling_potential=float(pairs(s[i:i+1],saved['theta'][i:i+1])[2].sum()),
            coupling_force_norm=float(np.linalg.norm(g[i]))) for i in range(l)])
    # A bounded actual-state quiet trajectory, not a fresh toy memory task.
    chosen=sorted(set((1,l//2-1,l-1)))
    sub={key:saved[key][chosen].copy() for key in ('theta','context','frequency_scale','impedance')}
    def multiple(x):
        y=p[chosen].copy()
        for _ in range(64):x,y=quiet(x,y,sub,dt)
        return x
    result['sixty_four_step_map']=mixed_contrast(multiple,s[chosen])
    x,y=s[chosen].copy(),p[chosen].copy()
    _,_,initial,_=signals(x,y,sub)
    histories=[initial.copy()]
    for t in range(steps):
        x,y=quiet(x,y,sub,dt)
        if (t+1)%16==0:histories.append(signals(x,y,sub)[2])
    path=np.stack(histories)
    norm=lambda q:np.linalg.norm(q,axis=-1)
    result['quiet_trajectory']=dict(layers=[i+1 for i in chosen],steps=steps,
        harmonic_endpoint_cosine=((initial*path[-1]).sum(-1)/np.maximum(norm(initial)*norm(path[-1]),1e-300)).tolist(),
        harmonic_relative_change=(norm(path[-1]-initial)/np.maximum(norm(initial),1e-300)).tolist(),
        finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),
        scope='frozen learned parameters and context, no input; finite evolution of actual state, not memory utility or chaos proof')
    mlp=[]
    for key,w in saved.items():
        if key.endswith('mlp.gate.weight'):
            block=np.linalg.norm(w.reshape(w.shape[0],h,-1),axis=-1)
            mlp.append(dict(layer=int(key.split('.layers.')[1].split('.')[0])+1,
                minimum_input_head_blocks_per_unit=int((block>0).sum(-1).min())))
    result['mlp_structural_fan_in']=mlp
    result['archive']=archive_topology(saved['config']['causal_archive']['snapshot'])
    result['optimizer_regulation']=optimizer_regulation(saved)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=256)
    args=parser.parse_args()
    saved=extract(args.checkpoint)
    result=diagnose(saved,args.steps)
    result['policy_regulation']=policy_regulation(args.checkpoint)
    args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('cycle','requested_bytes','finite_difference_gradient_error','quiet_trajectory','archive')}))


if __name__=='__main__':main()
