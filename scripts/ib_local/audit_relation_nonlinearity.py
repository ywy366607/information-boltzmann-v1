"""Frozen-body structural comparison on real archived mother interactions.

No world requests, optimizer steps, model construction, or CUDA allocation.
The chronological regressions below are readout diagnostics, not body training
or historical prospective scores: all records use one saved encoder.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expand(x, kind):
    if kind == 'condition':
        return x[:, :2]
    if kind == 'linear':
        return x
    c, a, b = x[:, 1], x[:, 2], x[:, 3]
    extra = np.stack((c*a, c*b, a*a, a*b, b*b, c*c), -1)
    return np.concatenate((x, extra), -1)


def ridge_predict(x, y, q, ridge):
    return q @ np.linalg.solve(x.T @ x + ridge*np.eye(x.shape[1]), x.T @ y)


def chronological_probe(x, residual, base, target, modes):
    """Earlier fit/validation chooses ridge; last quarter is untouched test."""
    n = len(x)
    fit, split = n//2, 3*n//4
    result = {}
    for kind in ('condition', 'linear', 'quadratic'):
        phi = expand(x, kind)
        options = []
        for lam in (.001, .01, .1, 1., 10., 100.):
            pred = base[fit:split] + ridge_predict(phi[:fit], residual[:fit], phi[fit:split], lam)
            options.append((float(np.square(pred-target[fit:split]).mean()), lam))
        valid, lam = min(options)
        pred = base[split:] + ridge_predict(phi[:split], residual[:split], phi[split:], lam)
        per = np.square(pred-target[split:]).mean(-1)
        result[kind] = dict(dimension=phi.shape[1], ridge=lam, validation_mse=valid,
            test_mse=float(per.mean()), by_action={str(m):dict(n=int((modes[split:]==m).sum()),
                mse=float(per[modes[split:]==m].mean())) for m in np.unique(modes[split:])},
            support_rank=int(np.linalg.matrix_rank(phi[:split])))
    result['base_test_mse'] = float(np.square(base[split:]-target[split:]).mean())
    result['sizes'] = dict(fit=fit, validation=split-fit, test=n-split)
    return result


def local_budget_probe(x, residual, base, target, source):
    """Same target, historical nearest contexts; isolate count from nonlinearity.

    New diagnostic only: this is NOT the production unknown/scope selector.
    Ridge is selected exclusively on the earlier validation quarter.
    """
    n=len(x)
    fit, split=n//2,3*n//4
    normalized=source/np.maximum(np.linalg.norm(source,axis=-1,keepdims=True),1e-12)
    neighbors={i:np.argsort(-(normalized[:i]@normalized[i]),kind='stable')[:128]
               for i in range(fit,n)}
    result={}
    for budget in (8,32,128):
        for kind in ('linear','quadratic'):
            phi=expand(x,kind)
            errors={}
            for lam in (.01,.1,1.,10.):
                per=[]
                for i in range(fit,n):
                    use=neighbors[i][:budget]
                    pred=base[i]+ridge_predict(phi[use],residual[use],phi[i:i+1],lam)[0]
                    per.append(float(np.square(pred-target[i]).mean()))
                errors[lam]=np.asarray(per)
            valid_count=split-fit
            lam=min(errors,key=lambda l:float(errors[l][:valid_count].mean()))
            result[f'{kind}_{budget}']=dict(ridge=lam,
                validation_mse=float(errors[lam][:valid_count].mean()),
                test_mse=float(errors[lam][valid_count:].mean()))
    return result


def representation_probe(x,residual,base,target,source,action):
    """Check compressed nonlinear capacity versus already available inputs."""
    n=len(x);fit=n//2;split=3*n//4
    raw_condition=np.concatenate((np.ones((n,1)),source),-1)
    raw_action=np.concatenate((raw_condition,action-source),-1)
    centers=x[np.linspace(0,fit-1,64,dtype=int),1:]
    distances=np.square(centers[:,None]-centers[None]).sum(-1)
    bandwidth=float(np.median(distances[distances>0]))
    radial=np.exp(-np.square(x[:,None,1:]-centers[None]).sum(-1)/bandwidth)
    designs=dict(shared_linear=x,shared_rbf=np.concatenate((x,radial),-1),
        raw_condition=raw_condition,raw_condition_action=raw_action)
    result={};held={}
    for name,phi in designs.items():
        # Same fitting-only column normalization for all expressive variants.
        phi=phi.copy()
        phi[:,1:]=(phi[:,1:]-phi[:fit,1:].mean(0))/np.maximum(phi[:fit,1:].std(0),1e-6)
        options=[]
        for lam in (.01,.1,1.,10.,100.,1000.):
            pred=base[fit:split]+ridge_predict(phi[:fit],residual[:fit],phi[fit:split],lam)
            options.append((float(np.square(pred-target[fit:split]).mean()),lam))
        valid,lam=min(options)
        pred=base[split:]+ridge_predict(phi[:split],residual[:split],phi[split:],lam)
        per=np.square(pred-target[split:]).mean(-1)
        held[name]=per
        result[name]=dict(dimension=phi.shape[1],ridge=lam,validation_mse=valid,test_mse=float(per.mean()))
    # Time blocks, rather than independent-coordinate pseudoreplication.
    blocks=[v for v in np.array_split(np.arange(n-split),16) if len(v)]
    rng=np.random.default_rng(114)
    draws=rng.integers(len(blocks),size=(2000,len(blocks)))
    for name in held:
        change=held[name]-held['shared_linear']
        b=np.array([change[v].mean() for v in blocks])
        result[name]['test_mse_delta_vs_shared_linear']=float(change.mean())
        result[name]['block_bootstrap_delta_interval']=np.quantile(b[draws].mean(-1),[.025,.975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive-root', type=Path,
                        help='directory of unchanged manifest-addressed archive copies')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'preregistered.json').write_text(json.dumps(dict(
        question='Does nonlinear residual prediction improve held-out actual mother outcomes?',
        comparisons=['condition-only', 'current four-coordinate linear', 'same coordinates quadratic'],
        followup='same saved eight-fact linear/quadratic fit; nearest past context budgets 8/32/128',
        representation_followup='same chronological split: shared linear/RBF and full latent condition/condition-action readouts',
        split='time-ordered half fit, quarter ridge validation, quarter untouched test',
        fixed='one frozen encoder/prior/basis and identical archived actual labels',
        scope='retrospective readout diagnostic; not historical prospective or causal intervention',
        mutations='none to body, optimizer, world, RNG or archive'), indent=2), encoding='utf-8')
    # A standard trusted local checkpoint read, without restoring its full body
    # or building the large n-gram table a second time.
    import torch
    torch.set_num_threads(1)
    before = digest(args.checkpoint)
    with gzip.open(args.checkpoint, 'rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=False)
    cycle, manifest = saved['cycle'], saved['config']['causal_archive']
    sd, engine = saved['model'], saved['engine']
    weights = {k:v.detach().numpy().astype(np.float64).copy() for k,v in sd.items()
               if k == 'encoder.weight' or k.startswith(('long_predictor.', 'mother_from_boundary.', 'relation_basis.'))}
    d = engine['_program_forecasts']['path']['mother']
    arr = lambda t:t.detach().double().numpy().copy()
    exact = {k:arr(d[k]) for k in ('x', 'query', 'coefficient', 'mean', 'base')}
    exact['y']=arr(d['y'])
    local_ridge=float(d['ridge'])
    exact['target'] = arr(engine['_mother_boundary_target']) @ arr(engine['_mother_pre_encoder']).T
    chosen = int(engine['_pending_record']['choice'])
    del d, sd, engine, saved
    gc.collect()
    print(f'frozen checkpoint {cycle} extracted; no body/table rebuilt', flush=True)
    dbpath = Path(manifest['snapshot']).resolve()
    if args.archive_root is not None:
        dbpath=(args.archive_root/dbpath.name).resolve()
    archive_before = digest(dbpath)
    if archive_before != manifest['sha256']:
        raise ValueError('archive manifest mismatch')
    db = sqlite3.connect(dbpath.as_uri()+'?mode=ro', uri=True)
    schema = json.loads(db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0])
    layouts, offset = {}, 0
    for key in sorted(schema):
        spec = schema[key]
        count = int(np.prod(spec['shape'])) if spec['shape'] else 1
        layouts[key] = (offset, count, spec)
        offset += count*np.dtype(spec['dtype']).itemsize
    def unpack(payload, key):
        off, count, spec = layouts[key]
        return np.frombuffer(payload, dtype=spec['dtype'], offset=off, count=count).reshape(spec['shape']).copy()
    facts = []
    for eid, stamp, payload in db.execute('SELECT id,stamp,payload FROM episodes ORDER BY stamp,id'):
        if (bool(unpack(payload,'mother_valid')) and int(unpack(payload,'count')) == 1
                and int(unpack(payload,'mother_stamp')) == stamp):
            facts.append((eid, stamp, *(unpack(payload,k) for k in ('source','action','mother_next','option'))))
    db.close()
    source = np.stack([f[2] for f in facts]) @ weights['encoder.weight'].T
    action = np.stack([f[3] for f in facts]) @ weights['encoder.weight'].T
    target = np.stack([f[4] for f in facts]) @ weights['encoder.weight'].T
    modes = np.array([int(f[5]) for f in facts])
    lin = lambda x, name: x @ weights[name+'.weight'].T + weights.get(name+'.bias', 0.)
    hidden = lin(np.concatenate((source,action), -1),'long_predictor.0')
    hidden = hidden/(1+np.exp(-np.clip(hidden,-700,700)))
    base = lin(hidden,'long_predictor.2')
    base = base + lin(base, 'mother_from_boundary')
    c = np.tanh(lin(source,'relation_basis.condition'))
    a0 = lin(action-source,'relation_basis.action')
    gate = 2/(1+np.exp(-np.clip(lin(source,'relation_basis.gate'),-700,700)))
    u = np.tanh(a0)*gate
    x = np.concatenate((np.ones_like(c),c,u), -1)
    residual = target-base
    # Decompose the EXACT saved pre-response forecast, retaining its original
    # encoder, support, momentum contribution and target, not current replays.
    correction = exact['mean']-exact['base']
    common = correction.mean(0, keepdims=True)
    alternatives = [i for i in range(len(correction)) if i != chosen]
    variants = dict(base=exact['base'], full=exact['mean'],
        common_only=exact['base']+common, contrast_only=exact['base']+correction-common)
    outcomes = {}
    for name, pred in variants.items():
        mse = np.square(pred-exact['target'].reshape(1,-1)).mean(-1)
        outcomes[name] = dict(chosen_mse=float(mse[chosen]),
            margin=float(mse[alternatives].mean()-mse[chosen]))
    reconstruction=exact['base']+(exact['query']@exact['x'].T)@exact['coefficient']
    reproduction=float(np.abs(reconstruction-exact['mean']).max())
    if reproduction>1e-5:
        raise ValueError('saved original forecast decomposition mismatch')
    local_forms={}
    for kind in ('linear','quadratic'):
        phi,q=expand(exact['x'],kind),expand(exact['query'],kind)
        pred=exact['base']+ridge_predict(phi,exact['y'],q,local_ridge)
        mse=np.square(pred-exact['target'].reshape(1,-1)).mean(-1)
        held=[]
        for i in range(len(phi)):
            keep=np.arange(len(phi))!=i
            p=ridge_predict(phi[keep],exact['y'][keep],phi[i:i+1],local_ridge)
            held.append(float(np.square(p-exact['y'][i]).mean()))
        local_forms[kind]=dict(chosen_mse=float(mse[chosen]),
            support_leave_one_out_mse=float(np.mean(held)),
            dimension=phi.shape[1],rank=int(np.linalg.matrix_rank(phi)),ridge=local_ridge)
    xc = exact['x'][:,:2]
    xa = exact['x'][:,2:4]
    ra = xa-xc @ np.linalg.pinv(xc) @ xa
    # Rank of condition alone versus the richer proposed local fit is a
    # sample-identification check; never turn interpolated training fit into gain.
    singular = np.linalg.svd(x-x.mean(0),compute_uv=False)
    report = dict(checkpoint_cycle=cycle,checkpoint_sha256=before,
        archive=str(dbpath),archive_sha256=archive_before,facts=len(facts),
        exact_forecast=dict(chosen=chosen,support=len(exact['x']),variants=outcomes,
            reproduction_error=reproduction,nonlinear_same_support=local_forms,
            correction_common_energy=float(np.square(common).mean()),
            correction_contrast_energy=float(np.square(correction-common).mean()),
            conditional_action_rank=int(np.linalg.matrix_rank(ra,tol=1e-6)),
            conditional_action_singular=np.linalg.svd(ra,compute_uv=False).tolist()),
        shared_coordinates=dict(condition_std=c.std(0).tolist(),action_std=u.std(0).tolist(),
            action_tanh_saturated_fraction=float((np.abs(np.tanh(a0))>.95).mean()),
            gate_mean=gate.mean(0).tolist(),gate_std=gate.std(0).tolist(),
            centered_singular=singular.tolist()),
        chronological=chronological_probe(x,residual,base,target,modes),
        nearest_context_budget=local_budget_probe(x,residual,base,target,source),
        representation=representation_probe(x,residual,base,target,source,action),
        scope='Frozen current representation retrospective readouts, not historical forecasts or world interventions',
        training_mutations=0,world_calls=0)
    report['source_unchanged'] = digest(args.checkpoint)==before
    report['archive_unchanged'] = digest(dbpath)==archive_before
    assert report['source_unchanged'] and report['archive_unchanged']
    np.savez_compressed(args.output/'diagnostic_arrays.npz',x=x,source=source,action=action,
                        target=target,base=base,modes=modes)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
