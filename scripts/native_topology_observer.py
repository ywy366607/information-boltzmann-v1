"""Passive observation of actual native states and executed operations.

No five-phase graph, coactivity faces, filtered loss phases, RNG draws or
training objectives are introduced. Installation is external to production
execution_digest so original pending events can first be recovered unchanged.
"""
from __future__ import annotations

import functools
import gzip
import inspect
import dataclasses
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import torch

from scripts.audit_higher_order_dynamics import pairs, omega, incidence

SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def matrix_observation(s, p, theta, context, frequency_scale, impedance, previous=None):
    """All physical heads and unordered potential edges, in native coordinates.

    Complex phase is matrix-valued. Pair arguments are meaningful only together
    with their overlap amplitude; no scalar rotor is assigned to a weak pair.
    """
    s, p = np.asarray(s, dtype=np.float64), np.asarray(p, dtype=np.float64)
    layers, heads = s.shape[:2]
    z = np.asarray(impedance).reshape(layers, heads)
    saved = dict(theta=np.asarray(theta), context=np.asarray(context),
                 frequency_scale=np.asarray(frequency_scale), impedance=z)
    w = omega(saved, s)
    ii, jj, potential, gx, gy = pairs(s, saved['theta'])
    velocity = (w/z)[..., None, None]*p
    flux = ((velocity[:, ii]*gx).sum((-2, -1))-
            (velocity[:, jj]*gy).sum((-2, -1)))/2
    physical = z[..., None, None]**.5*s+1j*p/z[..., None, None]**.5
    flat = physical.reshape(layers, heads, -1)
    norm = np.linalg.norm(flat, axis=-1)
    gram = flat@flat.conj().swapaxes(-1, -2)
    denominator = norm[..., None]*norm[:, None, :]
    overlap = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 1e-24)
    phase_delta = np.full_like(norm, np.nan)
    temporal_overlap = np.zeros_like(norm)
    if previous is not None:
        old = np.asarray(previous).reshape(flat.shape)
        inner = (flat*old.conj()).sum(-1)
        den = norm*np.linalg.norm(old, axis=-1)
        valid = den > 1e-24
        temporal_overlap = np.divide(abs(inner), den, out=np.zeros_like(den), where=valid)
        phase_delta[valid] = np.angle(inner[valid])
    # Exact content dependence of the production pair coefficient.
    norm_s = (s*s).sum((-2, -1))
    correlation = (s[:, ii]*s[:, jj]).sum((-2, -1))/np.sqrt(
        (1+norm_s[:, ii])*(1+norm_s[:, jj]))
    logits = theta[:, heads+2:heads+3]*correlation+theta[:, heads+3:heads+4]*(
        np.log1p(norm_s[:, ii])+np.log1p(norm_s[:, jj]))
    weight = 2*np.exp(-np.logaddexp(0., -logits))
    energy = .5*(w/z*(p*p).sum((-2, -1))+w*z*(s*s).sum((-2, -1)))
    for edge, (i, j) in enumerate(zip(ii, jj)):
        energy[:, i] += potential[:, edge]/2
        energy[:, j] += potential[:, edge]/2
    b = incidence(heads, ii, jj)
    harmonic = np.eye(len(ii))-b.T@np.linalg.pinv(b@b.T)@b
    return dict(node_energy=energy, omega=w, impedance=z,
        complex_overlap_real=overlap.real, complex_overlap_imag=overlap.imag,
        complex_norm=norm, temporal_overlap=temporal_overlap, phase_delta=phase_delta,
        edge_potential=potential, edge_weight=weight, edge_flux=flux,
        edge_force_left=np.linalg.norm(gx.reshape(layers, len(ii), -1), axis=-1),
        edge_force_right=np.linalg.norm(gy.reshape(layers, len(ii), -1), axis=-1),
        edge_graph_harmonic=flux@harmonic, theta=np.asarray(theta).copy()), physical


class TensorStore:
    """Lossless bounded-cache payload store, including exact broadcast shapes."""
    def __init__(self,path,cache_size=8192):
        self.stream=Path(path).open('wb')
        self.cache=OrderedDict();self.cache_size=cache_size

    def put(self,value):
        value=value.detach()
        shape=list(value.shape)
        for dim,(size,stride) in enumerate(zip(value.shape,value.stride())):
            if size>1 and stride==0:value=value.narrow(dim,0,1)
        array=value.cpu().contiguous().numpy()
        raw=array.tobytes()
        sha=hashlib.sha256(raw).hexdigest()
        key=(array.dtype.str,tuple(array.shape),sha)
        ref=self.cache.get(key)
        if ref is None:
            payload=zlib.compress(raw,1);offset=self.stream.tell();self.stream.write(payload)
            ref=dict(offset=offset,compressed_bytes=len(payload),dtype=array.dtype.str,
                stored_shape=list(array.shape),sha256=sha)
            self.cache[key]=ref
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        else:self.cache.move_to_end(key)
        return dict(tensor_ref=dict(ref,shape=shape))

    def flush(self):self.stream.flush()
    def close(self):self.stream.close()


def load_tensor(path,reference):
    r=reference['tensor_ref']
    with Path(path).open('rb') as f:
        f.seek(r['offset']);raw=zlib.decompress(f.read(r['compressed_bytes']))
    if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('tensor payload checksum mismatch')
    array=np.frombuffer(raw,dtype=r['dtype']).reshape(r['stored_shape'])
    return np.broadcast_to(array,r['shape'])


def _json(value,store=None):
    if torch.is_tensor(value):
        if store is not None:return store.put(value)
        a = value.detach().cpu().numpy()
        # Exact bound port vectors retained, not just a norm.
        return dict(shape=list(a.shape), values=a.tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json(x,store) for x in value]
    if isinstance(value, dict):
        return {str(k):_json(v,store) for k,v in value.items()}
    if dataclasses.is_dataclass(value):
        return {f.name:_json(getattr(value,f.name),store) for f in dataclasses.fields(value)}
    if hasattr(value, 'value'):
        return value.value
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


class NativeTopologyObserver:
    """All-layer physical traces plus real operator/learning/world receipts."""
    def __init__(self, engine, directory, *, raw_anchor_limit=12, chunk_size=64):
        self.engine, self.directory = engine, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = gzip.open(self.directory/'operations.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
        self.tensors=TensorStore(self.directory/'tensors.bin')
        self.clock, self.quiet_time, self.update_clock = 0, 0., 0
        self.rows, self.chunk_index, self.chunk_size = [], 0, chunk_size
        self.raw_anchor_limit, self.anchors = raw_anchor_limit, 0
        self.undo, self.handles, self.closed = [], [], False
        from scripts.native_topology_compute import install
        self.restore_compute=install()
        self.external_cycle = None
        self.module = engine.model.head_coupling
        self.first = engine.model.body.base.layers[0]
        self.record('contract', dict(schema=1, layers=self.module.layers, heads=self.module.heads,
            native_edges='actual unordered head-pair potential; no filled faces',
            scope='all real internal/external/learning events; graph harmonic is graph-only',
            dirac_status='no independently identified native edge phase: not evaluated',
            functions='matrix_observation; actual typed mechanisms; actual optimizer/world boundaries'))
        self.record('compute_workspace',dict(direction_chunk=2,production_chunk=4,
            retained='all local JVP directions and exact original update; complete event equality required'))
        self._install()
        from scripts.native_relation_signals import install_relation_signals, SOURCE_SHA256 as signal_sha
        install_relation_signals(self)
        self.record('relation_signal_contract', dict(schema=1, source_sha256=signal_sha,
            source='existing calculations only; no additional model/world calls',
            missing='not zero; no conversion of node differences into native edge activity',
            branch_identity='call nesting, functional evidence, bound ports and parameter update clock',
            admission='full event equality must be rerun before any future live collection'))

    def record(self, kind, data):
        self.events.write(json.dumps(dict(kind=kind, event=self.clock,
            external_cycle=self.external_cycle, quiet_time=self.quiet_time,
            update_clock=self.update_clock, data=_json(data,self.tensors)), ensure_ascii=False, allow_nan=False)+'\n')

    def _replace(self, owner, name, replacement):
        original = getattr(owner, name)
        self.undo.append((owner, name, original))
        setattr(owner, name, replacement)
        return original

    @torch.no_grad()
    def arrays(self, recurrent=None, momenta=None):
        e, m, f = self.engine, self.module, self.first
        recurrent = e.recurrent if recurrent is None else recurrent
        momenta = e.bfss_momenta if momenta is None else momenta
        s = recurrent[..., :f.matrix_size].reshape(m.layers, m.heads, m.dk, m.dv)
        p = torch.stack(momenta).reshape_as(s)
        context=m.context
        if not bool(m.context_valid):
            from scripts.ib_local.adaptive_hamiltonian import invariants
            corr,_=invariants(s)
            context=corr.mean(-1).reshape_as(m.context)
        return {k:v.detach().cpu().numpy().copy() for k,v in dict(S=s,P=p,
            theta=m.theta,context=context,frequency_scale=m.frequency_scale,impedance=m.impedance).items()}

    def sample(self, kind, recurrent=None, momenta=None, *, previous=None, dt=0.):
        a = self.arrays(recurrent, momenta)
        values, physical = matrix_observation(a['S'],a['P'],a['theta'],a['context'],
            a['frequency_scale'],a['impedance'],previous)
        values={k:v.astype(np.float32) for k,v in values.items()}
        values.update(kind=np.asarray(kind), event=np.asarray(self.clock),
            quiet_time=np.asarray(self.quiet_time), dt=np.asarray(dt), updates=np.asarray(self.update_clock))
        self.rows.append(values)
        if (self.anchors < self.raw_anchor_limit and kind=='event_after'
                and self.clock>0 and self.clock & (self.clock-1)==0):
            np.savez_compressed(self.directory/f'anchor_{self.anchors:03}.npz', **a,
                kind=np.asarray(kind),event=np.asarray(self.clock),quiet_time=np.asarray(self.quiet_time))
            self.anchors += 1
        if len(self.rows) >= self.chunk_size:
            self.flush()
        return physical

    def flush(self):
        if self.rows:
            arrays = {k:np.stack([row[k] for row in self.rows]) for k in self.rows[0]}
            np.savez_compressed(self.directory/f'physical_{self.chunk_index:05}.npz', **arrays)
            self.chunk_index += 1
            self.rows.clear()
        self.events.flush()
        self.tensors.flush()

    def _install(self):
        m, e = self.module, self.engine
        quiet_original = m.quiet
        @functools.wraps(quiet_original)
        def quiet(*args, **kw):
            # One original call may have solver subdivisions; record its actual
            # dt/steps and final state, without invoking additional evolution.
            caller=inspect.currentframe().f_back.f_code.co_name
            committed=bool(kw.get('capture',False))
            if committed and caller not in ('quiet_packed','continue_internal'):
                raise ValueError(f'unclassified actual quiet caller: {caller}')
            prefix='' if committed else 'preview_'
            before = self.sample(prefix+'quiet_before', args[1], args[2])
            result = quiet_original(*args, **kw)
            dt = float(kw['dt'])*int(kw.get('steps',1))
            if committed:self.quiet_time += dt
            self.sample(prefix+'quiet_after',result[0],result[1],previous=before,dt=dt)
            self.record('quiet_execution',dict(caller=caller,committed_SP=committed,
                steps=int(kw.get('steps',1)),dt=float(kw['dt']),
                solver_substeps=int(m.last_solver_substeps)))
            return result
        self._replace(m, 'quiet', quiet)
        original_advance = type(e)._advance
        @functools.wraps(original_advance)
        def advance(owner, *args, **kw):
            self.sample('update_before')
            value = original_advance(owner, *args, **kw)
            self.sample('update_after')
            return value
        self._replace(type(e), '_advance', advance)
        original_step = e.optimizer.step
        parameter_names={id(p):name for name,p in e.model.named_parameters()}
        @functools.wraps(original_step)
        def step(*args, **kw):
            before=m.theta.detach().clone()
            grad = None if m.theta.grad is None else m.theta.grad.detach().clone()
            participants=[]
            with torch.no_grad():
                for group in e.optimizer.param_groups:
                    for parameter in group['params']:
                        if parameter.grad is None:continue
                        n=min(64,parameter.numel())
                        index=torch.arange(n,device=parameter.device)*(parameter.numel()-1)//max(n-1,1)
                        participants.append((parameter_names[id(parameter)],parameter,index,
                            parameter.detach().flatten()[index].clone(),parameter.grad.detach().norm(),float(group['lr'])))
            value=original_step(*args,**kw)
            callback = getattr(self, 'after_optimizer_proposal', None)
            if callback is not None:
                callback(m, before)
            self.update_clock += 1
            state=e.optimizer.state.get(m.theta,{})
            self.record('optimizer',dict(theta_before=before,theta_after=m.theta,
                actual_gradient=grad,applied_delta=m.theta.detach()-before,
                budget_gate=state.get('budget_gate'), exp_avg=state.get('exp_avg'),
                exp_avg_sq=state.get('exp_avg_sq'), step=state.get('step')))
            with torch.no_grad():
                statistics=torch.stack([torch.stack((gn,
                    (parameter.detach().flatten()[index]-old).square().mean().sqrt()))
                    for _,parameter,index,old,gn,_ in participants]).cpu().tolist()
            self.record('all_parameter_update_participation',dict(
                meaning='every actual gradient-bearing parameter; full gradient norm and deterministic <=64-coordinate applied-step RMS',
                parameters=[dict(name=name,lr=lr,gradient_norm=stats[0],sampled_applied_step_rms=stats[1])
                    for (name,parameter,index,old,gn,lr),stats in zip(participants,statistics)]))
            return value
        self._replace(e.optimizer,'step',step)
        original_take=e.world.take
        @functools.wraps(original_take)
        def take(*args,**kw):
            utterance=args[0] if args else kw.get('utterance',b'')
            result=original_take(*args,**kw)
            self.record('world',dict(utterance_hex=bytes(utterance).hex(),response_hex=bytes(result[0]).hex(),
                holdout_hex=bytes(result[1]).hex(),info=result[2]))
            return result
        self._replace(e.world,'take',take)
        # Preserve every imported alias of the actual function, not a duplicate
        # predictor or an imagined five-port incidence matrix.
        from scripts.ib_local import causal_relation_program as cp
        originals={name:getattr(cp,name) for name in ('mechanism','execute','revise')}
        replacements={}
        for name, function in originals.items():
            def make(label, fn):
                @functools.wraps(fn)
                def wrapper(*args,**kwargs):
                    result=fn(*args,**kwargs)
                    if label=='mechanism':
                        state=args[1]
                        self.record('mechanism',dict(identity=state.identity,generation=state.generation,
                            version=state.version,kind=kwargs.get('kind',cp.Port.MOTHER),
                            source=args[2],action=args[3],
                            mother=kwargs.get('mother'),momentum=kwargs.get('momentum'),
                            mean=result.get('mean'),base=result.get('base'),
                            variable_uses=getattr(state,'variable_uses',())))
                    elif label=='execute':
                        self.record('typed_execution',dict(calls=[dict(operation=c.operation.name,
                            inputs=[(v.identity,v.port.value) for v in c.arguments],
                            output=(c.result.identity,c.result.port.value)) for c in args[0]],
                            actual_values=[dict(identity=v.identity,port=v.port.value,tensor=p.tensor)
                                           for v,p in result[0].items()],
                            alternative_outputs=[dict(operation=c.operation.name,
                                identity=c.result.identity,port=c.result.port.value,tensor=p.tensor)
                                for c,p in result[1]]))
                    else:
                        self.record('relation_revision',dict(identity=args[0].identity,
                            version_before=args[0].version,version_after=result.version,
                            observed=args[1].observed,kind=args[1].kind,fact_id=args[1].fact_id,
                            source=args[1].source,action=args[1].action,outcome=args[1].outcome))
                    return result
                return wrapper
            replacements[name]=make(name,function)
        for module in list(sys.modules.values()):
            if module is None or not getattr(module,'__name__','').startswith('scripts.ib_local'):
                continue
            for name, original in originals.items():
                if getattr(module,name,None) is original:
                    self._replace(module,name,replacements[name])

    def begin_event(self, external_cycle):
        self.clock += 1
        self.external_cycle=external_cycle
        self.sample('event_before')

    def end_event(self, row):
        self.sample('event_after')
        # Exact original outcome/choice/credit/relationship receipts, including
        # version identities. No reconstruction from a sliding visit window.
        self.record('event', {k:v for k,v in row.items() if k not in ('films',)})
        if row.get('autonomous_event',{}).get('kind')=='external':
            db=self.engine.model._causal_archive.db
            self.record('structural_relation_dependencies',dict(
                edges=db.execute('SELECT relation_id,unknown_id,port,generation,seen_version '
                                 'FROM relation_variables ORDER BY relation_id,unknown_id').fetchall(),
                meaning='persisted exact dependencies, distinct from current activation and physical head edges'))
        self.flush()

    def close(self):
        if self.closed:return
        for owner,name,original in reversed(self.undo):setattr(owner,name,original)
        for handle in self.handles:handle.remove()
        self.restore_compute()
        self.flush();self.events.close();self.tensors.close();self.closed=True
