"""Three locked causal event/read trajectories; no production modification.

Registration is CPU only. Execution needs a separate content-bound independent
approval. One original saved-Adam update is disposable; thirteen32-event
forwards (416ticks) and one backward are the hard ceiling. No retry or .pt save.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OLD_PATH = ROOT/'scripts/ib/diagnose_fly_pipeline_repair.py'
OLD_SHA = 'c9e88111a410b08caccf90aef0766971d63669b539195e0c95dd2892d48b37f7'
WIDTH = 32
CHUNK = 65536
EPS = np.finfo(np.float32).eps
TINY = np.finfo(np.float32).tiny
DESIGN = 'results/published/fly_pipeline_event_plan_review_20261007.json'
PRIOR = 'results/published/fly_pipeline_repair_diagnostic_20261007.json'
COUNTS = {'reconstruct_observe': 1, 'A': 4, 'B': 4, 'C': 4}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(CHUNK), b''):
            value.update(piece)
    return value.hexdigest()


if digest(OLD_PATH) != OLD_SHA:
    raise ValueError('Immutable original diagnostic helper SHA mismatch')
import diagnose_fly_pipeline_repair as old

training, pipeline, reservoir = old.training, old.pipeline, old.reservoir
DEPENDENCIES = tuple(dict.fromkeys((*old.DEPENDENCIES,
    'scripts/ib/diagnose_fly_event_observable.py', 'tests/test_fly_event_observable.py')))


def safe_json(value):
    """Failure journals remain writable even if a failed computation is NaN."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {'invalid_numeric': repr(value)}
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(item) for item in value]
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    training.atomic_json(Path(path), safe_json(value))


def hashes():
    return {name: digest(ROOT/name) for name in DEPENDENCIES}


def precision_floor(scale, repeat=0.):
    """Own coordinate scale, without an artificial unit-amplitude floor."""
    return max(64*EPS*max(float(scale),TINY),10*float(repeat))


class Budget:
    def __init__(self):
        self.value = {'window_forwards_by_role': {}, 'physical_ticks': 0,
                      'backwards': 0, 'disposable_optimizer_updates': 0}

    def charge(self, role, *, observe=False):
        counts = self.value['window_forwards_by_role']
        if role not in COUNTS or counts.get(role,0) >= COUNTS[role]:
            raise ValueError('Fixed forward-role budget exceeded')
        if observe and (role!='reconstruct_observe' or self.value['backwards']):
            raise ValueError('Only one original reconstruction backward/update allowed')
        counts[role] = counts.get(role,0)+1
        self.value['physical_ticks'] += WIDTH
        self.value['backwards'] += int(observe)
        self.value['disposable_optimizer_updates'] += int(observe)

    def finish(self):
        if (self.value['window_forwards_by_role']!=COUNTS or self.value['physical_ticks']!=416
                or self.value['backwards']!=1 or self.value['disposable_optimizer_updates']!=1):
            raise ValueError('Incomplete registered execution ledger')


def motor_slice(value, index, width):
    return value[...,index] if torch.is_tensor(value) and value.numel()>1 and value.shape[-1]==width else value


def native_pulse(spike, state, context, index, use_stp=True):
    if not use_stp:
        return spike
    u0, _, _, norm = context['stp_params']
    u0 = motor_slice(u0,index,state.h.shape[-1])
    norm = motor_slice(norm,index,state.h.shape[-1])
    u = state.u[:,index]
    active = u+u0*(1-u)*spike
    return torch.clamp((active*state.x[:,index]/norm)*spike,max=3.)


class Trajectory:
    """Disk arrays; copying never stages more than65536 GPU coordinates."""
    def __init__(self, directory, label, motor, neurons, dimension, vocabulary, width=WIDTH):
        self.label, self.width = label, width
        self.arrays, self.paths = {}, {}
        shapes = {name:(width,motor) for name in ('v','threshold','s','p','h')}
        shapes.update(r=(width,dimension),z=(width,dimension),logits=(width,vocabulary),
                      margins=(width,motor+neurons),masks=(width,motor+neurons))
        for name,shape in shapes.items():
            path = Path(directory)/f'{label}_{name}.bin'
            if path.exists():
                raise FileExistsError('Never overwrite a previous trajectory')
            self.paths[name] = path
            self.arrays[name] = np.memmap(path,mode='w+',shape=shape,
                dtype=np.bool_ if name in ('s','masks') else np.float32)
        self.calls = self.tokens = self.norm_calls = self.decoder_calls = 0
        self.commits = 0
        self.pulse_identity = []
        self.state = None
        self.scores = None
        self.norm_metadata = None

    def put(self, name, value, row=None, columns=None):
        destination = self.arrays[name] if row is None else self.arrays[name][row]
        if columns is not None:
            destination = destination[columns]
        if torch.is_tensor(value):
            source = value.detach().reshape(-1)
            if source.numel()!=destination.size:
                raise ValueError(f'Telemetry shape mismatch: {name}')
            flat = destination.reshape(-1)
            for offset in range(0,len(flat),CHUNK):
                piece = source[offset:offset+CHUNK].cpu().numpy()
                if not np.isfinite(piece).all():
                    raise ValueError(f'Nonfinite telemetry: {name}')
                flat[offset:offset+len(piece)] = piece
        else:
            source = np.asarray(value).reshape(-1)
            if source.size!=destination.size or not np.isfinite(source).all():
                raise ValueError(f'Invalid telemetry: {name}')
            destination.reshape(-1)[:] = source

    def tape(self):
        motor = self.arrays['v'].shape[1]
        return [self.arrays['masks'][tick,part].reshape(1,-1) for tick in range(self.width)
                for part in (slice(None,motor),slice(motor,None))]

    def validate(self, scores, state):
        if (self.calls!=2*self.width or self.tokens!=self.width
                or self.norm_calls!=1 or self.decoder_calls!=1):
            raise ValueError('Native two-stage/batched-head telemetry incomplete')
        self.scores = np.asarray(scores,dtype=np.float64)
        if self.scores.shape!=(self.width,) or not np.isfinite(self.scores).all():
            raise ValueError('Invalid native scores')
        self.state = old.clone_state(state)
        for name,value in self.state.state_dict().items():
            for tensor in value if name=='ring' else (value,):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'Nonfinite physical field: {name}')
        for array in self.arrays.values():
            array.flush()

    def close(self):
        for array in self.arrays.values():
            array.flush()
            array._mmap.close()
        self.arrays.clear()


def array_error(first, second):
    if first.shape!=second.shape:
        raise ValueError('Repeated telemetry shape changed')
    maximum = scale = square = 0.
    a,b = first.reshape(-1),second.reshape(-1)
    for offset in range(0,len(a),CHUNK):
        x = np.asarray(a[offset:offset+CHUNK],dtype=np.float64)
        y = np.asarray(b[offset:offset+CHUNK],dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite repeated array')
        difference = x-y
        maximum = max(maximum,float(np.max(np.abs(difference),initial=0)))
        scale = max(scale,float(np.max(np.abs(x),initial=0)),float(np.max(np.abs(y),initial=0)))
        square += float(difference@difference)
    floor = 0. if first.dtype==np.bool_ else precision_floor(scale)
    return {'max_error':maximum,'rms_error':(square/max(a.size,1))**.5,
            'scale':scale,'floor':floor,'passed':maximum<=floor}


def state_repeat(first,second):
    result = {}
    for name,x in first.state_dict().items():
        y = getattr(second,name)
        xs,ys = (x,y) if name=='ring' else ((x,),(y,))
        rows=[]
        for a,b in zip(xs,ys):
            scale = max(float(a.abs().max()) if a.numel() else 0.,float(b.abs().max()) if b.numel() else 0.)
            error = float((a-b).abs().max()) if a.numel() else 0.
            floor=precision_floor(scale)
            rows.append({'max_error':error,'scale':scale,'floor':floor,'passed':error<=floor})
        result[name] = rows
    return result


def repeat_gate(first, second):
    fields={key:array_error(value,second.arrays[key]) for key,value in first.arrays.items()}
    fields['scores']=array_error(first.scores.astype(np.float32),second.scores.astype(np.float32))
    physical=state_repeat(first.state,second.state)
    passed=all(row['passed'] for row in fields.values()) and all(row['passed'] for rows in physical.values() for row in rows)
    return {'passed':passed,'arrays':fields,'physical':physical}


@contextmanager
def observe_arrays(model, trajectory, *, fixed=None):
    """Read original prepared context and original batched head; no extra tick."""
    handles=[]
    with old.instrument(model,fixed=fixed) as events:
        original_begin,original_finish = pipeline.begin_fly_prediction,model.finish_coba_tick
        inner_spike = pipeline.SpikeFn
        motor=model.n_read
        class RecordMargin:
            @staticmethod
            def apply(margin):
                call=trajectory.calls
                tick,stage=divmod(call,2)
                if tick>=trajectory.width:
                    raise ValueError('Unexpected extra SpikeFn call')
                expected=(1,motor if stage==0 else model.n_neurons)
                if tuple(margin.shape)!=expected:
                    raise ValueError('SpikeFn stage/order/shape changed')
                columns=slice(None,motor) if stage==0 else slice(motor,None)
                trajectory.put('margins',margin,row=tick,columns=columns)
                value=inner_spike.apply(margin)
                trajectory.put('masks',events['masks'][-1],row=tick,columns=columns)
                trajectory.calls+=1
                return value

        def begin(instance,state,**kwargs):
            value,pending=original_begin(instance,state,**kwargs)
            tick=trajectory.tokens
            if trajectory.calls!=2*tick+1:
                raise ValueError('Prediction issue must precede the current observation')
            with torch.no_grad():
                c,index=pending.context,instance.read_indices
                v=c['alpha'][:,index]*state.h[:,index]+c['beta_int'][:,index]*c['base_current'][:,index]
                threshold=motor_slice(c['eff_threshold'],index,state.h.shape[-1])
                threshold=torch.zeros_like(v)+threshold
                s=torch.as_tensor(events['masks'][-1],device=v.device,dtype=v.dtype)
                for name,tensor in {'v':v,'threshold':threshold,'s':s,'p':native_pulse(s,state,c,index,instance.use_stp),
                                    'h':v*(1-s),'r':pending.feature}.items():
                    trajectory.put(name,tensor,row=tick)
            trajectory.tokens+=1
            return value,pending

        def finish(instance,context,drive,*,return_biophysics=False):
            ret,bio=original_finish(context,drive,return_biophysics=True)
            tick=trajectory.tokens-1
            if trajectory.calls!=2*(tick+1):
                raise ValueError('Commit must be the second event stage of this tick')
            index=instance.read_indices
            errors={}
            for name,tensor in {'v':bio['v_pre'][:,index],'s':ret[1][:,index],
                                'p':ret[2][0][:,index],'h':ret[0][:,index]}.items():
                array=tensor.detach().cpu().numpy().reshape(-1)
                errors[name]=array_error(trajectory.arrays[name][tick],array.astype(trajectory.arrays[name].dtype))
                if not errors[name]['passed']:
                    raise ValueError(f'Sealed motor/commit identity failed: {name}')
            trajectory.pulse_identity.append(errors)
            trajectory.commits+=1
            return (ret,bio) if return_biophysics else ret

        def norm_hook(module,inputs,output):
            trajectory.norm_calls+=1
            trajectory.put('z',output)
            epsilon=module.eps if module.eps is not None else torch.finfo(output.dtype).eps
            trajectory.norm_metadata={'epsilon':float(epsilon),'gain':module.weight.detach().cpu().tolist(),
                                      'input_dtype':str(inputs[0].dtype)}

        def decoder_hook(module,inputs,output):
            trajectory.decoder_calls+=1
            trajectory.put('logits',output)

        pipeline.SpikeFn=reservoir.SpikeFn=RecordMargin
        pipeline.begin_fly_prediction=begin
        model.finish_coba_tick=types.MethodType(finish,model)
        handles=[model.read_norm.register_forward_hook(norm_hook),model.decoder.register_forward_hook(decoder_hook)]
        try:
            yield events
        finally:
            for handle in handles:
                handle.remove()
            pipeline.begin_fly_prediction=original_begin
            model.finish_coba_tick=original_finish
            pipeline.SpikeFn=reservoir.SpikeFn=inner_spike


def probability_contrast(reference, changed, targets):
    """FP64 exact finite CE change = KL(p_reference||p_changed) - alignment."""
    rows=[]
    mean_delta=np.zeros(reference.shape[1],dtype=np.float64)
    mean_error=np.zeros_like(mean_delta)
    centered_deltas=[]
    for tick,target in enumerate(targets):
        a=torch.as_tensor(np.array(reference[tick],dtype=np.float64))
        b=torch.as_tensor(np.array(changed[tick],dtype=np.float64))
        delta=b-a
        delta=delta-delta.mean()
        logp,logq=a.log_softmax(-1),b.log_softmax(-1)
        p=logp.exp()
        expected=float(p@delta)
        alignment=float(delta[int(target)])-expected
        kl=float(p@(logp-logq))
        direct=float(-logq[int(target)]+logp[int(target)])
        error=abs(direct-(kl-alignment))
        bound=64*np.finfo(np.float64).eps*(1+abs(float(logp[int(target)]))+abs(float(logq[int(target)]))
              +abs(alignment)+abs(kl)+float((p*delta).abs().sum()))
        if not all(np.isfinite(value) for value in (alignment,kl,direct,error)) or error>bound or kl < -bound:
            raise ValueError('Finite probability contrast closure failed')
        e=p.numpy().copy(); e[int(target)]-=1
        mean_error+=e/len(targets)
        mean_delta+=delta.numpy()/len(targets)
        centered_deltas.append(delta.numpy())
        rows.append({'alignment':alignment,'kl':kl,'risk_change':direct,'closure_error':error,
                     'arithmetic_floor':bound,'centered_logit_norm':float(delta.norm()),
                     'reference_ce':float(-logp[int(target)]),'changed_ce':float(-logq[int(target)])})
    common_linear=float(mean_error@mean_delta)
    total_linear=-float(np.mean([row['alignment'] for row in rows]))
    shared=[]
    for tick,target in enumerate(targets):
        a=torch.as_tensor(np.array(reference[tick],dtype=np.float64))
        shift=torch.from_numpy(mean_delta)
        shared.append(float(-(a+shift).log_softmax(-1)[int(target)]+a.log_softmax(-1)[int(target)]))
    total=float(np.mean([row['risk_change'] for row in rows]))
    return {'per_target':rows,'mean_risk_change':total,
        'mean_alignment':float(np.mean([row['alignment'] for row in rows])),
        'mean_kl':float(np.mean([row['kl'] for row in rows])),
        'common_linear_cost':common_linear,'covariance_linear_cost':total_linear-common_linear,
        'shared_only_retrospective_risk_change':float(np.mean(shared)),
        'ordered_residual_with_interactions':total-float(np.mean(shared)),
        'mean_delta_norm':float(np.linalg.norm(mean_delta)),
        'centered_time_delta_energy':float(sum(np.sum((row-mean_delta)**2) for row in centered_deltas)/len(targets)),
        'scope':'Current fixed head finite probability accounting; whole-window common is retrospective, not an online predictor.'}


def rms_geometry(reference, changed, metadata):
    """Exact finite endpoint map and separately labeled local radial derivative."""
    a,b=np.asarray(reference,dtype=np.float64),np.asarray(changed,dtype=np.float64)
    gain=np.asarray(metadata['gain'],dtype=np.float64)
    epsilon=metadata['epsilon']
    rho_a=np.sqrt(np.mean(a*a,axis=-1,keepdims=True)+epsilon)
    rho_b=np.sqrt(np.mean(b*b,axis=-1,keepdims=True)+epsilon)
    delta=b-a
    dot=np.sum(a*delta,axis=-1,keepdims=True)
    aa=np.sum(a*a,axis=-1,keepdims=True)
    radial=a*np.divide(dot,aa,out=np.zeros_like(dot),where=aa>0)
    jac=gain*(delta/rho_a-a*dot/(a.shape[-1]*rho_a**3))
    finite=gain*(b/rho_b-a/rho_a)
    return {'finite':finite,'reference_endpoint':gain*a/rho_a,'changed_endpoint':gain*b/rho_b,
            'per_target':{'radial_fraction':(np.sum(radial**2,axis=-1)/np.maximum(np.sum(delta**2,axis=-1),np.finfo(np.float64).tiny)).tolist(),
            'finite_response_norm':np.linalg.norm(finite,axis=-1).tolist(),
            'local_J_response_norm':np.linalg.norm(jac,axis=-1).tolist(),
            'local_J_finite_remainder_norm':np.linalg.norm(finite-jac,axis=-1).tolist()},
            'scope':'Finite map uses actual gain/epsilon. Local J is descriptive for finite event jumps.'}


def comparison_floor(actual,expected,*,reduction_floor=0.,risk_floor=None):
    if isinstance(expected,int):
        return actual==expected,0.
    floor=max(64*EPS*max(abs(float(expected)),TINY),float(reduction_floor))
    if risk_floor is not None:
        floor=max(floor,risk_floor)
    return abs(float(actual)-float(expected))<=floor,floor


def geometry(reference,changed,projection,repeat_errors):
    """Finite native map accounting with explicit dot-product error bounds."""
    fields={}
    for name in ('v','s','p','h','r','z'):
        row=array_error(reference.arrays[name],changed.arrays[name])
        row['contrast_floor']=0. if name=='s' else precision_floor(row['scale'],repeat_errors.get(name,0.))
        row['resolved']=row['max_error']>row['contrast_floor']
        fields[name]=row
    h_a=np.asarray(reference.arrays['h'],dtype=np.float64)
    h_b=np.asarray(changed.arrays['h'],dtype=np.float64)
    r_a=np.asarray(reference.arrays['r'],dtype=np.float64)
    r_b=np.asarray(changed.arrays['r'],dtype=np.float64)
    predicted_a=np.empty_like(r_a); predicted_b=np.empty_like(r_b)
    bounds_a=np.empty_like(r_a); bounds_b=np.empty_like(r_b)
    n=projection.shape[-1]
    gamma=n*EPS/(1-n*EPS)
    for offset in range(0,projection.shape[0],16):
        w=projection[offset:offset+16].double().numpy()
        predicted_a[:,offset:offset+len(w)]=h_a@w.T
        predicted_b[:,offset:offset+len(w)]=h_b@w.T
        bounds_a[:,offset:offset+len(w)]=gamma*(np.abs(h_a)@np.abs(w).T)
        bounds_b[:,offset:offset+len(w)]=gamma*(np.abs(h_b)@np.abs(w).T)
    bound=bounds_a+bounds_b+precision_floor(max(np.max(np.abs(r_a)),np.max(np.abs(r_b))),repeat_errors.get('r',0.))
    residual=(r_b-r_a)-(predicted_b-predicted_a)
    if np.any(np.abs(residual)>bound):
        raise ValueError('Native projection/FP64 finite-map identity exceeded dot bound')
    norm=rms_geometry(r_a,r_b,reference.norm_metadata)
    z_a=np.asarray(reference.arrays['z'],dtype=np.float64)
    z_b=np.asarray(changed.arrays['z'],dtype=np.float64)
    gain=np.asarray(reference.norm_metadata['gain'],dtype=np.float64)
    epsilon=reference.norm_metadata['epsilon']
    def norm_bound(r,endpoint):
        dim=r.shape[-1]
        gamma_norm=(dim+2)*EPS/(1-(dim+2)*EPS)
        square=np.mean(r*r,axis=-1,keepdims=True)+epsilon
        square_error=gamma_norm*(np.mean(r*r,axis=-1,keepdims=True)+abs(epsilon))
        lower=np.sqrt(np.maximum(square-square_error,np.finfo(np.float64).tiny))
        rho=np.sqrt(square)
        rho_error=square_error/(rho+lower)
        return np.abs(gain*r)*rho_error/(rho*lower)+precision_floor(np.max(np.abs(endpoint)),repeat_errors.get('z',0.))
    norm_bounds=norm_bound(r_a,norm['reference_endpoint'])+norm_bound(r_b,norm['changed_endpoint'])
    norm_residual=(z_b-z_a)-norm['finite']
    if np.any(np.abs(norm_residual)>norm_bounds):
        raise ValueError('Native RMSNorm/FP64 finite-map identity exceeded error bound')
    return {'fields':fields,
        'projection_identity':{'maximum_error':float(np.max(np.abs(residual))),
            'maximum_dot_error_bound':float(np.max(bound)), 'gamma_n':gamma,
            'cpu64_finite_response_norm':np.linalg.norm(predicted_b-predicted_a,axis=1).tolist()},
        'normalization_identity':{'maximum_error':float(np.max(np.abs(norm_residual))),
            'maximum_error_bound':float(np.max(norm_bounds))},
        'normalization':norm['per_target'],
        'geometric_loss_witness_at_reset':any(fields[name]['resolved'] for name in ('v','s','p')) and not fields['h']['resolved'],
        'scope':'Resolved differences of this fixed direction; no coordinate-independent cross-layer norm ratio, semantic information or decoder-class claim.'}


def idle_device():
    try:
        response=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    except (OSError,subprocess.SubprocessError) as error:
        raise RuntimeError('Cannot establish idle GPU process gate') from error
    if response.stdout.strip():
        raise RuntimeError('A GPU compute process is active; this diagnostic must run alone')


def register(args):
    design=json.loads((ROOT/DESIGN).read_text(encoding='utf-8'))
    prior=json.loads((ROOT/PRIOR).read_text(encoding='utf-8'))
    if design['status']!='approved_design_only' or not prior['complete']:
        raise ValueError('Approved design and completed immutable prior report required')
    if digest(args.checkpoint)!=design['fixed_estimand']['expected_checkpoint_sha256']:
        raise ValueError('Wrong full mature checkpoint')
    for name,expected in design['input_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'Approved design input changed: {name}')
    saved=torch.load(args.checkpoint,mmap=True,map_location='cpu',weights_only=False)
    if saved['format']!=training.FORMAT or saved['ledger']['train_cursor']!=96000 or saved['ledger']['val_cursor']!=1536:
        raise ValueError('Wrong pipeline source/cursor')
    if saved['config']['read_centering'] or saved['config']['learn_stp']:
        raise ValueError('Only original native/frozen-STP source is registered')
    train=training.load_tokens(ROOT/'data/ib_owt_gpt2_31m/train.npy')
    targets={key:train[start:start+WIDTH].astype(np.int64).tolist() for key,start in [('fit',96000),('future',96032)]}
    if targets!= {key:prior['context_contrast']['same_future_targets'] if key=='future' else prior['baseline'].get('targets',targets['fit']) for key in targets}:
        raise ValueError('Target source disagrees with immutable report')
    paths={'train':ROOT/'data/ib_owt_gpt2_31m/train.npy','graph':ROOT/'data/malecns_v1/fly_reservoir_coba.npz',
           'sensory_partitions':ROOT/'data/malecns_v1/sensory_partitions.npz'}
    source=hashes()
    plan={'status':'preregistered_pending_independent_execution_approval','date':'2026-10-07',
        'scope':'Result-informed finite event/read trace, not capability training or primary live evaluation',
        'design':{'path':DESIGN,'sha256':digest(ROOT/DESIGN)},'prior':{'path':PRIOR,'sha256':digest(ROOT/PRIOR)},
        'checkpoint':{'path':str(args.checkpoint.resolve()),'sha256':digest(args.checkpoint),'format':saved['format'],
                      'source_events':saved['learner']['events'],'source_updates':saved['learner']['updates']},
        'immutable_helper_sha256':OLD_SHA,'script_sha256':digest(__file__),'source_hashes':source,
        'data':{key:{'path':str(path.resolve()),'sha256':digest(path)} for key,path in paths.items()},
        'targets':targets,'unique_target_positions':64,'scales':[1.],
        'budget':{'roles':COUNTS,'window_events':WIDTH,'physical_ticks':416,'backwards':1,'updates':1,'vram_mib':3900},
        'execution':'One eager ORIGINAL observe reconstruction. A/B/C each fit/future initial+repeat; future always OLD A-fit terminal. No retry and no checkpoints.',
        'trace':'Native prepared-context motor v/threshold/s/STPpulse/h/r; native batched32 RMSNorm z/decoder logits. All64 begin/finish masks and margins; motor pulse identity to commit.',
        'floors':{'repeat_array':'64*float32eps*actual-own-field-scale; discrete masks exact; no max1 scale',
          'attribution_array':'max(predeclared own-scale bound,10*maximum repeated array error across all A/B/C)',
          'reconstruction':'64*float32eps*absolute recorded nonzero quantity + recorded dot reduction bound; integers exact; native risks additionally use immutable prior decision_floor',
          'risk':'max(64*float32eps*maximum native target CE,10*maximum native/reconstructed/decomposition replay error)',
          'probability':'FP64 CE=KL-alignment per target;64eps64 times the recorded arithmetic term magnitudes',
          'projection':'CPU64 endpoint/contrast identity uses gamma_n*abs(W)abs(h) FP32 dot bound plus repeated-field error; no cross-layer norm-ratio claim'},
        'interpretation':design['binding_next_action'],'resources':'CPU/disk parameter staging on F:; idle GPU and3900MiB hardstop',
        'execution_authorized':False,'gpu_runs':0}
    if args.protocol.exists():
        raise FileExistsError('Preserve a previous registration; choose a new path')
    write_json(args.protocol,plan)
    print('CPU registration only:13windows,416ticks,1backward,1disposable update; pending execution approval')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true'); mode.add_argument('--run',action='store_true')
    parser.add_argument('--checkpoint',type=Path,default=Path('F:/fly_checkpoints/q8_fly_pipeline14_joint_96k/last.pt'))
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--approval',type=Path); parser.add_argument('--output',type=Path); parser.add_argument('--scratch',type=Path)
    args=parser.parse_args()
    if args.register:
        register(args); return
    if not all((args.approval,args.output,args.scratch)):
        parser.error('--run requires independent approval, output and F: scratch')
    run(args)


def run(args):
    """Execution path intentionally cannot be entered by CPU registration."""
    budget=Budget()
    report={'status':'invalid_or_incomplete','complete':False,'execution_ledger':budget.value,
        'scope':'One mature-state conditional event/read contrast; not production training or a capability claim',
        'source_unchanged':False,'reconstruction':{},'repeats':{},'trajectories':{},'contrasts':{}}
    traces=[]; gradient_disk=None; snapshot=None; learner=None
    try:
        if args.output.exists() or args.output.with_name(args.output.stem+'_partial.json').exists():
            raise FileExistsError('Preserve any earlier result/failure; no auto-retry')
        plan=json.loads(args.protocol.read_text(encoding='utf-8'))
        approval=json.loads(args.approval.read_text(encoding='utf-8'))
        if (approval.get('status')!='approved' or approval.get('prereg_sha256')!=digest(args.protocol)
                or approval.get('script_sha256')!=digest(__file__)
                or approval.get('source_hashes')!=plan['source_hashes'] or hashes()!=plan['source_hashes']):
            raise ValueError('Independent new execution approval/content lock failed')
        if (plan['budget']!={'roles':COUNTS,'window_events':WIDTH,'physical_ticks':416,'backwards':1,'updates':1,'vram_mib':3900}
                or plan['scales']!=[1.] or plan['unique_target_positions']!=64):
            raise ValueError('Locked design/budget changed')
        for item in (plan['design'],plan['prior']):
            if digest(ROOT/item['path'])!=item['sha256']:
                raise ValueError('Reviewed design or immutable prior report changed')
        if digest(args.checkpoint)!=plan['checkpoint']['sha256']:
            raise ValueError('Wrong mature source')
        for item in plan['data'].values():
            if digest(item['path'])!=item['sha256']:
                raise ValueError('Registered data/graph changed')
        if args.scratch.resolve().drive.upper()!='F:':
            raise ValueError('Use independent F: disk staging')
        if args.scratch.exists():
            raise FileExistsError('Scratch directory must be new; no hidden retry')
        idle_device()
        if not torch.cuda.is_available():
            raise ValueError('CUDA is required for separately approved execution')
        torch.set_num_threads(1)
        torch.cuda.reset_peak_memory_stats()
        saved=torch.load(args.checkpoint,mmap=True,map_location='cpu',weights_only=False)
        cfg=saved['config']
        if saved['format']!=training.FORMAT or cfg['read_centering'] or cfg['learn_stp']:
            raise ValueError('Source protocol/native read/parameter scope mismatch')
        model=reservoir.FlyReservoirLM(plan['data']['graph']['path'],vocab_size=training.VOCAB,d_model=768,
            injection='topographic',read_surface='output',synapse_model='coba',use_alif=True,use_stp=True,
            decoder_bias=True,read_centering=False,use_read_gamma_trace=False,use_latent_predictor=False).cuda()
        model.dan_plastic_lr=0.
        learner=pipeline.FlyPipelineLearner(model,training.build_rest(model,'cuda'),
            lr=cfg['lr'],lr_synapse=cfg['lr'],lr_sensory=cfg['lr'],max_grad_norm=cfg['max_grad_norm'],
            plasticity_optimizer='adamw',settle_ticks=0,writer_baseline_clock='input',learn_stp=False)
        training.restore_learner(saved,learner)
        training.memory_gate(3900)
        entering=old.clone_state(learner.state)
        named={name:parameter for name,parameter in model.named_parameters() if parameter.requires_grad}
        frozen_head={'output_read.weight','read_norm.weight','decoder.weight','decoder.bias'}
        groups={'head':frozen_head-{'decoder.bias'},'bias':{'decoder.bias'},
                'body':set(named)-frozen_head,'full':set(named)}
        if not frozen_head.issubset(named):
            raise ValueError('Actual parameter partition changed')
        original={name:saved['model'][name] for name in named}
        args.scratch.mkdir(parents=True)
        # Two trainable maps, thirteen bounded tapes, and a safety allowance.
        array_bytes=WIDTH*(4*(5*model.n_read+2*768+training.VOCAB+model.n_read+model.n_neurons)
                    +model.n_read+model.n_read+model.n_neurons)
        required=2*sum(parameter.numel()*4 for parameter in named.values())+13*array_bytes+256*2**20
        if shutil.disk_usage(args.scratch).free<required:
            raise OSError('Insufficient space for bounded parameter/trajectory staging')
        gradient_disk=old.DiskTensor(named,args.scratch/'raw_gradient.bin')
        fit=torch.tensor(plan['targets']['fit'],device='cuda',dtype=torch.long)
        future=torch.tensor(plan['targets']['future'],device='cuda',dtype=torch.long)
        prior=json.loads((ROOT/PRIOR).read_text(encoding='utf-8'))
        def new_trace(label):
            trace=Trajectory(args.scratch,label,model.n_read,model.n_neurons,768,training.VOCAB)
            traces.append(trace)
            return trace
        def forward(label,role,initial,tokens,tape=None):
            budget.charge(role)
            learner.state=old.clone_state(initial)
            model.zero_grad(set_to_none=True)
            trace=new_trace(label)
            with torch.no_grad(),observe_arrays(model,trace,fixed=tape):
                scores,state,_=learner.forward_window(tokens[None],tokens[None])
            trace.validate(scores.detach().cpu().numpy(),state)
            training.memory_gate(3900)
            return trace
        A_fit=forward('A_fit','A',entering,fit)
        A_future=forward('A_future','A',A_fit.state,future)
        reconstruction=new_trace('reconstruction_fit')
        learner.state=old.clone_state(entering)
        budget.charge('reconstruct_observe',observe=True)
        with old.before_clip(named,gradient_disk),old.clamp_instrument(learner,gradient_disk.values,original) as clamp,observe_arrays(model,reconstruction):
            scores,metrics=learner.observe(fit)
        reconstruction.validate(scores,learner.state)
        report['reconstruction']['native_identity']=repeat_gate(A_fit,reconstruction)
        if not report['reconstruction']['native_identity']['passed']:
            raise ValueError('Original observe/native forward identity failed')
        if (learner.events!=saved['learner']['events']+WIDTH or learner.updates!=saved['learner']['updates']+1
                or learner.physical_ticks!=saved['learner']['physical_ticks']+WIDTH):
            raise ValueError('Original learning-life cadence mismatch')
        values,storage=old.parameter_snapshot(named,args.scratch/'updated_parameters.bin')
        snapshot=(values,storage)
        directions={key:old.directional_summary(gradient_disk.values,original,values,names) for key,names in groups.items()}
        clip_factor=min(1.,learner.max_grad_norm/(metrics['grad_norm_before_clip']+1e-6))
        report['reconstruction'].update(metrics=metrics,clamp=clamp,clip_factor=clip_factor,
                                        directions=directions,completed_original_observe=True)
        gates=[]
        def verify(name,actual,expected,*,reduction_floor=0.,risk=False):
            passed,floor=comparison_floor(actual,expected,reduction_floor=reduction_floor,
                risk_floor=prior['decision_floor'] if risk else None)
            gates.append({'quantity':name,'actual':actual,'expected':expected,'floor':floor,'passed':passed})
            if not passed:
                raise ValueError(f'Reconstruction conformance failed: {name}')
        report['reconstruction']['conformance']=gates
        for name,value in metrics.items():
            verify('metric/'+name,value,prior['ordinary_metrics'][name])
        verify('clip_factor',clip_factor,prior['ordinary_clip_factor'])
        for name,value in clamp.items():
            verify('clamp/'+name,value,prior['ordinary_clamp'][name])
        for group,row in directions.items():
            expected=prior['ordinary_surrogate_direction'][group]
            for key in ('gradient_norm','displacement_norm','changed_coordinates','gradient_dot_actual_displacement'):
                reduction=max(row['float64_accumulation_floor'],expected['float64_accumulation_floor']) if key=='gradient_dot_actual_displacement' else 0.
                verify(group+'/'+key,row[key],expected[key],reduction_floor=reduction)
        model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        old.restore_parameters(named,original)
        A_repeat_fit=forward('A_fit_repeat','A',entering,fit)
        A_repeat_future=forward('A_future_repeat','A',A_fit.state,future)
        old.restore_parameters(named,values,selected=groups['body'],origin=original)
        # Exact frozen-head identity after masked restoration.
        for name in frozen_head:
            verify('frozen/'+name,old.realized_norm(named,original,{name}),0.)
        B_fit=forward('B_fit','B',entering,fit)
        B_future=forward('B_future','B',A_fit.state,future)
        B_repeat_fit=forward('B_fit_repeat','B',entering,fit)
        B_repeat_future=forward('B_future_repeat','B',A_fit.state,future)
        C_fit=forward('C_fit','C',entering,fit,A_fit.tape())
        C_future=forward('C_future','C',A_fit.state,future,A_future.tape())
        C_repeat_fit=forward('C_fit_repeat','C',entering,fit,A_fit.tape())
        C_repeat_future=forward('C_future_repeat','C',A_fit.state,future,A_future.tape())
        primary={'A':(A_fit,A_future),'B':(B_fit,B_future),'C':(C_fit,C_future)}
        repeats={'A':(A_repeat_fit,A_repeat_future),'B':(B_repeat_fit,B_repeat_future),'C':(C_repeat_fit,C_repeat_future)}
        array_repeats={}; score_repeat=0.
        for key in primary:
            for phase,first,second in zip(('fit','future'),primary[key],repeats[key]):
                row=repeat_gate(first,second)
                report['repeats'][f'{key}_{phase}']=row
                for field,result in row['arrays'].items():
                    array_repeats[field]=max(array_repeats.get(field,0.),result['max_error'])
                score_repeat=max(score_repeat,row['arrays']['scores']['max_error'])
                if not row['passed']:
                    raise ValueError(f'Repeated hard events/physical fields/stage arrays unstable: {key}/{phase}')
        for phase,a,b,c in zip(('fit','future'),primary['A'],primary['B'],primary['C']):
            verify('A/'+phase+'/risk',float(a.scores.mean()),prior['baseline'][phase+'_nll'],risk=True)
            verify('B/'+phase+'/risk',float(b.scores.mean()),prior['controls']['body_1'][phase+'_nll'],risk=True)
            changes=old.branches(a.tape(),b.tape())
            expected=prior['controls']['body_1'][phase+'_spikes']
            for field in ('begin_motor_changes','finish_body_changes','total_changes'):
                verify('B/'+phase+'/'+field,changes[field],expected[field])
            if changes['changed_by_call']!=expected['changed_by_call']:
                raise ValueError('Reconstructed event-change tape counts do not conform')
            if old.branches(a.tape(),c.tape())['total_changes']:
                raise ValueError('Fixed-event trajectory did not preserve ALL old masks')
        verify('C/fit/risk',float(C_fit.scores.mean()),prior['controls']['body_1']['fixed_event_fit_nll'],risk=True)
        max_score=max(float(trace.scores.max()) for trace in traces)
        risk_floor=precision_floor(max_score,score_repeat)
        probability_repeat=0.
        for phase,tokens in [('fit',plan['targets']['fit']),('future',plan['targets']['future'])]:
            index=0 if phase=='fit' else 1
            for label,first,second in [('B-A','A','B'),('C-A','A','C'),('B-C','C','B')]:
                a,b=primary[first][index],primary[second][index]
                ar,br=repeats[first][index],repeats[second][index]
                accounting=probability_contrast(a.arrays['logits'],b.arrays['logits'],tokens)
                repeated=probability_contrast(ar.arrays['logits'],br.arrays['logits'],tokens)
                native_delta=float((b.scores-a.scores).mean())
                if abs(native_delta-accounting['mean_risk_change'])>risk_floor:
                    raise ValueError('Native CE and FP64 contrast disagree beyond floor')
                error=max(abs(x[key]-y[key]) for x,y in zip(accounting['per_target'],repeated['per_target'])
                          for key in ('alignment','kl','risk_change'))
                probability_repeat=max(probability_repeat,error)
                accounting.update(native_risk_change=native_delta,replay_error=error,
                    geometry=geometry(a,b,saved['model']['output_read.weight'],array_repeats))
                report['contrasts'][f'{phase}/{label}']=accounting
        risk_floor=max(risk_floor,10*probability_repeat)
        report['decision_floors']={'risk':risk_floor,'maximum_probability_replay_error':probability_repeat,
            'maximum_array_replay_error':array_repeats}
        for row in report['contrasts'].values():
            row['risk_direction']='adverse' if row['mean_risk_change']>risk_floor else (
                'beneficial' if row['mean_risk_change'] < -risk_floor else 'unresolved')
            row['alignment_direction']='favorable' if row['mean_alignment']>risk_floor else (
                'adverse' if row['mean_alignment'] < -risk_floor else 'unresolved')
        budget.finish()
        if hashes()!=plan['source_hashes'] or digest(args.checkpoint)!=plan['checkpoint']['sha256']:
            raise ValueError('Code/source changed during trace')
        for trace in traces:
            report['trajectories'][trace.label]={'native_scores':trace.scores.tolist(),
                'pulse_commit_identity':trace.pulse_identity,'norm_metadata':trace.norm_metadata,
                'arrays':{key:{'path':str(path.resolve()),'shape':list(trace.arrays[key].shape),
                    'dtype':str(trace.arrays[key].dtype),'sha256':digest(path)} for key,path in trace.paths.items()},
                'complete_begin_finish_calls':trace.calls,'complete_commit_calls':trace.commits}
        report.update(status='complete_local_trace',complete=True,source_unchanged=True,
            protocol_sha256=digest(args.protocol),source_sha256=plan['checkpoint']['sha256'],
            resources=training.memory_gate(3900),interpretation=plan['interpretation'])
        write_json(args.output,report)
        print('Completed fixed event/read trace; no production change or checkpoint written')
    except BaseException as error:
        report.update(status='invalid_or_ambiguous',complete=False,error=repr(error),
            completed_commit_calls=sum(trace.commits for trace in traces),
            attempted_trace_calls={trace.label:trace.calls for trace in traces},auto_retry=False)
        write_json(args.output.with_name(args.output.stem+'_partial.json'),report)
        raise
    finally:
        for trace in traces:
            trace.close()
        if gradient_disk is not None:
            gradient_disk.close()
            (args.scratch/'raw_gradient.bin').unlink(missing_ok=True)
        if snapshot is not None:
            values,storage=snapshot
            values.clear(); storage.flush(); storage._mmap.close()
            (args.scratch/'updated_parameters.bin').unlink(missing_ok=True)


if __name__=='__main__':
    main()
