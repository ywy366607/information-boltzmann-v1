"""Locked, finite-update diagnosis on a mature pipeline checkpoint.

Counterfactual forks are not primary active evaluation. Two saved-Adam updates
are disposable; the source and production individual are never changed. No
checkpoint is written. --register only reads CPU metadata and writes the plan.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_fly_pipeline_stream as training
from information_boltzmann.core import fly_pipeline as pipeline
from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from diagnose_fly_harness_update import (
    clone_state, parameter_snapshot, restore_parameters, directional_summary, state_distance,
)

EPS = np.finfo(np.float32).eps
WIDTH = 32
PLAN_COUNTS = {
    'baseline_and_two_identity_fit_future': 6,
    'B_prefix_then_same_future': 2,
    'ordinary_observe': 1,
    'fixed_event_backward': 1,
    'ordinary_registered_masks_fit_future': 10,
    'ordinary_body_two_scales_fixed_fit': 2,
    'reset_candidate_future_identity': 1,
    'reset_candidate_observe': 1,
    'reset_candidate_registered_masks_fit_future': 4,
    'two_body_matched_norm_fit_future': 4,
    'selection_control_repeats': 12,
}
DEPENDENCIES = (*training.SOURCE_FILES,
                'scripts/ib/diagnose_fly_harness_update.py',
                'scripts/ib/diagnose_fly_pipeline_repair.py')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(65536), b''):
            value.update(piece)
    return value.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    training.atomic_json(Path(path), json_native(value))


def json_native(value):
    """Preserve NumPy scalar values while making both final/error reports writable."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    return value


def source_hashes():
    return {name: digest(ROOT/name) for name in dict.fromkeys(DEPENDENCIES)}


def register(args):
    """No physical evolution, model construction, gradient, optimizer or CUDA."""
    saved = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=False)
    if saved['format'] != training.FORMAT:
        raise ValueError('Complete pipeline continuation required')
    ledger = saved['ledger']
    if ledger['train_cursor'] != 96000 or ledger['val_cursor'] != 1536:
        raise ValueError('This registered diagnosis fixes the completed96k source boundary')
    train = training.load_tokens(ROOT/'data/ib_owt_gpt2_31m/train.npy')
    validation = training.load_tokens(ROOT/'data/ib_owt_gpt2_31m/validation.npy')
    values = {'fit': train[96000:96032].astype(np.int64).tolist(),
              'future': train[96032:96064].astype(np.int64).tolist(),
              'B_prefix': validation[1536:1568].astype(np.int64).tolist()}
    if any(len(v) != WIDTH for v in values.values()):
        raise ValueError('Insufficient registered real OWT targets')
    data_paths = {'train': ROOT/'data/ib_owt_gpt2_31m/train.npy',
                  'validation': ROOT/'data/ib_owt_gpt2_31m/validation.npy',
                  'reference_train': Path(saved['config']['fixed_reference']['train']),
                  'graph': ROOT/'data/malecns_v1/fly_reservoir_coba.npz',
                  'sensory_partitions': ROOT/'data/malecns_v1/sensory_partitions.npz'}
    plan = {'status': 'preregistered_pending_independent_approval', 'date': '2026-10-07',
        'scope': 'Disposable mature-state mechanism forks; not primary active evaluation or capability training',
        'source': {'path': str(args.checkpoint.resolve()), 'sha256': digest(args.checkpoint),
                   'format': saved['format'], 'events': saved['learner']['events'],
                   'updates': saved['learner']['updates'], 'ledger': ledger},
        'script_sha256': digest(__file__), 'source_hashes': source_hashes(),
        'data': {key: {'path': str(path.resolve()), 'sha256': digest(path)} for key,path in data_paths.items()},
        'targets': values, 'A_prefix': 'fit: actual contiguous train[96000:96032]',
        'parameter_groups': {'head': ['output_read.weight', 'read_norm.weight', 'decoder.weight'],
                             'bias': ['decoder.bias'], 'body': 'all other actual trainable parameters'},
        'scales': {'ordinary_body': [1.0, 0.1], 'all_other_masks': [1.0]},
        'selection_replays': 'Repeat both fit and future at each ordinary/reset joint, body, and matched-norm parameter point; twelve windows. Hard mask instability makes selection ambiguous.',
        'window_events': WIDTH, 'unique_target_positions': 96,
        'budget': {'window_forwards_by_role': PLAN_COUNTS, 'window_forwards': sum(PLAN_COUNTS.values()),
                   'forward_physical_ticks': 1408, 'backwards': 3, 'disposable_optimizer_updates': 2,
                   'vram_limit_mib': 3900, 'no_cuda_graph_capture': True},
        'ordinary': 'Original observe and both saved AdamW states, original clip and post-step clamps',
        'C_reset': 'Detach spike ONLY in membrane reset at begin motor and finish body. Pulse/STP/ALIF spike paths keep original ATan SG. Forward state and scores must be identical.',
        'fixed_event': 'Hold every begin+finish hard mask constant, no spike derivative; absent derivatives count zero. Writer derivative zero is expected, not failure.',
        'matched_norm': {'radius': 'r=min(||ordinary_body_actual_delta||,||reset_body_actual_delta||)',
                         'scales': 'r/norm per direction; zero norm produces a zero step',
                         'common_head': 'same ordinary full-step head AND bias displacement in both arms',
                         'realized_norm_gate': 'Each copied FP32 norm must match r within64float32eps*r; require r>10*(64float32eps*max(1,r)). These are numerical-resolution gates, not model or biological parameters.',
                         'norm_units': 'Euclidean mixed parameter coordinates; not biological distance'},
        'future_state': 'Always start at OLD-parameter fit terminal; new-parameter prefix never replaces it',
        'identity_gate': 'All physical fields including all4 ring slots and same-input hard masks; score/state error <=64float32eps*scale plus observed repeat floor',
        'decision_floor': 'tau=max(1e-6,10*maximum_score_error_across_old_and_selection_parameter_replays,64float32eps*max(1,maximum_registered_selection_score))',
        'selection': 'At fixed scale1: reset joint future improves ordinary joint beyond tau and fit is nonworse; reset body fit improves ordinary body beyond tau and is nonworse than old fit; raw body-only future improves ordinary body beyond tau and is nonworse than old future; reset matched-norm/common-head future improves ordinary beyond tau and matched fit is nonworse. Raw head gradients must match, relative FP32 matched-norm and resolved-radius gates must pass. All identity gates must pass. Any selection-point hard-mask or continuous-state replay instability, head-gradient mismatch or unresolved norm matching is ambiguous, never candidate rejection/support. Otherwise failed predictive gates are locally unsupported. Scale.1 is descriptive, not an alternative selection endpoint.',
        'interpretation': ['Original SG dot, fixed-event derivative dot, finite branch remainder and hard-mask switches remain distinct.',
            'Future harm alone is not SG error; fixed-event branch derivatives exclude spike crossings and untested clamp branches.',
            'Common/covariance saved-head algebra is retrospective and does not establish motor information sufficiency.',
            'A/B histories are real prefixes before scoring common future targets; no future feature swapping or input-to-output shortcut.',
            'No long-history gradient or capability/convergence claim; any selected candidate needs separate follow-up review.'],
        'stopping': 'Abort on source/data/code change, nonfinite, identity failure, incomplete gradient coverage, count overrun or memory limit; no adaptive expansion.',
        'prior_access': 'Completed training risk and all6 active B outcomes known; this is a prospective result-informed mechanism diagnosis, not blind prediction.',
        'execution': {'gpu_runs': 0, 'new_results': False}}
    if args.recovery_history is not None:
        history=json.loads(args.recovery_history.read_text(encoding='utf-8'))
        if len(history['runs'])!=1 or history['runs'][0]['status']!='report_serialization_failed':
            raise ValueError('Recovery registration requires exactly one report-loss attempt')
        plan['execution_recovery']={'history_path':str(args.recovery_history.resolve()),
            'history_sha256':digest(args.recovery_history),'maximum_replays':1,
            'cumulative_budget':history['cumulative_allowed_budget_if_recovery_approved'],
            'change':'NumPy scalar JSON representation only; no scientific endpoint, source, target, step or optimizer change',
            'prior_outcome_seen':False}
        plan['execution']={'previous_gpu_runs':1,'previous_reports_lost':1,'new_results':False}
    write_json(args.protocol, plan)
    print(f'Registered {sum(PLAN_COUNTS.values())} forwards,1408ticks,3backwards,2fork updates; no GPU run')


@contextmanager
def instrument(model, *, fixed=None, reset_only=False):
    """Cover both SpikeFn globals; reset detach never detaches pulse/STP/ALIF."""
    original_spike = reservoir.SpikeFn
    old_pipeline_spike = pipeline.SpikeFn
    original_begin = pipeline.begin_fly_prediction
    original_finish = model.finish_coba_tick
    result = {'masks': [], 'reset_units': [], 'calls': 0}

    class SpikeInstrument:
        @staticmethod
        def apply(margin):
            index = result['calls']
            result['calls'] += 1
            if fixed is None:
                value = original_spike.apply(margin)
            else:
                if index >= len(fixed) or tuple(fixed[index].shape) != tuple(margin.shape):
                    raise ValueError('Fixed-event shape/call-order mismatch')
                value = torch.as_tensor(fixed[index], device=margin.device, dtype=margin.dtype)
            result['masks'].append(value.detach().bool().cpu().numpy().copy())
            return value

    def unit_stats(v, spike, threshold, stage):
        with torch.no_grad():
            phi = 1/(1+(torch.pi*(v-threshold)).square())
            jac = 1-spike-v*phi
            after = v*(1-spike)
            result['reset_units'].append({'stage': stage,
                'pre_energy': float(v.double().square().sum()),
                'post_energy': float(after.double().square().sum()),
                'reset_discarded_energy': float((v.double().square()*spike).sum()),
                'ordinary_reset_v_jacobian_negative_fraction': float((jac<0).float().mean()),
                'ordinary_reset_v_jacobian_min': float(jac.min()),
                'ordinary_reset_v_jacobian_abs_mean': float(jac.abs().mean()),
                'scope': 'ATan reset Jacobian only; not the entire state Jacobian or energy flux'})

    def begin(m, state, *, decode=True, h_mean_decay=.99, base_rates=None, thresholds=None,
              conductance_gains=None, alif_params=None, stp_params=None):
        if not reset_only:
            value, pending = original_begin(m, state, decode=decode, h_mean_decay=h_mean_decay,
                base_rates=base_rates, thresholds=thresholds, conductance_gains=conductance_gains,
                alif_params=alif_params, stp_params=stp_params)
            with torch.no_grad():
                c, idx = pending.context, m.read_indices
                v = c['alpha'][:,idx]*state.h[:,idx]+c['beta_int'][:,idx]*c['base_current'][:,idx]
                threshold = c['eff_threshold']
                if torch.is_tensor(threshold) and threshold.numel()>1:
                    threshold = threshold[:,idx]
                spike = torch.as_tensor(result['masks'][-1],device=v.device,dtype=v.dtype)
                unit_stats(v, spike, threshold, 'begin_motor')
            return value, pending
        c = m.prepare_coba_tick(state.h, state.ring, state.ge, state.gi, state.b, state.x, state.u,
            base_rates=base_rates, thresholds=thresholds, conductance_gains=conductance_gains,
            alif_params=alif_params, stp_params=stp_params)
        idx = m.read_indices
        v = c['alpha'][:,idx]*state.h[:,idx]+c['beta_int'][:,idx]*c['base_current'][:,idx]
        threshold = c['eff_threshold']
        if torch.is_tensor(threshold) and threshold.numel()>1:
            threshold = threshold[:,idx]
        spike = SpikeInstrument.apply(v-threshold)
        unit_stats(v, spike, threshold, 'begin_motor')
        motor = v*(1-(spike.detach() if reset_only else spike))
        if m.read_centering:
            mean = state.h_mean[:,idx] if state.h_mean.numel() else torch.zeros_like(motor)
            motor = motor-(h_mean_decay*mean+(1-h_mean_decay)*motor)
        feature = m.output_read(motor)
        pending = pipeline.PendingFlyTick(state, c, feature, h_mean_decay)
        return (m.decoder(m.read_norm(feature)) if decode else feature), pending

    def finish(instance, context, drive, *, return_biophysics=False):
        ret, bio = original_finish(context, drive, return_biophysics=True)
        idx = instance.read_indices
        threshold = bio['eff_threshold']
        if torch.is_tensor(threshold) and threshold.numel()>1:
            threshold = threshold[:,idx]
        unit_stats(bio['v_pre'][:,idx], ret[1][:,idx], threshold, 'finish_motor')
        if reset_only:
            ret = (bio['v_pre']*(1-ret[1].detach()), *ret[1:])
        return (ret, bio) if return_biophysics else ret

    reservoir.SpikeFn = pipeline.SpikeFn = SpikeInstrument
    pipeline.begin_fly_prediction = begin
    model.finish_coba_tick = types.MethodType(finish, model)
    try:
        yield result
        if fixed is not None and result['calls'] != len(fixed):
            raise ValueError('Fixed-event incomplete mask coverage')
    finally:
        reservoir.SpikeFn, pipeline.SpikeFn = original_spike, old_pipeline_spike
        pipeline.begin_fly_prediction = original_begin
        model.finish_coba_tick = original_finish


class DiskTensor:
    """Bounded CPU staging for gradients; no full host or extra device clone."""
    def __init__(self, named, path):
        self.storage = np.memmap(path, mode='w+', dtype=np.float32,
                                 shape=(sum(p.numel() for p in named.values()),))
        self.values = {}
        start = 0
        for name,p in named.items():
            self.values[name] = torch.from_numpy(self.storage[start:start+p.numel()]).view(p.shape)
            start += p.numel()

    def capture(self, named, *, missing_zero=False):
        missing = []
        for name,p in named.items():
            dst = self.values[name].view(-1)
            if p.grad is None:
                if not missing_zero:
                    raise ValueError(f'Missing actual gradient {name}')
                missing.append(name)
                dst.zero_()
            else:
                if p.dtype != torch.float32 or not p.grad.is_contiguous():
                    raise ValueError('Expected contiguous original FP32 gradient')
                src = p.grad.detach().view(-1)
                for offset in range(0,p.numel(),65536):
                    dst[offset:offset+65536].copy_(src[offset:offset+65536])
                    if not torch.isfinite(dst[offset:offset+65536]).all():
                        raise ValueError(f'Nonfinite captured gradient {name}')
        self.storage.flush()
        return missing

    def close(self):
        self.values.clear()
        self.storage.flush()
        self.storage._mmap.close()


@contextmanager
def before_clip(named, disk):
    original = torch.nn.utils.clip_grad_norm_
    calls = [0]
    expected = {id(p) for p in named.values()}

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        if {id(p) for p in parameters} != expected:
            raise ValueError('Original clipping parameter coverage changed')
        calls[0] += 1
        disk.capture(named)
        return original(parameters, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = capture
    try:
        yield
        if calls[0] != 1:
            raise ValueError('Actual observe did not clip exactly once')
    finally:
        torch.nn.utils.clip_grad_norm_ = original


def branches(a,b):
    if len(a)!=len(b) or any(x.shape!=y.shape for x,y in zip(a,b)):
        raise ValueError('Spike instrument alignment mismatch')
    changes = [int(np.count_nonzero(x!=y)) for x,y in zip(a,b)]
    return {'begin_motor_changes': sum(changes[::2]), 'finish_body_changes': sum(changes[1::2]),
            'changed_by_call': changes, 'total_changes': sum(changes), 'total_calls': len(changes),
            'first_changed_call': next((i for i,n in enumerate(changes) if n),None)}


def realized_norm(named, old, names):
    """Measure the copied FP32 step, including interpolation rounding."""
    square = 0.
    for name in sorted(names):
        current, origin = named[name].detach().view(-1), old[name].view(-1)
        for offset in range(0,current.numel(),65536):
            delta = current[offset:offset+65536].cpu().double()-origin[offset:offset+65536].double()
            square += float(delta.square().sum())
    return square**.5


def gradient_equality(first, second, names):
    maximum = scale = square = 0.
    coordinates = 0
    for name in sorted(names):
        a, b = first[name].view(-1), second[name].view(-1)
        for offset in range(0,a.numel(),65536):
            x, y = a[offset:offset+65536], b[offset:offset+65536]
            difference = x.double()-y.double()
            maximum = max(maximum,float(difference.abs().max()))
            scale = max(scale,float(x.abs().max()))
            square += float(difference.square().sum())
            coordinates += x.numel()
    floor = 64*EPS*max(1.,scale)
    return {'maximum_absolute_error':maximum,'rms_error':(square/max(coordinates,1))**.5,
            'floor':floor,'passed':maximum<=floor}


@contextmanager
def clamp_instrument(learner, raw_gradient, old):
    """Account actual FP32 edge clipping without changing the original call."""
    original = learner.clamp_edges
    report = {'calls':0,'changed_coordinates':0,'unclamped_edge_displacement_square':0.,
              'clamp_correction_square':0.,'raw_gradient_dot_unclamped_edge_displacement':0.,
              'raw_gradient_dot_clamp_correction':0.}
    def capture():
        report['calls'] += 1
        for name, parameter in zip(learner.edge_names,learner.edges):
            flat, before, gradient = parameter.detach().view(-1),old[name].view(-1),raw_gradient[name].view(-1)
            for offset in range(0,flat.numel(),65536):
                pre = flat[offset:offset+65536].cpu()
                post = pre.clamp(0.,5.)
                delta = pre.double()-before[offset:offset+65536].double()
                correction = post.double()-pre.double()
                g = gradient[offset:offset+65536].double()
                report['changed_coordinates'] += int(correction.count_nonzero())
                report['unclamped_edge_displacement_square'] += float(delta.square().sum())
                report['clamp_correction_square'] += float(correction.square().sum())
                report['raw_gradient_dot_unclamped_edge_displacement'] += float((g*delta).sum())
                report['raw_gradient_dot_clamp_correction'] += float((g*correction).sum())
        return original()
    learner.clamp_edges = capture
    try:
        yield report
        if report['calls']!=1:
            raise ValueError('Original clamp cadence mismatch')
    finally:
        learner.clamp_edges = original


def identity(reference, actual, *, repeat_state=None, repeat_score=0.):
    error = max(abs(a-b) for a,b in zip(reference['scores'],actual['scores']))
    score_floor = max(repeat_score,64*EPS*max(1.,max(reference['scores'])))
    distance = state_distance(reference['state'],actual['state'])
    for name,row in distance.items():
        tensors = reference['state'].ring if name=='ring' else (getattr(reference['state'],name),)
        scale = max((float(t.abs().max()) for t in tensors if t.numel()),default=0.)
        floor = max(64*EPS*max(scale,1.), (repeat_state or {}).get(name,{}).get('max_absolute_difference',0.))
        if row['max_absolute_difference']>floor:
            raise ValueError(f'Identity physical field failed: {name}')
    flip = branches(reference['masks'],actual['masks'])
    if error>score_floor or flip['total_changes']:
        raise ValueError('Identity score or actual hard-mask gate failed')
    return {'max_score_error': error, 'score_floor': score_floor, 'state_distance':distance, 'branches':flip}


def merge_state_floor(previous, observed):
    """Use all old-parameter repeats, rather than the last observed replay."""
    return {name: {'max_absolute_difference': max(
        previous.get(name, {}).get('max_absolute_difference', 0.),
        row['max_absolute_difference'])} for name, row in observed.items()}


def selection_replay(reference, actual, *, repeat_state=None):
    """Measure local event instability at this particular updated parameter point."""
    distance = state_distance(reference['state'], actual['state'])
    switches = branches(reference['masks'], actual['masks'])
    error = max(abs(a-b) for a, b in zip(reference['scores'], actual['scores']))
    floors = {}
    for name in distance:
        tensors = reference['state'].ring if name=='ring' else (getattr(reference['state'],name),)
        scale = max((float(t.abs().max()) for t in tensors if t.numel()),default=0.)
        floors[name] = max(64*EPS*max(1.,scale),
            (repeat_state or {}).get(name,{}).get('max_absolute_difference',0.))
    state_stable = all(row['max_absolute_difference']<=floors[name]
        for name,row in distance.items())
    return {'max_score_error': error, 'state_distance': distance,
            'state_floors':floors,'continuous_state_stable':state_stable,
            'branches': switches, 'hard_masks_stable': switches['total_changes'] == 0}


def matched_norm_gate(radius, ordinary_norm, candidate_norm):
    """Reject unresolved radii and require relative, rather than absolute, matching."""
    absolute_floor = 64*EPS*max(1.,radius)
    relative_floor = 64*EPS*radius
    return {'radius_resolved':radius>10*absolute_floor,
            'norms_match':max(abs(ordinary_norm-radius),abs(candidate_norm-radius))<=relative_floor,
            'absolute_floor':float(absolute_floor),'relative_floor':float(relative_floor),
            'scope':'Floating-point identification only; not a physical parameter or functional norm match'}


def fixed_prior(path):
    """Identical train-only add-one counts with bounded CPU staging."""
    tokens = training.load_tokens(path)
    counts = np.ones(training.VOCAB,dtype=np.float64)
    for offset in range(0,len(tokens),65536):
        chunk = np.asarray(tokens[offset:offset+65536],dtype=np.int64)
        if len(chunk) and (chunk.min()<0 or chunk.max()>=training.VOCAB):
            raise ValueError('Reference target outside registered vocabulary')
        counts += np.bincount(chunk,minlength=training.VOCAB)
    return torch.tensor(np.log(counts/counts.sum()),dtype=torch.float32)


def head_algebra(features, targets, saved, fixed):
    """Exactly the frozen decoder gradient algebra, not body/Adam attribution."""
    w = saved['model']
    with torch.no_grad():
        q = F.rms_norm(features,(features.shape[-1],),w['read_norm.weight'])
        residual = F.linear(q,w['decoder.weight'])
        bias = w['decoder.bias']
        common = residual.mean(0,keepdim=True)
        logits = {'full':residual+bias, 'bias':bias[None].expand_as(residual),
                  'common':common.expand_as(residual)+bias,
                  'content_without_common':residual-common+bias,
                  'fixed_train_frequency':fixed[None].expand_as(residual)}
        scores = {key:F.cross_entropy(value,targets,reduction='none').tolist() for key,value in logits.items()}
        errors = logits['full'].softmax(-1).double()
        errors[torch.arange(len(targets)),targets]-=1
        q = q.double()
        em,qm = errors.mean(0),q.mean(0)
        ec,qc = errors-em,q-qm
        n = len(q)
        total = float(((errors@errors.T)*(q@q.T)).sum()/n**2)
        background = float(em.square().sum()*qm.square().sum())
        content = float(((ec@ec.T)*(qc@qc.T)).sum()/n**2)
        cross = float(((ec@em)*(qc@qm)).sum()/n)
        return {'scores':scores,'means':{key:float(np.mean(value)) for key,value in scores.items()},
            'per_event_norms': {'projected_feature':features.double().norm(dim=1).tolist(),
                'normalized_feature':q.norm(dim=1).tolist(),
                'residual_logits':residual.double().norm(dim=1).tolist(),
                'full_logits':logits['full'].double().norm(dim=1).tolist(),
                'content_logits':(residual-common).double().norm(dim=1).tolist()},
            'shared_norms': {'common_logits':float(common.double().norm()),
                'bias_logits':float(bias.double().norm()),
                'mean_normalized_feature':float(qm.norm())},
            'decoder_gradient':{'full_norm':max(total,0)**.5,'common_norm':max(background,0)**.5,
                'covariance_norm':max(content,0)**.5,'common_covariance_inner':cross,
                'relative_identity_error':abs(total-background-content-2*cross)/max(total,1e-30)},
            'normalized_feature_variation_fraction':float(qc.square().sum()/q.square().sum().clamp_min(1e-30)),
            'scope':'Same frozen head on actual causal features; common/content counterfactuals use the whole window retrospectively. Not legal alternative online predictors or proof of code sufficiency.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('F:/fly_checkpoints/q8_fly_pipeline14_joint_96k/last.pt'))
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--scratch',type=Path)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--recovery-history',type=Path)
    args=parser.parse_args()
    if args.register:
        register(args)
        return
    if args.approval is None or args.output is None or args.scratch is None:
        parser.error('Execution requires independently approved plan, output and scratch')
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    approval=json.loads(args.approval.read_text(encoding='utf-8'))
    if (approval.get('status')!='approved' or approval.get('prereg_sha256')!=digest(args.protocol)
            or approval.get('script_sha256')!=digest(__file__)
            or approval.get('source_hashes')!=plan['source_hashes'] or source_hashes()!=plan['source_hashes']):
        raise ValueError('Independent diagnostic approval/content lock failed')
    if digest(args.checkpoint)!=plan['source']['sha256']:
        raise ValueError('Mature source differs from locked checkpoint')
    for item in plan['data'].values():
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Registered data or graph changed')
    if 'execution_recovery' in plan:
        recovery=plan['execution_recovery']
        if digest(recovery['history_path'])!=recovery['history_sha256']:
            raise ValueError('Recovery exposure/history lock changed')
    if args.output.exists():
        raise FileExistsError('Preserve previous outcomes; output must be new')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required; registration alone is CPU')
    saved=torch.load(args.checkpoint,mmap=True,map_location='cpu',weights_only=False)
    cfg=saved['config']
    if saved['format']!=training.FORMAT or cfg['read_centering'] or cfg['learn_stp']:
        raise ValueError('This locked source is the native no-centering/frozen-STP arm')
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    model=reservoir.FlyReservoirLM(plan['data']['graph']['path'],vocab_size=50257,d_model=768,
        injection='topographic',read_surface='output',synapse_model='coba',use_alif=True,use_stp=True,
        decoder_bias=True,read_centering=False,use_read_gamma_trace=False,use_latent_predictor=False).cuda()
    model.dan_plastic_lr=0.

    def fresh_learner():
        result=pipeline.FlyPipelineLearner(model,training.build_rest(model,'cuda'),
            lr=cfg['lr'],lr_synapse=cfg['lr'],lr_sensory=cfg['lr'],max_grad_norm=cfg['max_grad_norm'],
            plasticity_optimizer='adamw',settle_ticks=0,writer_baseline_clock='input',learn_stp=False)
        training.restore_learner(saved,result)
        return result

    learner=fresh_learner()
    beginning=clone_state(learner.state)
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    groups={'head':set(plan['parameter_groups']['head']),'bias':set(plan['parameter_groups']['bias'])}
    groups['body']=set(named)-groups['head']-groups['bias']
    groups['full']=set(named)
    if not groups['head'].issubset(named) or not groups['bias'].issubset(named) or not groups['body']:
        raise ValueError('Actual parameter partition differs from registered groups')
    old={name:saved['model'][name] for name in named}
    args.scratch.mkdir(parents=True,exist_ok=True)
    required=sum(p.numel()*4 for p in named.values())*5+256*2**20
    if shutil.disk_usage(args.scratch).free<required:
        raise OSError('Insufficient disk for bounded gradient/snapshot staging')
    paths=[args.scratch/name for name in ('ordinary_gradient.bin','fixed_gradient.bin','reset_gradient.bin','ordinary_theta.bin','reset_theta.bin')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Scratch must not overwrite an earlier partial diagnostic')
    ordinary_g,fixed_g,reset_g=[DiskTensor(named,path) for path in paths[:3]]
    snapshots=[]
    ledger={'window_forwards_by_role':{},'forward_physical_ticks':0,'backwards':0,'disposable_optimizer_updates':0}
    report={'scope':plan['scope'],'protocol_sha256':digest(args.protocol),'source_sha256':plan['source']['sha256'],
        'parameter_groups':{key:sorted(value) for key,value in groups.items()},'controls':{},'candidate_controls':{},
        'execution_ledger':ledger,'complete':False}
    fit=torch.tensor(plan['targets']['fit'],device='cuda',dtype=torch.long)
    future=torch.tensor(plan['targets']['future'],device='cuda',dtype=torch.long)
    bprefix=torch.tensor(plan['targets']['B_prefix'],device='cuda',dtype=torch.long)

    def charge(role,backward=False,update=False):
        ledger['window_forwards_by_role'][role]=ledger['window_forwards_by_role'].get(role,0)+1
        ledger['forward_physical_ticks']+=WIDTH
        ledger['backwards']+=int(backward)
        ledger['disposable_optimizer_updates']+=int(update)
        if (ledger['forward_physical_ticks']>1408 or ledger['backwards']>3 or ledger['disposable_optimizer_updates']>2
                or ledger['window_forwards_by_role'][role]>PLAN_COUNTS[role]):
            raise ValueError('Preregistered diagnostic budget exceeded')

    def check_state(state):
        for name,value in state.state_dict().items():
            for tensor in value if name=='ring' else (value,):
                if not torch.isfinite(tensor).all():
                    raise ValueError('Nonfinite physical field')
        training.memory_gate(3900)

    def forward(initial,targets,role,*,fixed=None,reset=False,backward=False):
        charge(role,backward=backward)
        learner.state=clone_state(initial)
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward), instrument(model,fixed=fixed,reset_only=reset) as events:
            scores,state,features=learner.forward_window(targets[None],targets[None])
            if backward:
                scores.mean().backward()
            output={'scores':scores.detach().cpu().tolist(),'state':clone_state(state),
                    'features':features.detach().cpu(),'masks':events['masks'],'reset_units':events['reset_units']}
        if len(output['masks'])!=2*WIDTH or any(not np.isfinite(x) for x in output['scores']):
            raise ValueError('Incomplete threshold coverage or nonfinite scores')
        check_state(output['state'])
        return output

    def weights(snapshot,names=None,fraction=1.,head_reference=None):
        restore_parameters(named,snapshot,selected=names,origin=old,fraction=fraction)
        if head_reference is not None:
            restore_parameters(named,head_reference,selected=groups['head']|groups['bias'])

    repeat_error = maximum_score = 0.

    def control(snapshot,names,fraction,role,head_reference=None,*,repeat=False):
        nonlocal repeat_error, maximum_score
        weights(snapshot,names,fraction,head_reference)
        actual_body_norm = realized_norm(named,old,groups['body']) if head_reference is not None else None
        fitted=forward(beginning,fit,role)
        following=forward(base_fit['state'],future,role)
        maximum_score = max(maximum_score, max(fitted['scores']+following['scores']))
        result = {'fit_nll':float(np.mean(fitted['scores'])),'future_nll':float(np.mean(following['scores'])),
            'fit_scores':fitted['scores'],'future_scores':following['scores'],
            'realized_body_displacement_norm':actual_body_norm,
            'fit_spikes':branches(base_fit['masks'],fitted['masks']),
            'future_spikes':branches(base_future['masks'],following['masks'])}
        if repeat:
            again_fit=forward(beginning,fit,'selection_control_repeats')
            again_future=forward(base_fit['state'],future,'selection_control_repeats')
            a,b=selection_replay(fitted,again_fit,repeat_state=repeat_fit_state),selection_replay(following,again_future,repeat_state=repeat_future_state)
            repeat_error=max(repeat_error,a['max_score_error'],b['max_score_error'])
            maximum_score=max(maximum_score,max(again_fit['scores']+again_future['scores']))
            result['local_replay']={'fit':a,'future':b,
                'stable':all(row['hard_masks_stable'] and row['continuous_state_stable'] for row in (a,b))}
        return result

    try:
        training.memory_gate(3900)
        base_fit=forward(beginning,fit,'baseline_and_two_identity_fit_future')
        base_future=forward(base_fit['state'],future,'baseline_and_two_identity_fit_future')
        report['baseline']={'fit_scores':base_fit['scores'],'future_scores':base_future['scores'],
            'fit_nll':float(np.mean(base_fit['scores'])),'future_nll':float(np.mean(base_future['scores'])),
            'fit_reset_units':base_fit['reset_units'],'future_reset_units':base_future['reset_units']}
        reference_logits=fixed_prior(Path(saved['config']['fixed_reference']['train']))
        report['head_algebra_fit']=head_algebra(base_fit['features'],fit.cpu(),saved,reference_logits)
        report['head_algebra_future']=head_algebra(base_future['features'],future.cpu(),saved,reference_logits)
        report['repeat_identity']=[]
        repeat_error=0.
        maximum_score=max(base_fit['scores']+base_future['scores'])
        repeat_fit_state={}
        repeat_future_state={}
        for _ in range(2):
            repeated_fit=forward(beginning,fit,'baseline_and_two_identity_fit_future')
            repeated_future=forward(base_fit['state'],future,'baseline_and_two_identity_fit_future')
            a,b=identity(base_fit,repeated_fit),identity(base_future,repeated_future)
            repeat_error=max(repeat_error,a['max_score_error'],b['max_score_error'])
            repeat_fit_state=merge_state_floor(repeat_fit_state,a['state_distance'])
            repeat_future_state=merge_state_floor(repeat_future_state,b['state_distance'])
            report['repeat_identity'].append({'fit':a,'future':b})
        other_prefix=forward(beginning,bprefix,'B_prefix_then_same_future')
        other_future=forward(other_prefix['state'],future,'B_prefix_then_same_future')
        report['context_contrast']={'A_prefix':'actual fit32','B_prefix':'fresh validation32',
            'same_future_targets':plan['targets']['future'],'A_future_scores':base_future['scores'],
            'B_future_scores':other_future['scores'],
            'B_minus_A_future_nll':float(np.mean(other_future['scores'])-np.mean(base_future['scores'])),
            'scope':'Actual pre-observation scores after different real histories; one fixed learned head, not a code sufficiency proof'}
        del other_prefix,other_future,repeated_fit,repeated_future
        learner.state=clone_state(beginning)
        with before_clip(named,ordinary_g),clamp_instrument(learner,ordinary_g.values,old) as ordinary_clamp,instrument(model) as ordinary_events:
            charge('ordinary_observe',backward=True,update=True)
            ordinary_scores,ordinary_metrics=learner.observe(fit)
        actual={'scores':ordinary_scores,'state':clone_state(learner.state),'masks':ordinary_events['masks']}
        report['ordinary_observe_identity']=identity(base_fit,actual,repeat_state=repeat_fit_state,repeat_score=repeat_error)
        if learner.events!=saved['learner']['events']+WIDTH or learner.updates!=saved['learner']['updates']+1:
            raise ValueError('Original observe update cadence mismatch')
        ordinary_new,ordinary_storage=parameter_snapshot(named,paths[3])
        snapshots.append((ordinary_new,ordinary_storage))
        report['ordinary_metrics']=ordinary_metrics
        report['ordinary_clamp']=ordinary_clamp
        report['ordinary_clip_factor']=min(1.,learner.max_grad_norm/(ordinary_metrics['grad_norm_before_clip']+1e-6))
        report['ordinary_surrogate_direction']={key:directional_summary(ordinary_g.values,old,ordinary_new,names) for key,names in groups.items()}
        if abs(report['ordinary_surrogate_direction']['full']['gradient_norm']-ordinary_metrics['grad_norm_before_clip'])>64*EPS*max(1.,ordinary_metrics['grad_norm_before_clip']):
            raise ValueError('Captured preclip gradient norm identity failed')
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        for key in ('full','body','head','bias'):
            for scale in ((1.,.1) if key=='body' else (1.,)):
                row=control(ordinary_new,groups[key],scale,'ordinary_registered_masks_fit_future',repeat=key in ('full','body') and scale==1.)
                if key=='body':
                    fixed_row=forward(beginning,fit,'ordinary_body_two_scales_fixed_fit',fixed=base_fit['masks'])
                    row['fixed_event_fit_nll']=float(np.mean(fixed_row['scores']))
                    row['hard_minus_fixed_event_fit']=row['fit_nll']-row['fixed_event_fit_nll']
                    del fixed_row
                report['controls'][f'{key}_{scale:g}']=row
        weights(old)
        fixed_result=forward(beginning,fit,'fixed_event_backward',fixed=base_fit['masks'],backward=True)
        report['fixed_backward_identity']=identity(base_fit,fixed_result,repeat_state=repeat_fit_state,repeat_score=repeat_error)
        report['fixed_event_missing_gradient_names']=fixed_g.capture(named,missing_zero=True)
        report['ordinary_fixed_derivative_direction']={key:directional_summary(fixed_g.values,old,ordinary_new,names) for key,names in groups.items()}
        del fixed_result
        model.zero_grad(set_to_none=True)
        del learner
        gc.collect(); torch.cuda.empty_cache()
        learner=fresh_learner()
        candidate_identity_future=forward(base_fit['state'],future,'reset_candidate_future_identity',reset=True)
        report['reset_forward_identity']={'future':identity(base_future,candidate_identity_future,repeat_state=repeat_future_state,repeat_score=repeat_error)}
        del candidate_identity_future
        learner.state=clone_state(beginning)
        with before_clip(named,reset_g),clamp_instrument(learner,reset_g.values,old) as candidate_clamp,instrument(model,reset_only=True) as candidate_events:
            charge('reset_candidate_observe',backward=True,update=True)
            candidate_scores,candidate_metrics=learner.observe(fit)
        candidate_actual={'scores':candidate_scores,'state':clone_state(learner.state),'masks':candidate_events['masks']}
        report['reset_observe_identity']=identity(base_fit,candidate_actual,repeat_state=repeat_fit_state,repeat_score=repeat_error)
        if learner.events!=saved['learner']['events']+WIDTH or learner.updates!=saved['learner']['updates']+1:
            raise ValueError('Candidate observe update cadence mismatch')
        reset_new,reset_storage=parameter_snapshot(named,paths[4])
        snapshots.append((reset_new,reset_storage))
        report['candidate_metrics']=candidate_metrics
        report['candidate_clamp']=candidate_clamp
        report['candidate_clip_factor']=min(1.,learner.max_grad_norm/(candidate_metrics['grad_norm_before_clip']+1e-6))
        report['raw_head_gradient_equality']=gradient_equality(ordinary_g.values,reset_g.values,groups['head']|groups['bias'])
        report['candidate_surrogate_direction']={key:directional_summary(reset_g.values,old,reset_new,names) for key,names in groups.items()}
        report['candidate_original_surrogate_direction']={key:directional_summary(ordinary_g.values,old,reset_new,names) for key,names in groups.items()}
        report['ordinary_candidate_surrogate_direction']={key:directional_summary(reset_g.values,old,ordinary_new,names) for key,names in groups.items()}
        report['candidate_fixed_derivative_direction']={key:directional_summary(fixed_g.values,old,reset_new,names) for key,names in groups.items()}
        if abs(report['candidate_surrogate_direction']['full']['gradient_norm']-candidate_metrics['grad_norm_before_clip'])>64*EPS*max(1.,candidate_metrics['grad_norm_before_clip']):
            raise ValueError('Candidate preclip gradient norm identity failed')
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        for key in ('full','body'):
            report['candidate_controls'][f'{key}_1']=control(reset_new,groups[key],1.,'reset_candidate_registered_masks_fit_future',repeat=True)
        ordinary_norm=report['ordinary_surrogate_direction']['body']['displacement_norm']
        candidate_norm=report['candidate_surrogate_direction']['body']['displacement_norm']
        radius=min(ordinary_norm,candidate_norm)
        scales={'ordinary':radius/ordinary_norm if ordinary_norm else 0.,'reset':radius/candidate_norm if candidate_norm else 0.}
        report['matched_norm']={'radius':radius,'scales':scales,
            'common_head':'ordinary full-step head+bias, identical in both arms',
            'ordinary':control(ordinary_new,groups['body'],scales['ordinary'],'two_body_matched_norm_fit_future',ordinary_new,repeat=True),
            'reset':control(reset_new,groups['body'],scales['reset'],'two_body_matched_norm_fit_future',ordinary_new,repeat=True)}
        base_joint=report['controls']['full_1']; candidate_joint=report['candidate_controls']['full_1']
        base_body=report['controls']['body_1']; candidate_body=report['candidate_controls']['body_1']
        matched_base=report['matched_norm']['ordinary']; matched_reset=report['matched_norm']['reset']
        norm_gate=matched_norm_gate(radius,matched_base['realized_body_displacement_norm'],matched_reset['realized_body_displacement_norm'])
        report['matched_norm']['numerical_gate']=norm_gate
        tau=max(1e-6,10*repeat_error,64*EPS*max(1.,maximum_score))
        report['decision_floor']=tau
        tests={'joint_future_improved':candidate_joint['future_nll']<base_joint['future_nll']-tau,
               'joint_fit_nonworse':candidate_joint['fit_nll']<=base_joint['fit_nll']+tau,
               'body_fit_improved':candidate_body['fit_nll']<base_body['fit_nll']-tau,
               'body_fit_nonworse_than_old':candidate_body['fit_nll']<=report['baseline']['fit_nll']+tau,
               'body_future_improved':candidate_body['future_nll']<base_body['future_nll']-tau,
               'body_future_nonworse_than_old':candidate_body['future_nll']<=report['baseline']['future_nll']+tau,
               'matched_body_future_improved':matched_reset['future_nll']<matched_base['future_nll']-tau,
               'matched_body_fit_nonworse':matched_reset['fit_nll']<=matched_base['fit_nll']+tau,
               'matched_norm_valid':norm_gate['norms_match'],
               'matched_radius_resolved':norm_gate['radius_resolved']}
        tests['raw_head_gradients_equal']=report['raw_head_gradient_equality']['passed']
        tests['selection_replays_stable']=all(row['local_replay']['stable'] for row in
            (base_joint,candidate_joint,base_body,candidate_body,matched_base,matched_reset))
        numerical_ok=all(tests[key] for key in ('matched_norm_valid','matched_radius_resolved',
            'raw_head_gradients_equal','selection_replays_stable'))
        status='ambiguous' if not numerical_ok else ('locally_supported' if all(tests.values()) else 'locally_unsupported')
        report['candidate_selection']={'status':status,'checks':tests,'selected_for_followup_review':all(tests.values()),
            'scope':'Fixed-scale1 local finite diagnosis only; not approval to change production or a capability claim'}
        if ledger['window_forwards_by_role']!=PLAN_COUNTS or ledger['forward_physical_ticks']!=1408 or ledger['backwards']!=3 or ledger['disposable_optimizer_updates']!=2:
            raise ValueError('Complete actual ledger does not match preregistered schedule')
        if source_hashes()!=plan['source_hashes'] or digest(args.checkpoint)!=plan['source']['sha256']:
            raise ValueError('Code or source changed during diagnosis')
        report['source_unchanged']=True
        report['complete']=True
        report['resources']=training.memory_gate(3900)
        write_json(args.output,report)
        print('Completed locked diagnostic; no production checkpoint or optimizer changed')
    except BaseException as error:
        report['error']=repr(error)
        report['complete']=False
        write_json(args.output.with_name(args.output.stem+'_partial.json'),report)
        raise
    finally:
        # Close only this diagnostic's map views before removing its own staging.
        for disk in (ordinary_g,fixed_g,reset_g):
            disk.close()
        for values,storage in snapshots:
            values.clear(); storage.flush(); storage._mmap.close()
        for path in paths:
            path.unlink(missing_ok=True)


if __name__=='__main__':
    main()
