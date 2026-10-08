"""Bounded motor-physics/unknown-boundary response assay, CPU only.

True boundary flux is used only in the explicitly labeled algebraic oracle
check. The autonomous student receives origin state and existing pulse queues;
future labels are loss-only. This arm adds physical equations, state and flux
supervision and is not a capacity-matched pure-neural ablation.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ib.check_fly_rtc_cpu import (
    responses, FlyReservoirLM, TickResponseStudent, initial_physical,
    copy_physical, quiet_teacher_step, physical_options,
)
from scripts.ib.check_fly_rtc_surface_cpu import surface_subset
from information_boltzmann.core.fly_rtc_surface import MotorSurfaceForecaster
from information_boltzmann.core.fly_rtc_motor_flux import MotorLocalState, MotorFluxForecaster


def concatenate_states(states):
    return MotorLocalState(**{f.name: (tuple(torch.cat([s.ring[d] for s in states]) for d in range(4))
                             if f.name == 'ring' else torch.cat([getattr(s, f.name) for s in states]))
                              for f in fields(MotorLocalState)})


def take_local(state, lanes):
    return MotorLocalState(**{f.name: tuple(p[lanes] for p in state.ring) if f.name == 'ring'
                             else getattr(state, f.name)[lanes] for f in fields(MotorLocalState)})


@torch.no_grad()
def batch_data(student, data, lanes):
    base = surface_subset(data, lanes)
    origins = concatenate_states([take_local(student.local_state(s), lanes) for s in data[8]])
    queue = torch.cat([student.known_queue(s)[:, lanes] for s in data[8]], 1)
    target = []
    for root, future in zip(data[8], data[9]):
        before = [root] + future[:-1]
        target.append(torch.stack([student.external_arrival(s)[lanes] for s in before]))
    return base, origins, queue, torch.cat(target, 1)


@torch.no_grad()
def ambiguity_check(model):
    """Two legal numerical states with identical existing student observations."""
    codec = model.rtc_student.codec
    sampled = set(codec.sample_indices[codec.sample_mask.bool()].tolist())
    unseen = [int(j) for j in model.read_indices if int(j) not in sampled]
    if not unseen:
        return dict(status='no_unsampled_motor_example', scope='Construct a null-space case separately')
    a = initial_physical(model, 1)
    a.h[:, model.read_indices] = .03
    b = copy_physical(a)
    b.ge[:, unseen[0]] += .005
    za, zb = codec.encode(a), codec.encode(b)
    next_a = quiet_teacher_step(model, a, physical_options(model))
    next_b = quiet_teacher_step(model, b, physical_options(model))
    return dict(status='same_observation_different_next_state', neuron_index=unseen[0],
                codec_max_delta=float((za-zb).abs().max()),
                raw_motor_max_delta=float((a.h[:, model.read_indices]-b.h[:, model.read_indices]).abs().max()),
                next_motor_max_delta=float((next_a.h[:, model.read_indices]-next_b.h[:, model.read_indices]).abs().max()),
                scope='Legal numerical initial states; reachability from one particular stream not established')


@torch.no_grad()
def oracle_closure(student, model, data, coefficients):
    maxima = {name: 0. for name in ('h', 'ge', 'gi', 'b', 'x', 'u', 'mean', 'ring')}
    for origin, future in zip(data[8], data[9]):
        local, physical = student.local_state(origin), origin
        for actual in future:
            local = student.integrate(local, student.external_arrival(physical), coefficients)
            target = student.local_state(actual)
            for name in maxima:
                if name == 'ring':
                    error = max(float((a-b).abs().max()) for a,b in zip(local.ring,target.ring))
                else:
                    error = float((getattr(local,name)-getattr(target,name)).abs().max())
                maxima[name] = max(maxima[name], error)
            physical = actual
    return dict(maximum_absolute_error=maxima,
                scope='Teacher-flux identity only, not autonomous forecast',
                passed=max(maxima.values()) <= 2e-6)


def evaluate(student, model, data, coefficients):
    base, origins, queue, target_flux = data
    history, target_z, target_y, root_y, motor, target_m, mean = base
    with torch.no_grad():
        z, m, means, flux = student(history, origins, coefficients, queue)
        response = MotorSurfaceForecaster.decode(model, m, means)
        errors = (response[1:]-target_y).square().mean((1,2))
        held = (target_y-root_y[None]).square().mean((1,2))
        raw = (m[1:]-target_m).square().mean((1,2))
        raw_held = (target_m-motor[None]).square().mean((1,2))
        return dict(motor_error_ratio=float(errors.mean()/held.mean()),
                    raw_motor_error_ratio=float(raw.mean()/raw_held.mean()),
                    motor_delta_mse=float(errors.mean()), hold_motor_delta_mse=float(held.mean()),
                    flux_mse=float((flux-target_flux).square().mean()),
                    zero_tick_max_error=float((response[0]-root_y).abs().max()),
                    per_tick=[dict(tick=k+1, motor=float(errors[k]), hold_motor=float(held[k]),
                                   ratio=float(errors[k]/held[k]), raw_motor=float(raw[k]),
                                   hold_raw_motor=float(raw_held[k])) for k in range(len(errors))])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates',type=int,default=120)
    parser.add_argument('--output',type=Path,default=ROOT/'results/published/fly_rtc_flux_cpu_20261007.json')
    args=parser.parse_args()
    graph=ROOT/'results/fly_rtc_cpu_numerical/real_induced_graph.npz'
    previous=json.loads((ROOT/'results/published/fly_rtc_cpu_response_20261007.json').read_text())
    if hashlib.sha256(graph.read_bytes()).hexdigest()!=previous['graph']['graph_sha256']:
        raise ValueError('Graph changed')
    prereg=dict(seed=11, updates=args.updates, horizon=14, train_lanes=[0,1,2,3], heldout_lanes=[4,5,6,7],
               review='rtc_contract_review approved local-state closure with known arrival queue',
               oracle_identity='True external flux must reproduce all local state fields within2e-6 numerical rounding.',
               autonomous_prediction='Learn only newly emitted outside flux; known prefix is origin-ring data. Future teacher states loss-only.',
               loss='Train-normalized raw motor, regional latent and external flux MSE',
               stopping='Fixed120updates, no validation tuning',
               comparison_limit='Adds local physical equations/state/flux supervision; improvement is hybrid system evidence, not a parameter-matched pure neural ablation.',
               failure_action='Oracle identity failure blocks autonomous fit; forecast failure narrows unknown boundary model/closure, not additional bath or read patches.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_name(args.output.stem+'_preregistered.json').write_text(json.dumps(prereg,indent=2),encoding='utf-8')
    start=time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model=FlyReservoirLM(graph,vocab_size=16,d_model=16,injection='sensory',read_surface='output',
                         synapse_model='coba',use_alif=True,use_stp=True)
    model.rtc_student=TickResponseStudent.from_model(model,latent_dim=32,sample_per_region=32,horizon=14,seed=11)
    model.eval()
    data=responses(model,14,11,return_surface=True,return_physical=True)
    student=MotorFluxForecaster(model)
    coefficients=student.coefficients(model)
    ambiguity=ambiguity_check(model)
    identity=oracle_closure(student,model,data,coefficients)
    print(json.dumps(dict(ambiguity=ambiguity,oracle_identity=identity)),flush=True)
    if not identity['passed']:
        raise RuntimeError('Local physical split failed; stop before learned fit')
    train=batch_data(student,data,slice(0,4))
    heldout=batch_data(student,data,slice(4,8))
    before=evaluate(student,model,heldout,coefficients)
    base=train[0]
    zscale=(base[1]-base[0][0][None]).square().mean().clamp_min(1e-12)
    mscale=(base[5]-base[4][None]).square().mean().clamp_min(1e-12)
    fscale=train[3].square().mean().clamp_min(1e-12)
    optimizer=torch.optim.AdamW(student.parameters(),lr=2e-3,weight_decay=0)
    curve=[]
    for update in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        z,m,_,flux=student(base[0],train[1],coefficients,train[2])
        loss=F.mse_loss(z[1:],base[1])/zscale+F.mse_loss(m[1:],base[5])/mscale+F.mse_loss(flux,train[3])/fscale
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite student fit')
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
        if not torch.isfinite(gradient): raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        if update==0 or (update+1)%20==0:
            point=dict(update=update+1,normalized_train_loss=float(loss.detach()))
            curve.append(point); print(json.dumps(point),flush=True)
    report=dict(preregistration=prereg,device='cpu',graph=previous['graph'],
                ambiguity=ambiguity,oracle_identity=identity,before=before,
                after=evaluate(student,model,heldout,coefficients),
                train_after=evaluate(student,model,train,coefficients),
                heldout_by_lane=[evaluate(student,model,batch_data(student,data,slice(i,i+1)),coefficients) for i in range(4,8)],
                curve=curve,parameters=sum(p.numel() for p in student.parameters()),updates=args.updates,
                checkpoints_written=0,teacher_motor_counterfactual_energy=data[4],
                elapsed_seconds=time.perf_counter()-start)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(heldout_ratio=report['after']['motor_error_ratio'],before_ratio=before['motor_error_ratio'],
                         elapsed_seconds=report['elapsed_seconds'],report=str(args.output))),flush=True)


if __name__=='__main__': main()
