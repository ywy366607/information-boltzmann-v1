"""Distinguish sensory pulse depletion, physical latency, and read geometry.

Read-only CPU forks of one mature individual. An eight-tick quiet pulse response
and one real 32-event continuing window diagnose mechanisms, not capabilities.
The continuing production learner, weights, optimizer and cursors are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_fly_current_token_causality import ROOT, load_mapped_model
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, advance_fly_input_event


def physical_copy(raw):
    return FlyPhysicalState(**{key: tuple(t.detach().clone() for t in value)
        if key == 'ring' else value.detach().clone() for key, value in raw.items()})


def gradient_parts(a, h):
    """Exact linear-weight gradient identity; no optimizer approximation."""
    mean_a, mean_h = a.mean(0), h.mean(0)
    common = mean_a[:, None]*mean_h[None, :]
    covariance = (a-mean_a).T@(h-mean_h)/len(h)
    total = a.T@h/len(h)
    return {
        'total_norm': float(total.norm()), 'common_norm': float(common.norm()),
        'covariance_norm': float(covariance.norm()),
        'common_to_covariance_norm_ratio': float(common.norm()/covariance.norm().clamp_min(1e-30)),
        'common_covariance_cosine': float((common*covariance).sum()/(common.norm()*covariance.norm()).clamp_min(1e-30)),
        'relative_identity_error': float((total-common-covariance).norm()/total.norm().clamp_min(1e-30)),
        'scope': 'Unclipped saved-head gradient; Adam moments and actual update displacement excluded.',
    }


def read_pair(model, ha, hb):
    za, zb = model.output_read(ha), model.output_read(hb)
    qa, qb = model.read_norm(za), model.read_norm(zb)
    la, lb = model.decoder(qa), model.decoder(qb)
    pa, pb = la.softmax(-1), lb.softmax(-1)
    delta = lb-la
    centered_delta = delta-(pa*delta).sum(-1, keepdim=True)
    dz = zb-za
    radial = za*(za*dz).sum()/za.square().sum().clamp_min(1e-30)
    relative = lambda a,b: float((a-b).norm()/((a.norm()+b.norm())*.5).clamp_min(1e-30))
    return {
        'motor_relative_contrast': relative(ha,hb),
        'projected_relative_contrast': relative(za,zb),
        'normalized_relative_contrast': relative(qa,qb),
        'projected_difference_tangent_fraction': float((dz-radial).square().sum()/dz.square().sum().clamp_min(1e-30)),
        'probability_fisher_squared_contrast': float((pa*centered_delta.square()).sum()),
        'symmetrized_probability_kl': float(((pa-pb)*(la.log_softmax(-1)-lb.log_softmax(-1))).sum()*.5),
        'max_logits_contrast': float(delta.abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/q8_fly_bptt32_adamw_continuous_100k'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.run/'last.pt', map_location='cpu', mmap=True, weights_only=False)
    model = load_mapped_model(saved)
    cfg, old = saved['config'], saved['learner']
    raw = old['physical']
    motor, sensory = model.read_indices, model.injection_index
    train = np.load(ROOT/cfg['data']/'train.npy', mmap_mode='r')
    cursor, previous = int(saved['train_cursor']), int(old['previous_token'])
    alternate = next(int(t) for t in train[cursor+1:cursor+128] if int(t) != previous)
    settle, clock = old.get('settle_ticks', 0), old.get('writer_baseline_clock','physical')
    report = {'scope': __doc__, 'checkpoint_tokens': int(saved['bptt_train_tokens']),
              'input_pair': [previous,alternate], 'normal_read_tick_after_input': settle,
              'physical_ticks_per_input': 1+settle, 'writer_baseline_clock': clock,
              'preregistration': 'results/published/fly_clock_readout_preregistered.json'}
    # Wrap transmission only inside this diagnostic process to observe A before
    # the synaptic filter. No production code/kernel or numerical result changes.
    step_original = model.step
    namespace = step_original.__func__.__globals__
    transmit_original = namespace['execute_delayed_synaptic_transmission']
    arrivals, traces = [], []

    def transmit_traced(*positional, **options):
        output = transmit_original(*positional, **options)
        arrivals.append(output[:,motor].clone())
        return output

    def step_traced(*positional, **options):
        arrivals.clear()
        ret, bio = step_original(*positional, **options, return_biophysics=True)
        traces.append({
            'arrival_e': arrivals[0], 'arrival_i': arrivals[1],
            'ge': ret[3][:,motor].clone(), 'gi': ret[4][:,motor].clone(),
            'v_pre': bio['v_pre'][:,motor].clone(), 'h_post': ret[0][:,motor].clone(),
            'spike_sensory': ret[1][:,sensory].clone(),
            'pulse_sensory': bio['transmitted_pulse'][:,sensory].clone(),
            'filter_e': (1-bio['leak_se'][:,motor]).clone(),
            'filter_i': (1-bio['leak_si'][:,motor]).clone(),
        })
        return ret

    namespace['execute_delayed_synaptic_transmission'] = transmit_traced
    model.step = step_traced
    try:
        a, b = physical_copy(raw), physical_copy(raw)
        ids_a, ids_b = torch.tensor([previous]), torch.tensor([alternate])
        pulse_rows = []
        with torch.no_grad():
            for tick in range(8):
                traces.clear()
                if tick == 0:
                    a = advance_fly_input_event(model,a,ids_a,settle_ticks=0,writer_baseline_clock=clock)
                    b = advance_fly_input_event(model,b,ids_b,settle_ticks=0,writer_baseline_clock=clock)
                else:
                    def quiet(s, ids):
                        h, _, ring, ge, gi, adapt, x, u = model.step(s.h,ids,
                            spike_ring=s.ring,ge=s.ge,gi=s.gi,b=s.b,x=s.x,u=s.u,
                            sensory_drive=torch.zeros_like(s.h))
                        return FlyPhysicalState(h,ring,ge,gi,adapt,x,u,s.baseline)
                    a, b = quiet(a,ids_a), quiet(b,ids_b)
                ta, tb = traces
                delta = {name: tb[name]-ta[name] for name in ('arrival_e','arrival_i','ge','gi','v_pre','h_post','spike_sensory','pulse_sensory')}
                row = {'tick_after_input': tick, 'difference_norms': {k:float(v.norm()) for k,v in delta.items()},
                       'read_response': read_pair(model,ta['h_post'],tb['h_post'])}
                if tick == 0:
                    changed = delta['spike_sensory'].bool()
                    report['sensory_pulse_conversion'] = {
                        'changed_spike_decisions': int(changed.sum()),
                        'pulse_to_spike_contrast_norm_ratio': float(delta['pulse_sensory'].norm()/delta['spike_sensory'].norm().clamp_min(1e-30)),
                        'pre_input_resource_x_mean_on_changed_neurons': float(raw['x'][:,sensory][changed].mean()),
                        'pre_input_release_u_mean_on_changed_neurons': float(raw['u'][:,sensory][changed].mean()),
                        'scope': 'Same pre-input x/u and STP parameters; pulse attenuation is exact amplitude accounting, not semantic information loss.',
                    }
                if tick == 1:
                    report['first_arrival_filter_identity'] = {
                        'e_max_abs_residual': float((delta['ge']-ta['filter_e']*delta['arrival_e']).abs().max()),
                        'i_max_abs_residual': float((delta['gi']-ta['filter_i']*delta['arrival_i']).abs().max()),
                        'e_response_to_arrival_norm_ratio': float(delta['ge'].norm()/delta['arrival_e'].norm().clamp_min(1e-30)),
                        'i_response_to_arrival_norm_ratio': float(delta['gi'].norm()/delta['arrival_i'].norm().clamp_min(1e-30)),
                    }
                pulse_rows.append(row)
                print(f'Quiet response tick {tick}/7 measured',flush=True)
        report['quiet_pulse_response'] = pulse_rows
        # Restore hooks before the real sequence. All events share one saved
        # head; labels are consulted only after forward collection completes.
        namespace['execute_delayed_synaptic_transmission'] = transmit_original
        model.step = step_original
        state = physical_copy(raw)
        inputs = [previous]+[int(t) for t in train[cursor+1:cursor+32]]
        targets = torch.tensor(np.asarray(train[cursor+1:cursor+33],dtype=np.int64))
        hs = []
        with torch.no_grad():
            for i,token in enumerate(inputs):
                state = advance_fly_input_event(model,state,torch.tensor([token]),
                    settle_ticks=settle,writer_baseline_clock=clock)
                hs.append(state.h[:,motor].clone())
                if (i+1)%8 == 0:
                    print(f'Real continuing window {i+1}/32 measured',flush=True)
        h = torch.cat(hs)
        z = model.output_read(h).detach().requires_grad_(True)
        q = model.read_norm(z)
        logits = model.decoder(q)
        loss = F.cross_entropy(logits,targets,reduction='sum')
        derivative, = torch.autograd.grad(loss,z)
        report['motor_projection_gradient'] = gradient_parts(derivative.double(),h.double())
        with torch.no_grad():
            q_bar = q.mean(0,keepdim=True)
            common_logits = model.decoder(q_bar).expand_as(logits)
            report['same_head_real_window'] = {
                'events': 32, 'full_nll': float(loss/32),
                'common_read_background_nll': float(F.cross_entropy(common_logits,targets)),
                'conditional_gain_nll': float(F.cross_entropy(common_logits,targets)-loss/32),
                'scope': 'Fixed mature-head real continuation diagnosis; not primary active-learning performance.',
            }
            report['real_window_variation'] = {
                name: float((values-values.mean(0,keepdim=True)).square().sum()/values.square().sum().clamp_min(1e-30))
                for name,values in (('motor',h),('projected',z),('normalized',q))}
    finally:
        namespace['execute_delayed_synaptic_transmission'] = transmit_original
        model.step = step_original
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
