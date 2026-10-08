"""Inspect an existing jointly trained medium on real held-out local contexts.

No optimization or new capability experiment: parameter heterogeneity, operator
activity/ledgers and checkpoint-conditioned inference interventions only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.evaluation import field_energy_statistics
from information_boltzmann.runtime.training import clone_belief, quiet_training_chunk
from scripts.ib.train_plastic_conductance import unpack_belief


def spatial_summary(value, dimensions):
    value = value.double()
    mean = value.mean(dimensions, keepdim=True)
    residual = value-mean
    total = value.square().sum()
    variation = residual.square().mean(dimensions).sqrt()
    scale = value.square().mean(dimensions).sqrt().clamp_min(1e-30)
    return {'minimum':float(value.min()),'maximum':float(value.max()),
            'rms':float(value.square().mean().sqrt()),
            'spatial_rms':float(residual.square().mean().sqrt()),
            'spatial_fraction':float(residual.square().sum()/total) if total else 0.,
            'mean_spatial_relative_std':float((variation/scale).mean())}


def rank_summary(field):
    matrix = field[0].double().flatten(0,2)
    matrix -= matrix.mean(0,keepdim=True)
    power = torch.linalg.svdvals(matrix).square()
    fraction = power/power.sum().clamp_min(1e-30)
    return {'spatial_power_participation_rank':float(1/fraction.square().sum()),
            'top1_power_fraction':float(fraction[0]),
            'top8_power_fraction':float(fraction[:8].sum())}


def relative_change(before, after):
    numerator = sum((x-y).double().square().sum() for x,y in zip(
        (before.field,*before.flux),(after.field,*after.flux)))
    denominator = sum(x.double().square().sum() for x in (before.field,*before.flux))
    return float((numerator/denominator.clamp_min(1e-30)).sqrt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace-tokens',type=int,default=32)
    parser.add_argument('--write-exchange',choices=('global','contact_mode'))
    args = parser.parse_args()
    if args.trace_tokens < 1:
        parser.error('Positive trace length required')
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config = saved['config']
    checkpoint_exchange = config['constructor'].get('write_exchange','global')
    config['constructor'].setdefault('port_scope', 'global')
    config['constructor'].setdefault('activity_adaptation', False)
    config['constructor'].setdefault('pre_decoder_norm', False)
    config['constructor']['write_exchange'] = args.write_exchange or checkpoint_exchange
    del saved['optimizer']
    torch.manual_seed(config['seed'])
    model = PlasticMediumPorts3D(**config['constructor']).eval()
    initial_probes = model.readout.probe_coords.detach().clone()
    model.load_state_dict(saved['model'])
    base = unpack_belief(saved['belief'],'cpu')
    step = saved['step']
    del saved
    data = np.load(Path(config['data']['directory'])/'validation.npy',mmap_mode='r')
    duration, substeps = config['event_duration'],config['substeps']
    medium = model.medium
    with torch.no_grad():
        table = F.normalize(model.source.embedding.weight,dim=-1)
        prepared = medium.prepare_evolution()
        response = prepared.response
        summary = {
            'field':field_energy_statistics(base.medium.field),
            'field_rank':rank_summary(base.medium.field),
            'material':spatial_summary(prepared.material,(0,1,2)),
            'capacitance':spatial_summary(response.capacitance,(0,1,2)),
            'leak':spatial_summary(response.leak,(0,1,2)),
            'maximum_conductance':spatial_summary(response.maximum,(0,1,2)),
            'closing_rate':spatial_summary(response.closing,(0,1,2)),
            'conduction_offset':spatial_summary(base.medium.conduction,(1,2,3)),
            'receptors':spatial_summary(base.medium.receptors,(1,2,3)),
            'effective_speed':spatial_summary(medium.edge_log_speeds(
                base.medium,prepared.material,prepared.baseline_log_speed).exp(),(1,2,3)),
            'membrane_time':spatial_summary(response.capacitance[None]/(
                response.leak[None]+(response.maximum[None]*base.medium.receptors).sum(-2)),(1,2,3))}
        summary['energy_in_field'] = float(0.5*base.medium.field.square().sum(-1).mean())
        summary['energy_in_edge_responses'] = float(sum(0.5*x.square().sum(-1).mean() for x in base.medium.flux))
        metric = prepared.plasticity[2]
        summary['content_metric'] = spatial_summary(metric,(0,1,2))
        drift = (model.readout.probe_coords-initial_probes+0.5).remainder(1)-0.5
        summary['read_probe_drift_mean'] = float(drift.norm(dim=-1).mean())
        summary['read_head_scales'] = model.readout.head_log_scale.exp().flatten().tolist()
        summary['read_probe_scales'] = model.readout.probe_log_scale.exp().flatten().tolist()
        if model.port_scope == 'compact':
            summary['local_ports'] = {key: value.tolist() for key, value in
                                      model.port_diagnostics(base.medium.field).items()}

        traces = []
        interventions = []
        rank_samples = []
        paired_boundary = []
        requested_exchange = model.write_agent.exchange
        for start in config['validation']['site_starts']:
            token = torch.tensor([int(data[start])])
            row = {'site':start, 'state':'identical saved mature belief'}
            for exchange in ('global','contact_mode'):
                model.write_agent.exchange = exchange
                _, info = model.assimilate(base,token,token_features=table,
                                           diagnostics=True,training_terms=False)
                row[exchange] = {key:float(info[key]) for key in (
                    'old_field_direct_energy_retention','incident_energy',
                    'write_balance_residual','write_angle_abs_mean')}
            paired_boundary.append(row)
        model.write_agent.exchange = requested_exchange
        for start in config['validation']['site_starts']:
            warm, score = config['validation']['warm_in_tokens'],config['validation']['score_tokens']
            ids = torch.from_numpy(np.array(data[start:start+warm+score+1],dtype=np.int64))[None]
            current = clone_belief(base)
            for index in range(warm):
                current,_ = model.assimilate(current,ids[:,index],token_features=table,
                                             diagnostics=False,training_terms=False)
                current,_ = model.advance(current,duration,substeps=substeps,
                                          prepared=prepared,diagnostics=False)
            warm_state = clone_belief(current)
            for index in range(warm,warm+min(score,args.trace_tokens)):
                posterior = []
                hook = model.write_agent.action_posterior.register_forward_hook(
                    lambda module,inputs,output: posterior.append(output.detach()))
                written, wi = model.assimilate(current,ids[:,index],token_features=table,
                                               diagnostics=True,training_terms=False)
                hook.remove()
                action_mean = posterior[0].chunk(2,-1)[0]
                write_angle = torch.atan(F.softplus(action_mean)*wi['innovation_norm'])
                retention = write_angle.cos()[:,None,None,None,:]
                old_field_energy_retention = (current.medium.field.square()*retention.square()).sum() / current.medium.field.square().sum().clamp_min(1e-30)
                amplitude_retention = float(retention.mean())
                if model.write_agent.exchange == 'contact_mode':
                    old_field_energy_retention = wi['old_field_direct_energy_retention']
                    amplitude_retention = float(old_field_energy_retention.sqrt())
                first = medium.adapt_conduction(written.medium,0.5*duration,
                                                prepared.material,prepared.plasticity)
                streamed = medium.transport(first,duration,prepared.material,prepared.baseline_log_speed)
                scattered = medium.collide(streamed,duration,prepared.material)
                released, ri = medium.respond(scattered,duration,coefficients=response,diagnostics=True)
                last = medium.adapt_conduction(released,0.5*duration,
                                               prepared.material,prepared.plasticity)
                last = replace(last,elapsed=current.medium.elapsed+duration)
                current = PlasticBelief(last,written.precision)
                feature, read = model.read(current,decode=False,diagnostics=True)
                logits = model.decode(feature)
                angles = (medium.collision_rate(medium._features(
                    medium._pack(streamed),prepared.material,streamed.receptors))*duration).abs()
                speeds = medium.edge_log_speeds(first,prepared.material,prepared.baseline_log_speed).exp()
                transport_angles = speeds*torch.tensor([8,8,4])*math.sqrt(2)*duration
                attention = read['read_attention_weights'][0]
                entropy = -(attention*attention.clamp_min(1e-30).log()).sum(-1)
                head = attention.mean(1)
                pair_tv = [float((head[i]-head[j]).abs().sum()/2)
                           for i in range(head.shape[0]) for j in range(i)]
                # Remove one operator only within this same current token,
                # retaining the identical written state and all other operators.
                logit_changes = {}
                for name,flags in [('transport',{'transport':False}),('collision',{'collision':False})]:
                    alt,_ = medium.advance(written.medium,duration,substeps=substeps,
                                           prepared=prepared,diagnostics=False,**flags)
                    alt_logits,_ = model.read(PlasticBelief(alt,written.precision))
                    logit_changes[name] = float((alt_logits-logits).square().mean().sqrt())
                traces.append({
                    'site':start,'token_offset':index-warm,
                    'write_angle_mean':float(wi['write_angle_abs_mean']),
                    'write_transmitted_fraction':float(wi['accepted_fraction']),
                    'old_field_direct_amplitude_retention':amplitude_retention,
                    'old_field_direct_energy_retention':float(old_field_energy_retention),
                    'write_balance_residual':float(wi['write_balance_residual']),
                    'write_chart_entropy':float(wi['port_chart_entropy']),
                    'transport_angle_mean':float(transport_angles.mean()),
                    'transport_relative_change':relative_change(first,streamed),
                    'transport_field_relative_change':float((streamed.field-first.field).norm()/first.field.norm().clamp_min(1e-30)),
                    'transport_energy_residual':float((medium.energy(streamed)-medium.energy(first)).abs().max()),
                    'collision_angle_mean':float(angles.mean()),
                    'collision_relative_change':relative_change(streamed,scattered),
                    'collision_field_relative_change':float((scattered.field-streamed.field).norm()/streamed.field.norm().clamp_min(1e-30)),
                    'collision_energy_residual':float((medium.energy(scattered)-medium.energy(streamed)).abs().max()),
                    'response_relative_change':relative_change(scattered,released),
                    'response_source_work':float(ri['response_source_work'][0]),
                    'response_joule_heat':float(ri['response_joule_heat'][0]),
                    'response_balance_residual':float((medium.energy(released)-medium.energy(scattered)
                        -ri['response_source_work']+ri['response_joule_heat']).abs().max()),
                    'field_spatial_share':field_energy_statistics(last.field)['spatial_share'],
                    'read_entropy_per_head':entropy.mean(1).tolist(),
                    'read_head_pairwise_tv_mean':sum(pair_tv)/len(pair_tv),
                    'read_max_site_weight':float(attention.max()),
                    'transport_logit_rms_change':logit_changes['transport'],
                    'collision_logit_rms_change':logit_changes['collision']})
                if model.port_scope == 'compact':
                    traces[-1]['local_ports'] = {key: value.tolist() for key, value in
                        model.port_diagnostics(last.field).items()
                        if key not in ('write_port_weights', 'read_port_weights')}
            rank_samples.append({'site':start,**rank_summary(current.medium.field)})
            site = {'site':start}
            for name,flags in [('full',{}),('no_transport',{'transport':False}),
                               ('no_collision',{'collision':False}),('no_response',{'bath':False})]:
                trial = clone_belief(warm_state)
                features = []
                for index in range(warm,warm+score):
                    trial,_ = model.assimilate(trial,ids[:,index],token_features=table,
                                               diagnostics=False,training_terms=False)
                    state,_ = medium.advance(trial.medium,duration,substeps=substeps,
                                              prepared=prepared,diagnostics=False,**flags)
                    trial = PlasticBelief(state,trial.precision)
                    feature,_ = model.read(trial,decode=False)
                    features.append(feature)
                loss = F.cross_entropy(model.decode(torch.stack(features,1)).flatten(0,1),
                                       ids[:,warm+1:warm+score+1].flatten())
                site[name] = float(loss)
            for name in ('no_transport','no_collision','no_response'):
                site[name+'_minus_full'] = site[name]-site['full']
            interventions.append(site)
            print(json.dumps({'site_completed':site}),flush=True)

    # Direct CE versus auxiliary gradients: inspect active mechanism learning,
    # with the same mature belief, without stepping an optimizer.
    inputs = torch.from_numpy(np.array(data[8192:8201],dtype=np.int64))[None]
    loss,_,nll = quiet_training_chunk(model,inputs[:,:-1],inputs[:,1:],clone_belief(base),
                                     event_duration=duration,substeps=substeps)
    names,parameters = zip(*model.named_parameters())
    ce_grad = torch.autograd.grad(nll,parameters,retain_graph=True,allow_unused=True)
    aux_grad = torch.autograd.grad(loss-nll,parameters,allow_unused=True)
    gradients = {}
    groups = {'material':'medium.material.', 'transport':'medium.log_speed.',
              'pathway_metric':'medium.conduction_plasticity.metric_material.',
              'collision':'medium.collision_rate.', 'response':'medium.conductance_response.',
              'write':'write_agent.', 'read':'readout.'}
    for group,prefix in groups.items():
        indices = [i for i,name in enumerate(names) if name.startswith(prefix)]
        def norm(values):
            return math.sqrt(sum(float(values[i].double().square().sum())
                                 for i in indices if values[i] is not None))
        gradients[group] = {'next_token_ce_norm':norm(ce_grad),'write_auxiliary_norm':norm(aux_grad)}
    numeric = [k for k,v in traces[0].items() if isinstance(v,(float,int)) and k not in ('site','token_offset')]
    aggregate = {k:{'mean':sum(r[k] for r in traces)/len(traces),'min':min(r[k] for r in traces),
                    'max':max(r[k] for r in traces)} for k in numeric}
    aggregate['read_entropy_per_head'] = np.mean([r['read_entropy_per_head'] for r in traces],axis=0).tolist()
    report = {'checkpoint':str(args.checkpoint),'step':step,
              'scope':'trained-checkpoint mechanism/ledger audit; no new optimization, no convergence claim',
              'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'checkpoint_write_exchange':checkpoint_exchange,
              'audited_write_exchange':model.write_agent.exchange,
              'law_changed_without_training':checkpoint_exchange != model.write_agent.exchange,
              'event_duration':duration,'substeps':substeps,
              'validation_policy':'full mature state per site; warm256, score128; no reset',
              'state_and_coefficients':summary,'traced_real_tokens':len(traces),
              'trace_aggregate':aggregate,'spatial_rank_samples':rank_samples,
              'checkpoint_interventions':interventions,'direct_gradients':gradients,
              'paired_boundary_same_state':paired_boundary,
              'interpretation_notes':['State change shows operator activity; NLL interventions describe reliance at this checkpoint, not superiority of a separately trained architecture.',
                                      'Direct old-field retention freezes the boundary action; it is not the full state-dependent Jacobian or a lifetime bound for stored edge responses.',
                                      'Contact-mode amplitude retention is a field-norm ratio; global amplitude retention is the mean channel multiplier.',
                                      'mean_spatial_relative_std is spatial standard deviation divided by spatial RMS, averaged across remaining components.',
                                      'no_response removes receptor updates, reversal work and Joule heat together; it is not an isolated dissipation intervention.',
                                      'Trace reconstructs the one-substep trained split; additional substeps would need an explicit refined trace.']}
    if substeps != 1:
        raise ValueError('This stage trace is only defined for the one-substep launch')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
