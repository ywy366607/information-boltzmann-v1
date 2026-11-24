"""Read-only causal-model diagnosis; no world calls or optimizer updates.

Small algebraic reproductions call the production posterior, not an invented
memory algorithm. The real-body audit re-encodes archived facts with ONE frozen
4752 body and excludes the target event and all later events from support.
This is a structural reconstruction, not a replay of historical training.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.causal_relation_program import (
    Port, Variable, mechanism, posterior, role_features)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def minimal_models():
    """Three falsifiable implications of the actual conditional residual fit."""
    dtype = torch.float64
    # A world with no action effect. All eight observations are exact.
    x = torch.tensor([[1., 0.]], dtype=dtype).repeat(8, 1)
    y = torch.ones(8, 1, dtype=dtype)
    raw_q = torch.tensor([[1., 0.], [1., 1.]], dtype=dtype)
    q = F.normalize(raw_q, dim=-1)
    d = posterior(x, y, q, .1)
    radial = dict(prediction=d['mean'].flatten().tolist(),
        true_action_difference=0.,
        invented_action_difference=float(d['mean'][1]-d['mean'][0]),
        condition_norm_fixed_prediction=posterior(x,y,raw_q,.1)['mean'].flatten().tolist())

    # Two mechanisms fit every historical observation exactly but disagree
    # on the action derivative. More samples on this same line cannot help.
    t = torch.tensor([-2., -1., 1., 2.], dtype=dtype)
    c, a = t, t
    confounding = dict(observation_max_difference=float((2*c-2*a).abs().max()),
        mechanism_1_action_slope=0., mechanism_2_action_slope=2.,
        observation_rank=int(torch.linalg.matrix_rank(torch.stack((c,a),-1))),
        intervention_predictions=[2., -2.], intervention=dict(condition=1.,action=-1.))
    # Positive control: cross the same condition with BOTH actions. The same
    # production update now learns the action direction, without a larger net.
    crossed_raw=torch.tensor([[1.,-1.],[1.,1.]],dtype=dtype).repeat(4,1)
    crossed_x=F.normalize(crossed_raw,dim=-1)
    crossed_y=crossed_raw[:,1:]
    crossed=posterior(crossed_x,crossed_y,crossed_x[:2],.1)['mean'].flatten()
    confounding['crossed_observation_rank']=int(torch.linalg.matrix_rank(crossed_x))
    confounding['crossed_action_predictions']=crossed.tolist()
    assert crossed[0]<-.97 and crossed[1]>.97

    # Even ZERO residual leaves a unit inverse-gamma prior in each output.
    zero = posterior(x, torch.zeros_like(y), q, .1)
    noise = dict(residual_mse=0., noise=zero['noise'].tolist(),
        predictive_scale2=zero['scale2'].flatten().tolist(),
        leverage=zero['leverage'].tolist(),
        information_64_coordinates=(32*torch.log1p(zero['leverage'])).tolist())

    # Constant correction alone improves total squared error yet reverses
    # which action best predicts the same factual outcome. Action difference
    # is EXACTLY unchanged, disproving that this margin identifies erasure.
    p = torch.tensor([[0.],[-1.]],dtype=dtype)
    target = torch.tensor([1.],dtype=dtype)
    correction = torch.tensor([2.5],dtype=dtype)
    err = (p-target).square().flatten()
    corrected = (p+correction-target).square().flatten()
    margin = dict(original_mse=err.tolist(), corrected_mse=corrected.tolist(),
        mean_mse_before=float(err.mean()), mean_mse_after=float(corrected.mean()),
        actual_action=0, margin_before=float(err[1]-err[0]),
        margin_after=float(corrected[1]-corrected[0]),
        action_difference_before=float(p[1]-p[0]),
        action_difference_after=float((p+correction)[1]-(p+correction)[0]))
    assert radial['invented_action_difference'] < -.28
    assert confounding['observation_max_difference'] == 0.
    assert abs(noise['noise'][0]-.2) < 1e-12
    assert margin['margin_before'] > 0 > margin['margin_after']
    assert margin['mean_mse_after'] < margin['mean_mse_before']
    return dict(radial_leak=radial, nonidentification=confounding,
                residual_prior_floor=noise, factual_margin_counterexample=margin)


def centered(x):
    return x-x.mean(0,keepdim=True)


def forecast_audit(d, target, selected):
    """Exact decompositions, with no changed parameters or fitted labels."""
    x, q, coefficient = d['x'].double(), d['query'].double(), d['coefficient'].double()
    base, full, target = d['base'].double(), d['mean'].double(), target.double().reshape(1,-1)
    correction = full-base
    width = x.shape[1]//2
    rc = (q[:,:width]@x[:,:width].T)@coefficient
    ra = (q[:,width:]@x[:,width:].T)@coefficient
    alternatives = [i for i in range(len(q)) if i != selected]
    mse = lambda v: (v-target).square().mean(-1)
    margin = lambda v: float(mse(v)[alternatives].mean()-mse(v)[selected])
    # Fix only the query condition norm; otherwise retain the exact action
    # coordinate, evidence, residuals, and posterior coefficient.
    cq = q[:,:width].mean(0,keepdim=True).expand(len(q),-1)
    fixed_condition = base+(cq@x[:,:width].T)@coefficient+ra
    predictions = dict(base=base, full=full, common_only=base+correction.mean(0),
        contrast_only=base+centered(correction), condition_only=base+rc,
        action_only=base+ra, fixed_query_condition=fixed_condition)
    variants = {k:dict(chosen_mse=float(mse(v)[selected]),
        all_mse=float(mse(v).mean()), margin=margin(v)) for k,v in predictions.items()}
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    keep = singular > singular.max()*1e-8 if len(singular) else singular.bool()
    dq = centered(q)
    contrast_power = float(dq.square().sum())
    projected = dq@vh[keep].T
    covered = float(projected.square().sum())/max(contrast_power,1e-30)
    action_x = x[:,width:]
    # Residual action excitation after conditioning on the source features.
    condition_rank = int(torch.linalg.matrix_rank(x[:,:width],tol=1e-8))
    residual_action = action_x-x[:,:width]@torch.linalg.pinv(x[:,:width],rtol=1e-8)@action_x
    # Construct two residual operators with indistinguishable historical fits
    # but opposite (+1/-1) current action contrasts in one output coordinate.
    candidate=alternatives[0]
    contrast=q[selected]-q[candidate]
    null=contrast-(contrast@vh[keep].T)@vh[keep]
    k=float(contrast@null)
    w=torch.zeros(full.shape[-1],dtype=torch.float64)
    w[0]=1
    initial=float(full[selected,0]-full[candidate,0])
    indistinguishable=[]
    for desired in (-1.,1.):
        delta=null[:,None]*w[None,:]*((desired-initial)/max(k,1e-30))
        actual_contrast=initial+float(contrast@delta[:,0])
        indistinguishable.append(dict(desired_contrast=desired,
            achieved_contrast=actual_contrast,historical_prediction_change=float((x@delta).abs().max()),
            operator_change_norm=float(delta.norm())))
    energy = lambda z:float(z.square().mean())
    actual_error = float(mse(full)[selected])
    expected_variance = d['scale2'][selected].double().mean()*d['df']/(d['df']-2)
    # Hold query condition block fixed: the remaining variation is explicitly
    # in the action role. This does not assert it is the true world response.
    return dict(variants=variants, support=len(x), condition_rank=condition_rank,
        conditional_action_excitation=float(residual_action.norm()/action_x.norm().clamp_min(1e-30)),
        query_action_contrast_span_coverage=covered,
        correction_common_energy=energy(correction.mean(0)),
        correction_contrast_energy=energy(centered(correction)),
        condition_contrast_energy=energy(centered(rc)),
        action_contrast_energy=energy(centered(ra)),
        decomposition_error=float((full-base-rc-ra).abs().max()),
        query_condition_norm=q[:,:width].norm(dim=-1).tolist(),
        leverage=d['leverage'].tolist(),
        noise_mean=float(d['noise'].mean()),
        evidence_residual_mse=float(d['y'].square().mean()),
        unit_noise_prior_fraction=float((torch.ones_like(d['noise']) /
            (d['noise']*(1+len(x)/2))).mean()),
        predicted_variance=float(expected_variance),
        observed_error=actual_error,
        variance_to_observed_error=float(expected_variance)/max(actual_error,1e-30),
        same_history_opposite_action_operators=indistinguishable,
        epistemic_fraction_of_unconditioned_query_energy=1-covered)


def summarize(rows):
    names=list(rows[0]['variants'])
    result=dict(n=len(rows), variants={})
    for name in names:
        result['variants'][name]={key:float(np.mean([r['variants'][name][key] for r in rows]))
                                  for key in ('chosen_mse','all_mse','margin')}
    for key in rows[0]:
        if isinstance(rows[0][key],(int,float)):
            result[key]=dict(mean=float(np.mean([r[key] for r in rows])),
                             median=float(np.median([r[key] for r in rows])))
    return result


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    before=digest(args.checkpoint)
    with gzip.open(args.checkpoint,'rb') as f:
        saved=torch.load(f,map_location='cpu',weights_only=False)
    cfg,e=saved['config'],saved['engine']
    manifest=cfg['causal_archive']
    original_archive_sha=digest(manifest['snapshot'])
    d=e['_program_forecasts']['path']['mother']
    target=F.linear(e['_mother_boundary_target'],e['_mother_pre_encoder'])
    exact=forecast_audit(d,target,e['_pending_record']['choice'])
    frozen_extra=dict(cycle=saved['cycle'],
        original_hypotheses_error=float((e['_program_forecasts']['hypotheses']-
            target[None,None]).square().mean()),
        target_coordinate_energy=float(target.square().mean()),
        support_singular_values=torch.linalg.svdvals(d['x']).tolist())
    del e,saved,d
    gc.collect()
    print('frozen saved forecast decomposed',flush=True)

    from scripts.ib_local.run_causal_water_reply import build
    from scripts.ib_local.causal_archive import attach_archive
    from scripts.ib_local.shared_unknown_bindings import _evidence
    engine,_,_,source=build(args.checkpoint,device='cpu',
        table_storage=args.output/'table_work',allow_external_archive=True)
    model=engine.model
    model.eval()
    budgets=manifest['budgets']
    archive=attach_archive(engine,args.output/'archive_work',manifest=manifest,
        episode_capacity=budgets['episodes'], relation_capacity=budgets['relations'],
        max_bytes=budgets['max_bytes'])
    metrics=[json.loads(line) for line in args.metrics.read_text(encoding='utf-8').splitlines()]
    template=engine._program_question_state
    records=[]
    for row in metrics:
        question=row['shared_unknown']
        if 'observed_episode' not in question['revision']:
            continue
        eid=question['revision']['observed_episode']
        actual=_evidence(archive,eid)[0]
        # Present support rows are immutable observed facts, but their list
        # is not a historical snapshot. Exclude all facts unavailable then.
        ids=[v[0] for v in archive.db.execute('SELECT episode_id FROM unknown_evidence '
             'WHERE unknown_id=? ORDER BY ordinal',(question['identity'],)) if v[0]<eid]
        evidence=tuple(e for old in ids for e in _evidence(archive,old))
        mother_evidence=[v for v in evidence if v.kind==Port.MOTHER]
        if len(mother_evidence)<2:
            continue
        state=replace(template,binding=Variable(question['identity'],Port(question['role'])),
            constraints=evidence,extra=())
        # These SAME remembered action differences are production's finite
        # inverse-planning candidates. They are not fabricated world outcomes.
        alternatives=[actual.source+v.action-v.source for v in mother_evidence]
        actions=torch.cat([actual.action,actual.source]+alternatives)
        s=actual.source.expand_as(actions)
        dist=mechanism(model,state,s,actions,ridge=engine.relation_ridge)
        result=forecast_audit(dist,model.encoder(actual.outcome),0)
        result.update(cycle=row['cycle'],role=question['role'],episode=eid,evidence_ids=ids,
            reconstruction='fixed current parameters; current support subset strictly older than target; no historic momentum')
        records.append(result)
    print('real episodes reconstructed',len(records),flush=True)
    # Read-only intervention on the complete planner, not on a component's
    # loss: keep its body, actions, mean mechanism, evidence and reward fixed.
    from unittest.mock import patch
    from scripts.ib_local import causal_relation_program as program
    from scripts.ib_local.navigation_gain_units import wood_choice_logits
    real_posterior=program.posterior
    actions=engine._internal_action
    sources=actions[:1].expand_as(actions)  # first option is real silence=h
    momentum=engine._momentum_decision_read[None].expand(len(actions),-1)
    planner=[]
    for scale in (1.,.02,0.):
        def scaled_posterior(*pos,**kw):
            value=real_posterior(*pos,**kw)
            value['noise']=value['noise']*scale
            value['scale2']=value['scale2']*scale
            return value
        with patch.object(program,'posterior',scaled_posterior):
            result=program.imagine(model,template,sources,actions,ridge=engine.relation_ridge,
                reward_valid=engine.reward_label_valid(wood=True),momentum=momentum)
        probs=wood_choice_logits(engine,result['expected_after_value']).softmax(0)
        planner.append(dict(noise_multiplier=scale,
            expected_after_value=result['expected_after_value'].tolist(),
            future_delta=result['future_delta'].tolist(),
            wood_choice_probability=probs.tolist(),
            tv_to_uniform=float((probs-1/len(probs)).abs().sum()/2),
            consequence_mean=result['path']['mother']['mean'].tolist()))
    # Forecast mean equality establishes what this intervention held fixed.
    planner_mean_error=max(float((torch.tensor(v['consequence_mean'])-
        torch.tensor(planner[0]['consequence_mean'])).abs().max()) for v in planner)
    for value in planner:
        del value['consequence_mean']
    output=dict(minimal=minimal_models(),frozen_last_forecast=exact,
        frozen_last_extra=frozen_extra,
        retrospective_current_body=dict(summary=summarize(records),records=records),
        full_planner_noise_intervention=dict(cases=planner,
            unchanged_mother_mean_max_error=planner_mean_error,
            meaning='current frozen body last option set; simultaneous local-posterior noise intervention throughout U/transport; wood contribution before habit/other calibrated offsets; not a trained repair'),
        source=source,contract='0 world calls, 0 optimizer updates, frozen current-body structural analysis',
        checkpoint_before=before,checkpoint_after=digest(args.checkpoint),
        archive_before=original_archive_sha,archive_after=digest(manifest['snapshot']))
    archive.close()
    (args.output/'report.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    # Research-private fast-table files are scratch storage, not a new body.
    import shutil
    for name in ('rows','count','tag','err','epoch','last_write'):
        value=getattr(engine.table,name,None)
        if isinstance(value,np.memmap):
            value._mmap.close()
    for name in ('table_work','archive_work'):
        path=(args.output/name).resolve()
        assert path.parent==args.output.resolve()
        shutil.rmtree(path)
    assert output['checkpoint_before']==output['checkpoint_after']
    assert output['archive_before']==output['archive_after']
    print(json.dumps(dict(last=exact,cohort=output['retrospective_current_body']['summary']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('results/cgb_self_action_observation_20261114_recovered/body/last.pt.gz'))
    parser.add_argument('--metrics',type=Path,default=Path('results/cgb_self_action_observation_20261114_recovered/body/metrics.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('results/cgb_relation_cause_20261114'))
    run(parser.parse_args())
