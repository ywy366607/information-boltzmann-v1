"""Report the two preregistered requirements on complete real-world traces."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.closed_loop_information import mean_interval


def read(path):
    if (path/'INVALID.json').exists():
        raise ValueError(f'invalid run: {path}')
    return [json.loads(line) for line in (path/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]


def block_effect(values, block=16):
    """Paired turn-block differences; no claim of independent seed replications."""
    values=np.asarray(values,dtype=float)
    n=len(values)//block
    if n<2:
        return dict(mean=float(values.mean()),scope='too few turn blocks')
    result=mean_interval(values[:n*block].reshape(n,block).mean(1))
    result['scope']='16-turn blocks of one continuing individual, conditional on its exposure'
    result['turns_omitted_at_tail']=len(values)-n*block
    return result


def predictive(rows):
    late=rows[len(rows)//2:]
    out={}
    for coordinate,name in [(0,'reply_nll4'),(1,'reply_nll32'),(3,'gain')]:
        def errors(key): return np.array([r[key][coordinate] for r in late])
        actual=errors('episodic_real_error')
        out[name]=dict(real_mse=float(actual.mean()),
            shuffled_mse=float(errors('episodic_shuffled_error').mean()),
            parametric_mse=float(errors('episodic_plain_error').mean()),
            action_base_mse=float(errors('wood_base_error').mean()),
            shuffle_minus_real=block_effect(errors('episodic_shuffled_error')-actual),
            parametric_minus_real=block_effect(errors('episodic_plain_error')-actual),
            action_base_minus_real=block_effect(errors('wood_base_error')-actual),
            used_real_mse=float(errors('episodic_used_real_error').mean()),
            used_shuffled_mse=float(errors('episodic_used_shuffled_error').mean()),
            used_shuffle_minus_real=block_effect(errors('episodic_used_shuffled_error')-errors('episodic_used_real_error')))
    return out


def read_ages(source, rows, final_checkpoint):
    """Reconstruct actual selected ages, not the oldest available bank entry."""
    with gzip.open(source,'rb') as f:
        state=torch.load(f,map_location='cpu',weights_only=False)
    clock=int(state['model']['memory.clock'])
    writes=state['model']['memory.last_write'].numpy().copy()
    del state
    weighted_ages=[];over512=[];maximum=0;channel_ages=[];channel_over=[]
    for row in rows:
        slots=np.asarray(row['episodic_read_slots'],dtype=int)
        weights=np.asarray(row['episodic_read_weights'],dtype=float)
        ages=clock-writes[slots]
        live=weights>0
        if live.any(): maximum=max(maximum,int(ages[live].max()))
        weighted_ages.append(float(weights@ages))
        over512.append(float(weights[ages>512].sum()))
        if 'causal_channel_read_weights' in row:
            w=np.array(row['causal_channel_read_weights'])
            channel_ages.append((w*ages[:,None]).sum(0).tolist())
            channel_over.append(w[ages>512].sum(0).tolist())
        clock+=row['external_bytes']+row['self_bytes']
        writes[row['episode_slot']]=clock
    with gzip.open(final_checkpoint,'rb') as f:
        final=torch.load(f,map_location='cpu',weights_only=False)
    exact=(clock==int(final['model']['memory.clock']) and
           np.array_equal(writes,final['model']['memory.last_write'].numpy()))
    del final
    assert exact,'read-age reconstruction does not match actual continued state'
    return dict(mean_weighted_age_bytes=float(np.mean(weighted_ages)),
        mean_read_mass_older_than_512=float(np.mean(over512)),maximum_selected_age_bytes=maximum,
        channel_mean_weighted_age=np.mean(channel_ages,axis=0).tolist() if channel_ages else None,
        channel_mass_older_than_512=np.mean(channel_over,axis=0).tolist() if channel_over else None,
        final_clock_and_writes_reconstruction_exact=exact)


def analyze(root):
    names=('train_real_causal','train_blocked_causal')
    rows={name:read(root/name) for name in names}
    configs={name:json.loads((root/name/'config.json').read_text(encoding='utf-8')) for name in names}
    for key in ('source','seed','cycles','validation_offset','validation_bytes','validation_sha256','sources',
                'variant','observer_mode','wood_gain_mode'):
        if key not in configs[names[0]] and key not in configs[names[1]]:continue
        assert configs[names[0]][key]==configs[names[1]][key],key
    assert all(len(rows[n])==configs[n]['cycles'] for n in names)
    real,blocked=(rows[n] for n in names)
    curve={n:[dict(cycle=r['new_cycle'],nll=r['validation_nll']) for r in rows[n] if 'validation_nll' in r] for n in names}
    differences=[b['nll']-a['nll'] for a,b in zip(curve[names[0]],curve[names[1]])]
    def first(key):
        return next((i+1 for i,(a,b) in enumerate(zip(real,blocked)) if key(a)!=key(b)),None)
    summaries={n:json.loads((root/n/'summary.json').read_text(encoding='utf-8')) for n in names}
    prediction=predictive(real)
    direct=configs[names[0]].get('wood_gain_mode')=='direct_occurrence'
    result=dict(source=configs[names[0]]['source'],
        config_match=True,summary=summaries,validation_curve=curve,
        blocked_minus_real_checkpoint_nll=differences,
        mean_validation_improvement=float(np.mean(differences)),final_validation_improvement=differences[-1],
        prediction_real_path=prediction,prediction_blocked_path=predictive(blocked),
        chain=dict(first_utterance_difference=first(lambda x:x['utterance_hex']),
            first_action_difference=first(lambda x:x['choice']),
            first_world_position_difference=first(lambda x:x['world']['start']),
            utterances_differ=sum(a['utterance_hex']!=b['utterance_hex'] for a,b in zip(real,blocked)),
            world_positions_differ=sum(a['world']['start']!=b['world']['start'] for a,b in zip(real,blocked)),
            max_predictive_policy_tv=max(r['episodic_policy_tv'] for r in real),
            late_gain_confidence_mean=float(np.mean([r['wood_confidence_used'][3] for r in real[len(real)//2:]])),
            max_read_age=max(r['episodic_max_bank_age'] for r in real)),
        acceptance=dict(prediction_increment=prediction['gain']['real_mse']<prediction['gain']['shuffled_mse']
            and (not direct or prediction['gain']['used_real_mse']<prediction['gain']['used_shuffled_mse']),
            whole_loop_gain=float(np.mean(differences))>0 and differences[-1]>=0,
            source_checkpoints_unchanged=all(s['source_checkpoint_unchanged'] for s in summaries.values())),
        units='NLL nats/byte; gain forecast squared errors use (10*nats/byte)^2; near NLL forecasts use (nats/byte/3)^2',
        scope='Single seed, fixed real-language continuing-world budget; not convergence or cross-world proof')
    result['selected_read_ages']=read_ages(Path(result['source']['checkpoint']),real,root/names[0]/'last.pt.gz')
    if direct:
        late=real[len(real)//2:]
        result['direct_calibration']=dict(
            mean_alpha=np.mean([r['direct_gain_alpha'] for r in late],axis=0).tolist(),
            qualified_counts=real[-1]['direct_gain_counts_next'],
            positive_delivery_turns=sum(max(abs(v) for v in r['direct_gain_offset'])>0 for r in real),
            maximum_centered_logit_increment=max(max(abs(v) for v in r['direct_gain_offset']) for r in real),
            primary='both raw and actually calibrated real predictions beat same-turn shuffled evidence',
            added_neural_parameters=0,added_moment_bytes=96)
    result['acceptance']['goal_requirements_supported']=all(result['acceptance'].values()) and result['chain']['world_positions_differ']>0
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=analyze(args.input)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['acceptance']),flush=True)


if __name__=='__main__':main()
