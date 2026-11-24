"""Summarize paired real-event impulse/factorial responses without Torch."""
import argparse
import json
import hashlib
import sqlite3
from functools import lru_cache
from pathlib import Path
import numpy as np


def external(report):
    return [r for r in report['events'] if r['autonomous_event']['kind']=='external']


def relative(delta, reference):
    return (np.linalg.norm(delta.reshape(32,-1),axis=1)/np.maximum(
        np.linalg.norm(reference.reshape(32,-1),axis=1),1e-30)).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    available={d.name:json.loads((d/'report.json').read_text(encoding='utf-8'))
        for d in a.work.iterdir() if d.is_dir() and not d.name.startswith(('invalid_','abandoned_')) and (d/'report.json').exists()}
    result=dict(schema=1, source_cycle=1791, completed_arms=list(available),
        comparisons={},nonadditivity={},
        scope='paired transient interventions, not significance, trained superiority, chaos or homological memory proof')
    aliases={}
    certificates={}
    for directory in a.work.iterdir():
        if directory.is_dir() and (directory/'certificate.json').exists():
            cert=json.loads((directory/'certificate.json').read_text())
            aliases[directory.name]=cert['equivalent_to']
            certificates[directory.name]=cert
            available[directory.name]=copy_report=json.loads(json.dumps(available[cert['equivalent_to']]))
            copy_report['seconds']=0;copy_report['optimizer_updates']=0
    result['derived_equivalent_arms']=certificates
    def workpath(name):return a.work/aliases.get(name,name)
    @lru_cache(maxsize=144)
    def state(name,index):
        with np.load(workpath(name)/f'state_{index:02}.npz') as z:
            return {key:z[key].astype(np.float64) for key in ('S','P','parameters')}
    from scripts.audit_higher_order_dynamics import archive_topology
    result['final_archive_topology']={}
    database_hashes={}
    for name in available:
        print('archive '+name,flush=True)
        files=list((workpath(name)/'causal').glob('working_*.sqlite'))
        if len(files)!=1:raise RuntimeError('private final archive is not unique')
        result['final_archive_topology'][name]=archive_topology(files[0])
        db=sqlite3.connect(files[0].resolve().as_uri()+'?mode=ro',uri=True)
        tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        h=hashlib.sha256()
        for table in tables:
            h.update(table.encode())
            for row in sorted(db.execute(f'SELECT * FROM "{table}"').fetchall(),key=repr):
                h.update(repr(row).encode())
        database_hashes[name]=h.hexdigest()
        db.close()
    result['final_archive_all_tables_hash']=database_hashes
    for name,report in available.items():
        condition,mask=name.split('_')
        if not mask.isdigit():continue
        mask=int(mask)
        if not mask or condition+'_0' not in available:continue
        base=available[condition+'_0']; rows,original=external(report),external(base)
        sequence=[]
        for i,(row,old) in enumerate(zip(rows,original),1):
            record=dict(external=i,response_nll_delta=row['response_nll']-old['response_nll'])
            if condition!='gdn':
                prob=np.asarray(row['choice_probability']);ref=np.asarray(old['choice_probability'])
                record.update(action_tv=float(abs(prob-ref).sum()/2),
                    action_changed=row['choice']!=old['choice'],utterance_changed=row['utterance_hex']!=old['utterance_hex'],
                    world_start_changed=row['world']['start']!=old['world']['start'],
                    actual_observation_changed=(report['observations'][i-1]['response']!=base['observations'][i-1]['response']),
                    progress_delta=row['metal_g']-old['metal_g'],
                    relation_history_changed=report['checkpoints'][i]['archive']['logical_hash']!=base['checkpoints'][i]['archive']['logical_hash'])
            snapshot,before=state(name,i),state(f'{condition}_0',i)
            record['S_relative_by_layer']=relative(snapshot['S']-before['S'],before['S'])
            record['parameter_sample_delta_norm']=float(np.linalg.norm(snapshot['parameters']-before['parameters']))
            if condition!='gdn':record['P_relative_by_layer']=relative(snapshot['P']-before['P'],before['P'])
            sequence.append(record)
        result['comparisons'][name]=dict(pulse=report['pulse'],sequence=sequence,
            internal_events=sum(r['autonomous_event']['kind']=='internal' for r in report['events']),
            release_events=sum(r['autonomous_event']['kind']=='release' for r in report['events']),
            optimizer_updates=report['optimizer_updates'],seconds=report['seconds'],peak_cuda_mib=report['peak_cuda_mib'])
    for condition in ('closed','yoked','gdn'):
        if not all(f'{condition}_{i}' in available for i in range(4)):continue
        rows=[external(available[f'{condition}_{i}']) for i in range(4)]
        sequence=[]
        for t in range(1,len(rows[0])+1):
            record=dict(external=t,nll_mixed=float(rows[3][t-1]['response_nll']-rows[1][t-1]['response_nll']-rows[2][t-1]['response_nll']+rows[0][t-1]['response_nll']))
            if condition!='gdn':
                probs=[np.asarray(r[t-1]['choice_probability']) for r in rows]
                record['action_probability_mixed']=(probs[3]-probs[1]-probs[2]+probs[0]).tolist()
                record['progress_mixed']=rows[3][t-1]['metal_g']-rows[1][t-1]['metal_g']-rows[2][t-1]['metal_g']+rows[0][t-1]['metal_g']
            data=[state(f'{condition}_{i}',t) for i in range(4)]
            for key in ('S','P','parameters'):
                delta=data[3][key]-data[1][key]-data[2][key]+data[0][key]
                record[key+'_mixed_norm']=float(np.linalg.norm(delta))
                if key=='S':record['S_mixed_relative_by_layer']=relative(delta,data[0][key])
            sequence.append(record)
        result['nonadditivity'][condition]=sequence
    if 'closed' in result['nonadditivity'] and 'yoked' in result['nonadditivity']:
        result['world_feedback_interaction']=[dict(external=c['external'],
            action_probability_mixed_difference=(np.asarray(c['action_probability_mixed'])-y['action_probability_mixed']).tolist(),
            progress_mixed_difference=c['progress_mixed']-y['progress_mixed'],
            nll_mixed_difference=c['nll_mixed']-y['nll_mixed'])
            for c,y in zip(result['nonadditivity']['closed'],result['nonadditivity']['yoked'])]
    if all(name in available for name in ('closed_fire2','yoked_fire2')):
        full,yoke=available['closed_fire2'],available['yoked_fire2']
        result['first_action_world_feedback']=[]
        for t,(c,y) in enumerate(zip(external(full),external(yoke)),1):
            record=dict(external=t,closed_choice=c['choice'],yoked_choice=y['choice'],
                policy_tv=float(abs(np.asarray(c['choice_probability'])-y['choice_probability']).sum()/2),
                closed_start=c['world']['start'],yoked_start=y['world']['start'],
                closed_progress=c['metal_g'],yoked_progress=y['metal_g'],
                conditional_next_boundary_error=dict(closed=c.get('internal_rollout',{}).get('next_boundary_prequential_mse'),
                    yoked=y.get('internal_rollout',{}).get('next_boundary_prequential_mse')),
                same_observation=full['observations'][t-1]['response']==yoke['observations'][t-1]['response'],
                closed_unknown=c.get('shared_unknown'),yoked_unknown=y.get('shared_unknown'),
                closed_revision=c.get('joint_relation_update'),yoked_revision=y.get('joint_relation_update'))
            data=[state(name,t) for name in ('closed_fire2','yoked_fire2')]
            record.update(S_relative_by_layer=relative(data[0]['S']-data[1]['S'],data[1]['S']),
                P_relative_by_layer=relative(data[0]['P']-data[1]['P'],data[1]['P']),
                parameter_sample_delta_norm=float(np.linalg.norm(data[0]['parameters']-data[1]['parameters'])))
            result['first_action_world_feedback'].append(record)
    if (a.work/'yoked_0/replay_checks.json').exists():
        result['yoke_exact_checks']=json.loads((a.work/'yoked_0/replay_checks.json').read_text())
        result['yoke_exact_checks']['all_archive_tables']=database_hashes.get('closed_0')==database_hashes.get('yoked_0')
    a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed=result['completed_arms'],exact_yoke=result.get('yoke_exact_checks'),
        compared=list(result['comparisons'])),ensure_ascii=False))


if __name__=='__main__':main()
