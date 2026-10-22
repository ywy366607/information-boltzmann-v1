"""Project time-order evidence onto the same inherited material identities."""
import argparse
import gzip
import json
import itertools
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.compare_support_transport import digest
from scripts.ib.generative.native_learning_stream import atomic_json


def differences(a,b,original_ids):
    aa={n:i for i,n in enumerate(a['ids'])};bb={n:i for i,n in enumerate(b['ids'])}
    common=sorted(set(original_ids)&set(aa)&set(bb))
    result=dict(shared_inherited_domains=len(common),
        original_domains_missing_either=len(original_ids)-len(common))
    for key in ('charge','charged','depleted','free','carried','coordinates'):
        x=np.asarray(a[key])[np.array([aa[i] for i in common],dtype=int)]
        y=np.asarray(b[key])[np.array([bb[i] for i in common],dtype=int)]
        result[key+'_l1']=float(np.abs(x-y).sum())
        result[key+'_l2']=float(np.linalg.norm(x-y))
    ca,cb=a['cohort'],b['cohort']
    keys=set(ca)|set(cb)
    # These are material endpoint positions/configurations across branches,
    # not a claim that identical post-fork carrier counters share a history.
    result['contact_pattern_symmetric_difference']=len(set(ca)^set(cb))
    result['trace_configuration_l1']=sum(abs(ca.get(k,{}).get('trace',0)-cb.get(k,{}).get('trace',0)) for k in keys)
    result['support_configuration_l1']=sum(abs(ca.get(k,{}).get('bound',0)-cb.get(k,{}).get('bound',0)) for k in keys)
    result['world_voltage_l2']=float(np.linalg.norm(np.array(a['world_voltage'])-b['world_voltage']))
    result['boundary_conductance_l1']=float(np.abs(np.array(a['boundary_conductance'])-b['boundary_conductance']).sum())
    result['total_trace_signed_difference']=a['trace']-b['trace']
    result['total_support_signed_difference']=a['bound_support']-b['bound_support']
    return result


def main(run):
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    matched=json.loads((run/'matched_future.json').read_text(encoding='utf-8'))
    assert report['passed'] and matched['passed'] and matched['primary_unchanged']
    parent,meta=load_exchange(report['sources']['parent_archive'])
    original_ids=meta['growth_observer']['ids']
    results=[]
    for family in range(4):
        for closed in (False,True):
            prefix=f'f{family}_'+('closed_' if closed else 'native_')
            a,b=(json.loads((run/(prefix+order+'_summary.json')).read_text(encoding='utf-8')) for order in ('AB','BA'))
            ma,mb=(matched['arms'][prefix+order] for order in ('AB','BA'))
            for arm in (ma,mb):
                assert not any(arm['start']['charge']) and not any(arm['start']['world_voltage'])
            ra,rmeta=load_exchange(ma['start_archive']);rb,bmeta=load_exchange(mb['start_archive'])
            assert rmeta['diagnostic_future_bath']['shared_future']==bmeta['diagnostic_future_bath']['shared_future']
            for key in ('rng_reaction','rng_motion','rng_field'):
                assert getattr(ra,key).bit_generator.state==getattr(rb,key).bit_generator.state
            entries={}
            for name in (prefix+'AB',prefix+'BA'):
                counts={}
                with gzip.open(run/(name+'_events.jsonl.gz'),'rt',encoding='utf-8') as stream:
                    for line in stream:
                        row=json.loads(line)
                        if row.get('domain_record'):continue
                        key=('port_' if row.get('interface',-1)>=0 else '')+row['kind']+('+' if row['sign']>0 else '-')
                        counts[key]=counts.get(key,0)+1
                if closed:assert all(v==0 for k,v in counts.items() if 'write' in k)
                entries[name]=counts
            results.append(dict(noise_family=family,closed_write_barrier=closed,
                after_shared_64ms_tail=differences(a['ordered'],b['ordered'],original_ids),
                after_paid_charge_preparation=differences(ma['start'],mb['start'],original_ids),
                common_bath_future_32ms=differences(ma['future'],mb['future'],original_ids),
                order_source_work_difference=a['ordered']['source_work']-b['ordered']['source_work'],
                future_source_work_difference=(ma['future']['source_work']-ma['start']['source_work'])
                    -(mb['future']['source_work']-mb['start']['source_work']),
                original_event_counts=entries))
    aggregate={}
    for closed in (False,True):
        rows=[r for r in results if r['closed_write_barrier']==closed]
        aggregate['closed' if closed else 'native']={phase:{key:dict(
            values=[r[phase][key] for r in rows],mean=float(np.mean([r[phase][key] for r in rows])))
            for key in ('trace_configuration_l1','support_configuration_l1','charged_l1','world_voltage_l2',
                        'contact_pattern_symmetric_difference','boundary_conductance_l1')}
            for phase in ('after_shared_64ms_tail','after_paid_charge_preparation','common_bath_future_32ms')}
    noise_reference={}
    for closed in (False,True):
        mode='closed' if closed else 'native'
        rows=[r for r in results if r['closed_write_barrier']==closed]
        noise_reference[mode]={}
        for field,label in (('trace','trace_configuration_l1'),('bound','support_configuration_l1')):
            values=[]
            for order in ('AB','BA'):
                cohorts=[json.loads((run/f'f{i}_{mode}_{order}_summary.json').read_text())['ordered']['cohort'] for i in range(4)]
                for i,j in itertools.combinations(range(4),2):
                    a,b=cohorts[i],cohorts[j]
                    values.append(sum(abs(a.get(k,{}).get(field,0)-b.get(k,{}).get(field,0)) for k in a.keys()|b.keys()))
            noise_reference[mode][label]=dict(same_order_different_noise_values=values,
                same_order_different_noise_mean=float(np.mean(values)),
                swapped_order_shared_initial_rng_values=[r['after_shared_64ms_tail'][label] for r in rows],
                swapped_order_shared_initial_rng_mean=float(np.mean([r['after_shared_64ms_tail'][label] for r in rows])))
    atomic_json(run/'timing_analysis.json',dict(passed=True,all_families=results,aggregate=aggregate,
        same_order_noise_reference=noise_reference,
        original_identity_projection=original_ids,
        source_report_sha256=digest(run/'report.json'),matched_future_sha256=digest(run/'matched_future.json'),
        scope='Order-sensitive retained material and subsequent exchange. Four noise families descriptive only; not pure-r causation, equal-work comparison, full V2 migration or prediction gain'))
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
