"""Track live storage provenance from actual receipts, never feed it back."""
import argparse
import gzip
import heapq
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.trace_fate_audit import cohort
from scripts.ib.generative.native_learning_stream import atomic_json


def records(path):
    previous=float('-inf')
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            assert row['time_ms']>=previous
            previous=row['time_ms']
            yield row


def analyze(run,label):
    m,meta=load_exchange(run/(label+'_start.npz'))
    obs=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
    live={str(v['carrier_id']):dict(r=v['trace'],c=v['bound'],
        origin='preexisting_unknown' if v['trace'] else None,started=None)
        for v in cohort(m,obs).values()}
    start_ms=m.t
    groups={}
    def group(origin):
        return groups.setdefault(origin or 'unmarked',dict(created=0,ended=0,
            ended_known_age_count=0,ended_known_age_total_ms=0.,support_capture_events=0,
            field_downhill_creation=0,field_uphill_or_zero_creation=0,
            live_trace=0,live_supported_trace=0,live_known_age_total_ms=0.,live_known_age_count=0))
    thermal_count=charged_thermal=0;max_live=len(live)
    merged=heapq.merge(records(run/(label+'_events.jsonl.gz')),
        records(run/(label+'_domains.jsonl.gz')),key=lambda x:x['time_ms'])
    for row in merged:
        kind=row['kind']
        if kind in ('domain_birth','domain_recovery') or kind.startswith('domain_'):
            if not row['accepted']:continue
            uid=str(row['observed_carrier_id'])
            if kind=='domain_birth':
                assert uid not in live;live[uid]=dict(r=0,c=0,origin=None,started=None)
            else:
                old=live.pop(uid);assert old['r']==old['c']==0
            continue
        if 'observed_carrier_id' not in row:continue
        uid=str(row['observed_carrier_id']);s=row['sign']
        if kind=='assembly':
            if s>0:
                assert uid not in live;live[uid]=dict(r=0,c=0,origin=None,started=None)
            else:
                old=live.pop(uid);assert old['r']==old['c']==0
            max_live=max(max_live,len(live));continue
        state=live[uid]
        if kind in ('write','thermal'):
            if kind=='thermal':
                thermal_count+=1;charged_thermal+=bool(row['charge_changes'])
            if s>0:
                assert state['r']==0
                state.update(r=1,origin=kind,started=row['time_ms'])
                g=group(kind);g['created']+=1
                field='field_downhill_creation' if row['electrical_energy_delta']<0 else 'field_uphill_or_zero_creation'
                g[field]+=1
            else:
                assert state['r']==1
                g=group(state['origin']);g['ended']+=1
                if state['started'] is not None:
                    g['ended_known_age_count']+=1
                    g['ended_known_age_total_ms']+=row['time_ms']-state['started']
                state.update(r=0,origin=None,started=None)
        elif kind in ('binding','transport'):
            dc=s if kind=='binding' else -s
            assert state['c']==(0 if dc>0 else 1)
            state['c']+=dc
            if dc>0:group(state['origin'])['support_capture_events']+=1
    end,endmeta=load_exchange(run/(label+'_complete.npz'))
    endobs=CarrierObservation(end,endmeta['growth_observer'],endmeta['carrier_observer'])
    current=cohort(end,endobs)
    assert set(live)=={str(v['carrier_id']) for v in current.values()}
    for value in current.values():
        state=live[str(value['carrier_id'])]
        assert state['r']==value['trace'] and state['c']==value['bound']
        if state['r']:
            g=group(state['origin']);g['live_trace']+=1;g['live_supported_trace']+=state['c']
            if state['started'] is not None:
                g['live_known_age_count']+=1;g['live_known_age_total_ms']+=end.t-state['started']
    return dict(passed=True,start_ms=start_ms,end_ms=end.t,actual_elapsed_ms=end.t-start_ms,
        total_thermal_events=thermal_count,thermal_events_carrying_charge=charged_thermal,
        groups=groups,maximum_live_carriers=max_live,final_live_carriers=len(live),
        scope='Route provenance, field energy and observed support capture; write is not automatically useful and thermal is not automatically irrelevant',
        survival_scope='Ended episodes and right-censored live episodes reported separately; no unconditional lifetime estimate')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.run/'report.json').read_text())
    result={label:analyze(args.run,label) for label in report['arms']}
    atomic_json(args.run/'storage_sources.json',dict(passed=True,arms=result,
        observer_state_cost='O(live carriers) plus fixed route aggregates; no new physical brain state'))
    print(json.dumps(result,ensure_ascii=False,indent=2))
