"""Reproject all conditions onto the SAME archived parent cohort."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.trace_fate_audit import cohort
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import atomic_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def shared_fate(reference,live):
    same={k:v for k,v in live.items() if k in reference and v['carrier_id']==reference[k]['carrier_id']}
    rebuilt={k:v for k,v in live.items() if k in reference and v['carrier_id']!=reference[k]['carrier_id']}
    groups={}
    for name,field,value in [('traced','trace',1),('untraced','trace',0),('bound','bound',1),('unbound','bound',0)]:
        keys=[k for k,v in reference.items() if v[field]==value]
        retained=[k for k in keys if k in same]
        groups[name]=dict(initial=len(keys),retained=len(retained),
            current_trace=sum(same[k]['trace'] for k in retained),current_bound=sum(same[k]['bound'] for k in retained))
    return dict(source_carriers=len(reference),retained=len(same),rebuilt_old_positions=len(rebuilt),groups=groups)


def main(run):
    report=read(run/'report.json');assert report['passed']
    parent,meta=load_exchange(report['sources']['parent_archive'])
    observer=CarrierObservation(parent,meta['growth_observer'],meta['carrier_observer'])
    reference=cohort(parent,observer)
    setups={k:read(run/(k+'_setup.json')) for k in report['arms']}
    arms={}
    for label,a in report['arms'].items():
        first,first_meta=load_exchange(run/(label+'_start.npz'))
        obs=CarrierObservation(first,first_meta['growth_observer'],first_meta['carrier_observer'])
        rows=[dict(time_ms=first.t,phase=None,nll_nats_per_byte=None,
                   fate=shared_fate(reference,cohort(first,obs)))]
        for window in read(run/(label+'_windows.json')):
            rows.append(dict(time_ms=window['time_ms'],phase=window['phase'],
                nll_nats_per_byte=window['nll_nats_per_byte'],fate=shared_fate(reference,window['current_cohort'])))
        arms[label]=dict(windows=rows,final=rows[-1]['fate'],
                        scope='One shared pre-intervention parent cohort, including the 24 originally traced carriers in erase conditions')
    mechanisms={}
    for kept,erased in [('kept','erase'),('kept_no_capture','erase_no_capture')]:
        rates_a,rates_b=[setups[k]['initial_rates'] for k in (kept,erased)]
        condition=setups[kept]['setup']['condition']
        all_rows=[]
        for row in condition['receipts']:
            if not row['applied']:continue
            key=row['key']
            for kind in ('binding+','binding-'):
                ra,rb=rates_a['contacts'][key],rates_b['contacts'][key]
                if kind in ra and kind in rb:
                    all_rows.append(dict(key=key,kind=kind,kept_rate=ra[kind],erased_rate=rb[kind],ratio=ra[kind]/rb[kind]))
        if kept.endswith('_no_capture'):
            assert all(x['kept_rate']==x['erased_rate'] for x in all_rows)
        a,b=[load_exchange(run/(k+'_complete.npz'))[0] for k in (kept,erased)]
        records=[[json.loads(x) for x in (run/(k+'_intervals.jsonl')).read_text().splitlines()] for k in (kept,erased)]
        assert records[0][0]['pre_arrival_logits']==records[1][0]['pre_arrival_logits']
        np.testing.assert_array_equal([x['target'] for x in records[0]],[x['target'] for x in records[1]])
        mechanisms[kept+'__'+erased]=dict(inspected=condition['inspected_traced_carriers'],
            applied=condition['applied'],binding_rate_pairs=all_rows,
            conductance_difference_l2=float(np.linalg.norm(np.asarray(rates_a['bridge_conductance'])-rates_b['bridge_conductance'])),
            boundary_conductance_difference_l2=float(np.linalg.norm(np.asarray(rates_a['boundary_conductance'])-rates_b['boundary_conductance'])),
            initial_forecast_exactly_matched=True,
            first_4ms_later_forecast_logit_difference_l2=float(np.linalg.norm(np.asarray(records[0][1]['pre_arrival_logits'])-records[1][1]['pre_arrival_logits'])),
            final_world_charge_difference_l2=float(np.linalg.norm(a.world_charge-b.world_charge)),
            final_brain_port_voltage_difference_l2=float(np.linalg.norm(a.charge[a.port_nodes]/a.capacitance[a.port_nodes]-b.charge[b.port_nodes]/b.capacitance[b.port_nodes])),
            scope='Exact current-rate/finite-path delivery; single paths do not establish stable learning benefit or mediation')
    old_run=Path(report['sources']['parent_archive']).parent
    old=[json.loads(x) for x in (old_run/'symmetric_dt2_intervals.jsonl').read_text().splitlines()][128:]
    current=[json.loads(x) for x in (run/'natural_intervals.jsonl').read_text().splitlines()][128:]
    assert [x['target'] for x in old]==[x['target'] for x in current]
    before={key:float(np.mean([x['scores'][key] for x in old])) for key in old[0]['scores']}
    after=report['arms']['natural']['nll_by_phase']['revisit_A']
    revisit=dict(first_actual_pass=before,actual_revisit=after,
        material_delta=after['material']-before['material'],same_128_actual_bytes=True,
        scope='Prequential active exposure and revisit; no reset, no convergence or memory-capacity conclusion')
    atomic_json(run/'cohort_analysis.json',dict(parent_cohort=reference,arms=arms,mechanisms=mechanisms,
        actual_revisit=revisit,setup_interventions={k:v['setup'] for k,v in setups.items()},
        original_report_cohort_definition='Each raw arm starts after its setup, so erased-arm original_trace count is 0. This derived analysis uses the SAME archived pre-intervention parent cohort for all arms.',
        no_physical_state_or_raw_record_changed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
