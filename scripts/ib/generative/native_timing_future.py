"""Shared future bath after temporal histories and paid charge preparation."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.material_fork import fork_material
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json
from scripts.ib.generative.native_timing_audit import panel


def main(run):
    assert json.loads((run/'progress.json').read_text())['status']=='completed'
    sources=json.loads((run/'sources.json').read_text())
    parent,parent_meta=load_exchange(sources['parent_archive'])
    assert digest(sources['parent_archive'])==sources['parent_sha256']
    primary_hash=digest(run/'natural_complete.npz')
    data=np.load(run/'actual_train_4096_4136.npy',allow_pickle=False)[32:]
    result={}
    for family in range(4):
        for closed in (False,True):
            for order in ('AB','BA'):
                name=f'f{family}_'+('closed_' if closed else 'native_')+order
                m,meta=load_exchange(run/(name+'_neutral.npz'))
                assert not np.any(m.charge) and not np.any(m.world_charge)
                old_rng={key:copy.deepcopy(getattr(m,key).bit_generator.state)
                         for key in ('rng_reaction','rng_motion','rng_field')}
                for key in old_rng:
                    rng=copy.deepcopy(getattr(parent,key))
                    rng.bit_generator.advance(10000000+1000000*family)
                    setattr(m,key,rng)
                new_rng={key:copy.deepcopy(getattr(m,key).bit_generator.state) for key in old_rng}
                meta=copy.deepcopy(meta)
                meta['diagnostic_future_bath']=dict(previous=old_rng,shared_future=new_rng,
                    offset=10000000+1000000*family,
                    scope='Counterfactual future bath choice only; no physical elapsed time skipped')
                start=run/(name+'_matched_start.npz')
                save_exchange(start,m,meta)
                restored,rmeta=load_exchange(start)
                session=copy.deepcopy(meta['session']);rsession=copy.deepcopy(rmeta['session'])
                sham=fork_material(m);ssession=copy.deepcopy(session)
                obs=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
                robs=CarrierObservation(restored,rmeta['growth_observer'],rmeta['carrier_observer'])
                m.event_observer,m.domain_observer=obs.native,obs.domain
                restored.event_observer,restored.domain_observer=robs.native,robs.domain
                before=panel(m,obs);rows=[];maximum=0.
                for target in data:
                    row=advance_byte(m,session,int(target),'matched_diagnostic_common_future')
                    assert row==advance_byte(restored,rsession,int(target),'matched_diagnostic_common_future')
                    assert row==advance_byte(sham,ssession,int(target),'matched_diagnostic_common_future')
                    rows.append(row);maximum=max(maximum,max(abs(float(v)) for v in m.balances().values()))
                same(m.state_dict(),restored.state_dict())
                same(m.state_dict(),sham.state_dict())
                assert session==rsession==ssession and obs.state==robs.state and obs.carriers==robs.carriers
                assert maximum<1e-8
                meta.update(session=session,growth_observer=obs.state,carrier_observer=obs.carriers,
                    last_actual_byte=int(data[-1]),actual_matched_future_intervals=8)
                finish=run/(name+'_matched_future.npz')
                save_exchange(finish,m,meta)
                result[name]=dict(start=before,future=panel(m,obs),scored_records=rows,
                    maximum_error=maximum,exact_restored_future_intervals=8,
                    observer_only_sham_intervals=8,
                    start_archive=str(start),final_archive=str(finish),final_sha256=digest(finish))
    assert digest(run/'natural_complete.npz')==primary_hash
    atomic_json(run/'matched_future.json',dict(passed=True,primary_unchanged=True,arms=result,
        future_bath_shared_within_each_family=True,
        protocol=str(ROOT/'docs/information_boltzmann/CGB_NATIVE_TIMING_FUTURE_CONTROL_20261022.md'),
        scope='Full retained slow/material/geometry state plus common future bath; not a pure trace ablation'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
