"""Independent full-provenance replay from the saved observation midpoint."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.readout_observation import FrozenBitReadout,before_arrival_features
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json
from scripts.ib.generative.prediction_observation_audit import observe
from scripts.ib.generative.compare_support_transport import same,digest


def main(run):
    final,final_meta=load_exchange(run/'natural_complete.npz')
    m,meta=load_exchange(run/'validation_mid.npz')
    parent_hash=digest(run/'natural_complete.npz')
    session=copy.deepcopy(meta['session'])
    observer=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
    m.event_observer,m.domain_observer=observer.native,observer.domain
    readouts={k:FrozenBitReadout.from_state_dict(v) for k,v in meta['frozen_external_readouts'].items()}
    frequencies=copy.deepcopy(meta['matched_training_frequencies'])
    aggregates=copy.deepcopy(meta['observed_aggregates'])
    last_byte=meta['last_actual_byte']
    data=np.load(run/'validation_bytes.npy',allow_pickle=False)
    original=[json.loads(x) for x in (run/'natural_intervals.jsonl').read_text().splitlines()][1536:]
    assert len(original)==512
    for offset,target in enumerate(data[512:]):
        features=before_arrival_features(m,session,last_byte)
        predictions,scores=observe(features,readouts,frequencies,int(target))
        row=advance_byte(m,session,int(target),'fresh_validation_observation')
        expected=original[offset]
        assert all(expected[key]==value for key,value in row.items())
        assert scores==expected['external_observation_scores']
        for key,value in predictions.items():
            np.testing.assert_array_equal(value,expected['external_frozen_observer_logits'][key])
        bucket=aggregates['fresh_validation_observation'];bucket['n']+=1
        for key,value in row['scores'].items():bucket['raw'][key]+=value
        for key,value in scores.items():bucket['observed'][key]+=value
        last_byte=int(target)
    same(m.state_dict(),final.state_dict())
    assert session==final_meta['session'] and last_byte==final_meta['last_actual_byte']
    assert observer.state==final_meta['growth_observer'] and observer.carriers==final_meta['carrier_observer']
    assert aggregates==final_meta['observed_aggregates']
    assert frequencies==final_meta['matched_training_frequencies']
    assert {k:v.state_dict() for k,v in readouts.items()}==final_meta['frozen_external_readouts']
    assert digest(run/'natural_complete.npz')==parent_hash
    atomic_json(run/'full_metadata_resume.json',dict(passed=True,actual_diagnostic_replay_bytes=512,
        all_physics_three_rngs_session_observers_frozen_readouts_frequencies_aggregates_lastbyte_exact=True,
        source='All reference arrays and counters loaded only from validation_mid metadata',main_candidate_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
