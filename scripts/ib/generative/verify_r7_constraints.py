"""Independent source, censoring, locked-score and native-rate checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.r7_generator_bridge import interval_mass,retention_sector,objective
from scripts.ib.generative.r7_lifetime_constraints import observed_intervals,score


def main(args):
    base=args.base
    audit=base/'cgb_r7_observation_audit_20261022'
    life=base/'cgb_r7_lifetime_constraints_20261022'
    bridge=base/'cgb_r7_generator_bridge_20261022'
    source=base/'cgb_inverse_sources_20261022/Filo2'
    obs=json.loads((audit/'observations.json').read_text())
    author=loadmat(source/'Data/AllData.mat',simplify_cells=True)['Output']
    crosschecks={}
    for genotype in ['Control','Atg6_mutant','Atg7_mutant']:
        actual=[]
        for w in obs['workbooks']:
            if w['genotype']!=genotype or w['kind']!='filopodium':continue
            import re
            gc=int(re.search(r'GC(\d+)',w['file']).group(1))
            actual.extend((gc,r['first_frame'],r['last_frame'],r['reported_duration_minutes'])
                for r in w['records'] if r['bulbous_frame_count']==0)
        expected=[]
        for k in ['F','sF']:
            m=author[genotype]['P60'][k]
            expected.extend(tuple(map(int,row)) for row in zip(*[
                np.atleast_1d(m[x]) for x in ['GC','StartTimes','EndTimes','LTimes']]))
        assert sorted(actual)==sorted(expected)
        crosschecks[genotype]=len(actual)
    report=json.loads((life/'report.json').read_text())
    locked=json.loads((life/'parameters_locked.json').read_text())
    assert locked['observations_sha256']==hashlib.sha256((audit/'observations.json').read_bytes()).hexdigest()
    control=sorted([w for w in obs['workbooks'] if w['kind']=='filopodium' and w['genotype']=='Control'],key=lambda w:w['file'])
    for family,model in report['models'].items():
        assert model['parameters']==locked['parameters'][family]
        for key,group in [('train',control[:2]),('test',control[2:])]:
            actual=score(model['parameters'],*observed_intervals(group))
            assert actual==model['scores'][key]
    native=json.loads((bridge/'report.json').read_text())
    parent,_=load_exchange(native['native_parameter_source'])
    assert hashlib.sha256(Path(native['native_parameter_source']).read_bytes()).hexdigest()==native['parent_sha256']
    for row in native['rows']:
        matrix,rates=retention_sector(parent.params,row['total_support'])
        np.testing.assert_array_equal(matrix,np.asarray(row['transient_generator']))
        assert rates==row['actual_native_rates_per_reference_ms']
        theta=[np.log(row['clock_reference_ms_per_minute']),row['initially_unbound_probability']]
        for key,group in [('train',control[:2]),('test',control[2:])]:
            d,e=observed_intervals(group)
            value=objective(theta,matrix,d.astype(int),e.astype(bool))
            assert abs(value-row[key+'_nll'])<1e-12
        survival,events=interval_mass(matrix,row['clock_reference_ms_per_minute'],theta[1])
        assert abs(events[:59].sum()+survival[59]-1)<1e-12
    result=dict(passed=True,author_mat_nonbulb_rows_exact=crosschecks,
        locked_model_scores_reproduced=True,native_rate_projections_exact=True,
        normalized_absorbing_probability=True,parent_archive_sha256_verified=True,
        author_code_executed=False,physical_individual_advanced=False)
    atomic_json(life/'independent_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--base',type=Path,default=Path('F:/fly_stream_overflow'))
    main(p.parse_args())
