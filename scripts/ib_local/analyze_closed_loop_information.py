"""Sampling-bias controls on an already collected closed-loop trajectory.

Block-shuffle surrogates are statistical controls, not new worlds or training.
The cut must be selected anew on every surrogate to account for partition search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.ib_local.closed_loop_information import analyze_trace
from scripts.ib_local.audit_closed_loop_information import GROUPS


def block_surrogate(trace, groups, rng, block=8):
    if len(trace) % block:
        raise ValueError('whole equal-sized time blocks required')
    result=np.empty_like(trace)
    count=len(trace)//block
    for columns in groups.values():
        # Preserve the two coordinates of a role together; independently
        # rearrange its time blocks, preserving local within-role dynamics.
        source=trace[:,columns].reshape(count,block,len(columns))
        result[:,columns]=source[rng.permutation(count)].reshape(len(trace),len(columns))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--surrogates',type=int,default=39)
    p.add_argument('--single-coordinate',action='store_true',
                   help='Post-primary sensitivity: first declared coordinate of each role')
    args=p.parse_args()
    rows=[json.loads(line) for line in (args.input/'full.jsonl').read_text(encoding='utf-8').splitlines()]
    trace=np.array([r['observables'] for r in rows])
    groups=GROUPS
    if args.single_coordinate:
        trace=trace[:,::2]
        groups={name:[i] for i,name in enumerate(GROUPS)}
        primary=analyze_trace(trace,groups)
    else:
        primary=json.loads((args.input/'result.json').read_text(encoding='utf-8'))['information']['1']
    phi=primary['minimum_projected_phi_g']['phi_g_nats']
    rng=np.random.default_rng(514)
    values=[]
    for i in range(args.surrogates):
        result=analyze_trace(block_surrogate(trace,groups,rng),groups)
        values.append(result['minimum_projected_phi_g']['phi_g_nats'])
        if (i+1)%10==0:
            print('SURROGATES',i+1,flush=True)
    report=dict(primary_phi_g_nats=phi,surrogate_minimum_phi_g_nats=values,
                surrogate_count=args.surrogates,block_size=8,
                empirical_upper_tail_p=(1+sum(v>=phi for v in values))/(1+len(values)),
                surrogate_median=float(np.median(values)),
                interpretation='Heuristic block-shuffle control, not a formal causal null; preserves within-role 8-turn blocks.',
                analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                regularization_sensitivity={str(reg):analyze_trace(trace,groups,regularization=reg)
                                            for reg in (1e-5,1e-3)})
    if args.single_coordinate:
        report.update(status='Post-primary exploratory dimension sensitivity; no new body execution',
                      channels=list(range(0,12,2)),analysis=primary)
    source=Path(__file__)
    shutil.copyfile(source,args.input/'executed_sources'/(
        source.stem+'_'+report['analyzer_source_sha256'][:12]+source.suffix))
    target='six_coordinate_sensitivity.json' if args.single_coordinate else 'sampling_controls.json'
    (args.input/target).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in
                      ('regularization_sensitivity','surrogate_minimum_phi_g_nats','analysis')},indent=2),flush=True)


if __name__=='__main__':
    main()
