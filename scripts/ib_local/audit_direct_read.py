"""Audit the real continuing transition and a pure-observer optimization."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.audit_water_wood_resume import audit, digest, live_parts
from scripts.ib_local.navigation_checkpoint import snapshot, restore
from scripts.ib_local.navigation_direct_read import configure_direct_read
from scripts.ib_local.run_causal_water_reply import build


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=8)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);gdn_fused.enable()
    e,_,_,source=build(args.checkpoint)
    configure_direct_read(e,'real','full')
    state=snapshot(e);rows={};reports={}
    for mode in ('full','joint_only'):
        restore(e,state);e.observer_mode=mode
        start=time.perf_counter();rows[mode]=[]
        for _ in range(args.cycles):
            rows[mode].append(e.step(torch.randn(64,32,device=e.device)))
        parts=live_parts(e)
        # These two fields only say whether unused witness scores were read.
        parts['engine']={k:v for k,v in parts['engine'].items()
                         if k not in ('observer_mode','_witness_readings')}
        reports[mode]=dict(seconds=time.perf_counter()-start,
            hashes={k:digest(v) for k,v in parts.items()})
        del parts;gc.collect()
    equal={k:reports['full']['hashes'][k]==reports['joint_only']['hashes'][k]
           for k in reports['full']['hashes']}
    causal_fields=('choice','utterance_hex','wood_target','metal_g','joint_progress',
                   'direct_gain_mean','direct_gain_alpha','direct_gain_counts_next','choice_probability')
    same_rows=all(all(a[k]==b[k] for k in causal_fields)
                  for a,b in zip(rows['full'],rows['joint_only']))
    assert all(equal.values()) and same_rows,(equal,same_rows)
    # Restore audit also covers the new occurrence moments and RNG.
    strict=audit(e)
    report=dict(source=source,cycles=args.cycles,observer_modes=reports,
        equal_physical_parts=equal,equal_actual_choices_gains_and_forecasts=same_rows,
        excluded_observer_fields=['observer_mode','_witness_readings'],strict_resume=strict,
        forecast_before_outcome=True,
        finite_budget_bytes=sum(v.nbytes for v in e.read_gain_state.values()
                               if hasattr(v,'nbytes')),
        policy_tv_max=max(r['episodic_policy_tv'] for r in rows['joint_only']),
        qualified_counts=e.read_gain_state['count'].tolist(),
        scope='engineering audit on actual full learning/world transitions; no capability claim')
    for mode,data in rows.items():
        (args.output/f'{mode}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in data),encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
