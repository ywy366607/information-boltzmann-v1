"""Frozen independent language sidecars for a saved continuing individual."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.run_causal_water_reply import build, sha256_file
from scripts.ib_local.run_language_navigation_v2 import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mother-validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owt-offset',type=int,default=262144)
    p.add_argument('--owt-bytes',type=int,default=32768)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    gdn_fused.enable()
    engine,raw,offset,source=build(args.checkpoint)
    data=np.asarray(raw[:2*(args.owt_offset+args.owt_bytes+1)])
    owt=(data[data>=offset]-offset)[args.owt_offset:args.owt_offset+args.owt_bytes+1].astype(np.uint8)
    mother=np.frombuffer(args.mother_validation.read_bytes(),dtype=np.uint8)
    if len(owt)!=args.owt_bytes+1 or len(mother)<513:
        raise ValueError('independent validation stream too short')
    # The exact validation program already used for the 6 Nov body adapts a
    # journalled sidecar fast table and memory, never the live individual.
    owt_nll=validate(engine.model,engine.table,owt)
    mother_nll=validate(engine.model,engine.table,mother)
    report=dict(status='complete',checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256_file(args.checkpoint),
        cycle=source['cycle'],owt_nats_per_byte=owt_nll,owt_bytes=len(owt)-1,
        owt_sha256=hashlib.sha256(owt.tobytes()).hexdigest(),
        mother_nats_per_byte=mother_nll,mother_bytes=len(mother)-1,
        mother_sha256=hashlib.sha256(mother.tobytes()).hexdigest(),
        scope='cold recurrent validation sidecars with adaptive journalled table and latent memory',
        caveat='mother validation uses previous research-side fork replies, not an independent benchmark')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
