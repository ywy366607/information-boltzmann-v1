"""Score the saved stopped individual; no world calls or optimizer updates."""
import argparse
import gc
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.run_causal_water_reply import build,sha256_file
from scripts.ib_local.causal_archive import attach_archive
from scripts.ib_local.run_language_navigation_v2 import validate
from scripts.ib_local import gdn_fused


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    gdn_fused.enable()
    before=sha256_file(args.checkpoint)
    with gzip.open(args.checkpoint,'rb') as stream:
        saved=torch.load(stream,map_location='cpu',weights_only=False)
    config,cycle=saved['config'],saved['cycle']
    del saved;gc.collect()
    engine,raw,offset,meta=build(args.checkpoint,'cuda',table_storage=args.output/'table',allow_external_archive=True)
    attach_archive(engine,args.output/'archive',manifest=config['causal_archive'])
    # Match the running learner's declared independent offset, not the first
    # bytes of the validation file (a different sidecar).
    prefix=np.asarray(raw[:2*(262144+8193)])
    valid=(prefix[prefix>=offset]-offset)[262144:262144+8193].astype(np.uint8)
    mother=np.fromfile('results/cgb_mother_continuation_20261110/validation/mother_validation.bin',np.uint8)[:8193]
    with torch.no_grad():
        result=dict(owt_nll=validate(engine.model,engine.table,valid),
                    mother_nll=validate(engine.model,engine.table,mother),bytes_each=8192)
    steps=[]
    for parameter,state in engine.optimizer.state.items():
        if 'step' in state:steps.append(int(state['step']))
    report=dict(cycle=cycle,source_sha256=before,source_unchanged=sha256_file(args.checkpoint)==before,
        optimizer_updates=0,world_calls=0,validation=result,
        validation_owt_decoded_offset=262144,
        parameter_count=sum(p.numel() for p in engine.model.parameters()),
        optimizer_parameter_steps=dict(min=min(steps),max=max(steps)),
        transition_memory_clock=int(engine.model.memory.clock),
        clock_scope='observations since this memory was attached; not a reconstruction of all pretraining',
        native_config=engine.model.base.config_dict())
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
