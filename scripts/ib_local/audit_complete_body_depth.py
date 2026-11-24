"""Paired frozen depth ablation using the production independent validation.

Only the new residual outputs are bypassed. Every arm starts from the same
complete model and private validation-sidecar convention. No training or save.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior audit')
    torch.set_num_threads(1)
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.run_causal_water_reply import build,sha256_file
    from scripts.ib_local.run_language_navigation_v2 import validate
    gdn_fused.enable()
    engine,unused,offset,provenance=build(args.checkpoint,device='cuda',allow_external_archive=True)
    del unused
    gc.collect()
    root=Path(__file__).resolve().parents[2]
    raw=np.load(root/'data/ib_owt_10000/validation.npy',mmap_mode='r')
    raw=np.asarray(raw[:2*(262144+8193)])
    owt=(raw[raw>=offset]-offset)[262144:262144+8193].astype(np.uint8)
    metadata=json.loads((args.checkpoint.parent/'last.metadata.json').read_text(encoding='utf-8'))
    manifest=metadata['config']['natural_text_corpus']
    fineweb=np.fromfile(manifest['validation_path'],dtype=np.uint8,count=8193)
    before_hash=sha256_file(args.checkpoint)
    result=dict(cycle=metadata['cycle'],checkpoint_sha256=before_hash,bytes_each=8192,
        scope='production fixed cold sidecars, weights frozen; table learns only inside reversible sidecar; no navigation',
        enabled={},new_residual_outputs_bypassed={})
    with torch.no_grad():
        for name,data in [('owt',owt),('fineweb',fineweb)]:
            result['enabled'][name]=validate(engine.model,engine.table,data)
        handles=[]
        def bypass(module,inputs,output):
            return inputs[0],output[1],output[2]
        try:
            for layer in engine.model.body.base.layers[4:]:
                handles.append(layer.register_forward_hook(bypass))
            for name,data in [('owt',owt),('fineweb',fineweb)]:
                result['new_residual_outputs_bypassed'][name]=validate(engine.model,engine.table,data)
        finally:
            for handle in handles:handle.remove()
    result['new_depth_nll_benefit']={name:result['new_residual_outputs_bypassed'][name]-result['enabled'][name]
        for name in result['enabled']}
    if before_hash!=sha256_file(args.checkpoint):raise RuntimeError('source changed during frozen audit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
