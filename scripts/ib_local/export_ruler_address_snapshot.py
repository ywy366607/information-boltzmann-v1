"""CPU-only, read-only export of trained address geometry, never a body update."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall
    torch.set_num_threads(1)
    digest=sha(args.checkpoint)
    owner=load_readonly_checkpoint(args.checkpoint)
    state=owner['learner']
    result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=digest,
        source_updates=state['updates'],source_seen=state['cache']['seen_tokens'],
        pending_inputs=len(owner['stream']['inputs']),driver=owner['branch_driver'],
        cuda_initialized=torch.cuda.is_initialized(),layers={},body_updates=0)
    arrays={}
    reference=load_readonly_checkpoint(args.reference) if args.reference else None
    reference_sha=sha(args.reference) if args.reference else None
    for layer,stored in state['memories'].items():
        cfg=stored['memory._extra_state'];extra=stored['_extra_state']
        wrapper=LayerExperienceRecall(cfg['width'],capacity=cfg['capacity'],
            latent_dim=stored['encoder.weight'].shape[0],topk=cfg['topk'],
            replay_tokens=extra['replay_tokens'],device='cpu',read_interval=extra['read_interval'])
        wrapper.memory.mode=cfg['mode']
        wrapper.load_state_dict(stored)
        m=wrapper.memory
        queries=m.history[m.history_valid]
        with torch.no_grad():
            read=wrapper.read(queries)
        prefix=f'l{layer}_'
        fields=dict(source=m.source,occupied=m.occupied,birth=m.birth,
            history=queries,history_clock=m.history_clock[m.history_valid],
            encoder_weight=wrapper.encoder.weight,encoder_bias=wrapper.encoder.bias,
            baseline_slots=read.slots,baseline_weights=read.weights,
            read_credit_mean=m.read_credit_mean)
        for name,value in fields.items():
            arrays[prefix+name]=value.detach().cpu().numpy().copy()
        if reference is not None:
            previous=reference['learner']['memories'][layer]
            for name in ('weight','bias'):
                arrays[prefix+'reference_encoder_'+name]=previous['encoder.'+name].numpy().copy()
        # Source at each stored window's index64 is the actual unmodified
        # pre-layer activity, independently of input/output recall placement.
        result['layers'][str(layer)]=dict(human_layer=layer+1,config=cfg,
            clock=int(m.clock),coordinate_offset=state['cache']['seen_tokens']-
                len(owner['stream']['inputs'])-int(m.clock),
            occupied=int(m.occupied.sum()),read_interval=extra['read_interval'],
            recall_placement=extra['recall_placement'],null_read_enabled=extra['null_read_enabled'],
            history_count=len(queries),encoder_dim=wrapper.encoder.out_features)
    np.savez(args.output/'addresses.npz',**arrays)
    result.update(checkpoint_unchanged=sha(args.checkpoint)==digest,
                  exported_arrays_sha256=sha(args.output/'addresses.npz'))
    if reference is not None:
        result['reference']=dict(path=str(args.reference),sha256=reference_sha,
            unchanged=sha(args.reference)==reference_sha)
    if not result['checkpoint_unchanged'] or torch.cuda.is_initialized():
        raise AssertionError('read-only CPU export changed source or initialized CUDA')
    (args.output/'snapshot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
