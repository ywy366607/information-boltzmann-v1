"""After the GPU owner ends, inspect real-body expression without learning/world calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-status',type=Path,required=True)
    p.add_argument('--expected-cycle',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='waiting_training',expected_cycle=args.expected_cycle,
        optimizer_updates=0,world_calls=0,changed_production_files=False)
    def write():
        t=args.output/'report.json.tmp'
        t.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        t.replace(args.output/'report.json')
    write()
    while True:
        s=json.loads(args.wait_status.read_text(encoding='utf-8'))
        if s['status'] in ('complete','stopped','failed','prior_did_not_complete_expected_boundary'):
            if s['status']!='complete' or s.get('final_cycle')!=args.expected_cycle:
                report.update(status='aborted_training_not_complete',training_status=s);write();return
            break
        time.sleep(5)
    from scripts.run_guarded_mother_continuation import resources
    # CPU inference omits the CUDA runtime/compiler and all backwards graphs.
    # Reserve 2 GiB rather than the 3.25 GiB GPU admission budget.
    minimum=2048 if args.device=='cpu' else 3328
    while resources(args.output)['available_commit_mib']<minimum:
        report.update(status='waiting_resources',resources=resources(args.output));write();time.sleep(5)
    # No Torch/CUDA import before the sole production GPU owner has finished.
    import numpy as np
    import torch
    import gzip
    import gc
    from scripts.ib_local.run_causal_water_reply import build,sha256_file
    from scripts.ib_local.causal_archive import attach_archive
    from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
    from scripts.ib_local.joint_relation_update import configure_joint_relation_update
    from scripts.ib_local.language_action import peek
    from scripts.ib_local.table_branch import table_branch
    torch.set_num_threads(1)
    checkpoint=args.wait_status.parent/'body/last.pt.gz'
    source_hash=sha256_file(checkpoint)
    # The sidecar config describes initialization. Only the committed body's
    # embedded manifest tracks its atomically replaced rolling archive.
    with gzip.open(checkpoint,'rb') as stream:
        saved=torch.load(stream,map_location='cpu',weights_only=False)
    cfg=saved['config']
    assert saved['cycle']==args.expected_cycle
    del saved;gc.collect()
    report.update(status='loading_frozen_body',source=str(checkpoint.resolve()),source_sha256=source_hash);write()
    report.update(device=args.device,minimum_commit_mib=minimum)
    engine,valid_raw,offset,meta=build(checkpoint,device=args.device,table_storage=args.output/'table',allow_external_archive=True)
    archive=attach_archive(engine,args.output/'archive',manifest=cfg['causal_archive'])
    configure_shared_unknowns(engine)
    configure_joint_relation_update(engine,revision=cfg['joint_relation_update']['implementation_revision'])
    table=engine.table
    # Prepare the native question/search on a private loaded body. Roll back
    # every speculative table write; no actual response is requested.
    archive.prepare()
    with table_branch(table) as branch:
        engine.table=branch
        try:engine._utter()
        finally:engine.table=table
    query=engine.model._question_context.detach().clone()
    state=engine.recurrent.detach().clone();pending=engine.pending
    report.update(status='frozen_inference',source_cycle=meta['cycle']);write()

    @torch.no_grad()
    def emit(initial,byte,prefix,context,seed):
        engine.model._question_context=None
        engine.decision.start(initial)
        with table_branch(table) as branch:
            for target in prefix:
                engine.decision.step(byte,peek(branch,byte))
                branch.run_credit(np.array([byte,int(target)],np.uint8),max_window=1)
                byte=int(target)
            engine.model._question_context=context
            generator=torch.Generator(device=args.device).manual_seed(seed)
            words=[]
            for _ in range(128):
                logits,_=engine.decision.step(byte,peek(branch,byte))
                chosen=int(torch.multinomial(logits.softmax(-1),1,generator=generator))
                words.append(chosen)
                branch.run_credit(np.array([byte,chosen],np.uint8),max_window=1)
                byte=chosen
        return bytes(words)

    # Test the actual trained question route at two fixed input contexts:
    # changing q must reveal whether it can directly change state or only logits.
    @torch.no_grad()
    def ports(initial,byte):
        engine.model._question_context=None;engine.decision.start(initial)
        off,h0=engine.decision.step(byte,peek(table,byte));s0=engine.decision.state.clone()
        engine.model._question_context=query;engine.decision.start(initial)
        on,h1=engine.decision.step(byte,peek(table,byte));s1=engine.decision.state.clone()
        return on-off,s0,float((s1-s0).abs().max()),float((h1-h0).abs().max())
    d0,next_s,s_error,h_error=ports(state,pending)
    d1,_,s_error1,h_error1=ports(next_s,32)
    report['query_route']=dict(state_change=max(s_error,s_error1),hidden_change=max(h_error,h_error1),
        logit_shift_context_difference=float((d0-d1).abs().max()),logit_shift_norm=float(d0.norm()),
        contract='same forced input and state; indirect changes from sampled words are separate')
    raw=np.asarray(valid_raw[:2*(262144+129)])
    owt=(raw[raw>=offset]-offset)[262144:262144+128].astype(np.uint8).tobytes()
    mother=Path('results/cgb_mother_continuation_20261110/validation/mother_validation.bin').read_bytes()[:128]
    cases=(('live_question',b'',query),('live_no_question',b'',None),
           ('real_mother_prefix',mother,None),('real_owt_prefix',owt,None))
    report['samples']=[]
    for name,prefix,context in cases:
        for seed in (110,111,112,113):
            text=emit(state,pending,prefix,context,seed)
            report['samples'].append(dict(condition=name,seed=seed,prefix=prefix.decode('utf-8','replace'),
                first32=text[:32].decode('utf-8','replace'),full128=text.decode('utf-8','replace'),
                hex=text.hex(),punctuation_within32=any(c in text[:32] for c in (b'.',b'?',b'!')),
                punctuation_within128=any(c in text for c in (b'.',b'?',b'!'))))
            write()
    report.update(status='complete',source_unchanged=sha256_file(checkpoint)==source_hash,
        interpretation='Frozen native body, paired same-prefix lengths; punctuation is not a grammaticality score. No parameter scaling comparison.')
    archive.close();write()


if __name__=='__main__':
    main()
