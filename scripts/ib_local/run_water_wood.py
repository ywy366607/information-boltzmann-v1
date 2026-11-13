"""Continue one complete real-language individual; one declared link differs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.navigation_episodic_wood import configure_episodic_wood
from scripts.ib_local.navigation_causal_wood import configure_causal_wood
from scripts.ib_local.navigation_direct_read import configure_direct_read
from scripts.ib_local.navigation_checkpoint import save
from scripts.ib_local.run_causal_water_reply import build, resume_check
from scripts.ib_local.run_language_navigation_v2 import validate


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8')


def run(args):
    if (args.checkpoint.parent/'INVALID.json').exists():
        raise ValueError('source run is marked INVALID; restore the declared predecessor')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    fused = gdn_fused.enable()
    engine, raw, offset, source = build(args.checkpoint)
    if args.variant is None:
        args.variant=('direct' if hasattr(engine,'read_gain_state') else
                      'causal' if hasattr(engine.model.kanli,'causal_input') else 'episodic')
    if args.variant!='direct' and hasattr(engine,'read_gain_state'):
        raise ValueError('cannot discard a restored direct-read calibration; restore its predecessor')
    if args.variant=='episodic' and hasattr(engine.model.kanli,'causal_input'):
        raise ValueError('cannot discard a restored causal F input; restore its predecessor instead')
    if args.variant == 'direct': configure_direct_read(engine,args.mode,args.observer_mode)
    elif args.variant == 'causal': configure_causal_wood(engine, args.mode)
    else: configure_episodic_wood(engine, args.mode)
    selected = np.asarray(raw[:2*(args.validation_offset+args.validation_bytes+1)])
    valid = (selected[selected >= offset]-offset)[args.validation_offset:
        args.validation_offset+args.validation_bytes+1].astype(np.uint8)
    del raw, selected
    paths = {Path(m.__file__) for n,m in sys.modules.copy().items()
             if n.startswith('scripts.ib_local.') and getattr(m, '__file__', None)}
    paths.add(Path(__file__))
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    frozen = args.output/'executed_sources'; frozen.mkdir()
    for p in paths:
        shutil.copyfile(p, frozen/p.name)
    cfg = dict(stage='candidate-conditioned episodic wood in the continuing OWT loop',
        source=source, source_raw_bounds=source['source_raw_bounds'], topic_window=engine.world.window,
        seed=11, mode=args.mode, variant=args.variant, phase=args.phase, cycles=args.cycles, fused=fused,
        wood_memory_features=14 if args.variant in ('causal','direct') else 0,
        wood_evidence_schema=2 if args.variant in ('causal','direct') else 0,
        wood_gain_mode='direct_occurrence' if args.variant=='direct' else 'legacy_value',
        observer_mode=getattr(engine,'observer_mode','full'),
        read_moment_bytes=sum(v.nbytes for v in getattr(engine,'read_gain_state',{}).values()
                             if hasattr(v,'nbytes')),
        policy_replay='bounded one-byte action fallback on existing replay guard' if args.variant=='direct' else 'legacy',
        policy_boundary_bytes=1024 if args.variant=='direct' else 0,
        neural_parameters=sum(p.numel() for p in engine.model.parameters()),
        common_settlement=engine.settlement_kind,
        validation_offset=args.validation_offset, validation_bytes=len(valid)-1,
        validation_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
        sources=hashes, stopping_rule='fixed preregistered budget; stop on nonfinite/replay/resume failure',
        prediction='before action; deterministic same-option/provenance/age-band shuffled evidence is diagnostic only',
        control='block only new episodic forecast delivery; preserve other water paths and predictor learning')
    write(args.output/'config.json', cfg)
    if args.resume_check:
        checked = resume_check(engine)
        write(args.output/'resume_check.json', checked)
        print('RESUME',json.dumps(checked),flush=True)
    initial = validate(engine.model, engine.table, valid)
    write(args.output/'initial.json',dict(validation_nll=initial))
    print('START',args.mode,initial,flush=True)
    updates, original_step = [0], engine.optimizer.step
    def count_step(*a, **kw):
        updates[0] += 1
        return original_step(*a, **kw)
    engine.optimizer.step = count_step
    start, rows = time.perf_counter(), []
    prefix=[] if args.verify_prefix is None else [json.loads(line) for line in
        args.verify_prefix.read_text(encoding='utf-8').splitlines()]
    with (args.output/'metrics.jsonl').open('w',encoding='utf-8') as stream:
        for i in range(args.cycles):
            row = engine.step(torch.randn(64,32,device=engine.device))
            if i<len(prefix):
                for key in ('choice','utterance_hex','wood_target','choice_probability','direct_gain_counts_next'):
                    if row[key]!=prefix[i][key]:raise AssertionError(('prefix changed',i+1,key))
            row.update(new_cycle=i+1,cycle=source['cycle']+i+1,
                       cumulative_optimizer_updates=updates[0],seconds=time.perf_counter()-start)
            if (i+1)%args.validate_every == 0 or i+1==args.cycles:
                row['validation_nll'] = validate(engine.model,engine.table,valid)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            rows.append(row)
            if i<3 or (i+1)%32==0:
                print('STEP',i+1,row['choice'],row['episodic_policy_tv'],
                    row['wood_confidence_used'][3],row.get('validation_nll'),flush=True)
    half = rows[len(rows)//2:]
    summary = dict(mode=args.mode,cycles=len(rows),external_bytes=sum(r['external_bytes'] for r in rows),
        self_bytes=sum(r['self_bytes'] for r in rows),optimizer_updates=updates[0],
        validation_initial=initial,validation_final=rows[-1]['validation_nll'],
        validation_checkpoint_mean=float(np.mean([r['validation_nll'] for r in rows if 'validation_nll' in r])),
        late_gain_real_mse=float(np.mean([r['episodic_real_error'][3] for r in half])),
        late_gain_shuffle_mse=float(np.mean([r['episodic_shuffled_error'][3] for r in half])),
        late_gain_plain_mse=float(np.mean([r['episodic_plain_error'][3] for r in half])),
        policy_tv_max=max(r['episodic_policy_tv'] for r in rows),
        policy_tv_sum=sum(r['episodic_policy_tv'] for r in rows),
        replay_error_max=max(r['policy_replay_max_error'] for r in rows),
        replay_fallback_turns=sum(r.get('policy_replay_fallback',{}).get('used',False) for r in rows),
        verified_prior_prefix_turns=min(len(rows),len(prefix)),
        max_read_age=max(r['episodic_max_bank_age'] for r in rows),
        choices={a:sum(r['choice']==a for r in rows) for a in ('silence','generate','reuse')},
        seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        source_checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==source['sha256'])
    assert summary['source_checkpoint_unchanged']
    # Remove a local counting closure before persisting ordinary optimizer state.
    engine.optimizer.step = original_step
    if args.phase == 'train':
        save(engine,args.output,source['cycle']+args.cycles,cfg)
    write(args.output/'summary.json',summary)
    print('COMPLETE',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('real','blocked'),required=True)
    p.add_argument('--variant',choices=('episodic','causal','direct'),help='Default: preserve restored F architecture')
    p.add_argument('--observer-mode',choices=('full','joint_only'),help='Direct variant: optional proven observer-only omission')
    p.add_argument('--phase',choices=('audit','train'),default='audit')
    p.add_argument('--cycles',type=int,default=8)
    p.add_argument('--validation-bytes',type=int,default=32768)
    p.add_argument('--validation-offset',type=int,default=131072)
    p.add_argument('--validate-every',type=int,default=128)
    p.add_argument('--resume-check',action='store_true')
    p.add_argument('--verify-prefix',type=Path,help='Check a preserved pre-repair actual prefix without altering the new transition')
    run(p.parse_args())


if __name__=='__main__':
    main()
