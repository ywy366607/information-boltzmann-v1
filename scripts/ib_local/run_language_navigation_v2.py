"""Continuous real OWT navigation run and independent sidecar evaluation.

Each cycle contains a query, response, unseen continuation, and two learner
updates. No test data is used to choose a query or update the live individual.
The corpus-index environment is a surrogate, not natural conversation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_action import digest
from scripts.ib_local.language_navigation_v2 import DialogueEngine
from scripts.ib_local.language_navigation_world import LanguageNavigationWorld
from scripts.ib_local.latent_navigation_v2 import LatentNavigationV2, TransitionRead
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_ngram_fast_table import protocol_stream
from scripts.ib_local.train_gdn_fast_table import FLOOR, TableGDN


@torch.no_grad()
def validate(model, table, valid, chunk=512):
    """Fixed weights; clone tables, score each byte before clone assimilation.

    Cold recurrent context in the validation sidecar; live body remains intact.
    The latent sidecar also commits observed validation only after scoring.
    """
    clone = RetrievedCreditTable.from_state_dict(table.state_dict())
    clone.history = np.uint64(0)
    old_memory = model.memory
    model.memory = copy.deepcopy(old_memory)
    state = model.initial_state(1, device=next(model.parameters()).device)
    total = count = 0
    try:
        for i in range(0, len(valid)-1, chunk):
            raw = valid[i:min(i+chunk+1,len(valid))]
            _, prob, _ = clone.run_credit(raw, max_window=chunk)
            ids = torch.tensor(raw[:-1].astype(np.int64), device=state.device)[None]
            targets = torch.tensor(raw[1:].astype(np.int64), device=state.device)[None]
            logp = torch.tensor(np.log(np.maximum(prob,FLOOR)), device=state.device)[None]
            logits, state, _, features, read = model.predict_observed(ids,state,logp)
            total += float(F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='sum'))
            count += len(raw)-1
            model.commit_observed(features, read)
    finally:
        model.memory = old_memory
    return total/count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('active','random','silence'), required=True)
    p.add_argument('--memory', choices=('merge','fifo'), default='merge')
    p.add_argument('--cycles', type=int, default=2500)
    p.add_argument('--stop-after',type=int,help='cleanly checkpoint after this absolute cycle')
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--validation-bytes', type=int, default=32768)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--threshold', type=float, default=.98)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--query-bytes', type=int, default=32)
    p.add_argument('--response-bytes', type=int, default=480)
    p.add_argument('--holdout-bytes', type=int, default=480)
    p.add_argument('--disable-latent', action='store_true')
    args = p.parse_args()
    if min(args.cycles,args.validate_every,args.validation_bytes) <= 0:
        raise ValueError('invalid evaluation budgets')
    end_cycle = min(args.cycles,args.stop_after) if args.stop_after is not None else args.cycles
    if end_cycle <= 0: raise ValueError('stop-after must be positive')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    fused = gdn_fused.enable()
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    train = np.load(root/'train.npy', mmap_mode='r')
    source_start, source_stop = 8_000_000, min(24_000_000, len(train))
    raw = np.asarray(train[source_start:source_stop])
    raw_valid = raw[raw >= offset]-offset
    if np.any(raw_valid > 255): raise ValueError('not a byte corpus')
    source = raw_valid.astype(np.uint8); del raw, raw_valid
    world = LanguageNavigationWorld(source)
    validraw = np.load(root/'validation.npy', mmap_mode='r')
    raw = np.asarray(validraw[:2*(args.validation_bytes+1)])
    valid = (raw[raw>=offset]-offset)[:args.validation_bytes+1].astype(np.uint8)
    if len(valid) != args.validation_bytes+1: raise ValueError('insufficient validation bytes')
    source_files = [Path(__file__),Path('scripts/ib_local/language_navigation_v2.py'),
                    Path('scripts/ib_local/language_navigation_world.py'),
                    Path('scripts/ib_local/latent_navigation_v2.py'),
                    Path('scripts/ib_local/latent_navigation.py'),
                    Path('scripts/ib_local/gated_deltanet.py'),
                    Path('scripts/ib_local/train_gdn_fast_table.py'),
                    Path('scripts/ib_local/retrieved_table_credit.py')]
    cfg = dict(mode=args.mode,memory=args.memory,cycles=args.cycles,seed=args.seed,
        validate_every=args.validate_every, validation_bytes=args.validation_bytes,
        query_bytes=args.query_bytes, response_bytes=args.response_bytes, holdout_bytes=args.holdout_bytes,
        threshold=args.threshold, disable_latent=args.disable_latent,
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        corpus_sha256=digest(root/'train.npy'), source_raw_bounds=[source_start,source_stop],
        source_byte_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        validation_sha256=hashlib.sha256(valid.tobytes()).hexdigest(), fused=fused,
        sources={str(x):digest(x) for x in source_files},
        attribution='stale action-time score gradient after one update, only for previously unexposed heldout positions; no full derivative through learning',
        world='bounded exact-byte 4-gram indexed OWT continuation; no natural-dialogue claim')
    saved = torch.load(args.output/'last.pt' if args.resume else args.checkpoint,
                       map_location='cpu',weights_only=False)
    body = TableGDN(GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,
                                   decay_init='log_timescale').cuda(),True).cuda()
    if not args.resume: body.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(body.parameters(),lr=3e-4,foreach=True)
    if not args.resume: optimizer.load_state_dict(saved['optimizer'])
    model = LatentNavigationV2(body,threshold=args.threshold,mode=args.memory).cuda()
    new = [p for n,p in model.named_parameters() if not n.startswith('body.')]
    optimizer.add_param_group(dict(params=new,lr=optimizer.param_groups[0]['lr']))
    if args.resume:
        if saved['config'] != cfg: raise ValueError('resume source/config mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, pending, baseline = saved['cycle'],saved['pending'],saved['baseline']
        world.load_state_dict(saved['world'])
        rng = np.random.default_rng(); rng.bit_generator.state = saved['world_rng']
    else:
        old = saved['config']
        if old['mode']!='off' or saved['step']!=old['steps']: raise ValueError('requires completed baseline')
        stream,_ = protocol_stream(old['steps'],old['tokens'])
        if hashlib.sha256(stream.tobytes()).hexdigest()!=old['data_sha256']:
            raise ValueError('original individual stream mismatch')
        pending, start, baseline = int(stream[-1]), 0, 0.
        del stream
        rng = np.random.default_rng(args.seed+2000)
    if start >= end_cycle: raise ValueError('requested stop does not advance checkpoint')
    table = RetrievedCreditTable.from_state_dict(saved['table'])
    recurrent = saved['recurrent'].cuda()
    torch.set_rng_state(saved['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    del saved
    if args.disable_latent:
        model.read_gain.requires_grad_(False)
        with torch.no_grad(): model.read_gain.zero_()
    engine = DialogueEngine(model,optimizer,recurrent,table,world,pending,rng,args.mode,
                            args.query_bytes,args.response_bytes,args.holdout_bytes,baseline)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    logfile = args.output/'metrics.jsonl'
    if args.resume:
        lines = [json.loads(s) for s in logfile.read_text().splitlines() if s.strip()]
        lines = [line for line in lines if line['cycle']<=start]
        if [line['cycle'] for line in lines] != list(range(1,start+1)):
            raise ValueError('checkpoint/log mismatch')
        logfile.write_text(''.join(json.dumps(x)+'\n' for x in lines))
    else:
        initial = dict(validation_nll=validate(model,table,valid),parameters=sum(p.numel() for p in model.parameters()),
                       byte_table_bytes=table.nbytes,latent_memory_bytes=model.memory.nbytes,
                       world_index_bytes=world.positions.nbytes,world_corpus_bytes=source.nbytes,
                       world_provenance_bytes=world.seen.nbytes)
        (args.output/'initial.json').write_text(json.dumps(initial,indent=2))
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with logfile.open('a') as log:
        for cycle in range(start,end_cycle):
            directions = torch.randn(64,32,device='cuda')
            record = engine.step(directions,age_ablate=((cycle+1)%args.validate_every == 0 or cycle+1 == end_cycle))
            record.update(cycle=cycle+1,optimizer_updates=2*(cycle+1),
                recurrent_events=int(model.memory.clock),
                merge_count=int(model.memory.merged_total), admit_count=int(model.memory.admitted_total),
                evict_count=int(model.memory.evicted_total), dropped_count=int(model.memory.dropped_total),
                occupied=int(model.memory.occupied.sum()), read_gain=float(model.read_gain.detach()))
            if (cycle+1)%args.validate_every == 0 or cycle+1 == end_cycle:
                record['validation_nll'] = validate(model,table,valid)
            log.write(json.dumps(record)+'\n'); log.flush()
            if (cycle+1)%args.validate_every == 0 or cycle+1 == end_cycle:
                snapshot=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    recurrent=engine.recurrent.cpu(),table=table.state_dict(),pending=engine.pending,
                    baseline=engine.baseline,world_cursor=world.cursor,world=world.state_dict(),
                    world_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    config=cfg,cycle=cycle+1)
                tmp=args.output/'last.pending.pt'
                torch.save(snapshot,tmp,pickle_protocol=5); tmp.replace(args.output/'last.pt')
                print(json.dumps(record),flush=True)
    summary=dict(status='complete' if end_cycle == args.cycles else 'checkpointed',cycles=end_cycle,updates=2*end_cycle,
        external_bytes=end_cycle*(args.response_bytes+args.holdout_bytes),
        self_bytes=sum(json.loads(s)['self_bytes'] for s in logfile.read_text().splitlines()),
        elapsed_seconds=time.perf_counter()-started, cuda_peak=torch.cuda.max_memory_allocated(),
        final_validation_nll=record['validation_nll'],
        claim='budget-limited indexed real-text surrogate; not independent dialogue or survival')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
