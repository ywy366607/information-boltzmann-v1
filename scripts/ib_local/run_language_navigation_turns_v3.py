"""V3 turn-taking navigation run on the continuing language individual.

Claude, 2026-11-04. Same individual and validation sidecar as v2
(`run_language_navigation_v2.py`); changes: the broad reservoir-indexed world
(`language_navigation_world_v3.BroadLanguageWorld`, ~40 MB source), the body's
own speak/listen and stop decisions (`language_navigation_turns_v3`), and the
experience table written once per experience (`latent_navigation_v3`).
Run only after `check_navigation_turns_v3_gate.py` has passed (design doc §11.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_action import digest
from scripts.ib_local.language_navigation_turns_v3 import TurnDialogueEngine
from scripts.ib_local.language_navigation_world_v3 import BroadLanguageWorld
from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_language_navigation_v2 import validate
from scripts.ib_local.run_ngram_fast_table import protocol_stream
from scripts.ib_local.train_gdn_fast_table import TableGDN

CHECKPOINT = Path('results/cgb_retrieved_credit_20261103/off/last.pt')
SOURCE_RAW = (8_000_000, 56_000_000)


def build(checkpoint=CHECKPOINT, mode='active', seed=11, validation_bytes=32768,
          disable_latent=False, max_utterance=32, response_bytes=480, holdout_bytes=480, capture=True):
    """The same continuing individual as v2 at its start, in the broad world."""
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    train = np.load(root/'train.npy', mmap_mode='r')
    raw = np.asarray(train[SOURCE_RAW[0]:min(SOURCE_RAW[1], len(train))])
    data = raw[raw >= offset]-offset
    if np.any(data > 255):
        raise ValueError('not a byte corpus')
    source = data.astype(np.uint8); del raw, data
    world = BroadLanguageWorld(source, seed=seed)
    validraw = np.load(root/'validation.npy', mmap_mode='r')
    raw = np.asarray(validraw[:2*(validation_bytes+1)])
    valid = (raw[raw >= offset]-offset)[:validation_bytes+1].astype(np.uint8)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    old = saved['config']
    if old['mode'] != 'off' or saved['step'] != old['steps']:
        raise ValueError('requires the completed source individual')
    stream, _ = protocol_stream(old['steps'], old['tokens'])
    if hashlib.sha256(stream.tobytes()).hexdigest() != old['data_sha256']:
        raise ValueError('original individual stream mismatch')
    pending = int(stream[-1]); del stream
    body = TableGDN(GatedDeltaNetLM(256, 128, 4, 4, 'decay_then_delta', 256, 4, 32.,
                                    decay_init='log_timescale').cuda(), True).cuda()
    body.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(body.parameters(), lr=3e-4, foreach=True)
    optimizer.load_state_dict(saved['optimizer'])
    model = LatentNavigationV3(body).cuda()
    new = [p for n, p in model.named_parameters() if not n.startswith('body.')]
    optimizer.add_param_group(dict(params=new, lr=optimizer.param_groups[0]['lr']))
    table = RetrievedCreditTable.from_state_dict(saved['table'])
    recurrent = saved['recurrent'].cuda()
    torch.set_rng_state(saved['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    if disable_latent:
        model.read_gain.requires_grad_(False)
        with torch.no_grad():
            model.read_gain.zero_()
    rng = np.random.default_rng(seed+3000)
    engine = TurnDialogueEngine(model, optimizer, recurrent, table, world, pending, rng, mode,
                                max_utterance, response_bytes, holdout_bytes, 0., capture=capture)
    files = [Path(__file__), Path('scripts/ib_local/language_navigation_turns_v3.py'),
             Path('scripts/ib_local/language_navigation_world_v3.py'),
             Path('scripts/ib_local/latent_navigation_v3.py'),
             Path('scripts/ib_local/language_navigation_v2.py'),
             Path('scripts/ib_local/latent_navigation_v2.py'),
             Path('scripts/ib_local/latent_navigation.py'), Path('scripts/ib_local/gated_deltanet.py')]
    cfg = dict(mode=mode, seed=seed, disable_latent=disable_latent, max_utterance=max_utterance,
               response_bytes=response_bytes, holdout_bytes=holdout_bytes,
               validation_bytes=validation_bytes, checkpoint=str(checkpoint),
               checkpoint_sha256=digest(checkpoint), source_raw_bounds=list(SOURCE_RAW),
               world_source_bytes=len(source), source_byte_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
               validation_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
               sources={str(x): digest(x) for x in files},
               world='broad reservoir-indexed OWT, world-side random choice; turn-taking; not natural dialogue')
    return engine, table, valid, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('active', 'silence', 'sequential'), required=True)
    p.add_argument('--cycles', type=int, default=2500)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--disable-latent', action='store_true')
    p.add_argument('--seed', type=int, default=11)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    fused = gdn_fused.enable()
    engine, table, valid, cfg = build(mode=args.mode, seed=args.seed, disable_latent=args.disable_latent)
    cfg.update(cycles=args.cycles, validate_every=args.validate_every, fused=fused)
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    model = engine.model
    initial = dict(validation_nll=validate(model, table, valid),
                   parameters=sum(p.numel() for p in model.parameters()),
                   latent_memory_bytes=model.memory.nbytes, world_index_bytes=engine.world.positions.nbytes)
    (args.output/'initial.json').write_text(json.dumps(initial, indent=2))
    print(json.dumps(initial), flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (args.output/'metrics.jsonl').open('w') as log:
        for cycle in range(args.cycles):
            directions = torch.randn(64, 32, device='cuda')
            check = (cycle+1) % args.validate_every == 0 or cycle+1 == args.cycles
            record = engine.step(directions, age_ablate=check)
            memory = model.memory
            record.update(cycle=cycle+1, recurrent_events=int(memory.clock),
                          episodes=int(memory.episodes), merge_count=int(memory.merged_total),
                          admit_count=int(memory.admitted_total), evict_count=int(memory.evicted_total),
                          occupied=int(memory.occupied.sum()),
                          progress_labelled=int((memory.progress_valid & memory.occupied).sum()),
                          read_gain=float(model.read_gain.detach()))
            if check:
                record['validation_nll'] = validate(model, table, valid)
                snapshot = dict(model=model.state_dict(), optimizer=engine.optimizer.state_dict(),
                                recurrent=engine.recurrent.cpu(), table=table.state_dict(),
                                pending=engine.pending, baseline=engine.baseline,
                                world=engine.world.state_dict(), config=cfg, cycle=cycle+1)
                tmp = args.output/'last.pending.pt'
                torch.save(snapshot, tmp, pickle_protocol=5)
                tmp.replace(args.output/'last.pt')
                print(json.dumps({k: record[k] for k in ('cycle', 'validation_nll', 'episodes',
                                                         'progress_labelled', 'read_gain')}), flush=True)
            log.write(json.dumps(record)+'\n'); log.flush()
    summary = dict(status='complete', cycles=args.cycles, updates=2*args.cycles,
                   elapsed_seconds=time.perf_counter()-started, cuda_peak=torch.cuda.max_memory_allocated(),
                   final_validation_nll=record['validation_nll'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
