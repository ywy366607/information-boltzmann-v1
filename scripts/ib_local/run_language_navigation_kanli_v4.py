"""V4 stage 1 (水火既济) run on the continuing language individual (Claude, 2026-11-04).

Same individual, validation sidecar and broad corpus as the v3 turn run; the
world's passive supply is a bounded topic window (`TopicWorld`), experiences
keep their words, and the body may reuse recalled words. Run only after
`check_navigation_kanli_v4_gate.py` has passed.
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
from scripts.ib_local.language_navigation_kanli_v4 import KanLiEngine, TopicWorld, attach
from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_language_navigation_turns_v3 import CHECKPOINT, SOURCE_RAW
from scripts.ib_local.run_language_navigation_v2 import validate
from scripts.ib_local.run_ngram_fast_table import protocol_stream
from scripts.ib_local.train_gdn_fast_table import TableGDN


def build(mode='active', seed=11, window=4096, validation_bytes=32768, capture=True, imagine=False, witness=False, fire=False, equal=False, water=False, clear=False, reply=False, film=False, quiet_digest=False):
    water = water or clear
    equal = equal or water or film or quiet_digest
    witness = witness or fire or equal
    imagine = imagine or witness
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    train = np.load(root/'train.npy', mmap_mode='r')
    raw = np.asarray(train[SOURCE_RAW[0]:min(SOURCE_RAW[1], len(train))])
    data = raw[raw >= offset]-offset
    source = data.astype(np.uint8); del raw, data
    if reply:
        from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
        world = ReplyTopicWorld(source, window=window, seed=seed)
    else:
        world = TopicWorld(source, window=window, seed=seed)
    validraw = np.load(root/'validation.npy', mmap_mode='r')
    raw = np.asarray(validraw[:2*(validation_bytes+1)])
    valid = (raw[raw >= offset]-offset)[:validation_bytes+1].astype(np.uint8)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
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
    if imagine:
        from scripts.ib_local.language_navigation_kanli_v4b import attach_imagination
        model = attach_imagination(LatentNavigationV3(body).cuda())
    else:
        model = attach(LatentNavigationV3(body).cuda())
    new = [p for n, p in model.named_parameters() if not n.startswith('body.')]
    optimizer.add_param_group(dict(params=new, lr=optimizer.param_groups[0]['lr']))
    table = RetrievedCreditTable.from_state_dict(saved['table'])
    recurrent = saved['recurrent'].cuda()
    torch.set_rng_state(saved['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    rng = np.random.default_rng(seed+4000)
    if quiet_digest:
        from scripts.ib_local.language_navigation_kanli_v4j import DigestEngine as Engine
    elif film:
        from scripts.ib_local.language_navigation_kanli_v4i import FilmEngine as Engine
    elif clear:
        from scripts.ib_local.language_navigation_kanli_v4g import ClearWaterEngine as Engine
    elif water:
        from scripts.ib_local.language_navigation_kanli_v4f import WaterEngine as Engine
    elif equal:
        from scripts.ib_local.language_navigation_kanli_v4e import EqualEngine as Engine
    elif fire:
        from scripts.ib_local.language_navigation_kanli_v4d import FireEngine as Engine
    elif witness:
        from scripts.ib_local.language_navigation_kanli_v4c import WitnessEngine as Engine
    elif imagine:
        from scripts.ib_local.language_navigation_kanli_v4b import ImaginingEngine as Engine
    else:
        Engine = KanLiEngine
    engine = Engine(model, optimizer, recurrent, table, world, pending, rng, mode,
                    32, 480, 480, 0., capture=capture)
    files = [Path(__file__), Path('scripts/ib_local/language_navigation_kanli_v4.py'),
             Path('scripts/ib_local/language_navigation_turns_v3.py'),
             Path('scripts/ib_local/language_navigation_world_v3.py'),
             Path('scripts/ib_local/latent_navigation_v3.py'), Path('scripts/ib_local/latent_navigation_v2.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4b.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4c.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4d.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4e.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4f.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4g.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4h.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4i.py'),
             Path('scripts/ib_local/language_navigation_kanli_v4j.py')]
    cfg = dict(stage='v4 stage 8: water metal digests in quiet (replay priority gain x need)' if quiet_digest else
               'v4 stage 7: wood is the film of the body itself (draft, imagined replies, entropy to the critic)' if film else
               'v4 stage 6b: earth sorts water before wood (clarity), wood plays films' if clear else
               'v4 stage 6: water overflows into wood (seeds, digest, release)' if water else
               'v4 stage 5b: equal-length speech window, no stop head' if equal else
               'v4 stage 5: fire has its own metal (self-normalized head advantage)' if fire else
               'v4 stage 4: metal has its own water (witness reservoir, TD return)' if witness else
               'v4 stage 2: wood-metal assist water-fire' if imagine else 'v4 stage 1: water-fire', mode=mode, seed=seed, topic_window=window,
               checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
               source_raw_bounds=list(SOURCE_RAW), world_source_bytes=len(source),
               validation_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
               sources={str(x): digest(x) for x in files},
               world=('bounded topic window read cyclically; the world takes up the END of the utterance (接话) and continues from there' if reply else
                      'bounded topic window read cyclically; understood speech (indexed 4-gram) moves it'))
    return engine, table, valid, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('active', 'silence'), required=True)
    p.add_argument('--cycles', type=int, default=2500)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--imagine', action='store_true')
    p.add_argument('--witness', action='store_true')
    p.add_argument('--fire', action='store_true')
    p.add_argument('--equal', action='store_true')
    p.add_argument('--water', action='store_true')
    p.add_argument('--clear', action='store_true')
    p.add_argument('--reply', action='store_true')
    p.add_argument('--film', action='store_true')
    p.add_argument('--digest', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    fused = gdn_fused.enable()
    engine, table, valid, cfg = build(mode=args.mode, seed=args.seed, imagine=args.imagine, witness=args.witness, fire=args.fire, equal=args.equal, water=args.water, clear=args.clear, reply=args.reply, film=args.film, quiet_digest=args.digest)
    cfg.update(cycles=args.cycles, validate_every=args.validate_every, fused=fused)
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    model = engine.model
    initial = dict(validation_nll=validate(model, table, valid))
    (args.output/'initial.json').write_text(json.dumps(initial, indent=2))
    started = time.perf_counter()
    with (args.output/'metrics.jsonl').open('w') as log:
        for cycle in range(args.cycles):
            check = (cycle+1) % args.validate_every == 0 or cycle+1 == args.cycles
            record = engine.step(torch.randn(64, 32, device='cuda'), age_ablate=check)
            memory = model.memory
            record.update(cycle=cycle+1, episodes=int(memory.episodes), merge_count=int(memory.merged_total),
                          evict_count=int(memory.evicted_total), occupied=int(memory.occupied.sum()),
                          words_stored=int((memory.word_count > 0).sum()),
                          world_moves=engine.world.moves, read_gain=float(model.read_gain.detach()))
            if check:
                record['validation_nll'] = validate(model, table, valid)
                snapshot = dict(model=model.state_dict(), optimizer=engine.optimizer.state_dict(),
                                recurrent=engine.recurrent.cpu(), table=table.state_dict(),
                                pending=engine.pending, baseline=engine.baseline,
                                world=engine.world.state_dict(), config=cfg, cycle=cycle+1)
                tmp = args.output/'last.pending.pt'
                torch.save(snapshot, tmp, pickle_protocol=5)
                tmp.replace(args.output/'last.pt')
                print(json.dumps({k: record[k] for k in ('cycle', 'validation_nll', 'world_moves', 'choice_probability')}),
                      flush=True)
            log.write(json.dumps(record)+'\n'); log.flush()
    summary = dict(status='complete', cycles=args.cycles, elapsed_seconds=time.perf_counter()-started,
                   final_validation_nll=record['validation_nll'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
