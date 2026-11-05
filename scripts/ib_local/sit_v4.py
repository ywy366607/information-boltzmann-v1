"""闭关 (sitting): the v4 individual with the world's input cut off (Claude, 2026-11-04).

User, 11-04: with no input from outside, water should rise by itself, become a
motive, act, and the act changes only the self. What happens to that loop in
the end; does everything disappear, like the emptiness after the films?

Nothing new is added. The individual's own machinery keeps running: it
predicts its next byte, samples it, hears it (recurrent state and byte table
assimilate it, the experience table is read into the shared GDN state), and
its usual learning stays on. Its own bytes never enter the language loss (as
in its whole life), so only the latent prediction, SIGReg, the experience
table and the byte table can change from this. No world, no progress label.
Observed, not steered: entropy, diversity, compressibility, state, and the
real-text validation sidecar.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.language_action import peek
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_language_navigation_kanli_v4 import build
from scripts.ib_local.run_language_navigation_v2 import validate


@torch.no_grad()
def free_run(engine, length):
    """Sample `length` bytes from the body's own predictions, self-hearing each."""
    engine.decision.start(engine.recurrent)
    byte = engine.pending
    out, probs, entropy = [], [], []
    for _ in range(length):
        p = peek(engine.table, byte)
        logits, _ = engine.decision.step(byte, p)
        logp = logits.log_softmax(-1)
        entropy.append(float(-(logp.exp()*logp).sum()))
        chosen = int(torch.multinomial(logp.exp(), 1))
        _, observed, _ = engine.table.run_credit(np.array([byte, chosen], np.uint8), max_window=1)
        if not np.array_equal(p, observed[0]):
            raise RuntimeError('self-hearing peek differs from prequential table read')
        probs.append(p)
        out.append(chosen)
        byte = chosen
    return np.array(out, np.uint8), np.asarray(probs, np.float32), entropy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--individual', type=Path, default=Path('results/cgb_language_navigation_v4_20261104/active/last.pt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunks', type=int, default=1000)
    p.add_argument('--chunk', type=int, default=480)
    p.add_argument('--validate-every', type=int, default=100)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    gdn_fused.enable()
    engine, _, valid, cfg = build()
    saved = torch.load(args.individual, map_location='cpu', weights_only=False)
    engine.model.load_state_dict(saved['model'])
    engine.optimizer.load_state_dict(saved['optimizer'])
    engine.table = table = RetrievedCreditTable.from_state_dict(saved['table'])
    engine.recurrent = saved['recurrent'].cuda()
    engine.pending = int(saved['pending'])
    del saved
    model, memory = engine.model, engine.model.memory
    cfg.update(individual=str(args.individual), chunks=args.chunks, chunk=args.chunk,
               experiment='sitting: no world input; own machinery unchanged')
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    start_validation = validate(model, table, valid)
    print(json.dumps(dict(start_validation=start_validation)), flush=True)
    open_experience = None
    started = time.perf_counter()
    with (args.output/'metrics.jsonl').open('w') as log, (args.output/'stream.bin').open('wb') as stream:
        for k in range(args.chunks):
            start_state = engine.recurrent.clone()
            source_time = int(memory.clock)
            words, probs, entropy = free_run(engine, args.chunk)
            stream.write(words.tobytes())
            raw = np.concatenate(([engine.pending], words)).astype(np.uint8)
            ids, targets, logp = engine._inputs(raw, probs)
            mask = torch.zeros_like(targets, dtype=torch.float32)  # own bytes: never language loss
            loss, next_state, features, diag = model.objective(
                ids, targets, start_state, logp, torch.randn(64, 32, device='cuda'), score_mask=mask)
            engine._advance(loss, features, diag, next_state)
            engine.pending = int(words[-1])
            # The previous chunk's experience is complete once its long successor exists.
            h = features.detach()
            if open_experience is not None:
                src, short, t0 = open_experience
                memory.commit_episode(src, src, short, h[memory.long-args.chunk], model.encoder,
                                      progress=None, source_time=t0)
            open_experience = (h[0].clone(), h[memory.short].clone(), source_time)
            text = words.tobytes()
            grams = {text[i:i+4] for i in range(len(text)-3)}
            state = engine.recurrent
            record = dict(chunk=k+1, bytes=(k+1)*args.chunk,
                          entropy_mean=float(np.mean(entropy)), entropy_min=float(np.min(entropy)),
                          gzip_ratio=len(gzip.compress(text))/len(text), distinct_4gram=len(grams)/(len(text)-3),
                          distinct_bytes=len(set(text)), space_newline_fraction=float(np.isin(words, [10, 32]).mean()),
                          latent_short=float(diag['latent']), latent_long=float(diag['latent_long']),
                          sigreg=float(diag['sigreg']), read_gain=float(model.read_gain.detach()),
                          state_norm=[float(state[l].norm()) for l in range(state.shape[0])],
                          feature_norm=float(h.norm(dim=-1).mean()),
                          read_top_weight=float(diag['read_weights'].max(-1).values.mean()),
                          occupied=int(memory.occupied.sum()), merges=int(memory.merged_total))
            if (k+1) % args.validate_every == 0 or k == 0:
                record['validation_nll'] = validate(model, table, valid)
                record['sample'] = text[:200].decode('utf-8', 'replace')
                print(json.dumps({x: record[x] for x in ('chunk', 'validation_nll', 'entropy_mean', 'gzip_ratio',
                                                          'distinct_4gram', 'read_gain')}), flush=True)
            log.write(json.dumps(record, ensure_ascii=False)+'\n'); log.flush()
    (args.output/'summary.json').write_text(json.dumps(dict(start_validation=start_validation,
        elapsed_seconds=time.perf_counter()-started), indent=2))


if __name__ == '__main__':
    main()
