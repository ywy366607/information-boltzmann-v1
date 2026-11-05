"""GPU engineering continuation on real OWT; deliberately no capability claim.

Restores the completed off individual's weights, moments, recurrent state, byte
table and RNG. Adds only the declared latent adapter. Keeps all observed pair
writes causal and records full continuation state. No 0/1 action experiment.
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
from scripts.ib_local.latent_navigation import LatentGraphTrainer, LatentNavigationBody
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_ngram_fast_table import protocol_stream
from scripts.ib_local.train_gdn_fast_table import TableGDN, FLOOR, LEXICAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--resume', type=Path, help='continue a complete calibration snapshot into a new output')
    parser.add_argument('--eager', action='store_true', help='disable CUDA Graph for debugging')
    args = parser.parse_args()
    if not 1 <= args.steps <= 64:
        raise ValueError('engineering calibration only: 1..64 updates')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    fused = gdn_fused.enable()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    old = saved['config']
    if old['mode'] != 'off' or saved['step'] != old['steps']:
        raise ValueError('requires completed off baseline; do not guess missing provenance')
    stream, _ = protocol_stream(old['steps'], old['tokens'])
    if hashlib.sha256(stream.tobytes()).hexdigest() != old['data_sha256']:
        raise ValueError('baseline source mismatch')
    pending = int(stream[-1]); del stream
    body = TableGDN(GatedDeltaNetLM(256, 128, 4, 4, 'decay_then_delta',
                    256, 4, 32., decay_init='log_timescale').cuda(), True).cuda()
    body.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(body.parameters(), lr=3e-4, foreach=True)
    optimizer.load_state_dict(saved['optimizer'])
    model = LatentNavigationBody(body).cuda()
    new_parameters = [p for name, p in model.named_parameters() if not name.startswith('body.')]
    optimizer.add_param_group(dict(params=new_parameters, lr=optimizer.param_groups[0]['lr']))
    state = saved['recurrent'].cuda()
    table = RetrievedCreditTable.from_state_dict(saved['table']) if args.resume is None else None
    torch_rng, cuda_rng, numpy_rng = saved['torch_rng'], saved['cuda_rng'], saved['numpy_rng']
    del saved
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    corpus = np.load(root/'train.npy', mmap_mode='r')
    cursor = 8_000_000
    config = dict(scope='engineering_only_not_capability_or_autonomy',
                  steps=args.steps, tokens=512, latent_dim=64, capacity=4096,
                  horizon=32, topk=8, temperature=.2, prediction_weight=.1, cuda_graph=not args.eager,
                  regularization_weight=.01, projections=32, source_start=cursor,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                  corpus_sha256=digest(root/'train.npy'), fused=fused,
                  sources={str(p): digest(p) for p in [Path(__file__),
                      Path('scripts/ib_local/latent_navigation.py'),
                      Path('scripts/ib_local/gated_deltanet.py'),
                      Path('scripts/ib_local/train_gdn_fast_table.py')]})
    start = 0
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        old_config = dict(resumed['config'])
        expected = dict(config)
        old_config.pop('steps'); expected.pop('steps')
        if old_config != expected or not 0 <= resumed['step'] < args.steps:
            raise ValueError('resume source, implementation or total budget mismatch')
        model.load_state_dict(resumed['model'])
        optimizer.load_state_dict(resumed['optimizer'])
        state = resumed['recurrent'].cuda()
        table = RetrievedCreditTable.from_state_dict(resumed['table'])
        cursor, pending, start = resumed['source_cursor'], resumed['pending'], resumed['step']
        torch_rng, cuda_rng, numpy_rng = resumed['torch_rng'], resumed['cuda_rng'], resumed['numpy_rng']
        del resumed
    torch.set_rng_state(torch_rng.cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in cuda_rng])
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    original_parameters = sum(p.numel() for p in body.parameters())
    named = list(body.named_parameters())
    groups = ([p for n,p in named if n.removeprefix('base.') in LEXICAL],
              [p for n,p in named if n.removeprefix('base.') not in LEXICAL], new_parameters)
    torch.cuda.reset_peak_memory_stats()
    runner = LatentGraphTrainer(model, optimizer, tokens=512, projections=32, capture=not args.eager)
    runner.state.copy_(state)
    started = time.perf_counter()
    rows = []
    for step in range(start, args.steps):
        start_cursor = cursor
        valid = np.flatnonzero(corpus[cursor:cursor+2048] >= offset)[:512]
        if len(valid) != 512:
            raise EOFError('insufficient real bytes')
        response = (corpus[cursor+valid]-offset).astype(np.uint8)
        cursor += int(valid[-1])+1
        raw = np.concatenate((np.array([pending], np.uint8), response))
        _, probs, _ = table.run_credit(raw, max_window=512)
        ids = torch.tensor(raw[:-1].astype(np.int64), device='cuda')[None]
        targets = torch.tensor(raw[1:].astype(np.int64), device='cuda')[None]
        logp = torch.tensor(np.log(np.maximum(probs, FLOOR)), device='cuda')[None]
        directions = torch.randn(64, 32, device='cuda')
        loss, _, _, diag = runner.step(ids, targets, logp, directions)
        norms = torch.stack([torch.nn.utils.clip_grad_norm_(g, 1., foreach=True) for g in groups])
        if not torch.isfinite(loss) or not torch.isfinite(norms).all():
            raise FloatingPointError('nonfinite same-body update')
        runner.finish()
        optimizer.step()
        pending = int(raw[-1])
        row = dict(step=step+1, source_start=start_cursor, source_stop=cursor,
                   ce=float(diag['ce']), latent=float(diag['latent']),
                   sigreg=float(diag['sigreg']), read_gain=float(model.read_gain.detach()),
                   gradient_norm=float(norms.norm()),
                   response_sha256=hashlib.sha256(response.tobytes()).hexdigest())
        rows.append(row)
        with (args.output/'metrics.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    snapshot = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    recurrent=runner.state.cpu(), table=table.state_dict(), pending=pending,
                    source_cursor=cursor, numpy_rng=numpy_rng, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), config=config, step=args.steps)
    temporary = args.output/'last.pending.pt'
    torch.save(snapshot, temporary, pickle_protocol=5)
    temporary.replace(args.output/'last.pt')
    summary = dict(status='complete', claim=config['scope'], updates=args.steps,
                   scored_bytes=args.steps*512, original_parameters=original_parameters,
                   resumed_from_step=start, updates_this_run=args.steps-start,
                   added_parameters=sum(p.numel() for p in new_parameters),
                   neural_state_bytes=runner.state.numel()*runner.state.element_size(),
                   byte_table_bytes=table.nbytes, latent_memory_bytes=model.memory.nbytes,
                   occupied_entries=int(model.memory.occupied.sum()),
                   cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                   elapsed_seconds=time.perf_counter()-started,
                   read_gain=rows[-1]['read_gain'], source_cursor=cursor,
                   no_capability_comparison=True, full_snapshot='last.pt')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
