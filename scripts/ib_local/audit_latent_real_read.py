"""Read-only matched intervention on a saved continuous real-text individual.

Two fixed-weight branches receive the same future bytes and byte-table reads;
only the latent read gain is set to zero in the second branch. Both start with
the same restored state. Latent memory is intentionally frozen during these two
windows: this isolates current read delivery, not online-learning performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_action import digest
from scripts.ib_local.latent_navigation import LatentNavigationBody
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.train_gdn_fast_table import TableGDN, FLOOR


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior probe')
    torch.set_num_threads(2)
    fused = gdn_fused.enable()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    base = GatedDeltaNetLM(256, 128, 4, 4, 'decay_then_delta', 256, 4, 32.,
                           decay_init='log_timescale').cuda()
    model = LatentNavigationBody(TableGDN(base, True).cuda(),
        cfg['latent_dim'], cfg['capacity'], cfg['horizon'], cfg['topk']).cuda()
    model.load_state_dict(saved['model'])
    model.eval()
    table = RetrievedCreditTable.from_state_dict(saved['table'])
    state_on = saved['recurrent'].cuda().clone()
    state_off = state_on.clone()
    gain = model.read_gain.detach().clone()
    pending, cursor = int(saved['pending']), int(saved['source_cursor'])
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    corpus = np.load(root/'train.npy', mmap_mode='r')
    results = []
    for window in range(2):
        valid = np.flatnonzero(corpus[cursor:cursor+2048] >= offset)[:512]
        if len(valid) != 512:
            raise EOFError('insufficient unseen source bytes')
        response = (corpus[cursor+valid]-offset).astype(np.uint8)
        cursor += int(valid[-1])+1
        raw = np.concatenate((np.array([pending], np.uint8), response))
        _, probs, _ = table.run_credit(raw, max_window=512)
        ids = torch.tensor(raw[:-1].astype(np.int64), device='cuda')[None]
        targets = torch.tensor(raw[1:].astype(np.int64), device='cuda')[None]
        logp = torch.tensor(np.log(np.maximum(probs, FLOOR)), device='cuda')[None]
        model.read_gain.copy_(gain)
        on, state_on, _ = model.predict(ids, state_on, logp)
        model.read_gain.zero_()
        off, state_off, _ = model.predict(ids, state_off, logp)
        results.append(dict(window=window+1, bytes=512,
            response_sha256=hashlib.sha256(response.tobytes()).hexdigest(),
            nll_on=float(F.cross_entropy(on.flatten(0,1), targets.flatten())),
            nll_off=float(F.cross_entropy(off.flatten(0,1), targets.flatten())),
            mean_abs_logit_change=float((on-off).abs().mean()),
            max_abs_logit_change=float((on-off).abs().max()),
            recurrent_l2_difference=float((state_on-state_off).norm())))
        pending = int(raw[-1])
    report = dict(scope='two matched frozen-memory real-text windows; wiring diagnostic only',
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        retained_gain=float(gain), frozen_latent_entries=int(model.memory.occupied.sum()),
        source_end=cursor, fused=fused, windows=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
