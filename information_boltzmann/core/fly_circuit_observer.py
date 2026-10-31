"""Which circuits take part in language, read off the living individual (read-only, never a gradient).

Every tick of the live pass adds each cell's transmitted event (ring slot 0 non-zero) to two counters:
coarse groups (connectome class; DANs split into PAM / PPL1 / other, APL and DPM named; superclass where the
class is empty) and fine groups (cell type; untyped cells by superclass). At each arrival the counts become
per-cell event rates over the word's interval and one row is written with what happened: the word, who said it,
its source, the interval's ticks, and the surprisal of the word under the read standing at its arrival (the brain's
prediction before hearing it). Dopamine at the MBONs (fast and slow, mean over the compartments) is kept per word.

Outputs in out_dir: circuits.jsonl (coarse rates + event), fine_XXXX.npz (float16 [words, types] rates per chunk)
and groups.json (labels and sizes). The observer only counts in the live pass (active flag), so a replay of the
same window never counts twice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def cell_groups(body_ids, annotations_path):
    import pandas as pd
    table = pd.read_feather(annotations_path, columns=['bodyId', 'class', 'superclass', 'type'])
    table = table.drop_duplicates('bodyId').set_index('bodyId')
    rows = table.reindex(np.asarray(body_ids, dtype=np.int64))
    coarse, fine = [], []
    for cls, sup, typ in zip(rows['class'].tolist(), rows['superclass'].tolist(), rows['type'].tolist()):
        cls = cls if isinstance(cls, str) and cls else None
        sup = sup if isinstance(sup, str) and sup else 'unannotated'
        typ = typ if isinstance(typ, str) and typ else None
        if typ is not None and typ.startswith('PAM'):
            label = 'DAN_PAM'
        elif typ is not None and typ.startswith('PPL1'):
            label = 'DAN_PPL1'
        elif cls == 'DAN':
            label = 'DAN_other'
        elif typ in ('APL', 'DPM'):
            label = typ
        elif cls == 'Kenyon_Cell':
            label = 'KC'
        else:
            label = cls or sup
        coarse.append(label)
        fine.append(typ or f'untyped:{sup}')
    return coarse, fine


class CircuitObserver:
    def __init__(self, model, body_ids, annotations_path, out_dir, chunk_words=256):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        coarse, fine = cell_groups(body_ids, annotations_path)
        device = model.output_read.weight.device
        self.coarse_names, coarse_id = np.unique(np.asarray(coarse), return_inverse=True)
        self.fine_names, fine_id = np.unique(np.asarray(fine), return_inverse=True)
        self.coarse_id = torch.as_tensor(coarse_id, device=device)
        self.fine_id = torch.as_tensor(fine_id, device=device)
        self.coarse_size = torch.bincount(self.coarse_id, minlength=len(self.coarse_names)).float()
        self.fine_size = torch.bincount(self.fine_id, minlength=len(self.fine_names)).float()
        mbon = np.flatnonzero(np.asarray(coarse) == 'MBON')
        self.mbon = torch.as_tensor(mbon, device=device)
        self.coarse_counts = torch.zeros(len(self.coarse_names), device=device)
        self.coarse_level = torch.zeros(len(self.coarse_names), device=device)
        self.fine_counts = torch.zeros(len(self.fine_names), device=device)
        self.da_sum = torch.zeros(2, device=device)
        self.ticks = 0
        self.active = False
        self.chunk_words = int(chunk_words)
        self.fine_rows, self.words = [], 0
        self.chunk = len(list(self.out_dir.glob('fine_*.npz')))
        (self.out_dir / 'groups.json').write_text(json.dumps({
            'coarse': self.coarse_names.tolist(), 'coarse_size': self.coarse_size.long().tolist(),
            'fine': self.fine_names.tolist(), 'fine_size': self.fine_size.long().tolist(),
            'event': 'ring slot 0 non-zero (a transmitted spike or graded release) per tick',
            'rate': 'events per cell per tick over the word interval'}, ensure_ascii=False), encoding='utf-8')

    @torch.no_grad()
    def __call__(self, model, state):
        if not self.active:
            return
        release = state.ring[0][0]
        events = (release != 0).float()
        self.coarse_counts.index_add_(0, self.coarse_id, events)
        self.coarse_level.index_add_(0, self.coarse_id, release.abs().float())
        self.fine_counts.index_add_(0, self.fine_id, events)
        dopamine = getattr(state, 'dopamine', None)
        if dopamine is not None and dopamine.numel() and self.mbon.numel() and dopamine.shape[0] >= 2:
            self.da_sum += dopamine[:2].index_select(1, self.mbon).mean(1)
        self.ticks += 1

    @torch.no_grad()
    def end_word(self, model, *, token, who, source, read, extra=None):
        ticks = max(self.ticks, 1)
        coarse = (self.coarse_counts / (self.coarse_size * ticks)).cpu().numpy()
        level = (self.coarse_level / (self.coarse_size * ticks)).cpu().numpy()
        fine = (self.fine_counts / (self.fine_size * ticks)).cpu().numpy().astype(np.float16)
        surprisal = None
        if read is not None:
            logits = model.decoder(model.read_norm(read[-1:].float()))
            surprisal = float(F.cross_entropy(logits, torch.tensor([int(token)], device=logits.device)))
        row = {'word': self.words, 'token': int(token), 'who': who, 'source': source, 'ticks': self.ticks,
               'surprisal': surprisal, 'da_mbon_fast': float(self.da_sum[0] / ticks),
               'da_mbon_slow': float(self.da_sum[1] / ticks),
               'rates': {str(n): float(v) for n, v in zip(self.coarse_names, coarse)},
               # graded cells (APL, photoreceptors) release on every tick: their signal is the release level
               'levels': {str(n): float(v) for n, v in zip(self.coarse_names, level)}}
        if extra:
            row.update(extra)
        with (self.out_dir / 'circuits.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        self.fine_rows.append(fine)
        self.words += 1
        if len(self.fine_rows) >= self.chunk_words:
            self.flush()
        self.coarse_counts.zero_(); self.coarse_level.zero_(); self.fine_counts.zero_(); self.da_sum.zero_()
        self.ticks = 0

    def flush(self):
        if not self.fine_rows:
            return
        np.savez(self.out_dir / f'fine_{self.chunk:04d}.npz', rates=np.stack(self.fine_rows))
        self.chunk += 1
        self.fine_rows = []


def attach_circuit_observer(learner, graph_path, annotations_path, out_dir):
    from . import fly_bptt_learning as fl
    with np.load(graph_path, allow_pickle=False) as graph:
        bodies = graph['neuron_body_ids']
    observer = CircuitObserver(learner.model, bodies, annotations_path, out_dir)
    fl.TICK_OBSERVERS.append(observer)
    learner.circuit_observer = observer
    return observer
