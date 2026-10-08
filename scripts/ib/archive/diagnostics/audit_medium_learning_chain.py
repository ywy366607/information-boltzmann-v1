"""Inspect task and port credit on real corpus windows, without updating a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.training import quiet_training_chunk
from scripts.ib.train_plastic_conductance import unpack_belief


def credit_summary(names, parameters, task, auxiliary):
    groups = {}
    unused = []
    for name, parameter, task_grad, aux_grad in zip(names, parameters, task, auxiliary):
        group = groups.setdefault(name.split('.')[0],
            dict(parameters=0, connected_parameters=0, task_squared=0.,
                 auxiliary_squared=0., dot=0.))
        group['parameters'] += parameter.numel()
        if task_grad is None and aux_grad is None:
            unused.append(name)
            continue
        group['connected_parameters'] += parameter.numel()
        if task_grad is not None:
            group['task_squared'] += float(task_grad.double().square().sum())
        if aux_grad is not None:
            group['auxiliary_squared'] += float(aux_grad.double().square().sum())
        if task_grad is not None and aux_grad is not None:
            group['dot'] += float((task_grad.double() * aux_grad.double()).sum())
    for group in groups.values():
        task_sq, aux_sq, dot = (group.pop(key) for key in
                               ('task_squared', 'auxiliary_squared', 'dot'))
        group.update(task_norm=math.sqrt(task_sq), auxiliary_norm=math.sqrt(aux_sq),
                     cosine=dot / math.sqrt(task_sq * aux_sq) if task_sq * aux_sq else None,
                     auxiliary_along_task_ratio=dot / task_sq if task_sq else None,
                     total_norm=math.sqrt(max(0., task_sq + aux_sq + 2 * dot)))
    return groups, unused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('Positive sample count required')
    torch.set_num_threads(1)
    # Load once: subsequent publication of last.pt cannot change this snapshot.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    constructor = dict(saved['config']['constructor'])
    constructor.update(medium_execution='native', port_execution='native')
    model = PlasticMediumPorts3D(**constructor).train()
    model.load_state_dict(saved['model'])
    belief = unpack_belief(saved['belief'], 'cpu')
    train = np.load(args.data_dir / 'train.npy', mmap_mode='r')
    length = saved['config']['bptt_chunk_tokens']
    pairs = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names, parameters = zip(*pairs)
    rows = []
    for index in range(args.samples):
        offset = saved['cursor'] + index * length
        targets = torch.from_numpy(np.array(train[offset:offset + length], dtype=np.int64))[None]
        carry = saved['learner']['carry_token'] if index == 0 else int(train[offset - 1])
        ids = torch.cat((targets.new_tensor([[carry]]), targets[:, :-1]), 1)
        loss, evolved, nll = quiet_training_chunk(model, ids, targets, belief,
            event_duration=saved['config']['event_duration'])
        task = torch.autograd.grad(nll, parameters, retain_graph=True, allow_unused=True)
        auxiliary = torch.autograd.grad(loss - nll, parameters, allow_unused=True)
        groups, unused = credit_summary(names, parameters, task, auxiliary)
        bias_only_nll = torch.nn.functional.cross_entropy(
            model.decoder.bias.detach()[None].expand(targets.numel(), -1), targets.flatten())
        rows.append(dict(offset=offset, task_nll=float(nll.detach()),
                         dynamic_bias_only_nll=float(bias_only_nll),
                         gain_over_dynamic_bias=float(bias_only_nll - nll.detach()),
                         write_loss=float((loss - nll).detach()), groups=groups,
                         disconnected_parameters=unused))
        belief = evolved.detach()
    report = dict(purpose='real-corpus credit/wiring audit; no optimizer update or capability claim',
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_step=saved['step'],
                  fresh_training_tokens=saved['cursor'] - 1,
                  actual_events=saved['learner']['events'], optimizer_updates=0,
                  event_duration=saved['config']['event_duration'], samples=rows,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
