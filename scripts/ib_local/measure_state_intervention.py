"""Paired held-out text interventions; never modifies the training process."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.window import LocalWindow
from scripts.ib_local.sampling import sample_window
from scripts.ib_bpe_window import clock_inputs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sites', type=int, default=4)
    p.add_argument('--zero-only', action='store_true')
    p.add_argument('--no-collision', action='store_true')
    args = p.parse_args()
    if args.zero_only and args.no_collision:
        p.error('Choose one intervention')
    torch.set_num_threads(2)
    device = 'cuda'
    torch.cuda.set_per_process_memory_fraction(.18)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    model = LocalWindow(hidden=cfg['hidden'], particles=cfg['particles']).to(device).eval()
    model.load_state_dict(saved['model'])
    model.requires_grad_(False)
    data = np.load(args.data / 'validation.npy', mmap_mode='r')
    n = cfg['particles']
    results = []
    gen = torch.Generator(device=device)

    def advance(state, previous, target, time, tables, noise):
        _, loss, x, v, _ = model(*state,
            torch.tensor([previous], device=device),
            torch.tensor([target], device=device),
            clock_inputs(time, 1, model.steps, device), noise, tables)
        return (x, v), float(loss)

    for site in range(args.sites):
        # Exclude the first 4224 tokens used for checkpoint selection.
        start = 8192 + site * 4096
        donor = start + 2048
        assert donor + 128 < len(data)
        gen.manual_seed(7100 + site)
        birth = model.core.initialize(torch.tensor([50256], device=device), gen)
        states = {'keep': (birth.x.clone(), birth.v.clone()),
                  'wrong_history': (birth.x.clone(), birth.v.clone())}
        histories = [('keep', start), ('wrong_history', donor)]
        if args.zero_only or args.no_collision:
            del states['wrong_history']
            histories = [('keep', start)]
        rng = np.random.default_rng(8100 + site)
        for t in range(128):
            tables, _ = sample_window(rng, 1, model.steps, n, device)
            noise = torch.randn(model.steps * 4, n, 4, device=device, generator=gen)
            for arm, pos in histories:
                prev = 50256 if t == 0 else int(data[pos + t - 1])
                states[arm], _ = advance(states[arm], prev, int(data[pos+t]), t, tables, noise)
        if args.no_collision:
            states['no_collision'] = tuple(a.clone() for a in states['keep'])
        elif args.zero_only:
            states['zero_reset'] = tuple(torch.zeros_like(a) for a in states['keep'])
            assert all(torch.count_nonzero(a).item() == 0 for a in states['zero_reset'])
        else:
            states['birth_reset'] = (birth.x.clone(), birth.v.clone())
        initial = {arm: {'kinetic': float(.5*v.square().sum(-1).mean()),
                         'potential': float(.5*x.square().sum(-1).mean())}
                   for arm, (x, v) in states.items()}
        losses = {arm: [] for arm in states}
        for t in range(64):
            tables, _ = sample_window(rng, 1, model.steps, n, device)
            noise = torch.randn(model.steps * 4, n, 4, device=device, generator=gen)
            cursor = start + 128 + t
            for arm in states:
                # Empty event layers are the identity collision operator.
                # OU bath, drive, clocks and transport are untouched.
                arm_tables = [[] for _ in tables] if arm == 'no_collision' else tables
                states[arm], loss = advance(states[arm], int(data[cursor-1]),
                    int(data[cursor]), 128+t, arm_tables, noise)
                losses[arm].append(loss)
        results.append(dict(site=site, start=start, donor=donor, initial=initial, losses=losses))
        report = dict(checkpoint_step=saved['step'], trained_tokens=saved['events'],
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            protocol=('128 history, 64 continuation; paired proposals/noise; frozen weights; '
                      + ('collision identity after shared history; T=0.1 unchanged' if args.no_collision
                         else 'all x,v zeroed; clock unchanged' if args.zero_only
                         else 'wrong history NOT energy matched')),
            sites=results, budget_limited=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({'site': site, 'nll': {k: float(np.mean(v)) for k,v in losses.items()}}), flush=True)


if __name__ == '__main__':
    main()
